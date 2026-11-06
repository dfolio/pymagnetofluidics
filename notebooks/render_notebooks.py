#!/usr/bin/python
"""Automated, robust rendering orchestrator for PINN Quarto verification notebooks.

Provides dependency-aware rendering, incremental compilation (mtime check),
CUDA device diagnostics, automatic Quarto daemon restarts, progress tracking,
and CLI argument management.

Example:
    Render only modified notebooks to HTML:
        $ python render_notebooks.py

    Force re-rendering of all notebooks to PDF and HTML with verbose logging:
        $ python render_notebooks.py --force --formats html pdf --verbose
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import platform

# Canonical sequence of verification notebooks in dependency order
DEFAULT_NOTEBOOKS: tuple[str, ...] = (
    "00_configuration/00_configuration.qmd",
    "01_baseline_flow/01_baseline_training.qmd",
    "01_baseline_flow/01_baseline_verification.qmd",
    "02_particle_flow/02_particle_training.qmd",
    "02_particle_flow/02_particle_verification.qmd",
    "03_resistance_problems/03_resistance_training.qmd",
    "03_resistance_problems/03_resistance_validation.qmd",
    "03_resistance_problems/03_resistance_gates.qmd",
    "04_warm_start/04_warm_start_verification.qmd",
    "05_trajectory/05_free_particle.qmd",
    "06_master_report/06_verification_summary.qmd",
)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"html", "pdf", "ipynb", "docx"})


@dataclass(frozen=True)
class RenderTask:
    """Immutable specification for an individual notebook format rendering task.

    Args:
        - `notebook_path`: Absolute path to the source `.qmd` file.
        - `output_format`: Target document format ('html', 'pdf', 'ipynb', etc.).
        - `output_path`: Destination artifact file path.
        - `restart_daemon`: Whether to append `--execute-daemon-restart` to Quarto.
        - `is_up_to_date`: Whether the artifact is already newer than the source.
        - `parameters`: Key-value parameter overrides passed to Quarto via `-P`.
    """

    notebook_path: Path
    output_format: str
    output_path: Path
    restart_daemon: bool
    is_up_to_date: bool
    # Added parameters mapping to support Quarto parameterized reporting (-P key:val)
    parameters: tuple[tuple[str, str | int | float | bool], ...] = ()


@dataclass(frozen=True)
class TaskResult:
    """Immutable execution outcome of an executed or skipped render task.

    Args:
        task: Associated `RenderTask`.
        success: Whether rendering succeeded or was up-to-date.
        skipped: Whether execution was bypassed because the target was fresh.
        duration_seconds: Elapsed wall-clock time in seconds.
        error_message: Error log or diagnostic details if failed.
    """

    task: RenderTask
    success: bool
    skipped: bool
    duration_seconds: float
    error_message: str | None = None


def format_duration(seconds: float) -> str:
    """Format an elapsed duration in seconds into a human-readable string.

    Args:
        seconds: Elapsed wall-clock time in seconds.

    Returns:
        Formatted duration string formatted as `Xh Ym Zs`, `Xm Ys`, or `Xs`.
    """
    total_sec = max(0, int(round(seconds)))
    hours, remainder = divmod(total_sec, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{total_sec}s"


def is_target_up_to_date(source_path: Path, target_path: Path) -> bool:
    """Check whether a target artifact exists and is strictly newer than its source.

    Args:
        source_path: Source file path.
        target_path: Generated output file path.

    Returns:
        True if target exists and its last-modified timestamp is greater than
        or equal to the source file; False otherwise.
    """
    if not source_path.exists() or not target_path.exists():
        return False
    return target_path.stat().st_mtime >= source_path.stat().st_mtime


def probe_cuda_environment() -> str:
    """Probe the host environment for CUDA hardware and Nvidia GPU availability.

    Returns:
        Descriptive string detailing active GPU hardware, or notice of CPU fallback.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return (
                f"CUDA available: {gpu_name} ({device_count} device(s), "
                f"{vram_gb:.2f} GB total memory)"
            )
        return "CUDA unavailable: PyTorch running on host CPU."
    except ImportError:
        return "PyTorch not detected in active Python environment."


def check_quarto_executable() -> str:
    """Verify that the `quarto` command-line utility is installed and accessible.

    Returns:
        Quarto version string.

    Raises:
        FileNotFoundError: If `quarto` is not found on the system PATH.
        RuntimeError: If `quarto --version` fails to execute.
    """
    quarto_path = shutil.which("quarto")
    if quarto_path is None:
        raise FileNotFoundError(
            "Quarto executable not found in PATH. Install Quarto from https://quarto.org/."
        )
    try:
        result = subprocess.run(
            [quarto_path, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.SubprocessError as error:
        raise RuntimeError(f"Failed to query Quarto version: {error}") from error


# CHANGED: Removed unsupported `--cache-dir` CLI flag and stripped `--output`/`--output-dir`
# to execute locally within the notebook's directory, avoiding Deno's asset resolution bug.
def build_quarto_command(
    task: RenderTask,
    log_level: str,
    embed_resources: bool = True,
) -> list[str]:
    """Assemble the CLI command list to invoke Quarto safely within the notebook directory.

    Args:
        task: Specification of the target render task.
        log_level: Quarto log level flag ('info', 'warning', 'error').
        embed_resources: Whether to produce self-contained HTML (embedding JS, CSS, MathJax).

    Returns:
        List of command tokens passed to `subprocess.run`.
    """
    cmd = [
        "quarto",
        "render",
        task.notebook_path.name,
        "--to",
        task.output_format,
        "--execute",
        "--log-level",
        log_level,
    ]
    if task.output_format == "html" and embed_resources:
        cmd.append("--embed-resources")
    if task.restart_daemon:
        cmd.append("--execute-daemon-restart")
    # Format parameter values to canonical lowercase strings for booleans
    for key, value in task.parameters:
        val_str = str(value).lower() if isinstance(value, bool) else str(value)
        cmd.extend(["-P", f"{key}:{val_str}"])
    return cmd


def stream_process_to_log(
    cmd: list[str],
    cwd: Path,
    log_file: Path,
    verbose: bool,
    task_name: str,
) -> tuple[int, str]:
    r"""Execute a subprocess streaming stdout and stderr directly to a log file.

    Writes metadata headers to `log_file` and flushes to disk prior to process launch.
    Emulates shell redirection `>> "${LOG_FILE}" 2>&1` via direct file-descriptor
    redirection, or multiplexes line-by-line to `sys.stdout` and disk when `verbose=True`.

    Args:
        cmd: Command arguments passed to the operating system.
        cwd: Working directory where the command is executed.
        log_file: Target log file path.
        verbose: If True, streams output simultaneously to `sys.stdout` and disk.
        task_name: Human-readable task identifier for header metadata.

    Returns:
        tuple[int, str]: A pair containing:
            * `exit_code`: Subprocess exit code (0 for success).
            * `error_summary`: Captured trailing error lines if non-zero exit code.
    """
    timestamp_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd_str = " ".join(cmd)

    with log_file.open("a", encoding="utf-8") as log_handle:
        # CHANGED: Write and flush metadata header block before subprocess execution begins
        log_handle.write(f"\n{'=' * 72}\n")
        log_handle.write(f"Task:        {task_name}\n")
        log_handle.write(f"Start Time:  {timestamp_start}\n")
        log_handle.write(f"Working Dir: {cwd}\n")
        log_handle.write(f"Command:     {cmd_str}\n")
        log_handle.write(f"{'-' * 72}\n")
        log_handle.flush()

        error_lines: list[str] = []

        if verbose:
            # CHANGED: Line-buffered multiplexer streaming live to console and log file
            with subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                if proc.stdout is not None:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_handle.write(line)
                        log_handle.flush()
                        error_lines.append(line)
                        if len(error_lines) > 20:
                            error_lines.pop(0)

                proc.wait()
                exit_code = proc.returncode
        else:
            # CHANGED: Direct kernel-level file descriptor redirection (zero Python memory overhead)
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            exit_code = proc.returncode

        timestamp_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"\n{'-' * 72}\n")
        log_handle.write(f"Exit Code:   {exit_code}\n")
        log_handle.write(f"End Time:    {timestamp_end}\n")
        log_handle.write(f"{'=' * 72}\n\n")
        log_handle.flush()

    error_summary = "".join(error_lines).strip() if error_lines else f"Exit code {exit_code}"
    return exit_code, error_summary


# CHANGED: Execute Quarto with cwd=task.notebook_path.parent so Deno resolves intermediate
# assets (like quarto.js) in-place without path errors, then atomically relocate outputs.
def execute_task(
    task: RenderTask,
    log_file: Path,
    verbose: bool,
    embed_resources: bool = True,
) -> TaskResult:
    """Execute a single Quarto render task inside the notebook's parent directory.

    Args:
        task: Target render task specification.
        log_file: Path to master execution log file.
        verbose: Whether to stream full stdout/stderr to console.
        embed_resources: Whether to embed assets directly in HTML files.

    Returns:
        TaskResult summarizing duration, exit status, and diagnostics.
    """
    if task.is_up_to_date:
        return TaskResult(task=task, success=True, skipped=True, duration_seconds=0.0)

    notebook_dir = task.notebook_path.parent
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    log_level = "info" if verbose else "warning"
    cmd = build_quarto_command(task, log_level, embed_resources)
    task_desc = f"{task.notebook_path.name} -> {task.output_format}"

    start_time = time.perf_counter()
    try:
        exit_code, error_summary = stream_process_to_log(
            cmd=cmd,
            cwd=notebook_dir,
            log_file=log_file,
            verbose=verbose,
            task_name=task_desc,
        )
        duration = time.perf_counter() - start_time
        success = exit_code == 0

        # Relocate generated artifact and companion files to destination directory
        if success:
            local_artifact = notebook_dir / f"{task.notebook_path.stem}.{task.output_format}"
            if local_artifact.exists():
                shutil.move(str(local_artifact), str(task.output_path))

            local_assets_dir = notebook_dir / f"{task.notebook_path.stem}_files"
            dest_assets_dir = task.output_path.parent / f"{task.notebook_path.stem}_files"
            if local_assets_dir.exists():
                if dest_assets_dir.exists():
                    shutil.rmtree(dest_assets_dir)
                shutil.move(str(local_assets_dir), str(dest_assets_dir))

        return TaskResult(
            task=task,
            success=success,
            skipped=False,
            duration_seconds=duration,
            error_message=None if success else error_summary,
        )
    except OSError as err:
        duration = time.perf_counter() - start_time
        return TaskResult(
            task=task,
            success=False,
            skipped=False,
            duration_seconds=duration,
            error_message=str(err),
        )


def parse_cli_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        argparse.Namespace holding validated CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Render PINN verification notebooks using Quarto with dependency caching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--formats",
        nargs="+",
        default=["html"],
        choices=sorted(SUPPORTED_FORMATS),
        help="Target document formats to render.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("rendered_output"),
        help="Directory to save rendered output documents.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-rendering even if output files are newer than source notebooks.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of PINN models from scratch by passing `-P retrain:true` to Quarto.",
    )
    parser.add_argument(
        "-n",
        "--notebooks",
        nargs="+",
        help="Optional subset of notebook paths or filenames to render.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display planned render actions and fresh/stale statuses without executing.",
    )
    # CHANGED: Replaced invalid `--cache-dir` with `--no-embed-resources` option
    parser.add_argument(
        "--no-embed-resources",
        action="store_true",
        help="Do not embed assets inside HTML (keeps external `<stem>_files/` directory).",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable desktop text notification upon completion.",
    )
    parser.add_argument(
        "--sound",
        action="store_true",
        help="Play an audio chime or alert upon completion.",
    )
    parser.add_argument(
        "--sound-file",
        type=Path,
        default=None,
        help="Optional path to an audio file (.wav, .mp3) to play instead of the system chime.",
    )
    return parser.parse_args()


# CHANGED: Added platform-independent desktop notification dispatcher
def notify_desktop(title: str, message: str) -> bool:
    """Send a platform-native desktop notification using standard OS utilities.

    Supports Linux (via `notify-send`), macOS (via `osascript`), and Windows
    (via PowerShell Windows Forms balloon tips), with graceful fallback in
    headless and SSH environments.

    Args:
        title: Notification header title.
        message: Notification body content.

    Returns:
        bool: True if the notification was dispatched successfully, False otherwise.
    """
    system_name = platform.system().lower()
    clean_title = title.replace('"', '\\"')
    clean_message = message.replace('"', '\\"')

    try:
        if "linux" in system_name:
            notify_send = shutil.which("notify-send")
            if notify_send is not None:
                subprocess.run(
                    [notify_send, clean_title, clean_message],
                    check=False,
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True

        elif "darwin" in system_name:
            osascript = shutil.which("osascript")
            if osascript is not None:
                script = f'display notification "{clean_message}" with title "{clean_title}"'
                subprocess.run(
                    [osascript, "-e", script],
                    check=False,
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True

        elif "windows" in system_name:
            powershell = shutil.which("powershell")
            if powershell is not None:
                ps_command = (
                    f'[reflection.assembly]::loadwithpartialname("System.Windows.Forms") | Out-Null;'
                    f"$notify = New-Object System.Windows.Forms.NotifyIcon;"
                    f"$notify.Icon = [System.Drawing.SystemIcons]::Information;"
                    f"$notify.Visible = $True;"
                    f'$notify.ShowBalloonTip(5000, "{clean_title}", "{clean_message}", [System.Windows.Forms.ToolTipIcon]::Info);'
                )
                subprocess.run(
                    [powershell, "-NoProfile", "-Command", ps_command],
                    check=False,
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True

    except (subprocess.SubprocessError, OSError):
        pass

    return False


# CHANGED: Added platform-independent audio alert / song player
def play_alert_sound(sound_file: Path | None = None) -> bool:
    """Play a system notification chime or custom audio file across platforms.

    Supports custom audio files (.wav, .mp3, .ogg) or default system chimes on
    Linux (`paplay`, `pw-play`, `aplay`), macOS (`afplay`), and Windows (`winsound`),
    with a terminal bell (`\\a`) fallback.

    Args:
        sound_file: Optional path to an audio file (.wav, .mp3, etc.). If None,
            plays the platform's default system completion sound.

    Returns:
        bool: True if audio playback succeeded, False otherwise.
    """
    system_name = platform.system().lower()

    try:
        # Path A: Custom user-provided song/audio file
        if sound_file is not None and sound_file.exists():
            sound_str = str(sound_file.resolve())

            if "darwin" in system_name:
                afplay = shutil.which("afplay")
                if afplay is not None:
                    subprocess.run([afplay, sound_str], check=False, timeout=30)
                    return True

            elif "linux" in system_name:
                for player in ("paplay", "pw-play", "aplay", "ffplay"):
                    player_bin = shutil.which(player)
                    if player_bin is not None:
                        cmd = [player_bin, sound_str] if player != "ffplay" else [player_bin, "-nodisp", "-autoexit", sound_str]
                        subprocess.run(cmd, check=False, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True

            elif "windows" in system_name:
                try:
                    import winsound  # type: ignore
                    if sound_file.suffix.lower() == ".wav":
                        winsound.PlaySound(sound_str, winsound.SND_FILENAME)
                        return True
                except ImportError:
                    pass

        # Path B: Standard system completion chime
        if "darwin" in system_name:
            mac_chime = Path("/System/Library/Sounds/Glass.aiff")
            afplay = shutil.which("afplay")
            if afplay is not None and mac_chime.exists():
                subprocess.run([afplay, str(mac_chime)], check=False, timeout=5)
                return True
            osascript = shutil.which("osascript")
            if osascript is not None:
                subprocess.run([osascript, "-e", "beep 1"], check=False, timeout=5)
                return True

        elif "windows" in system_name:
            try:
                import winsound  # type: ignore
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
                return True
            except ImportError:
                pass

        elif "linux" in system_name:
            # Common Freedesktop sound theme path
            linux_chime = Path("/usr/share/sounds/freedesktop/stereo/complete.oga")
            for player in ("paplay", "pw-play", "canberra-gtk-play"):
                player_bin = shutil.which(player)
                if player_bin is not None:
                    cmd = [player_bin, str(linux_chime)] if player != "canberra-gtk-play" else [player_bin, "-i", "complete"]
                    if player == "canberra-gtk-play" or linux_chime.exists():
                        subprocess.run(cmd, check=False, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True

    except (subprocess.SubprocessError, OSError):
        pass

    # Universal fallback: Standard ASCII terminal bell
    sys.stdout.write("\a")
    sys.stdout.flush()
    return True


# CHANGED: Added high-level completion notifier combining text and audio
def notify_completion(
    success: bool,
    total_tasks: int,
    failed_tasks: int,
    duration_str: str,
    enable_notification: bool = True,
    enable_sound: bool = False,
    sound_file: Path | None = None,
) -> None:
    """Notify the user of execution completion via desktop toast and audio chime.

    Args:
        success: Whether all render tasks completed without errors.
        total_tasks: Total number of tasks processed.
        failed_tasks: Number of tasks that failed.
        duration_str: Formatted execution duration string.
        enable_notification: Whether to trigger a desktop text notification.
        enable_sound: Whether to trigger an audio chime or song.
        sound_file: Optional path to a custom audio file.
    """
    status_label = "Succeeded" if success else "Failed"
    title = f"PINN Verification Suite: {status_label}"
    message = (
        f"Completed {total_tasks - failed_tasks}/{total_tasks} tasks in {duration_str}."
        if success
        else f"{failed_tasks}/{total_tasks} task(s) failed in {duration_str}."
    )

    if enable_notification:
        notify_desktop(title, message)

    if enable_sound:
        play_alert_sound(sound_file)


def main() -> int:
    """Main CLI execution entry point.

    Returns:
        0 on complete success, 1 on any render task failure.
    """
    args = parse_cli_arguments()
    base_dir = Path.cwd()
    output_dir = args.output_dir.resolve()
    cache_dir = None  # Not used with the new option
    embed_res = not args.no_embed_resources
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"render_log_{timestamp}.txt"

    print("=" * 70)
    print("PINN Verification Suite - High-Performance Notebook Renderer")
    print("=" * 70)
    print(f"Timestamp:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Directory:   {base_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Formats:          {', '.join(args.formats)}")
    print(f"Hardware status:  {probe_cuda_environment()}")

    try:
        quarto_ver = check_quarto_executable()
        print(f"Quarto version:   {quarto_ver}")
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # Filter selected notebooks
    raw_notebooks = DEFAULT_NOTEBOOKS if not args.notebooks else args.notebooks
    selected_notebooks = [
        (base_dir / nb).resolve() if not Path(nb).is_absolute() else Path(nb).resolve()
        for nb in raw_notebooks
    ]
    suffix = "r_weighted"
    # Generate plan
    tasks: list[RenderTask] = []
    first_render = True
    for nb_path in selected_notebooks:
        if not nb_path.exists():
            print(f"WARNING: Source notebook not found: {nb_path}", file=sys.stderr)
            continue
        is_training_notebook = "training" in nb_path.stem.lower()
        task_parameters = (("retrain", True),) if args.retrain else ()

        for fmt in args.formats:
            out_file = output_dir / fmt / f"{nb_path.stem}{suffix}.{fmt}"
            # Force re-render if global --force is set OR if --retrain is active on a training notebook
            force_execution = args.force or (args.retrain and is_training_notebook)
            up_to_date = False if force_execution else is_target_up_to_date(nb_path,
                                                                            out_file)
            restart_daemon = first_render and not up_to_date
            if restart_daemon:
                first_render = False
            tasks.append(
                RenderTask(
                    notebook_path=nb_path,
                    output_format=fmt,
                    output_path=out_file,
                    restart_daemon=restart_daemon,
                    is_up_to_date=up_to_date,
                    parameters=task_parameters
                )
            )

    if not tasks:
        print("No valid notebooks found to process.")
        return 0

    to_render_count = sum(1 for t in tasks if not t.is_up_to_date)
    print(
        f"Total tasks:      {len(tasks)} ({to_render_count} stale/forced, {len(tasks) - to_render_count} fresh)")
    print(f"Log file:         {log_file}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Planned tasks:")
        for t in tasks:
            status = "UP-TO-DATE (skip)" if t.is_up_to_date else "STALE (will render)"
            daemon_note = " [daemon-restart]" if t.restart_daemon else ""
            print(
                f"  * {t.notebook_path.name} -> {t.output_format}: {status}{daemon_note}")
        return 0

    results: list[TaskResult] = []
    overall_start = time.perf_counter()

    for idx, task in enumerate(tasks, start=1):
        rel_name = task.notebook_path.name
        print(f"\n[{idx}/{len(tasks)}] {rel_name} ({task.output_format})")
        if task.is_up_to_date:
            print("   -> Output is fresh; skipping render.")
            results.append(
                TaskResult(task=task, success=True, skipped=True, duration_seconds=0.0))
            continue

        print(f"   -> Rendering to {task.output_format}...")
        res = execute_task(task, log_file, args.verbose, embed_resources=embed_res)
        results.append(res)

        if res.success:
            print(f"   -> Completed in {format_duration(res.duration_seconds)}")
        else:
            print(f"   -> FAILED: {res.error_message}", file=sys.stderr)

    total_duration = time.perf_counter() - overall_start

    # Final Summary Table
    print("\n" + "=" * 70)
    print("RENDERING SUMMARY")
    print("=" * 70)
    print(f"Total Elapsed Time: {format_duration(total_duration)}")
    passed = sum(1 for r in results if r.success and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.success)
    print(
        f"Tasks: {len(results)} total | {passed} rendered | {skipped} up-to-date | {failed} failed\n")

    print(f"{'Notebook':<40} {'Format':<8} {'Status':<12} {'Duration':<10}")
    print("-" * 72)
    for res in results:
        status_str = "Passed" if res.success and not res.skipped else (
            "Cached" if res.skipped else "FAILED")
        dur_str = format_duration(res.duration_seconds) if not res.skipped else "-"
        print(
            f"{res.task.notebook_path.name:<40} {res.task.output_format:<8} {status_str:<12} {dur_str:<10}")
    print("-" * 72)

    notify_completion(
        success=(failed == 0),
        total_tasks=len(results),
        failed_tasks=failed,
        duration_str=format_duration(total_duration),
        enable_notification=not args.no_notify,
        enable_sound=args.sound,
        sound_file=args.sound_file,
    )
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
