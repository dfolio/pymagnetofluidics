#!/bin/bash
# render_notebooks.sh - Render all PINN verification notebooks using Quarto
# Enhanced with: computation timeline, elapsed times, progress tracking

# Set the base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOKS_DIR="${BASE_DIR}/"

# Output formats (can be customized)
# FORMATS=("html" "pdf" "ipynb")
FORMATS=("html")  # Uncomment to render only HTML

# List of all notebooks to render
NOTEBOOKS=(
    "00_configuration/00_configuration.qmd"
    "01_baseline_flow/01_baseline_training.qmd"
    "01_baseline_flow/01_baseline_verification.qmd"
    "02_particle_flow/02_particle_training.qmd"
    "02_particle_flow/02_particle_verification.qmd"
    "03_resistance_problems/03_resistance_training.qmd"
    "03_resistance_problems/03_resistance_validation.qmd"
    "03_resistance_problems/03_resistance_gates.qmd"
    "04_warm_start/04_warm_start_verification.qmd"
    "05_trajectory/05_free_particle.qmd"
    "06_master_report/06_verification_summary.qmd"
)

# Create output directory
OUTPUT_DIR="${BASE_DIR}/rendered_output"
mkdir -p "${OUTPUT_DIR}"

# Timing and progress variables
START_TIME=$(date +%s)
TOTAL_NOTEBOOKS=${#NOTEBOOKS[@]}
TOTAL_FORMATS=${#FORMATS[@]}
TOTAL_TASKS=$((TOTAL_NOTEBOOKS * TOTAL_FORMATS))
COMPLETED_TASKS=0

# Arrays to store timing data
declare -a NOTEBOOK_START_TIMES
declare -a NOTEBOOK_END_TIMES
declare -a NOTEBOOK_DURATIONS

# Log file
LOG_FILE="${BASE_DIR}/render_log_$(date +%Y%m%d_%H%M%S).txt"

# ==========================================
# FUNCTIONS
# ==========================================

echo_header() {
    echo "=========================================="
    echo "PINN Verification Suite - Notebook Renderer"
    echo "=========================================="
    echo ""
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Base directory: ${BASE_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Formats: ${FORMATS[*]}"
    echo "Total notebooks: ${TOTAL_NOTEBOOKS}"
    echo "Total tasks: ${TOTAL_TASKS}"
    echo "Log file: ${LOG_FILE}"
    echo ""
}

echo_progress() {
    local current=$1
    local total=$2
    local percentage=$((current * 100 / total))

    # Calculate ETA
    if [ $current -gt 0 ]; then
        local elapsed=$(( $(date +%s) - START_TIME ))
        local avg_time_per_task=$((elapsed / current))
        local remaining_tasks=$((total - current))
        local eta=$((remaining_tasks * avg_time_per_task))

        if [ $eta -gt 60 ]; then
            local eta_min=$((eta / 60))
            local eta_sec=$((eta % 60))
            ETA_STR="${eta_min}m ${eta_sec}s"
        else
            ETA_STR="${eta}s"
        fi
    else
        ETA_STR="Calculating..."
    fi

    # Format elapsed time
    local current_elapsed=$(( $(date +%s) - START_TIME ))
    if [ $current_elapsed -gt 3600 ]; then
        local hours=$((current_elapsed / 3600))
        local minutes=$(( (current_elapsed % 3600) / 60 ))
        local seconds=$((current_elapsed % 60))
        ELAPSED_STR="${hours}h ${minutes}m ${seconds}s"
    elif [ $current_elapsed -gt 60 ]; then
        local minutes=$((current_elapsed / 60))
        local seconds=$((current_elapsed % 60))
        ELAPSED_STR="${minutes}m ${seconds}s"
    else
        ELAPSED_STR="${current_elapsed}s"
    fi

    printf "\r[%-3s] %3d%% | Elapsed: %s | ETA: %s" \
        "$(progress_bar $percentage)" \
        "$percentage" \
        "$ELAPSED_STR" \
        "$ETA_STR"
}

progress_bar() {
    local percentage=$1
    local completed=$((percentage * 50 / 100))
    local remaining=$((50 - completed))
    printf "%${completed}s" | tr ' ' '=' | tr - '>'
    printf "%${remaining}s" | tr ' ' ' '
}

echo_notebook_header() {
    local notebook=$1
    local count=$2
    echo ""
    echo "[${count}/${TOTAL_NOTEBOOKS}] ==> ${notebook}"
    echo "   Started at: $(date '+%H:%M:%S')"
}

echo_notebook_footer() {
    local notebook=$1
    local duration=$2
    echo "   Completed at: $(date '+%H:%M:%S')"
    echo "   Duration: $(format_duration $duration)"
    echo ""
}

format_duration() {
    local seconds=$1
    if [ $seconds -gt 3600 ]; then
        local hours=$((seconds / 3600))
        local minutes=$(( (seconds % 3600) / 60 ))
        local secs=$((seconds % 60))
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $seconds -gt 60 ]; then
        local minutes=$((seconds / 60))
        local secs=$((seconds % 60))
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $seconds
    fi
}

check_quarto_installed() {
    if ! command -v quarto &> /dev/null; then
        echo "ERROR: Quarto is not installed or not in PATH"
        echo "Please install Quarto from https://quarto.org/"
        exit 1
    fi
    echo "Quarto version: $(quarto --version 2>&1 | head -n 1)"
}

render_notebook() {
    local notebook_path="$1"
    local notebook_name=$(basename "$notebook_path" .qmd)
    local notebook_dir=$(dirname "$notebook_path")
    local notebook_index=$2

    echo_notebook_header "$notebook_path" "$notebook_index"

    local notebook_start=$(date +%s)
    local all_success=true
    local quarto_extra_args=" --execute-daemon-restart"

    for format in "${FORMATS[@]}"; do
        local output_subdir="${OUTPUT_DIR}/${format}"
        mkdir -p "${output_subdir}"

        local output_file="${output_subdir}/${notebook_name}.${format}"

        echo "   Rendering ${notebook_path} to ${format}..."

        # Render with Quarto
        if [ "$format" == "pdf" ]; then
            quarto render "${NOTEBOOKS_DIR}/${notebook_path}" \
                --to pdf \
                --execute \
                --cache-dir "${BASE_DIR}/.quarto-cache" \
                --log-level warning >> "${LOG_FILE}" 2>&1
        elif [ "$format" == "ipynb" ]; then
            quarto render "${NOTEBOOKS_DIR}/${notebook_path}" \
                --to ipynb \
                --execute \
                --cache-dir "${BASE_DIR}/.quarto-cache" \
                --log-level warning >> "${LOG_FILE}" 2>&1
        elif [ "$format" == "docx" ]; then
            quarto render "${NOTEBOOKS_DIR}/${notebook_path}" \
                --to docx \
                --execute \
                --cache-dir "${BASE_DIR}/.quarto-cache" \
                --log-level warning >> "${LOG_FILE}" 2>&1
        else
            quarto render "${NOTEBOOKS_DIR}/${notebook_path}"\
                --to html \
                --execute \
                --log-level warning >> "${LOG_FILE}" 
                # --output "${output_file}" \
                #  --execute-daemon-restart 
        fi

        if [ $? -eq 0 ]; then
            echo "      ✓ ${format} rendered successfully"
            COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
        else
            echo "      ✗ ${format} failed to render"
            all_success=false
        fi

        echo_progress $COMPLETED_TASKS $TOTAL_TASKS
    done

    local notebook_end=$(date +%s)
    local notebook_duration=$((notebook_end - notebook_start))

    NOTEBOOK_START_TIMES[$notebook_index]=$(date -d @"$notebook_start" '+%H:%M:%S')
    NOTEBOOK_END_TIMES[$notebook_index]=$(date -d @"$notebook_end" '+%H:%M:%S')
    NOTEBOOK_DURATIONS[$notebook_index]=$notebook_duration

    echo_notebook_footer "$notebook_path" "$notebook_duration"

    return $([ "$all_success" = true ] && echo 0 || echo 1)
}

print_summary() {
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))

    echo ""
    echo "=========================================="
    echo "RENDERING SUMMARY"
    echo "=========================================="
    echo ""
    echo "Overall Statistics:"
    echo "  Start time: $(date -d @"$START_TIME" '+%Y-%m-%d %H:%M:%S')"
    echo "  End time:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Total duration: $(format_duration $total_duration)"
    echo "  Tasks completed: ${COMPLETED_TASKS}/${TOTAL_TASKS}"
    echo ""

    echo "Per-Notebook Timeline:"
    echo "  ┌─────────────────────────────────────────────────────────────┐"
    for i in "${!NOTEBOOKS[@]}"; do
        local notebook=${NOTEBOOKS[$i]}
        local start=${NOTEBOOK_START_TIMES[$i]}
        local end=${NOTEBOOK_END_TIMES[$i]}
        local duration=${NOTEBOOK_DURATIONS[$i]}
        local name=$(basename "$notebook" .qmd)

        printf "  │ %-40s %-8s → %-8s (%s) │\n" \
            "$name" \
            "$start" \
            "$end" \
            "$(format_duration $duration)"
    done
    echo "  └─────────────────────────────────────────────────────────────┘"
    echo ""

    # # Find slowest and fastest notebooks
    # local slowest_index=0
    # local fastest_index=0
    # for i in "${!NOTEBOOK_DURATIONS[@]}"; do
    #     if [ ${NOTEBOOK_DURATIONS[$i]} -gt ${NOTEBOOK_DURATIONS[$slowest_index]} ]; then
    #         slowest_index=$i
    #     fi
    #     if [ ${NOTEBOOK_DURATIONS[$i]} -lt ${NOTEBOOK_DURATIONS[$fastest_index]} ]; then
    #         fastest_index=$i
    #     fi
    # done
    # 
    # echo "Performance:"
    # echo "  Fastest: ${NOTEBOOKS[$fastest_index]} ($(format_duration ${NOTEBOOK_DURATIONS[$fastest_index]}))"
    # echo "  Slowest: ${NOTEBOOKS[$slowest_index]} ($(format_duration ${NOTEBOOK_DURATIONS[$slowest_index]}))"
    # echo ""
    # echo "Output directory: ${OUTPUT_DIR}"
    # echo "Log file: ${LOG_FILE}"
    # echo ""
}

# ==========================================
# MAIN EXECUTION
# ==========================================

# Clear screen for better visibility
clear

# Initialize
echo_header

# Check Quarto installation
check_quarto_installed
echo ""

# Start rendering
ALL_SUCCESS=true
for i in "${!NOTEBOOKS[@]}"; do
    if ! render_notebook "${NOTEBOOKS[$i]}" "$((i + 1))"; then
        ALL_SUCCESS=false
    fi
done

# Final progress (100%)
echo_progress $TOTAL_TASKS $TOTAL_TASKS
echo ""

# Print summary
print_summary

# Exit with appropriate code
if [ "$ALL_SUCCESS" = true ]; then
    echo "✓ All notebooks rendered successfully"
    exit 0
else
    echo "✗ Some notebooks failed to render (check log file for details)"
    exit 1
fi
