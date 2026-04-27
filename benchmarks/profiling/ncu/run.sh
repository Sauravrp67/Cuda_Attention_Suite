#!/usr/bin/env bash
# =============================================================================
# benchmarks/profiling/ncu/run.sh
#
# Nsight Compute launcher for deep kernel profiling:
#   - memory traffic / caches
#   - occupancy limits
#   - issue stall breakdown
#   - instruction mix / pipe activity
#
# The target process is harness-backed and uses benchmark case specs from
# benchmarks/harness/cases.py plus canonical kernel aliases from
# benchmarks/harness/baselines.py.
# =============================================================================

set -euo pipefail

OPERATION="attention"
KERNEL="naive_attention"
B=1; H=8; N=12288; D=64; M=512; K=2048; DTYPE="float32"
CAUSAL=false
DRY_RUN=false
KERNEL_REGEX=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/benchmarks/reports/ncu"
TARGET_SCRIPT="${SCRIPT_DIR}/target.py"
PARSER_SCRIPT="${SCRIPT_DIR}/parse_csv.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --operation) OPERATION="$2"; shift 2 ;;
        --kernel) KERNEL="$2"; shift 2 ;;
        --B)      B="$2";      shift 2 ;;
        --H)      H="$2";      shift 2 ;;
        --N)      N="$2";      shift 2 ;;
        --D)      D="$2";      shift 2 ;;
        --M)      M="$2";      shift 2 ;;
        --K)      K="$2";      shift 2 ;;
        --dtype)  DTYPE="$2";  shift 2 ;;
        --causal) CAUSAL=true; shift ;;
        --kernel-regex) KERNEL_REGEX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) awk '/^# ={10,}/ {count++} {print} count==2 {exit}' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${KERNEL_REGEX}" ]]; then
    KERNEL_REGEX="$(
        PROJECT_ROOT="${PROJECT_ROOT}" python3 - "$OPERATION" "$KERNEL" <<'PY'
import os
import sys
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
src_root = project_root / "src"
for path in (project_root, src_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmarks.profiling.common import profiler_kernel_regex, resolve_kernel_name

operation, kernel = sys.argv[1], sys.argv[2]
try:
    canonical = resolve_kernel_name(operation, kernel)
except Exception:
    canonical = kernel
regex = profiler_kernel_regex(operation, canonical)
print(regex or "")
PY
    )"
fi

if [[ -z "${KERNEL_REGEX}" ]]; then
    echo "ERROR: No default regex is known for operation='${OPERATION}' kernel='${KERNEL}'." >&2
    echo "       Pass --kernel-regex explicitly for PyTorch/internal kernels." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate ncu
# ---------------------------------------------------------------------------
NCU_BIN=""
for candidate in ncu /usr/local/cuda/bin/ncu; do
    if command -v "$candidate" &>/dev/null; then NCU_BIN="$candidate"; break; fi
done
if [[ -z "$NCU_BIN" ]]; then
    echo "ERROR: ncu not found. Add /usr/local/cuda/bin to PATH."; exit 1
fi
echo "ncu : $("$NCU_BIN" --version 2>&1 | head -1)"
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo ""

# ---------------------------------------------------------------------------
# Metrics
# smsp__warp_issue_stalled_math_throttle omitted — not present on sm_89.
# ---------------------------------------------------------------------------
METRICS="\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__m_xbar2l1tex_read_bytes.sum,\
l1tex__m_l1tex2xbar_write_bytes.sum,\
smsp__sass_inst_executed_op_global_ld.sum,\
smsp__sass_inst_executed_op_global_st.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
lts__t_sector_op_read_hit_rate.pct,\
lts__t_sector_op_write_hit_rate.pct,\
derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2,\
derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x4,\
derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
sm__cycles_elapsed.avg,\
smsp__cycles_active.avg,\
smsp__cycles_elapsed.avg.per_second,\
dram__cycles_elapsed.avg.per_second,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_per_block_size,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_blocks,\
launch__waves_per_multiprocessor,\
launch__registers_per_thread,\
smsp__pcsamp_sample_count,\
smsp__pcsamp_warps_issue_stalled_long_scoreboard,\
smsp__pcsamp_warps_issue_stalled_mio_throttle,\
smsp__pcsamp_warps_issue_stalled_no_instructions,\
smsp__pcsamp_warps_issue_stalled_not_selected,\
smsp__pcsamp_warps_issue_stalled_wait,\
smsp__pcsamp_warps_issue_stalled_lg_throttle,\
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle,\
smsp__pcsamp_warps_issue_stalled_tex_throttle,\
smsp__pcsamp_warps_issue_stalled_selected,\
launch__kernel_name,\
launch__grid_size,\
launch__block_size,\
launch__sm_count,\
launch__thread_count"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if [[ "${OPERATION}" == "attention" ]]; then
    STEM="${OPERATION}_${KERNEL}_B${B}_H${H}_N${N}_D${D}_${DTYPE}"
else
    STEM="${OPERATION}_${KERNEL}_M${M}_K${K}_N${N}_${DTYPE}"
fi
REP_FILE="${REPORT_DIR}/${STEM}.ncu-rep"
CSV_FILE="${REPORT_DIR}/${STEM}.csv"
SUMMARY_FILE="${REPORT_DIR}/summary_${STEM}_${TIMESTAMP}.txt"
EXTRACTED_CSV_FILE="${REPORT_DIR}/parsed_${STEM}_${TIMESTAMP}.csv"
mkdir -p "$REPORT_DIR"

# ---------------------------------------------------------------------------
# ncu command
#
# --kernel-name regex:<RE>  skip all kernels until name matches RE
# --launch-count 1          profile only first matching launch
# --target-processes all    follow child processes (Python → CUDA runtime)
# --replay-mode kernel      replay kernel per counter group (not full app)
# ---------------------------------------------------------------------------
NCU_CMD=(
    "$NCU_BIN"
    --target-processes all
    --kernel-name "regex:${KERNEL_REGEX}"
    --launch-count 1
    --replay-mode kernel
    # --metrics "${METRICS}"
    --set full
    --export "${REP_FILE}"
    --force-overwrite
    python3 "${TARGET_SCRIPT}"
        --operation "${OPERATION}"
        --kernel "${KERNEL}"
        --B "${B}" --H "${H}" --N "${N}" --D "${D}" --M "${M}" --K "${K}"
        --dtype "${DTYPE}"
)

if $CAUSAL; then
    NCU_CMD+=(--causal)
fi

NCU_CSV_CMD=(
    "$NCU_BIN"
    --import "${REP_FILE}"
    --csv
    --page raw
)

echo "============================================================"
echo "  CUDA ${OPERATION^} Benchmark — ncu Profile"
echo "============================================================"
echo "  operation  : ${OPERATION}"
echo "  kernel     : ${KERNEL}  (regex: ${KERNEL_REGEX})"
if [[ "${OPERATION}" == "attention" ]]; then
    echo "  shape      : B=${B} H=${H} N=${N} D=${D} causal=${CAUSAL}"
else
    echo "  shape      : M=${M} K=${K} N=${N}"
fi
echo "  dtype      : ${DTYPE}"
echo "  .ncu-rep   : ${REP_FILE}"
echo "============================================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] ncu command:"
    printf '  %s \\\n' "${NCU_CMD[@]}"
    exit 0
fi

# Perf paranoia check
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)
if [[ "$PARANOID" -gt 2 ]]; then
    echo "WARNING: perf_event_paranoid=${PARANOID} — ncu may fail without root."
    echo "         Fix: sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'"
    echo ""
fi

# Step 1: Profile
echo "[1/3] Profiling (skipping to '${KERNEL_REGEX}', then replaying per counter group)..."
echo "      Expected: 30-90s for N=512."
echo ""
"${NCU_CMD[@]}"
echo ""

# Step 2: Export CSV
echo "[2/3] Exporting CSV..."
"${NCU_CSV_CMD[@]}" > "${CSV_FILE}"
echo "      Saved: ${CSV_FILE}"
echo ""

echo "[3/3] Parsing wide-format CSV and generating summary..."

python3 - << 'PARSE_EOF'
import csv, sys, os

# Variables injected by the shell heredoc
CSV_FILE     = os.environ.get("_NCU_CSV",     "")
SUMMARY_FILE = os.environ.get("_NCU_SUMMARY", "")
KERNEL_NAME  = os.environ.get("_NCU_KERNEL",  "")
REP_FILE     = os.environ.get("_NCU_REP",     "")
B  = int(os.environ.get("_NCU_B",  "1"))
H  = int(os.environ.get("_NCU_H",  "8"))
N  = int(os.environ.get("_NCU_N",  "512"))
D  = int(os.environ.get("_NCU_D",  "64"))
DT = os.environ.get("_NCU_DT", "float32")
PARSE_EOF

# Use a second, cleaner Python block with env vars properly set
python3 "${PARSER_SCRIPT}" \
    --operation "${OPERATION}" \
    --csv "${CSV_FILE}" \
    --summary "${SUMMARY_FILE}" \
    --parsed-csv "${EXTRACTED_CSV_FILE}" \
    --rep "${REP_FILE}" \
    --kernel "${KERNEL}" \
    --dtype "${DTYPE}" \
    --B "${B}" --H "${H}" --N "${N}" --D "${D}" --M "${M}" --K "${K}"

echo ""
echo "============================================================"
echo "  .ncu-rep : ${REP_FILE}"
echo "  CSV      : ${CSV_FILE}"
echo "  Summary  : ${SUMMARY_FILE}"
echo "  Parsed   : ${EXTRACTED_CSV_FILE}"
echo "  GUI      : ncu-ui ${REP_FILE}"
echo "============================================================"
