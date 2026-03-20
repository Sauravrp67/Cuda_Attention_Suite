#!/usr/bin/env bash
# =============================================================================
# benchmarks/runners/run_ncu.sh  (v2 — fixed kernel targeting + CSV parser)
# Overview:
#   1. Kernel targeting: --kernel-name regex skips PyTorch internal kernels
#      (randn, memset, etc.) and lands on naive_attention_kernel.
#   2. CSV format: ncu --page raw is WIDE (one row per kernel, metrics as
#      columns). Parser reads row[col_name] not row["Metric Name"].
#   3. Removed smsp__warp_issue_stalled_math_throttle — absent on sm_89.
#
# USAGE:
#   chmod +x benchmarks/runners/run_ncu.sh
#   ./benchmarks/runners/run_ncu.sh                      # defaults: naive_v1 N=512
#   ./benchmarks/runners/run_ncu.sh --N 1024 --H 8
#   ./benchmarks/runners/run_ncu.sh --dry-run
#
# PERF EVENT PARANOIA FIX:
#   sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
# =============================================================================

set -euo pipefail

KERNEL="naive_v1"
B=1; H=8; N=512; D=64; DTYPE="float32"
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/benchmarks/reports/ncu"
TARGET_SCRIPT="${SCRIPT_DIR}/_ncu_target.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL="$2"; shift 2 ;;
        --B)      B="$2";      shift 2 ;;
        --H)      H="$2";      shift 2 ;;
        --N)      N="$2";      shift 2 ;;
        --D)      D="$2";      shift 2 ;;
        --dtype)  DTYPE="$2";  shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) awk '/^# ={10,}/ {count++} {print} count==2 {exit}' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Kernel name → regex for --kernel-name filter
#
# ncu matches against the CUDA kernel function name (the C++ symbol, possibly
# demangled). The regex must uniquely identify YOUR kernel and skip all
# PyTorch-internal launches (randn, memset, vectorized_elementwise, etc.)
# ---------------------------------------------------------------------------
declare -A KERNEL_REGEX
KERNEL_REGEX["naive_v1"]="naive_attention_kernel"

if [[ -z "${KERNEL_REGEX[$KERNEL]+_}" ]]; then
    echo "ERROR: Unknown kernel '$KERNEL'. Known: ${!KERNEL_REGEX[*]}" >&2
    exit 1
fi
KERNEL_RE="${KERNEL_REGEX[$KERNEL]}"

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
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
lts__t_bytes_equiv_l1sectordirty_is_rst.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__inst_executed_op_global_ld.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__theoretical_occupancy_pct,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,\
smsp__warp_issue_stalled_not_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STEM="${KERNEL}_B${B}_H${H}_N${N}_D${D}_${DTYPE}"
REP_FILE="${REPORT_DIR}/${STEM}.ncu-rep"
CSV_FILE="${REPORT_DIR}/${STEM}.csv"
SUMMARY_FILE="${REPORT_DIR}/summary_${STEM}_${TIMESTAMP}.txt"
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
    --kernel-name "regex:${KERNEL_RE}"
    --launch-count 1
    --replay-mode kernel
    # --metrics "${METRICS}"
    --set full
    --export "${REP_FILE}"
    --force-overwrite
    python3 "${TARGET_SCRIPT}"
        --kernel "${KERNEL}"
        --B "${B}" --H "${H}" --N "${N}" --D "${D}"
        --dtype "${DTYPE}"
)

NCU_CSV_CMD=(
    "$NCU_BIN"
    --import "${REP_FILE}"
    --csv
    --page raw
)

echo "============================================================"
echo "  CUDA Attention Benchmark — ncu Profile"
echo "============================================================"
echo "  kernel     : ${KERNEL}  (regex: ${KERNEL_RE})"
echo "  shape      : B=${B} H=${H} N=${N} D=${D}"
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
echo "[1/3] Profiling (skipping to '${KERNEL_RE}', then replaying per counter group)..."
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
export _NCU_CSV="${CSV_FILE}"
export _NCU_SUMMARY="${SUMMARY_FILE}"
export _NCU_KERNEL="${KERNEL}"
export _NCU_REP="${REP_FILE}"
export _NCU_B="${B}"
export _NCU_H="${H}"
export _NCU_N="${N}"
export _NCU_D="${D}"
export _NCU_DT="${DTYPE}"

python3 benchmarks/utils/parse_ncu_csv.py

echo ""
echo "============================================================"
echo "  .ncu-rep : ${REP_FILE}"
echo "  CSV      : ${CSV_FILE}"
echo "  Summary  : ${SUMMARY_FILE}"
echo "  GUI      : ncu-ui ${REP_FILE}"
echo "============================================================"