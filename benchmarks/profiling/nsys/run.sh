#!/usr/bin/env bash
# =============================================================================
# benchmarks/profiling/nsys/run.sh
#
# Nsight Systems launcher for full CUDA/NVTX/OS runtime traces.
# Intended for:
#   - call stack view / CPU-GPU timeline
#   - launch sequencing
#   - trace-level debugging of decode/prefill flows
# =============================================================================

set -euo pipefail

OPERATION="attention"
KERNEL="naive_attention"
B=1; H=8; N=512; D=64; M=128; K=1024; DTYPE="float32"
CAUSAL=false
WARMUP_ITERS=2
ITERS=5
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/benchmarks/reports/nsys"
TARGET_SCRIPT="${SCRIPT_DIR}/target.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

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
        --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '1,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

NSYS_BIN=""
for candidate in nsys /usr/local/cuda/bin/nsys; do
    if command -v "$candidate" &>/dev/null; then NSYS_BIN="$candidate"; break; fi
done
if [[ -z "$NSYS_BIN" ]]; then
    echo "ERROR: nsys not found. Add /usr/local/cuda/bin to PATH."; exit 1
fi

mkdir -p "$REPORT_DIR"
if [[ "${OPERATION}" == "attention" ]]; then
    STEM="${OPERATION}_${KERNEL}_B${B}_H${H}_N${N}_D${D}_${DTYPE}_${TIMESTAMP}"
else
    STEM="${OPERATION}_${KERNEL}_M${M}_K${K}_N${N}_${DTYPE}_${TIMESTAMP}"
fi
OUT_BASE="${REPORT_DIR}/${STEM}"

CMD=(
    "$NSYS_BIN"
    profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --force-overwrite=true
    --output="${OUT_BASE}"
    python3 "${TARGET_SCRIPT}"
        --operation "${OPERATION}"
        --kernel "${KERNEL}"
        --B "${B}" --H "${H}" --N "${N}" --D "${D}" --M "${M}" --K "${K}"
        --dtype "${DTYPE}"
        --warmup-iters "${WARMUP_ITERS}"
        --iters "${ITERS}"
)

if $CAUSAL; then
    CMD+=(--causal)
fi

echo "============================================================"
echo "  CUDA ${OPERATION^} Benchmark — nsys Trace"
echo "============================================================"
echo "  operation  : ${OPERATION}"
echo "  kernel     : ${KERNEL}"
if [[ "${OPERATION}" == "attention" ]]; then
    echo "  shape      : B=${B} H=${H} N=${N} D=${D} causal=${CAUSAL}"
else
    echo "  shape      : M=${M} K=${K} N=${N}"
fi
echo "  dtype      : ${DTYPE}"
echo "  warmup     : ${WARMUP_ITERS}"
echo "  traced     : ${ITERS}"
echo "  output     : ${OUT_BASE}"
echo "============================================================"

if $DRY_RUN; then
    echo "[DRY RUN] nsys command:"
    printf '  %s \\\n' "${CMD[@]}"
    exit 0
fi

"${CMD[@]}"

echo ""
echo "============================================================"
echo "  Trace base : ${OUT_BASE}"
echo "  GUI        : nsys-ui ${OUT_BASE}.nsys-rep"
echo "============================================================"

