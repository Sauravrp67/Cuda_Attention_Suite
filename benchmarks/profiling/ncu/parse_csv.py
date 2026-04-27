"""
benchmarks/profiling/ncu/parse_csv.py

Parses ncu CSV exports from:
  --metrics <specific>   → wide format  (one row per kernel, metrics as columns)
  --set full             → wide format  (same, but with ALL columns present)

Both produce wide format when --page raw is passed (which benchmarks/profiling/ncu/run.sh does).
The only difference is which columns are present.

sm_89 (Ada Lovelace / RTX 4050) specific metric names are used.
Key differences from older architectures:
  - smsp__sass_thread_inst_executed_op_f{add,mul,fma}_pred_on.sum  → NOT present
    Use derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2 instead
  - smsp__warp_issue_stalled_*_per_warp_active.pct                 → NOT present
    Use smsp__pcsamp_warps_issue_stalled_* sample counts instead
  - lts__t_bytes_equiv_l1sectordirty_is_rst.sum                    → NOT present on sm_89
  - smsp__inst_executed_op_global_ld.sum                           → NOT present
    Use smsp__sass_inst_executed_op_global_ld.sum instead
  - launch__theoretical_occupancy_pct                              → NOT present
    Use launch__occupancy_per_block_size instead
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a raw Nsight Compute CSV export into summary text and a flat CSV."
    )
    parser.add_argument("--operation", default=os.environ.get("_NCU_OPERATION", "attention"))
    parser.add_argument("--csv", dest="csv_file", default=os.environ.get("_NCU_CSV"))
    parser.add_argument("--summary", dest="summary_file", default=os.environ.get("_NCU_SUMMARY"))
    parser.add_argument("--parsed-csv", dest="parsed_csv_file", default=os.environ.get("_NCU_PARSED_CSV"))
    parser.add_argument("--rep", dest="rep_file", default=os.environ.get("_NCU_REP"))
    parser.add_argument("--kernel", dest="kernel_name", default=os.environ.get("_NCU_KERNEL", "unknown"))
    parser.add_argument("--B", type=int, default=int(os.environ.get("_NCU_B", "1")))
    parser.add_argument("--H", type=int, default=int(os.environ.get("_NCU_H", "8")))
    parser.add_argument("--N", type=int, default=int(os.environ.get("_NCU_N", "512")))
    parser.add_argument("--D", type=int, default=int(os.environ.get("_NCU_D", "64")))
    parser.add_argument("--M", type=int, default=int(os.environ.get("_NCU_M", "128")))
    parser.add_argument("--K", type=int, default=int(os.environ.get("_NCU_K", "1024")))
    parser.add_argument("--dtype", dest="dtype_name", default=os.environ.get("_NCU_DT", "float32"))
    return parser


args = _parser().parse_args()
csv_file = args.csv_file
summary_file = args.summary_file
rep_file = args.rep_file
parsed_csv_file = args.parsed_csv_file
kernel_name = args.kernel_name
operation = args.operation
B = args.B
H = args.H
N = args.N
D = args.D
M = args.M
K = args.K
DT = args.dtype_name

if not csv_file or not summary_file or not rep_file:
    print("ERROR: --csv, --summary, and --rep are required.")
    sys.exit(1)
if not parsed_csv_file:
    parsed_csv_file = str(Path(summary_file).with_suffix(".csv"))

# ---------------------------------------------------------------------------
# Metric name constants
# All verified against --set full CSV column dump on sm_89
# (RTX 4050 Laptop, Ada Lovelace)
# ---------------------------------------------------------------------------

# --- Memory ---
M_DRAM_READ             = "dram__bytes_read.sum"
M_DRAM_WRITE            = "dram__bytes_write.sum"
M_L1_GLOBAL_LD_SECTORS  = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
M_L1_GLOBAL_ST_SECTORS  = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
M_L1_TO_L2_READ         = "l1tex__m_xbar2l1tex_read_bytes.sum"
M_L1_TO_L2_WRITE        = "l1tex__m_l1tex2xbar_write_bytes.sum"
M_GLOBAL_LD_INST        = "smsp__sass_inst_executed_op_global_ld.sum"
M_GLOBAL_ST_INST        = "smsp__sass_inst_executed_op_global_st.sum"

# --- Cache ---
M_L1_HIT                = "l1tex__t_sector_hit_rate.pct"
M_L2_HIT                = "lts__t_sector_hit_rate.pct"
M_L2_HIT_READ           = "lts__t_sector_op_read_hit_rate.pct"
M_L2_HIT_WRITE          = "lts__t_sector_op_write_hit_rate.pct"

# --- Compute ---
# sm_89: raw .sum counts for fadd/fmul/ffma are NOT exported at all
# Use derived metrics — ncu pre-multiplies FMA by 2, HFMA by 4
M_FFMA_X2               = "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2"
M_HFMA_X4               = "derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x4"
M_DFMA_X2               = "derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2"
M_FMA_PIPE              = "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"


#----SOL--------
M_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_MEM_THROUGHPUT = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
M_L1_THROUGHPUT = "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
M_L2_THROUGHPUT = "lts__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM_THROUGHPUT = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"


#-----Timing-----
M_DURATION_MS = "gpu__time_duration.sum"
M_ELAPSED_CYCLES = "sm__cycles_elapsed.avg"
M_ACTIVE_CYCLES = "smsp__cycles_active.avg"
M_SM_FREQ_HZ = "smsp__cycles_elapsed.avg.per_second"
M_DRAM_FREQ_HZ = "dram__cycles_elapsed.avg.per_second"

# --- Occupancy ---
M_OCC_ACHIEVED          = "sm__warps_active.avg.pct_of_peak_sustained_active"
M_OCC_PER_BLOCK         = "launch__occupancy_per_block_size"
M_OCC_LIM_REGS          = "launch__occupancy_limit_registers"
M_OCC_LIM_SMEM          = "launch__occupancy_limit_shared_mem"
M_OCC_LIM_WARPS         = "launch__occupancy_limit_warps"
M_OCC_LIM_BLOCKS        = "launch__occupancy_limit_blocks"
M_WAVES_PER_SM          = "launch__waves_per_multiprocessor"
M_REGS_PER_THREAD       = "launch__registers_per_thread"

# --- Warp stalls (sm_89: pcsamp sample counts, NOT per_warp_active.pct) ---
# Convert to %: divide each count by M_PCSAMP_TOTAL
M_PCSAMP_TOTAL          = "smsp__pcsamp_sample_count"
M_STALL_LONG_SB         = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
M_STALL_MIO             = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
M_STALL_NO_INST         = "smsp__pcsamp_warps_issue_stalled_no_instructions"
M_STALL_NOT_SELECTED    = "smsp__pcsamp_warps_issue_stalled_not_selected"
M_STALL_WAIT            = "smsp__pcsamp_warps_issue_stalled_wait"
M_STALL_LG_THROTTLE     = "smsp__pcsamp_warps_issue_stalled_lg_throttle"
M_STALL_MATH_THROTTLE   = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"
M_STALL_TEX_THROTTLE    = "smsp__pcsamp_warps_issue_stalled_tex_throttle"
M_STALL_SELECTED        = "smsp__pcsamp_warps_issue_stalled_selected"

# --- Launch config ---
M_KERN_NAME             = "launch__kernel_name"
M_GRID_SIZE             = "launch__grid_size"
M_BLOCK_SIZE            = "launch__block_size"
M_SM_COUNT              = "launch__sm_count"
M_THREAD_CT             = "launch__thread_count"

# ---------------------------------------------------------------------------
# Read CSV — always wide format when --page raw is passed
# Row 0 = column headers (metric names)
# Row 1 = units
# Row 2+ = one row per profiled kernel instance
# ---------------------------------------------------------------------------
try:
    with open(csv_file, newline='', encoding='utf-8') as f:
        raw_rows = list(csv.reader(f))
except FileNotFoundError:
    print(f"ERROR: CSV not found: {csv_file}")
    sys.exit(1)

rows = []
for r in raw_rows:
    cleaned = [c.strip().strip('"') for c in r]
    if any(c for c in cleaned):
        rows.append(cleaned)

if len(rows) < 3:
    print(f"ERROR: Only {len(rows)} non-empty rows in CSV.")
    print(f"  ncu may not have matched any kernel.")
    print(f"  Debug: ncu --import {rep_file} --print-summary per-kernel")
    sys.exit(1)

header_row = rows[0]
units_row  = rows[1]
data_row   = rows[2]

if len(rows) > 3:
    print(f"  NOTE: {len(rows) - 2} kernel data rows. Using first (row index 2).")

print(f"  CSV columns: {len(header_row)}")

# ---------------------------------------------------------------------------
# Build col_map: lowercase_metric_name -> (float_value, unit_string)
# ---------------------------------------------------------------------------

def to_float(s):
    try:
        return float(s.replace(",", "").replace(" ", ""))
    except (ValueError, AttributeError):
        return None

col_map = {}
for i, col in enumerate(header_row):
    key  = col.strip().strip('"').lower()
    raw  = data_row[i] if i < len(data_row) else ""
    unit = units_row[i].strip().strip('"') if i < len(units_row) else ""
    col_map[key] = (to_float(raw), unit)

print(f"  Metrics in col_map: {len(col_map)}")

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def get_entry(metric):
    return col_map.get(metric.lower(), (None, ""))

def get_bytes(metric):
    val, unit = get_entry(metric)
    if val is None:
        return None
    u = unit.lower()
    if "kbyte" in u: return val * 1024
    if "mbyte" in u: return val * 1024 ** 2
    if "gbyte" in u: return val * 1024 ** 3
    return val

def get_pct(metric):
    val, _ = get_entry(metric)
    return val

def get_count(metric):
    val, unit = get_entry(metric)
    if val is None:
        return None
    u = unit.lower()
    if u in ("kunit", "k"): return val * 1_000
    if u in ("munit", "m"): return val * 1_000_000
    if u in ("gunit", "g"): return val * 1_000_000_000
    return val

def get_str(metric):
    val, _ = get_entry(metric)
    return str(val).strip() if val is not None else "unknown"

# ---------------------------------------------------------------------------
# Extract all values
# ---------------------------------------------------------------------------

dram_read        = get_bytes(M_DRAM_READ)   or 0.0
dram_write       = get_bytes(M_DRAM_WRITE)  or 0.0
dram_total       = dram_read + dram_write

sm_throughput = get_pct(M_SM_THROUGHPUT)
mem_throughput = get_pct(M_MEM_THROUGHPUT)
l1_throughput = get_pct(M_L1_THROUGHPUT)
l2_throughput = get_pct(M_L2_THROUGHPUT)
dram_throughput = get_pct(M_DRAM_THROUGHPUT)

duration_ms = get_count(M_DURATION_MS)
elapsed_cycles = get_count(M_ELAPSED_CYCLES)
active_cycles = get_count(M_ACTIVE_CYCLES)
sm_freq_ghz = get_count(M_SM_FREQ_HZ)
dram_freq_ghz = get_count(M_DRAM_FREQ_HZ)

# sm_freq_ghz     = (sm_freq_hz)  if sm_freq_hz   else None
# dram_freq_ghz   = (dram_freq_hz) if dram_freq_hz else None

l1_ld_sectors    = get_count(M_L1_GLOBAL_LD_SECTORS)
l1_st_sectors    = get_count(M_L1_GLOBAL_ST_SECTORS)
l1_to_l2_read    = get_bytes(M_L1_TO_L2_READ)
l1_to_l2_write   = get_bytes(M_L1_TO_L2_WRITE)
global_ld_inst   = get_count(M_GLOBAL_LD_INST)
global_st_inst   = get_count(M_GLOBAL_ST_INST)

l1_hit           = get_pct(M_L1_HIT)
l2_hit           = get_pct(M_L2_HIT)
l2_hit_read      = get_pct(M_L2_HIT_READ)
l2_hit_write     = get_pct(M_L2_HIT_WRITE)

ffma_x2          = get_count(M_FFMA_X2)
hfma_x4          = get_count(M_HFMA_X4)
dfma_x2          = get_count(M_DFMA_X2)
fma_pipe         = get_pct(M_FMA_PIPE)
hw_flops         = (ffma_x2 or 0.0) + (hfma_x4 or 0.0) + (dfma_x2 or 0.0)

occ_achieved     = get_pct(M_OCC_ACHIEVED)
occ_per_block    = get_count(M_OCC_PER_BLOCK)
occ_lim_regs     = get_count(M_OCC_LIM_REGS)
occ_lim_smem     = get_count(M_OCC_LIM_SMEM)
occ_lim_warps    = get_count(M_OCC_LIM_WARPS)
occ_lim_blocks   = get_count(M_OCC_LIM_BLOCKS)
waves_per_sm     = get_count(M_WAVES_PER_SM)
regs_per_thread  = get_count(M_REGS_PER_THREAD)

pcsamp_total     = get_count(M_PCSAMP_TOTAL) or 1.0

def stall_pct(metric):
    v = get_count(metric)
    return (v / pcsamp_total * 100.0) if v is not None else None

stall_long_sb      = stall_pct(M_STALL_LONG_SB)
stall_mio          = stall_pct(M_STALL_MIO)
stall_no_inst      = stall_pct(M_STALL_NO_INST)
stall_not_sel      = stall_pct(M_STALL_NOT_SELECTED)
stall_wait         = stall_pct(M_STALL_WAIT)
stall_lg_throttle  = stall_pct(M_STALL_LG_THROTTLE)
stall_math         = stall_pct(M_STALL_MATH_THROTTLE)
stall_tex          = stall_pct(M_STALL_TEX_THROTTLE)
stall_selected     = stall_pct(M_STALL_SELECTED)

stall_sum = sum(
    v for v in [
        stall_long_sb, stall_mio, stall_no_inst, stall_not_sel,
        stall_wait, stall_lg_throttle, stall_math, stall_tex,
    ]
    if v is not None
)

captured_kernel  = get_str(M_KERN_NAME)
grid_size        = get_count(M_GRID_SIZE)
block_size       = get_count(M_BLOCK_SIZE)
sm_count         = get_count(M_SM_COUNT)
thread_count     = get_count(M_THREAD_CT)

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
SECTOR_BYTES = 128   # Ada Lovelace cache line / sector size
elem_bytes   = {"float32": 4, "float16": 2, "bfloat16": 2}.get(DT, 4)
if operation == "attention":
    bytes_alg = 4 * B * H * N * D * elem_bytes
    bytes_hw_est = 2 * B * H * N * N * D * elem_bytes
    alg_flops = 4 * B * H * N * N * D
else:
    bytes_alg = (M * K + K * N + M * N) * elem_bytes
    bytes_hw_est = bytes_alg
    alg_flops = 2 * M * K * N

l1_ld_bytes  = (l1_ld_sectors * SECTOR_BYTES) if l1_ld_sectors else None
l1_st_bytes  = (l1_st_sectors * SECTOR_BYTES) if l1_st_sectors else None

# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
W = 54

def fb(v):
    if v is None: return "  (not found)"
    if v == 0:    return "         0 B"
    if v >= 1e12: return f"{v/1e12:>11.3f} TB"
    if v >= 1e9:  return f"{v/1e9:>11.3f} GB"
    if v >= 1e6:  return f"{v/1e6:>11.3f} MB"
    if v >= 1e3:  return f"{v/1e3:>11.3f} KB"
    return f"{v:>11.0f} B"

def fp(v):
    if v is None: return "  (not found)"
    return f"{v:>11.2f} %"

def fc(v):
    if v is None: return "  (not found)"
    return f"{v:>19,.0f}"

def frow(label, val_str): return f"    {label:<{W}}  {val_str}"
def brow(label, v):       return frow(label, fb(v))
def prow(label, v):       return frow(label, fp(v))
def crow(label, v):       return frow(label, fc(v))

# ---------------------------------------------------------------------------
# Build summary text
# ---------------------------------------------------------------------------
SEP  = "  " + "=" * 70
SEP2 = "  " + "-" * 70

shape_str = (
    f"B={B}  H={H}  N={N}  D={D}  dtype={DT}"
    if operation == "attention"
    else f"M={M}  K={K}  N={N}  dtype={DT}"
)

lines = [
    SEP,
    f"    CUDA {operation.title()} Suite — ncu Hardware Counter Summary",
    "    sm_89  Ada Lovelace  RTX 4050 Laptop GPU",
    SEP,
    frow("Kernel captured",   captured_kernel[:72]),
    frow("Shape",             shape_str),
    frow("Grid / Block",      f"{grid_size} / {block_size}"),
    frow("Registers/thread",  str(regs_per_thread)),
    frow("SMs active",        str(sm_count)),
    frow("Total threads",     fc(thread_count).strip()),
    frow("Waves per SM",      str(waves_per_sm)),
    SEP,
    "",
    "  ── DRAM TRAFFIC  (hardware ground truth) ───────────────────────────",
    SEP2,
    brow("DRAM Read",                      dram_read),
    brow("DRAM Write",                     dram_write),
    brow("DRAM Total  (read + write)",     dram_total),
    "",
    SEP,
    "",
    "  ── GPU SPEED OF LIGHT THROUGHPUT ────────────────────────────────────",
    "  Matches ncu-ui Summary tab top panel exactly.",
    SEP2,
    prow("Compute (SM) Throughput %",      sm_throughput),
    prow("Memory Throughput %",            mem_throughput),
    prow("L1/TEX Cache Throughput %",      l1_throughput),
    prow("L2 Cache Throughput %",          l2_throughput),
    prow("DRAM Throughput %",              dram_throughput),
    frow("Duration [ms]",                  f"{duration_ms:.2f}" if duration_ms else "(not found)"),
    crow("Elapsed Cycles",                 elapsed_cycles),
    crow("SM Active Cycles",               active_cycles),
    frow("SM Frequency [GHz]",             f"{sm_freq_ghz:.2f}" if sm_freq_ghz else "(not found)"),
    frow("DRAM Frequency [GHz]",           f"{dram_freq_ghz:.2f}" if dram_freq_ghz else "(not found)"),
    "",
    "  ── L1 ↔ L2 TRAFFIC ─────────────────────────────────────────────────",
    SEP2,
    brow("L1 → L2 read bytes   (L1 misses going to L2)", l1_to_l2_read),
    brow("L1 → L2 write bytes",                          l1_to_l2_write),
    brow("L1 global load bytes (sectors × 128)",         l1_ld_bytes),
    brow("L1 global store bytes (sectors × 128)",        l1_st_bytes),
    crow("Global load instructions",                     global_ld_inst),
    crow("Global store instructions",                    global_st_inst),
    "",
    "  ── CACHE HIT RATES ──────────────────────────────────────────────────",
    SEP2,
    prow("L1/TEX Hit Rate",                l1_hit),
    prow("L2 Hit Rate  (overall)",         l2_hit),
    prow("L2 Hit Rate  (reads only)",      l2_hit_read),
    prow("L2 Hit Rate  (writes only)",     l2_hit_write),
    "",
    "  ── COMPUTE THROUGHPUT ───────────────────────────────────────────────",
    "  (sm_89: raw .sum counts not exported; using derived×N metrics)",
    SEP2,
    crow("FP32 FMAs × 2  (derived__...ffma_pred_on_x2)",  ffma_x2),
    crow("FP16 FMAs × 4  (derived__...hfma_pred_on_x4)",  hfma_x4),
    crow("FP64 FMAs × 2  (derived__...dfma_pred_on_x2)",  dfma_x2),
    crow("Hardware FLOPs total  (sum of above)",           hw_flops if hw_flops > 0 else None),
    crow("Algorithmic FLOPs  (4·B·H·N²·D)",               alg_flops),
    prow("FP32 FMA Pipe utilisation  (% of peak elapsed)", fma_pipe),
    "",
    "  ── OCCUPANCY ────────────────────────────────────────────────────────",
    SEP2,
    prow("Achieved Occupancy  (sm__warps_active avg)",     occ_achieved),
    crow("Theoretical  (warps/SM per block)",              occ_per_block),
    crow("Limit — Registers",                              occ_lim_regs),
    crow("Limit — Shared Memory",                          occ_lim_smem),
    crow("Limit — Warps",                                  occ_lim_warps),
    crow("Limit — Blocks",                                 occ_lim_blocks),
    "",
    "  ── WARP STALL BREAKDOWN ─────────────────────────────────────────────",
    "  (sm_89 pcsamp counts ÷ total samples → %)",
    "  Long Scoreboard >> others → memory bound",
    "  Math Throttle   >> others → compute bound",
    SEP2,
    prow("Stall: Long Scoreboard  (L2/DRAM wait)",         stall_long_sb),
    prow("Stall: LG Throttle  (load/store queue full)",    stall_lg_throttle),
    prow("Stall: MIO Throttle  (mem I/O throttle)",        stall_mio),
    prow("Stall: Math Pipe Throttle",                      stall_math),
    prow("Stall: TEX Throttle",                            stall_tex),
    prow("Stall: No Instruction  (icache miss)",           stall_no_inst),
    prow("Stall: Not Selected  (warp not chosen by sched)",stall_not_sel),
    prow("Stall: Fixed Latency Wait",                      stall_wait),
    frow("Stall sum  (all non-active stalls)",             fp(stall_sum).strip()),
    prow("Active  (selected/issuing warps)",               stall_selected),
    "  ── OPEN IN GUI ──────────────────────────────────────────────────────",
    SEP2,
    f"    ncu-ui {rep_file}",
    SEP,
]

# ---------------------------------------------------------------------------
# Missing metrics section
# ---------------------------------------------------------------------------
all_checked = [
    M_DRAM_READ, M_DRAM_WRITE,
    M_L1_GLOBAL_LD_SECTORS, M_L1_GLOBAL_ST_SECTORS,
    M_L1_TO_L2_READ, M_L1_TO_L2_WRITE,
    M_GLOBAL_LD_INST, M_GLOBAL_ST_INST,
    M_L1_HIT, M_L2_HIT, M_L2_HIT_READ, M_L2_HIT_WRITE,
    M_FFMA_X2, M_HFMA_X4, M_DFMA_X2, M_FMA_PIPE,
    M_OCC_ACHIEVED, M_OCC_PER_BLOCK,
    M_OCC_LIM_REGS, M_OCC_LIM_SMEM, M_OCC_LIM_WARPS, M_OCC_LIM_BLOCKS,
    M_WAVES_PER_SM, M_REGS_PER_THREAD, M_PCSAMP_TOTAL,
    M_STALL_LONG_SB, M_STALL_MIO, M_STALL_NO_INST,
    M_STALL_NOT_SELECTED, M_STALL_WAIT,
    M_STALL_LG_THROTTLE, M_STALL_MATH_THROTTLE, M_STALL_TEX_THROTTLE,
    M_STALL_SELECTED,
]

missing = [m for m in all_checked if get_entry(m)[0] is None]

if missing:
    lines += [
        "",
        "  ── MISSING METRICS ──────────────────────────────────────────────────",
        "  Not found in CSV. If using --metrics, add them to METRICS in",
        "  benchmarks/profiling/ncu/run.sh. If using --set full, metric may not exist on this GPU.",
        SEP2,
    ]
    for m in missing:
        lines.append(f"    {m}")
    lines += [
        "",
        "  Dump all CSV column names to find correct names for your GPU:",
        f"    python3 -c \"import csv; r=list(csv.reader(open('{csv_file}')));",
        f"    [print(f'{{i:4d}}  {{c.strip()}}') for i,c in enumerate(r[0])]\"",
        "",
    ]

# Print the corrected METRICS string for benchmarks/profiling/ncu/run.sh

METRICS_LIST = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__m_xbar2l1tex_read_bytes.sum",
    "l1tex__m_l1tex2xbar_write_bytes.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sector_op_read_hit_rate.pct",
    "lts__t_sector_op_write_hit_rate.pct",
    "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",
    "derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x4",
    "derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_per_block_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
    "smsp__pcsamp_warps_issue_stalled_selected",
]

lines += [
    "",
    "  ── METRICS STRING FOR --metrics MODE (faster: ~8 passes vs 43) ──────",
    "  Replace METRICS in benchmarks/profiling/ncu/run.sh with this to skip --set full:",
    SEP2,
]

# wrap lines at 76 chars
line_buf = "METRICS=\""
for i, m in enumerate(METRICS_LIST):
    sep = "," if i < len(METRICS_LIST) - 1 else "\""
    chunk = m + sep
    if len(line_buf) + len(chunk) > 76:
        lines.append("  " + line_buf + "\\")
        line_buf = "  " + chunk
    else:
        line_buf += chunk
lines.append("  " + line_buf)
lines.append("")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
output = "\n".join(lines)
print(output)

with open(summary_file, "w") as f:
    f.write(output + "\n")
print(f"\n  Summary saved → {summary_file}")

parsed_fields = {
    "operation": operation,
    "requested_kernel": kernel_name,
    "captured_kernel": captured_kernel,
    "dtype": DT,
    "B": B,
    "H": H,
    "N": N,
    "D": D,
    "M": M,
    "K": K,
    "dram_read_bytes": dram_read,
    "dram_write_bytes": dram_write,
    "dram_total_bytes": dram_total,
    "algorithmic_bytes": bytes_alg,
    "hardware_byte_estimate": bytes_hw_est,
    "l1_to_l2_read_bytes": l1_to_l2_read,
    "l1_to_l2_write_bytes": l1_to_l2_write,
    "l1_global_load_bytes": l1_ld_bytes,
    "l1_global_store_bytes": l1_st_bytes,
    "l1_hit_pct": l1_hit,
    "l2_hit_pct": l2_hit,
    "l2_read_hit_pct": l2_hit_read,
    "l2_write_hit_pct": l2_hit_write,
    "sm_throughput_pct": sm_throughput,
    "memory_throughput_pct": mem_throughput,
    "l1_throughput_pct": l1_throughput,
    "l2_throughput_pct": l2_throughput,
    "dram_throughput_pct": dram_throughput,
    "duration_ms": duration_ms,
    "elapsed_cycles": elapsed_cycles,
    "active_cycles": active_cycles,
    "sm_freq_hz": sm_freq_ghz,
    "dram_freq_hz": dram_freq_ghz,
    "ffma_x2": ffma_x2,
    "hfma_x4": hfma_x4,
    "dfma_x2": dfma_x2,
    "hardware_flops": hw_flops,
    "algorithmic_flops": alg_flops,
    "achieved_occupancy_pct": occ_achieved,
    "occupancy_per_block": occ_per_block,
    "occupancy_limit_registers": occ_lim_regs,
    "occupancy_limit_shared_mem": occ_lim_smem,
    "occupancy_limit_warps": occ_lim_warps,
    "occupancy_limit_blocks": occ_lim_blocks,
    "waves_per_sm": waves_per_sm,
    "registers_per_thread": regs_per_thread,
    "stall_long_scoreboard_pct": stall_long_sb,
    "stall_mio_pct": stall_mio,
    "stall_no_instruction_pct": stall_no_inst,
    "stall_not_selected_pct": stall_not_sel,
    "stall_wait_pct": stall_wait,
    "stall_lg_throttle_pct": stall_lg_throttle,
    "stall_math_pct": stall_math,
    "stall_tex_pct": stall_tex,
    "stall_selected_pct": stall_selected,
    "grid_size": grid_size,
    "block_size": block_size,
    "sm_count": sm_count,
    "thread_count": thread_count,
    "csv_file": csv_file,
    "summary_file": summary_file,
    "rep_file": rep_file,
}

with open(parsed_csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(parsed_fields))
    writer.writeheader()
    writer.writerow(parsed_fields)
print(f"  Parsed CSV saved → {parsed_csv_file}")

# Sanity check
ck = captured_kernel.lower()
if not any(kw in ck for kw in ["naive_attention", "fused_attn", "fmha", "flash", "attn"]):
    print()
    print(f"  *** WARNING: captured kernel '{captured_kernel[:60]}' does not look")
    print(f"  ***          like an attention kernel. Check --kernel-name regex.")
    print(f"  ***          Run: ncu --import {rep_file} --print-summary per-kernel")
