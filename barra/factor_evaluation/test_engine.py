"""End-to-end test: FactorEvalEngine with real data + new metrics verification.

Mimics run.py data-loading pattern using qlib D.features(), BaseDataLoader,
and exposure_matrix.parquet.

Run:  conda run -n python3 python barra/factor_evaluation/test_engine.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import BENCHMARK_CONFIG
from utils import init_qlib, dt
from utils.io_utils import DataFrameIO
from qlib.data import D
from barra.base import BaseDataLoader
from barra.factor_evaluation import FactorEvalEngine
from barra.factor_evaluation.conf import DEFAULT_MAX_DECAY_LAG

CALC_DATE = "2026-04-24"
HISTORY_MONTHS = 12
N_GROUPS = 5
IC_PERIODS = (1,)
OUTPUT = Path(project_root) / "barra" / "factor_evaluation" / "data" / CALC_DATE
OUTPUT.mkdir(parents=True, exist_ok=True)

# ========================================================================
# Data Loading (mirrors run.py)
# ========================================================================

init_qlib()
data_loader = BaseDataLoader(market=BENCHMARK_CONFIG['market'])
start_date = dt.subtract_months(CALC_DATE, HISTORY_MONTHS)

# 1. Close data
instruments = data_loader.load_instruments(start_date, CALC_DATE)
close_df = D.features(instruments, ["$close"], start_date, CALC_DATE)
close = close_df["$close"]
close.sort_index(inplace=True)
print(f"Close: {close.shape}")

# 2. Benchmark (CSI300) close
bench_close_df = D.features([BENCHMARK_CONFIG['BENCHMARK']], ["$close"], start_date, CALC_DATE)
benchmark_close = bench_close_df["$close"]
benchmark_close.droplevel('instrument', inplace=True)
benchmark_close.sort_index(inplace=True)
print(f"Benchmark close: {benchmark_close.shape}")

# 3. Risk factors (CNE6 exposure)
exposure_path = project_root / "barra" / "factors" / "data" / CALC_DATE / "exposure_matrix.parquet"
risk_factors = DataFrameIO.read(str(exposure_path), "parquet")
risk_factors.sort_index(inplace=True)
print(f"Risk factors: {risk_factors.shape}")

# 4. Alpha factors
alpha_factors = data_loader.load_signal(start_date, CALC_DATE)
alpha_factors.columns = ['alpha1']
alpha_factors.sort_index(inplace=True)
print(f"Alpha factors: {alpha_factors.shape}")

# 5. Align indices
com_index = close.index.intersection(alpha_factors.index).intersection(risk_factors.index)
print(f"Common index size: {len(com_index)} (close={len(close)}, alpha={len(alpha_factors)}, risk={len(risk_factors)})")
close = close.loc[com_index]
risk_factors = risk_factors.loc[com_index]
alpha_factors = alpha_factors.loc[com_index]
print(f"After align: close={close.shape}, risk={risk_factors.shape}, alpha={alpha_factors.shape}")

# ========================================================================
# Test 1: Engine initialization + run (alpha with neutralize)
# ========================================================================

print(f"\n{'='*60}")
print("Test 1: Alpha factor — neutralize=True, all new metrics")
print(f"{'='*60}")

engine = FactorEvalEngine(
    close=close,
    risk_factors=risk_factors,
    alpha_factors=alpha_factors,
    ic_periods=IC_PERIODS,
    benchmark_close=benchmark_close,
)
result = engine.run(
    neutralize=True,
    n_groups=N_GROUPS,
    max_decay_lag=DEFAULT_MAX_DECAY_LAG,
    output=str(OUTPUT),
)

for name, res in result.get('alpha_factors', {}).items():
    for variant in ('raw', 'neutralized'):
        if variant not in res:
            continue
        v = res[variant]
        l1 = v['layer1'][1]
        l2 = v['layer2']
        print(f"  {name}[{variant}]:")
        print(f"    ICIR={l1['icir']:.4f}, RICIR={l1['ricir']:.4f}")
        print(f"    long_short mean={l2['long_short'].mean():.6f}")
        print(f"    monotonic_tstat={l2.get('monotonic_tstat', 'N/A'):.6f}")
        if 'layer3' in v:
            print(f"    half_life={v['layer3']['half_life']:.2f}")

# ========================================================================
# Test 2: Extract metrics — verify all 6 new fields are populated
# ========================================================================

print(f"\n{'='*60}")
print("Test 2: _extract_metrics — new fields")
print(f"{'='*60}")

for name, res in result.get('alpha_factors', {}).items():
    for variant in ('raw', 'neutralized'):
        if variant not in res:
            continue
        v = res[variant]
        engine.calc_date = CALC_DATE
        metrics = engine._extract_metrics(v, CALC_DATE, name, 'alpha')

        new_fields = [
            'monotonic_tstat', 'ls_alpha', 'ls_alpha_tstat', 'ls_beta',
            'ls_ir', 'ls_cum_return_1y',
        ]
        all_ok = True
        for f in new_fields:
            val = metrics.get(f)
            ok = val is not None and (not isinstance(val, float) or np.isfinite(val))
            status = "OK" if ok else "MISSING/NON-FINITE"
            if not ok:
                all_ok = False
            print(f"    {f}: {val} [{status}]")
        print(f"  {name}[{variant}] all new fields: {'PASS' if all_ok else 'FAIL'}")
        assert all_ok, f"New fields missing in {name}[{variant}]"

# ========================================================================
# Test 3: Static methods — edge cases
# ========================================================================

print(f"\n{'='*60}")
print("Test 3: Static helpers — edge cases")
print(f"{'='*60}")

# _compute_ir edge cases
ls_empty = pd.Series([], dtype=float)
ls_short = pd.Series([0.01, -0.02])
ls_zero = pd.Series([0.0, 0.0, 0.0])
ls_nan = pd.Series([0.01, np.nan, 0.02])
ls_normal = pd.Series(np.random.randn(100) * 0.02)

assert FactorEvalEngine._compute_ir(ls_empty) == 0.0, "Empty IR should be 0"
assert FactorEvalEngine._compute_ir(ls_short) == 0.0, "Short (<3) IR should be 0"
assert FactorEvalEngine._compute_ir(ls_zero) == 0.0, "Zero-var IR should be 0"
assert abs(FactorEvalEngine._compute_ir(ls_nan)) > 0, "NaN should be dropped in IR"
print(f"  _compute_ir: OK (normal={FactorEvalEngine._compute_ir(ls_normal):.4f})")

# _compute_cum_return edge cases
assert FactorEvalEngine._compute_cum_return(ls_empty) == 0.0, "Empty cum should be 0"
assert FactorEvalEngine._compute_cum_return(ls_empty, window=252) == 0.0, "Empty+window should be 0"
cum5 = FactorEvalEngine._compute_cum_return(pd.Series([0.01]*5), window=5)
expected_cum5 = 1.01**5 - 1
assert abs(cum5 - expected_cum5) < 1e-10, f"Cum 5x1%: {cum5:.6f} != {expected_cum5:.6f}"
print(f"  _compute_cum_return: OK")

# _compute_jensen_alpha with real benchmark (from loaded data)
ls_real = v['layer2']['long_short']
ja = FactorEvalEngine._compute_jensen_alpha(ls_real, benchmark_close)
assert 'alpha' in ja and 'alpha_tstat' in ja and 'beta' in ja
assert np.isfinite(ja['alpha']), "Jensen alpha should be finite"
assert np.isfinite(ja['alpha_tstat']), "Jensen tstat should be finite"
assert np.isfinite(ja['beta']), "Jensen beta should be finite"
print(f"  _compute_jensen_alpha: alpha={ja['alpha']:.6f}, tstat={ja['alpha_tstat']:.4f}, beta={ja['beta']:.4f}")

# _compute_jensen_alpha with too-short series
ja_short = FactorEvalEngine._compute_jensen_alpha(ls_real.iloc[:5], benchmark_close)
assert ja_short['alpha'] == 0.0 and ja_short['alpha_tstat'] == 0.0, "Short series should return zeros"
print(f"  _compute_jensen_alpha short-series guard: OK")

# ========================================================================
# Test 4: Field name consistency (save_to_mysql)
# ========================================================================

print(f"\n{'='*60}")
print("Test 4: Field name consistency")
print(f"{'='*60}")

# Reconstruct expected fields from the engine code
import inspect
src = inspect.getsource(engine.save_to_mysql)
mysql_fields = []
for line in src.split('\n'):
    if 'fields = [' in line:
        break
capture = False
for line in src.split('\n'):
    if 'fields = [' in line:
        capture = True
        continue
    if capture and ']' in line:
        break
    if capture:
        mysql_fields.extend([s.strip().strip("'").strip('"').strip(',') for s in line.split() if s.strip().strip("'").strip('"').strip(',')])

# Check all new fields in mysql_fields
new_fields = ['monotonic_tstat', 'ls_alpha', 'ls_alpha_tstat', 'ls_beta', 'ls_ir', 'ls_cum_return_1y']
for f in new_fields:
    assert f in mysql_fields, f"Missing '{f}' in save_to_mysql fields: {mysql_fields}"
print(f"  All {len(new_fields)} new fields present in save_to_mysql: PASS")

# ========================================================================
# Summary
# ========================================================================

print(f"\n{'='*60}")
print("ALL TESTS PASSED")
print(f"{'='*60}")
