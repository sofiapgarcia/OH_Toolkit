"""
OH Stats Testing Script
=======================

Demonstrates the full statistical analysis pipeline for EMG data:
1. Data preparation
2. Descriptive statistics & QA
3. Model fitting (LMM)
4. Post-hoc contrasts
5. FDR correction
6. Diagnostics
7. Report generation

Run from the OH_Parser directory:
    python testing_stats.py
"""
import sys
import os
import warnings

# Suppress some convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

from oh_parser import load_profiles, list_subjects

# =============================================================================
# SETUP: Load profiles
# =============================================================================
print("=" * 70)
print("OH STATS PIPELINE DEMONSTRATION")
print("=" * 70)

# Path can be set via environment variable or defaults to external drive
OH_PROFILES_PATH = os.environ.get(
    "OH_PROFILES_PATH",
    "/Volumes/NO NAME/Backup PrevOccupAI_PLUS Data/OH_profiles"
)

print("\n[1] Loading OH profiles...")
profiles = load_profiles(OH_PROFILES_PATH)
subjects = list_subjects(profiles)
print(f"    Loaded {len(subjects)} subjects: {subjects[:5]}...")

# =============================================================================
# STEP 1A: Data Discovery
# =============================================================================
print("\n" + "=" * 70)
print("[1A] DATA DISCOVERY")
print("=" * 70)

from oh_stats import discover_sensors, discover_questionnaires, get_profile_summary

# Discover available sensors
print("\n--- Available Sensors ---")
sensors = discover_sensors(profiles)
for sensor, metrics in sensors.items():
    print(f"  {sensor}: {len(metrics)} metrics")
    if metrics:
        print(f"    Sample: {list(metrics)[:3]}...")

# Discover questionnaires
print("\n--- Available Questionnaires ---")
questionnaires = discover_questionnaires(profiles)
for q_type, q_names in questionnaires.items():
    print(f"  {q_type}: {q_names}")

# Profile summary
print("\n--- Profile Summary ---")
profile_summary = get_profile_summary(profiles)
print(profile_summary[:500])  # First 500 chars

# =============================================================================
# STEP 1B: Data Preparation
# =============================================================================
print("\n" + "=" * 70)
print("[2] DATA PREPARATION")
print("=" * 70)

from oh_stats import prepare_daily_emg, prepare_daily_questionnaires

# Prepare daily EMG data (keep both sides)
ds = prepare_daily_emg(profiles, side="both")
print(f"\nDataset summary:")
print(f"  Shape: {ds['data'].shape}")
print(f"  Outcomes: {len(ds['outcome_vars'])} variables")
print(f"  ID var: {ds['id_var']}")
print(f"  Time var: {ds['time_var']}")
print(f"  Grouping: {ds['grouping_vars']}")
print(f"\nFirst 10 rows:")
print(ds['data'].head(10).to_string())

# Check if questionnaire data is available (conditionally activated)
qs = prepare_daily_questionnaires(profiles)
if qs is None:
    print("\n[Note] Daily questionnaire data not available - skipping")
else:
    print(f"\nQuestionnaire data: {qs['data'].shape}")

# =============================================================================
# STEP 1C: Dataset Utilities
# =============================================================================
print("\n" + "=" * 70)
print("[1C] DATASET UTILITIES")
print("=" * 70)

from oh_stats import (
    describe_dataset, validate_dataset, subset_dataset,
    get_n_subjects, get_n_observations, get_date_range, get_obs_per_subject
)

# Describe dataset
print("\n--- describe_dataset() ---")
print(describe_dataset(ds))

# Validation
print("\n--- validate_dataset() ---")
try:
    validated_ds = validate_dataset(ds)
    print(f"  Valid: True (no exceptions raised)")
    print(f"  Outcomes validated: {len(validated_ds['outcome_vars'])}")
except ValueError as e:
    print(f"  Valid: False")
    print(f"  Error: {e}")

# Quick accessors
print("\n--- Quick Accessors ---")
print(f"  N subjects: {get_n_subjects(ds)}")
print(f"  N observations: {get_n_observations(ds)}")
print(f"  Date range: {get_date_range(ds)}")
print(f"  Obs per subject (first 3): {dict(list(get_obs_per_subject(ds).items())[:3])}")

# Subset dataset
print("\n--- subset_dataset() ---")
if len(subjects) >= 2:
    ds_subset = subset_dataset(ds, subjects=subjects[:2])
    print(f"  Original: {get_n_subjects(ds)} subjects, {get_n_observations(ds)} obs")
    print(f"  Subset: {get_n_subjects(ds_subset)} subjects, {get_n_observations(ds_subset)} obs")

# =============================================================================
# STEP 2: Descriptive Statistics
# =============================================================================
print("\n" + "=" * 70)
print("[3] DESCRIPTIVE STATISTICS")
print("=" * 70)

from oh_stats import summarize_outcomes, check_normality, check_variance, missingness_report

# Select key outcomes for demonstration
primary_outcomes = [
    "EMG_intensity.mean_percent_mvc",
    "EMG_apdf.active.p50",
    "EMG_rest_recovery.rest_percent",
    "EMG_rest_recovery.gap_count",
]

# Summary statistics
print("\n--- Summary Statistics ---")
summary = summarize_outcomes(ds, outcomes=primary_outcomes)
print(summary.to_string(index=False))

# Normality check
print("\n--- Normality Assessment ---")
normality = check_normality(ds, outcomes=primary_outcomes)
print(normality[["outcome", "n", "skewness", "is_normal", "recommended_transform"]].to_string(index=False))

# Variance check (detect degenerate outcomes)
print("\n--- Variance Check (Degenerate Detection) ---")
variance = check_variance(ds, outcomes=primary_outcomes)
print(variance[["outcome", "n_unique", "pct_mode", "is_degenerate", "reason"]].to_string(index=False))

# Missingness
print("\n--- Missingness Summary ---")
miss = missingness_report(ds, outcomes=primary_outcomes)
print(f"Total missing: {miss['summary']['total_missing']} cells ({miss['summary']['pct_missing']:.1f}%)")

# =============================================================================
# STEP 3: Fit Linear Mixed Models
# =============================================================================
print("\n" + "=" * 70)
print("[4] LINEAR MIXED MODELS")
print("=" * 70)

from oh_stats import fit_lmm, fit_all_outcomes, summarize_lmm_result
from oh_stats.registry import get_outcome_info, OutcomeType

# Fit single model (primary outcome)
print("\n--- Single Model: EMG Mean %MVC ---")
result = fit_lmm(
    ds,
    outcome="EMG_intensity.mean_percent_mvc",
    fixed_effects=["C(day_index)", "C(side)"],  # Day + Side as categorical
    random_intercept="subject_id",
)
print(summarize_lmm_result(result))
print("\nCoefficients:")
print(result['coefficients'].to_string(index=False))

# Check model fit stats
print(f"\nRandom effects:")
group_var = result['random_effects'].get('group_var', 'NA')
resid_var = result['random_effects'].get('residual_var', 'NA')
icc = result['random_effects'].get('icc', 'NA')
print(f"  Subject variance: {group_var:.4f}" if isinstance(group_var, (int, float)) else f"  Subject variance: {group_var}")
print(f"  Residual variance: {resid_var:.4f}" if isinstance(resid_var, (int, float)) else f"  Residual variance: {resid_var}")
print(f"  ICC: {icc:.3f}" if isinstance(icc, (int, float)) else f"  ICC: {icc}")

# --- Transforms demonstration ---
print("\n--- Outcome Transforms ---")
from oh_stats import apply_transform
from oh_stats.registry import TransformType

# Apply log transform to a skewed outcome
test_outcome = "EMG_intensity.mean_percent_mvc"
original_values = ds['data'][test_outcome].dropna()
print(f"  Original range: {original_values.min():.2f} - {original_values.max():.2f}")

# Log transform
log_values = apply_transform(original_values, TransformType.LOG)
print(f"  Log-transformed range: {log_values.min():.2f} - {log_values.max():.2f}")

# Sqrt transform
sqrt_values = apply_transform(original_values, TransformType.SQRT)
print(f"  Sqrt-transformed range: {sqrt_values.min():.2f} - {sqrt_values.max():.2f}")

# =============================================================================
# STEP 4: Fit Multiple Outcomes
# =============================================================================
print("\n" + "=" * 70)
print("[5] BATCH MODEL FITTING")
print("=" * 70)

# Get non-degenerate continuous outcomes
from oh_stats.descriptive import get_non_degenerate_outcomes
from oh_stats.registry import list_outcomes

continuous_outcomes = list_outcomes(outcome_type=OutcomeType.CONTINUOUS)
print(f"\nRegistered continuous outcomes: {len(continuous_outcomes)}")

# Filter to non-degenerate
valid_outcomes = get_non_degenerate_outcomes(ds, continuous_outcomes[:10])  # Limit for demo
print(f"Non-degenerate outcomes: {len(valid_outcomes)}")

# Fit all
results = fit_all_outcomes(ds, outcomes=valid_outcomes[:5], skip_degenerate=True)
print(f"\nFitted {len(results)} models")

for name, r in results.items():
    status = "✓" if r['converged'] else "✗"
    aic = r['fit_stats'].get('aic', 'NA')
    icc_val = r['random_effects'].get('icc', 'NA')
    aic_str = f"{aic:.1f}" if isinstance(aic, (int, float)) else str(aic)
    icc_str = f"{icc_val:.3f}" if isinstance(icc_val, (int, float)) else str(icc_val)
    print(f"  {status} {name}: AIC={aic_str}, ICC={icc_str}")

# --- Model accessors demonstration ---
print("\n--- Model Accessors ---")
from oh_stats import get_residuals, get_fitted_values, get_random_effects

if result and result['converged']:
    residuals = get_residuals(result)
    fitted = get_fitted_values(result)
    ranef = get_random_effects(result)
    if residuals is not None:
        print(f"  Residuals: n={len(residuals)}, mean={residuals.mean():.4f}, std={residuals.std():.4f}")
    if fitted is not None:
        print(f"  Fitted values: n={len(fitted)}, range=[{fitted.min():.2f}, {fitted.max():.2f}]")
    if ranef is not None:
        print(f"  Random effects (first 3 subjects): {dict(list(ranef.items())[:3])}")

# --- Model comparison demonstration ---
print("\n--- Model Comparison ---")
from oh_stats import compare_models

# Fit a simpler model (no side effect) for comparison
result_simple = fit_lmm(
    ds,
    outcome="EMG_intensity.mean_percent_mvc",
    fixed_effects=["C(day_index)"],  # Only day, no side
    random_intercept="subject_id",
)
if result_simple['converged'] and result['converged']:
    comparison_df = compare_models([result_simple, result])
    print(comparison_df[["formula", "aic", "delta_aic"]].to_string(index=False))
    print(f"  Preferred model (lowest AIC): Row with delta_aic=0")

# =============================================================================
# STEP 5: Post-hoc Contrasts
# =============================================================================
print("\n" + "=" * 70)
print("[6] POST-HOC CONTRASTS")
print("=" * 70)

from oh_stats import pairwise_contrasts, compute_emmeans, summarize_contrast_result

# Pairwise day comparisons for primary outcome
result_main = results.get("EMG_intensity.mean_percent_mvc")
if result_main and result_main['converged']:
    print("\n--- Pairwise Day Contrasts: EMG Mean %MVC ---")
    contrast_result = pairwise_contrasts(result_main, factor="day_index", ds=ds, correction="holm")
    contrasts_df = contrast_result["contrasts"]  # ContrastResult is a TypedDict
    print("\nPairwise Contrasts (Holm-corrected):")
    if contrasts_df is not None and not contrasts_df.empty:
        cols_to_show = [c for c in ["contrast", "estimate", "std_error", "p_adjusted"] if c in contrasts_df.columns]
        print(contrasts_df[cols_to_show].head(10).to_string(index=False))
    else:
        print("  (No contrasts available)")
    
    print("\nEstimated Marginal Means:")
    emmeans_df = compute_emmeans(result_main, factor="day_index", ds=ds)
    print(emmeans_df.to_string(index=False))
    
    # Effect size demonstration
    print("\n--- Effect Sizes ---")
    from oh_stats import compute_effect_size
    
    effect = compute_effect_size(result_main, ds=ds)
    print(f"  Cohen's d (overall): {effect.get('cohens_d', 'NA')}")
    print(f"  Effect interpretation: {effect.get('interpretation', 'NA')}")
    
    # Linear trend test
    print("\n--- Linear Trend Test ---")
    from oh_stats import test_linear_trend
    
    trend = test_linear_trend(result_main, factor="day_index")
    if trend:
        print(f"  Linear coefficient: {trend.get('estimate', 'NA')}")
        print(f"  p-value: {trend.get('p_value', 'NA')}")
        print(f"  Significant: {trend.get('significant', 'NA')}")
    else:
        print("  (Trend test not available - factor may be categorical)")

# =============================================================================
# STEP 6: FDR Correction Across Outcomes
# =============================================================================
print("\n" + "=" * 70)
print("[7] MULTIPLICITY CORRECTION (FDR)")
print("=" * 70)

from oh_stats import apply_fdr, apply_holm, adjust_pvalues, significant_outcomes, fdr_summary

# Apply FDR correction for day_index effect across outcomes
fdr_results = apply_fdr(results, term="day_index", method="fdr_bh")

print("\n--- FDR Results (Benjamini-Hochberg) ---")
print(fdr_results[["outcome", "p_raw", "p_adjusted", "significant"]].to_string(index=False))

# Summary
n_sig = fdr_results['significant'].sum()
n_total = len(fdr_results)
print(f"\nSignificant outcomes: {n_sig}/{n_total} (FDR < 0.05)")

# FDR summary helper
print("\n--- FDR Summary ---")
fdr_summ = fdr_summary(fdr_results)
print(fdr_summ)  # Returns formatted string

# Holm correction comparison
print("\n--- Holm Correction (More Conservative) ---")
holm_results = apply_holm(results, term="day_index")
print(holm_results[["outcome", "p_raw", "p_adjusted", "significant"]].to_string(index=False))
n_sig_holm = holm_results['significant'].sum()
print(f"\nSignificant outcomes (Holm): {n_sig_holm}/{n_total}")

# Get significant outcomes list
print("\n--- Significant Outcomes Helper ---")
sig_list = significant_outcomes(fdr_results)
print(f"  Significant (FDR): {sig_list if sig_list else 'None'}")

# Raw p-value adjustment demo
print("\n--- adjust_pvalues() Demo ---")
raw_pvals = [0.001, 0.02, 0.04, 0.06, 0.10]
adj_pvals = adjust_pvalues(raw_pvals, method="fdr_bh")
print(f"  Raw: {raw_pvals}")
print(f"  Adjusted (FDR): {[round(p, 4) for p in adj_pvals]}")

# =============================================================================
# STEP 7: Model Diagnostics
# =============================================================================
print("\n" + "=" * 70)
print("[8] MODEL DIAGNOSTICS")
print("=" * 70)

from oh_stats import residual_diagnostics, check_assumptions, summarize_diagnostics

if result_main and result_main['converged']:
    print("\n--- Diagnostics: EMG Mean %MVC ---")
    diag = residual_diagnostics(result_main)
    print(summarize_diagnostics(diag))
    
    print("\nResidual Normality:")
    norm_result = diag.get('residuals_normality', {})
    shapiro_p = norm_result.get('shapiro_p', 'NA')
    print(f"  Shapiro-Wilk p: {shapiro_p:.4f}" if isinstance(shapiro_p, (int, float)) else f"  Shapiro-Wilk p: {shapiro_p}")
    print(f"  Is normal: {norm_result.get('is_normal', 'NA')}")
    
    print("\nOutliers:")
    outlier_result = diag.get('outliers', {})
    n_outliers = outlier_result.get('count', 'NA')
    print(f"  Count: {n_outliers}")
    print(f"  Percent: {outlier_result.get('percent', 'NA')}")
    
    print("\nHomoscedasticity:")
    homo_result = diag.get('homoscedasticity', {})
    print(f"  Levene p: {homo_result.get('levene_p', 'NA')}")
    print(f"  Is homoscedastic: {homo_result.get('is_homoscedastic', 'NA')}")
    
    # Diagnostic data accessor (for plotting)
    print("\n--- get_diagnostic_data() ---")
    from oh_stats import get_diagnostic_data
    diag_data = get_diagnostic_data(result_main)  # Takes LMMResult, not DiagnosticsResult
    if diag_data is not None:
        residual_df, qq_df = diag_data
        print(f"  Residual DataFrame: {residual_df.shape}")
        print(f"  QQ DataFrame: {qq_df.shape}")
        print(f"  Columns: {list(residual_df.columns)}")

# =============================================================================
# STEP 8: Report Generation
# =============================================================================
print("\n" + "=" * 70)
print("[9] REPORT GENERATION")
print("=" * 70)

from oh_stats import descriptive_table, coefficient_table, results_summary

# Table 1 style
print("\n--- Table 1: Descriptive Statistics ---")
table1 = descriptive_table(ds, outcomes=primary_outcomes[:3])
print(table1.to_string(index=False))

# Coefficient table
if result_main and result_main['converged']:
    print("\n--- Coefficient Table ---")
    coef_table = coefficient_table(result_main)
    print(coef_table.to_string(index=False))

# Results summary
print("\n--- Results Summary ---")
res_summary = results_summary(results, fdr_results)
print(res_summary.to_string(index=False))

# Multiple coefficient table
print("\n--- Coefficient Table (Multiple Models) ---")
from oh_stats import coefficient_table_multiple
coef_multi = coefficient_table_multiple(results)
print(coef_multi.head(10).to_string(index=False))

# Formatted descriptive table
print("\n--- Formatted Descriptive Table ---")
from oh_stats import descriptive_table_formatted
table1_fmt = descriptive_table_formatted(ds, outcomes=primary_outcomes[:2])
print(table1_fmt.to_string(index=False))

# =============================================================================
# STEP 9A: Export Functions
# =============================================================================
print("\n" + "=" * 70)
print("[9A] EXPORT FUNCTIONS")
print("=" * 70)

from oh_stats import export_to_csv, export_to_latex, print_results_summary, print_coefficient_summary
import tempfile

# Export to CSV (to temp file for demo)
print("\n--- export_to_csv() ---")
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
    csv_path = f.name
export_to_csv(res_summary, csv_path)
print(f"  Exported results summary to: {csv_path}")
print(f"  Preview:")
with open(csv_path, 'r') as f:
    lines = f.readlines()[:4]
    for line in lines:
        print(f"    {line.strip()}")

# Export to LaTeX
print("\n--- export_to_latex() ---")
with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
    tex_path = f.name
export_to_latex(table1, tex_path)
print(f"  Exported descriptive table to: {tex_path}")
with open(tex_path, 'r') as f:
    content = f.read()
    print(f"  Preview (first 200 chars):")
    print(f"    {content[:200]}...")

# Print helpers
print("\n--- print_results_summary() ---")
print_results_summary(results, fdr_results)

if result_main and result_main['converged']:
    print("\n--- print_coefficient_summary() ---")
    print_coefficient_summary(result_main)

# =============================================================================
# STEP 9BB: Registry Demonstration
# =============================================================================
print("\n" + "=" * 70)
print("[10] OUTCOME REGISTRY")
print("=" * 70)

from oh_stats.registry import (
    list_outcomes, get_outcome_info, register_outcome,
    OutcomeType, AnalysisLevel, TransformType
)

print("\n--- Registry Contents ---")
print(f"Total registered outcomes: {len(list_outcomes())}")
print(f"Continuous: {len(list_outcomes(outcome_type=OutcomeType.CONTINUOUS))}")
print(f"Proportion: {len(list_outcomes(outcome_type=OutcomeType.PROPORTION))}")
print(f"Count: {len(list_outcomes(outcome_type=OutcomeType.COUNT))}")
print(f"Primary: {len(list_outcomes(primary_only=True))}")

# Show info for a specific outcome
info = get_outcome_info("EMG_apdf.active.p50")
print(f"\n--- Outcome Info: EMG_apdf.active.p50 ---")
if info:
    print(f"  Type: {info['outcome_type'].name}")
    print(f"  Level: {info['level']}")
    print(f"  Transform: {info['transform'].name}")
    print(f"  Is Primary: {info.get('is_primary', False)}")
    print(f"  Description: {info.get('description', 'N/A')}")
else:
    print("  (Not found in registry)")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE DEMONSTRATION COMPLETE")
print("=" * 70)
print("""
This script demonstrated the full oh_stats pipeline:

✓ Data Discovery: discover_sensors, discover_questionnaires, get_profile_summary
✓ Data Preparation: prepare_daily_emg, prepare_daily_questionnaires
✓ Dataset Utilities: describe_dataset, validate_dataset, subset_dataset, accessors
✓ Descriptive Stats: summarize_outcomes, check_normality, check_variance, missingness
✓ Transforms: apply_transform (log, sqrt, etc.)
✓ LMM Fitting: fit_lmm, fit_all_outcomes, model accessors
✓ Model Comparison: compare_models (AIC, likelihood ratio test)
✓ Post-hoc: pairwise_contrasts, compute_emmeans, compute_effect_size, test_linear_trend
✓ Multiplicity: apply_fdr, apply_holm, adjust_pvalues, significant_outcomes
✓ Diagnostics: residual_diagnostics, check_assumptions, get_diagnostic_data
✓ Reporting: descriptive_table, coefficient_table, results_summary
✓ Export: export_to_csv, export_to_latex, print helpers
✓ Registry: list_outcomes, get_outcome_info, register_outcome

Not demonstrated (specialized use cases):
- prepare_from_dataframe (for pre-extracted data)
- prepare_sensor_data (generic sensor preparation)
- prepare_weekly_emg, prepare_baseline_questionnaires (alternative aggregations)
- compute_composite_score, align_sensor_questionnaire (data harmonization)
- reset_registry (testing only)
""")
