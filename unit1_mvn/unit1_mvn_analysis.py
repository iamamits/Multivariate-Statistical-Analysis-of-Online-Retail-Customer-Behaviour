"""
Unit 1: Multivariate Normal Distribution (MVN)
Computational Statistics Coursework — Customer Behaviour Analysis & Segmentation
Dataset: Online Retail II (N = 5,878 customers)

Syllabus Topics Covered:
  1. Multivariate normal distribution and its probability density function.
  2. Mean vector, covariance matrix, and correlation matrix estimation.
  3. Basic descriptive statistics for selected customer features.
  4. Appropriate distribution plots and normal Q-Q plots (raw vs. log1p).
  5. Conditional multivariate normal distribution.
  6. Relationship between conditional MVN and linear regression.
  7. Numerical demonstration using the actual customer dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# Paths
DATA_PATH = Path("phase1_output/customer_master.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("../phase1_output/customer_master.csv")

OUTPUT_DIR = Path("unit1_mvn/outputs")
if not OUTPUT_DIR.parent.exists():
    OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. DATA LOADING & FEATURE SELECTION
# ============================================================================
print("--- Step 1: Loading Data and Selecting Features ---")
df = pd.read_csv(DATA_PATH)
n_customers = len(df)
print(f"Loaded {n_customers:,} customer records from {DATA_PATH}.")

# Selected 5 continuous behavioural features:
# - RecencyDays: Days since most recent purchase (inactivity measure)
# - Frequency: Total number of distinct purchase invoices (loyalty measure)
# - GrossMonetary: Total customer purchase spend in GBP (monetary scale)
# - AvgOrderValue: Average spend per invoice (basket expenditure size)
# - UniqueProducts: Number of distinct catalog items purchased (basket breadth)
features = [
    "RecencyDays",
    "Frequency",
    "GrossMonetary",
    "AvgOrderValue",
    "UniqueProducts",
]

X_raw = df[features].copy()

# Retail customer data has extreme positive skewness (Pareto tail).
# The natural log transformation ln(1 + x) stabilizes variance and brings
# the distributions closer to symmetry.
X_log = np.log1p(X_raw)
X_log.columns = [f"ln_{col}" for col in features]


# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n--- Step 2: Descriptive Statistics ---")


def get_summary_table(data_df):
    """Compute basic univariate descriptive statistics."""
    rows = []
    for col in data_df.columns:
        vals = data_df[col].values
        rows.append({
            "Feature": col,
            "Mean": np.mean(vals),
            "StdDev": np.std(vals, ddof=1),
            "Median": np.median(vals),
            "Min": np.min(vals),
            "Max": np.max(vals),
            "Skewness": pd.Series(vals).skew(),
            "Kurtosis": pd.Series(vals).kurt(),
        })
    return pd.DataFrame(rows)


stats_raw = get_summary_table(X_raw)
stats_raw["Transformation"] = "Raw"

stats_log = get_summary_table(X_log)
stats_log["Transformation"] = "Log1p"

desc_stats = pd.concat([stats_raw, stats_log], ignore_index=True)
desc_stats_path = OUTPUT_DIR / "descriptive_statistics.csv"
desc_stats.to_csv(desc_stats_path, index=False)
print(f"Saved descriptive statistics to: {desc_stats_path}")
print(desc_stats[["Feature", "Mean", "StdDev", "Skewness", "Kurtosis"]].round(3).to_string(index=False))


# ============================================================================
# 3. PARAMETER ESTIMATION (MEAN VECTOR, COVARIANCE, CORRELATION)
# ============================================================================
print("\n--- Step 3: Multivariate Parameter Estimation ---")
n, p = X_log.shape

# 1. Sample Mean Vector (mu_hat)
# Formula: x_bar = (1 / n) * sum(x_i)
mean_vec = X_log.mean()
mean_vec_df = pd.DataFrame({"Feature": X_log.columns, "SampleMean": mean_vec.values})
mean_vec_path = OUTPUT_DIR / "mean_vector.csv"
mean_vec_df.to_csv(mean_vec_path, index=False)
print(f"Saved sample mean vector to: {mean_vec_path}")

# 2. Sample Covariance Matrix (S)
# Formula: S = (1 / (n - 1)) * sum( (x_i - x_bar)(x_i - x_bar)^T )
# Note: Bessel's correction uses (n - 1) for unbiased estimation.
# Population MLE uses (1 / n), differing here by only (5877 / 5878) = 0.99983.
cov_matrix = X_log.cov()
cov_matrix_path = OUTPUT_DIR / "covariance_matrix.csv"
cov_matrix.to_csv(cov_matrix_path)
print(f"Saved sample covariance matrix to: {cov_matrix_path}")

# 3. Sample Correlation Matrix (R)
# Formula: r_jk = s_jk / sqrt(s_jj * s_kk)
corr_matrix = X_log.corr()
corr_matrix_path = OUTPUT_DIR / "correlation_matrix.csv"
corr_matrix.to_csv(corr_matrix_path)
print(f"Saved sample correlation matrix to: {corr_matrix_path}")


# ============================================================================
# 4. NORMALITY DIAGNOSTIC PLOTS
# ============================================================================
print("\n--- Step 4: Generating Normality Diagnostic Plots ---")

# Plot 1: Feature Distributions (Raw vs Log1p Transformed)
fig, axes = plt.subplots(5, 2, figsize=(11, 14))
fig.suptitle("Feature Distributions: Raw vs. Log1p Transformed", fontsize=14, y=0.995)

for i, col in enumerate(features):
    # Raw on left
    axes[i, 0].hist(X_raw[col], bins=40, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    axes[i, 0].set_title(f"Raw {col} (Skewness: {stats_raw.loc[i, 'Skewness']:.2f})")
    axes[i, 0].set_ylabel("Density")

    # Log1p on right with fitted Gaussian curve
    vals_log = X_log[f"ln_{col}"]
    axes[i, 1].hist(vals_log, bins=40, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
    mu_col, std_col = vals_log.mean(), vals_log.std()
    x_axis = np.linspace(vals_log.min(), vals_log.max(), 100)
    normal_curve = (1.0 / (std_col * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - mu_col) / std_col) ** 2)
    axes[i, 1].plot(x_axis, normal_curve, "r--", lw=2, label="Gaussian Fit")
    axes[i, 1].set_title(f"Log1p {col} (Skewness: {stats_log.loc[i, 'Skewness']:.2f})")
    axes[i, 1].legend(loc="upper right", fontsize=8)

plt.tight_layout()
dist_plot_path = OUTPUT_DIR / "feature_distributions.png"
plt.savefig(dist_plot_path, dpi=200)
plt.close()
print(f"Saved distribution comparison plot to: {dist_plot_path}")

# Plot 2: Normal Q-Q Plots
fig, axes = plt.subplots(5, 2, figsize=(11, 14))
fig.suptitle("Normal Q-Q Plots: Raw vs. Log1p Transformed", fontsize=14, y=0.995)

# Vectorised theoretical normal quantiles
probs = (np.arange(1, n + 1) - 0.5) / n
q = 2 * probs - 1.0
a = 0.147
term1 = 2.0 / (np.pi * a) + np.log(1.0 - q**2) / 2.0
term2 = np.log(1.0 - q**2) / a
erf_inv = np.sign(q) * np.sqrt(np.maximum(np.sqrt(np.maximum(term1**2 - term2, 0.0)) - term1, 0.0))
z_theor = np.sqrt(2.0) * erf_inv

for i, col in enumerate(features):
    # Raw Q-Q
    sorted_raw = np.sort((X_raw[col] - X_raw[col].mean()) / X_raw[col].std())
    axes[i, 0].scatter(z_theor, sorted_raw, s=3, color="#1f77b4", alpha=0.4)
    axes[i, 0].plot([-3.5, 3.5], [-3.5, 3.5], "r--", lw=1.5)
    axes[i, 0].set_title(f"Q-Q: Raw {col}")
    axes[i, 0].set_xlabel("Theoretical Normal Quantiles")
    axes[i, 0].set_ylabel("Standardized Sample")

    # Log1p Q-Q
    sorted_log = np.sort((X_log[f"ln_{col}"] - X_log[f"ln_{col}"].mean()) / X_log[f"ln_{col}"].std())
    axes[i, 1].scatter(z_theor, sorted_log, s=3, color="#2ca02c", alpha=0.4)
    axes[i, 1].plot([-3.5, 3.5], [-3.5, 3.5], "r--", lw=1.5)
    axes[i, 1].set_title(f"Q-Q: Log1p {col}")
    axes[i, 1].set_xlabel("Theoretical Normal Quantiles")
    axes[i, 1].set_ylabel("Standardized Sample")

plt.tight_layout()
qq_plot_path = OUTPUT_DIR / "normal_qq_plots.png"
plt.savefig(qq_plot_path, dpi=200)
plt.close()
print(f"Saved normal Q-Q plots to: {qq_plot_path}")

# Plot 3: Correlation Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="Blues", vmin=-0.6, vmax=1.0, cbar=True, ax=ax)
ax.set_title("Pearson Correlation Matrix (Transformed Space)", fontsize=12)
plt.tight_layout()
heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=200)
plt.close()
print(f"Saved correlation heatmap to: {heatmap_path}")


# ============================================================================
# 5. CONDITIONAL MVN DISTRIBUTION & LINEAR REGRESSION
# ============================================================================
print("\n--- Step 5: Conditional MVN Distribution and Linear Regression ---")

# Theoretical Setup:
# Partition the random vector X into:
#   Y = ln_GrossMonetary (Target: Customer spend scale)
#   X_2 = [ln_RecencyDays, ln_Frequency, ln_UniqueProducts] (Conditioning predictors)
target_name = "ln_GrossMonetary"
pred_names = ["ln_RecencyDays", "ln_Frequency", "ln_UniqueProducts"]

y = X_log[target_name].values
X_preds = X_log[pred_names].values

mu_y = float(mean_vec[target_name])
mu_x = mean_vec[pred_names].values

# Partition the covariance matrix:
# S = [ s_yy   S_yx ]
#     [ S_xy   S_xx ]
s_yy = cov_matrix.loc[target_name, target_name]
S_yx = cov_matrix.loc[target_name, pred_names].values
S_xy = cov_matrix.loc[pred_names, target_name].values
S_xx = cov_matrix.loc[pred_names, pred_names].values

# 1. Theoretical MVN Conditional Mean & Variance:
# Formula: E[Y | X = x] = mu_y + S_yx * S_xx^(-1) * (x - mu_x)
#          = beta_0 + beta^T * x
# where beta = S_xx^(-1) * S_xy,  beta_0 = mu_y - beta^T * mu_x
# Conditional Variance: Var(Y | X) = s_yy - S_yx * S_xx^(-1) * S_xy
inv_S_xx = np.linalg.inv(S_xx)
beta_mvn = inv_S_xx @ S_xy
beta0_mvn = mu_y - float(beta_mvn @ mu_x)
var_cond_mvn = float(s_yy - S_yx @ inv_S_xx @ S_xy)

# 2. Empirical Ordinary Least Squares (OLS) Regression:
# Model: Y = beta_0 + beta_1 * X_1 + beta_2 * X_2 + beta_3 * X_3 + epsilon
X_design = np.column_stack([np.ones(n), X_preds])
beta_ols = np.linalg.lstsq(X_design, y, rcond=None)[0]
beta0_ols = beta_ols[0]
beta_slopes_ols = beta_ols[1:]

y_pred = X_design @ beta_ols
residuals = y - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - mu_y) ** 2)
r_squared = 1.0 - (ss_res / ss_tot)
residual_var_ols = ss_res / (n - len(pred_names) - 1)

# Tabulate comparison
reg_comparison = pd.DataFrame({
    "Parameter": ["Intercept (beta_0)"] + [f"Slope_{col}" for col in pred_names] + ["Residual_Variance"],
    "Theoretical_MVN": [beta0_mvn] + list(beta_mvn) + [var_cond_mvn],
    "Empirical_OLS": [beta0_ols] + list(beta_slopes_ols) + [residual_var_ols],
})
reg_comp_path = OUTPUT_DIR / "conditional_regression.csv"
reg_comparison.to_csv(reg_comp_path, index=False)
print(f"Saved conditional regression results to: {reg_comp_path}")
print(reg_comparison.to_string(index=False))

max_slope_diff = np.max(np.abs(beta_mvn - beta_slopes_ols))
print(f"\nVerification of Equivalence:")
print(f"  Maximum absolute difference in slopes: {max_slope_diff:.2e}")
print(f"  Empirical R-squared (R^2): {r_squared:.4f}")

# Plot 4: Conditional Regression (Actual vs Predicted)
fig, ax = plt.subplots(figsize=(7, 6))
sample_idx = np.random.choice(n, size=min(1200, n), replace=False)
ax.scatter(y_pred[sample_idx], y[sample_idx], s=10, alpha=0.35, color="#1f77b4")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="1:1 Reference Line")
ax.set_title(f"Conditional Expectation Model: Actual vs Predicted Spend\n(R^2 = {r_squared:.4f})")
ax.set_xlabel(r"Predicted Conditional Spend: $E[Y \mid X = x]$")
ax.set_ylabel(r"Actual Spend: $Y = \ln(\mathrm{GrossMonetary})$")
ax.legend(loc="upper left")
plt.tight_layout()
reg_plot_path = OUTPUT_DIR / "conditional_regression_plot.png"
plt.savefig(reg_plot_path, dpi=200)
plt.close()
print(f"Saved conditional regression plot to: {reg_plot_path}")

print("\n" + "=" * 60)
print("UNIT 1 ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
