"""
Unit 2: Linear Discriminant Analysis (LDA)
Computational Statistics Coursework — Customer Behaviour Analysis & Segmentation
Dataset: Online Retail II (N = 5,878 customers)

Syllabus Topics Covered:
  1. Statistical background of Linear Discriminant Analysis (within-class vs. between-class variation).
  2. Class mean vectors and pooled within-class covariance matrix.
  3. Fisher's linear discriminant function and Bayes classification rule.
  4. Assumptions and limitations of LDA.
  5. Distinction between discriminant analysis, regression, and clustering.
  6. Practical demonstration using retail customer data with strict leakage prevention.
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

# Paths: support execution from repository root or unit directory
ROOT_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "unit2_discriminant_analysis" else Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "phase1_output" / "customer_master.csv"
OUTPUT_DIR = ROOT_DIR / "unit2_discriminant_analysis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ============================================================================
# 1. DATA LOADING & TRANSPARENT CLASS DEFINITION
# ============================================================================
print("=" * 65)
print("UNIT 2: LINEAR DISCRIMINANT ANALYSIS (LDA)")
print("=" * 65)
print("\n--- Step 1: Loading Data and Defining Target Classes ---")

df = pd.read_csv(DATA_PATH)
n_total = len(df)
print(f"Loaded customer master with {n_total:,} customers.")

# The raw retail dataset does not contain pre-existing ground-truth class labels.
# For demonstrating discriminant analysis, we operationally define a binary target:
# - Calculate median GrossMonetary across all eligible customers.
# - Lower-spend customers: GrossMonetary < median
# - Higher-spend customers: GrossMonetary >= median
median_threshold = float(df["GrossMonetary"].median())
print(f"Operational Median Spend Threshold: £{median_threshold:.2f}")

df["SpendClass"] = np.where(df["GrossMonetary"] >= median_threshold, "Higher-spend", "Lower-spend")

class_summary = pd.DataFrame({
    "Class": ["Lower-spend (Class 0)", "Higher-spend (Class 1)"],
    "Definition": [f"GrossMonetary < £{median_threshold:.2f}", f"GrossMonetary >= £{median_threshold:.2f}"],
    "Count": [(df["SpendClass"] == "Lower-spend").sum(), (df["SpendClass"] == "Higher-spend").sum()],
})
class_summary["Proportion"] = class_summary["Count"] / n_total

class_summary_path = OUTPUT_DIR / "class_summary.csv"
class_summary.to_csv(class_summary_path, index=False)
print(f"Saved class definition summary to: {class_summary_path}")
print(class_summary.to_string(index=False))


# ============================================================================
# 2. PREDICTORS SELECTION & LEAKAGE PREVENTION
# ============================================================================
print("\n--- Step 2: Selecting Predictors with Leakage Prevention ---")

# Strict Leakage Prevention Rule:
# Because target classes are derived directly from GrossMonetary:
# - We STRICTLY EXCLUDE 'GrossMonetary' and 'AvgOrderValue' (monetary variables).
# - We use non-monetary behavioural predictors:
#   1. RecencyDays: Days since most recent purchase (inactivity measure)
#   2. Frequency: Count of distinct purchase invoices (engagement measure)
#   3. UniqueProducts: Count of distinct catalog items purchased (breadth measure)
predictor_names = ["RecencyDays", "Frequency", "UniqueProducts"]
print(f"Selected non-monetary predictors: {predictor_names}")

# Retail count features exhibit high positive skew; we apply log1p: ln(1 + x)
X_raw = df[predictor_names].values
X_log = np.log1p(X_raw)
transformed_names = [f"ln_{p}" for p in predictor_names]

# Binary target array: 0 = Lower-spend, 1 = Higher-spend
y = (df["SpendClass"] == "Higher-spend").astype(int).values


# ============================================================================
# 3. STRATIFIED TRAIN-TEST SPLIT (75% Train, 25% Test)
# ============================================================================
print("\n--- Step 3: Stratified Train-Test Split (Seed = 42) ---")

# To ensure strict statistical integrity:
# Model parameters must be estimated solely on the training set.
# The test set remains completely untouched until final evaluation.
rng = np.random.default_rng(RANDOM_SEED)

idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]

rng.shuffle(idx_0)
rng.shuffle(idx_1)

n0_train = int(len(idx_0) * 0.75)
n1_train = int(len(idx_1) * 0.75)

train_idx = np.concatenate([idx_0[:n0_train], idx_1[:n1_train]])
test_idx = np.concatenate([idx_0[n0_train:], idx_1[n1_train:]])

rng.shuffle(train_idx)
rng.shuffle(test_idx)

X_train, y_train = X_log[train_idx], y[train_idx]
X_test, y_test = X_log[test_idx], y[test_idx]

print(f"Total Customers:   {n_total:,}")
print(f"Training Set (75%): {len(train_idx):,} (Class 0: {(y_train == 0).sum():,}, Class 1: {(y_train == 1).sum():,})")
print(f"Testing Set (25%):  {len(test_idx):,} (Class 0: {(y_test == 0).sum():,}, Class 1: {(y_test == 1).sum():,})")


# ============================================================================
# 4. LINEAR DISCRIMINANT ANALYSIS (ESTIMATION ON TRAINING DATA)
# ============================================================================
print("\n--- Step 4: Parameter Estimation on Training Set ---")

n0 = int((y_train == 0).sum())
n1 = int((y_train == 1).sum())
n_train = n0 + n1

# 1. Class-Specific Mean Vectors:
# mu_0 = (1 / n0) * sum_{i in C0} x_i
# mu_1 = (1 / n1) * sum_{i in C1} x_i
mu_0 = np.mean(X_train[y_train == 0], axis=0)
mu_1 = np.mean(X_train[y_train == 1], axis=0)

mean_vectors_df = pd.DataFrame({
    "Predictor": transformed_names,
    "Mean_LowerSpend_Class0": mu_0,
    "Mean_HigherSpend_Class1": mu_1,
    "Difference (Class1 - Class0)": mu_1 - mu_0,
})
mean_vectors_path = OUTPUT_DIR / "class_mean_vectors.csv"
mean_vectors_df.to_csv(mean_vectors_path, index=False)
print(f"Saved class mean vectors to: {mean_vectors_path}")
print(mean_vectors_df.to_string(index=False))

# 2. Individual Class Covariance Matrices:
S_0 = np.cov(X_train[y_train == 0], rowvar=False, ddof=1)
S_1 = np.cov(X_train[y_train == 1], rowvar=False, ddof=1)

# 3. Pooled Within-Class Covariance Matrix:
# Assumption of LDA: Equal covariance matrices in both classes (Sigma_0 = Sigma_1 = Sigma).
# S_pool = ( (n0 - 1)*S_0 + (n1 - 1)*S_1 ) / (n0 + n1 - 2)
S_pool = ((n0 - 1) * S_0 + (n1 - 1) * S_1) / (n_train - 2)

pooled_cov_df = pd.DataFrame(S_pool, index=transformed_names, columns=transformed_names)
pooled_cov_path = OUTPUT_DIR / "pooled_covariance_matrix.csv"
pooled_cov_df.to_csv(pooled_cov_path)
print(f"\nSaved pooled within-class covariance matrix to: {pooled_cov_path}")
print(pooled_cov_df.round(4).to_string())

# 4. Fisher's Linear Discriminant Coefficients:
# Maximizes between-class variance relative to within-class variance:
# w = S_pool^(-1) * (mu_1 - mu_0)
inv_S_pool = np.linalg.inv(S_pool)
w = inv_S_pool @ (mu_1 - mu_0)

# Prior probabilities
pi_0 = n0 / n_train
pi_1 = n1 / n_train

# 5. Optimal Bayes Classification Rule Threshold:
# Discriminant function: d(x) = w^T * x - c
# Threshold: c = 0.5 * w^T * (mu_1 + mu_0) - ln(pi_1 / pi_0)
# Since pi_1 = pi_0 = 0.5, ln(pi_1 / pi_0) = 0, so c is the exact midpoint.
c = 0.5 * float(w @ (mu_1 + mu_0)) - float(np.log(pi_1 / pi_0))

lda_coeff_df = pd.DataFrame({
    "Predictor": transformed_names,
    "Discriminant_Weight_w": w,
    "Decision_Threshold_c": c,
})
lda_coeff_path = OUTPUT_DIR / "lda_coefficients.csv"
lda_coeff_df.to_csv(lda_coeff_path, index=False)
print(f"\nSaved LDA discriminant weights to: {lda_coeff_path}")
print(lda_coeff_df.to_string(index=False))
print(f"Optimal Decision Threshold (c): {c:.4f}")


# ============================================================================
# 5. EVALUATION ON UNTOUCHED TEST SET
# ============================================================================
print("\n--- Step 5: Evaluating Model on Untouched Test Set ---")

# Compute discriminant score for test observations:
# score(x) = w^T * x
test_scores = X_test @ w

# Classification rule:
# If score >= c -> Class 1 (Higher-spend)
# If score <  c -> Class 0 (Lower-spend)
y_pred_test = (test_scores >= c).astype(int)

# 1. Confusion Matrix
# Rows: Actual (0: Lower, 1: Higher), Columns: Predicted (0: Lower, 1: Higher)
tn = int(np.sum((y_test == 0) & (y_pred_test == 0)))
fp = int(np.sum((y_test == 0) & (y_pred_test == 1)))
fn = int(np.sum((y_test == 1) & (y_pred_test == 0)))
tp = int(np.sum((y_test == 1) & (y_pred_test == 1)))

conf_matrix_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual Lower-spend (0)", "Actual Higher-spend (1)"],
    columns=["Predicted Lower-spend (0)", "Predicted Higher-spend (1)"]
)
conf_matrix_path = OUTPUT_DIR / "confusion_matrix.csv"
conf_matrix_df.to_csv(conf_matrix_path)
print(f"Saved confusion matrix to: {conf_matrix_path}")
print(conf_matrix_df.to_string())

# 2. Performance Metrics
n_test = len(y_test)
accuracy = (tp + tn) / n_test

# Majority class baseline
baseline_accuracy = max(np.mean(y_test == 0), np.mean(y_test == 1))

# Precision, Recall, F1 for Higher-spend (Class 1)
prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

# Precision, Recall, F1 for Lower-spend (Class 0)
prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

metrics_df = pd.DataFrame([
    {"Metric": "Overall Test Accuracy", "Value": round(accuracy, 4)},
    {"Metric": "Majority-Class Baseline Accuracy", "Value": round(baseline_accuracy, 4)},
    {"Metric": "Accuracy Gain over Baseline", "Value": round(accuracy - baseline_accuracy, 4)},
    {"Metric": "Precision (Higher-spend)", "Value": round(prec_1, 4)},
    {"Metric": "Recall (Higher-spend)", "Value": round(rec_1, 4)},
    {"Metric": "F1-Score (Higher-spend)", "Value": round(f1_1, 4)},
    {"Metric": "Precision (Lower-spend)", "Value": round(prec_0, 4)},
    {"Metric": "Recall (Lower-spend)", "Value": round(rec_0, 4)},
    {"Metric": "F1-Score (Lower-spend)", "Value": round(f1_0, 4)},
])
metrics_path = OUTPUT_DIR / "test_evaluation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"\nSaved test evaluation metrics to: {metrics_path}")
print(metrics_df.to_string(index=False))


# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n--- Step 6: Generating Diagnostic Visualizations ---")

# Plot 1: 1D Linear Discriminant Score Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
    data=pd.DataFrame({"Score": test_scores, "Class": np.where(y_test == 1, "Higher-spend (Actual)", "Lower-spend (Actual)")}),
    x="Score", hue="Class", bins=35, kde=True, palette=["#2ca02c", "#1f77b4"], alpha=0.5, ax=ax
)
ax.axvline(c, color="red", linestyle="--", linewidth=2, label=f"Decision Cutoff (c = {c:.2f})")
ax.set_title("Test-Set 1D Linear Discriminant Score Distribution", fontsize=12)
ax.set_xlabel(r"Discriminant Score: $y = \mathbf{w}^T \mathbf{x}$")
ax.set_ylabel("Customer Count")
ax.legend(loc="upper right")
plt.tight_layout()
score_plot_path = OUTPUT_DIR / "discriminant_score_distribution.png"
plt.savefig(score_plot_path, dpi=200)
plt.close()
print(f"Saved discriminant score distribution plot to: {score_plot_path}")

# Plot 2: Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 13})
ax.set_title(f"Test-Set Confusion Matrix (Accuracy: {accuracy*100:.2f}%)", fontsize=12)
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
conf_plot_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(conf_plot_path, dpi=200)
plt.close()
print(f"Saved confusion matrix heatmap to: {conf_plot_path}")

# Plot 3: 2D Bivariate Class Separation in Predictor Space
fig, ax = plt.subplots(figsize=(8, 6))
plot_rng = np.random.default_rng(RANDOM_SEED)
scatter_sample = plot_rng.choice(n_test, size=min(800, n_test), replace=False)
scatter_df = pd.DataFrame({
    "ln_Frequency": X_test[scatter_sample, 1],
    "ln_UniqueProducts": X_test[scatter_sample, 2],
    "Class": np.where(y_test[scatter_sample] == 1, "Higher-spend", "Lower-spend"),
})
sns.scatterplot(
    data=scatter_df, x="ln_Frequency", y="ln_UniqueProducts", hue="Class",
    palette={"Lower-spend": "#1f77b4", "Higher-spend": "#2ca02c"}, alpha=0.6, s=30, ax=ax
)
ax.set_title("Class Separation: Order Frequency vs. Catalog Breadth", fontsize=12)
ax.set_xlabel(r"$\ln(\mathrm{Frequency} + 1)$")
ax.set_ylabel(r"$\ln(\mathrm{UniqueProducts} + 1)$")
plt.tight_layout()
separation_plot_path = OUTPUT_DIR / "lda_class_separation.png"
plt.savefig(separation_plot_path, dpi=200)
plt.close()
print(f"Saved 2D class separation plot to: {separation_plot_path}")

print("\n" + "=" * 65)
print("UNIT 2 DISCRIMINANT ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)
