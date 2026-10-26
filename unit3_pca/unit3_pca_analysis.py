"""
Unit 3: Principal Component Analysis (PCA)
Computational Statistics Coursework — Customer Behaviour Analysis & Segmentation
Dataset: Online Retail II (N = 5,878 customers)

Syllabus Topics Covered:
  1. Basic mathematical concept of Principal Component Analysis.
  2. Standardization and spectral decomposition of the correlation matrix.
  3. Eigenvalues, explained variance, and cumulative explained variance.
  4. Determining the number of components to retain (Kaiser criterion, scree plot, variance threshold).
  5. Component loadings and behavioural interpretation.
  6. Projection of customer observations onto the first two principal components.
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
ROOT_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "unit3_pca" else Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "phase1_output" / "customer_master.csv"
OUTPUT_DIR = ROOT_DIR / "unit3_pca" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ============================================================================
# 1. DATA LOADING & FEATURE SELECTION
# ============================================================================
print("=" * 65)
print("UNIT 3: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 65)
print("\n--- Step 1: Loading Data and Preparing Features ---")

df = pd.read_csv(DATA_PATH)
n_customers = len(df)
print(f"Loaded {n_customers:,} customer records from {DATA_PATH}.")

# Selected continuous behavioural features (p = 5):
# 1. RecencyDays: Days since last order (inactivity)
# 2. Frequency: Count of distinct purchase invoices (order rate)
# 3. GrossMonetary: Total customer lifetime spend in GBP (revenue scale)
# 4. AvgOrderValue: Mean spend per invoice (basket expenditure)
# 5. UniqueProducts: Count of distinct catalog product codes (breadth)
features = [
    "RecencyDays",
    "Frequency",
    "GrossMonetary",
    "AvgOrderValue",
    "UniqueProducts",
]

X_raw = df[features].copy()

# Step 1a: Log1p transformation to mitigate severe power-law skewness
X_log = np.log1p(X_raw)
X_log.columns = [f"ln_{col}" for col in features]

# Step 1b: Standardization (zero mean, unit variance)
# z_ij = (x_ij - mean_j) / std_j
# This ensures variables on different units/scales contribute equally to variance.
means = X_log.mean()
stds = X_log.std(ddof=1)
Z = (X_log - means) / stds

print(f"Standardized {Z.shape[1]} features across all {Z.shape[0]:,} customers.")


# ============================================================================
# 2. SPECTRAL DECOMPOSITION (EIGEN-ANALYSIS OF CORRELATION MATRIX)
# ============================================================================
print("\n--- Step 2: Eigen-Decomposition of Correlation Matrix ---")

# Because features are standardized, the sample covariance of Z is
# identically the sample correlation matrix R of X_log:
# R = (1 / (n - 1)) * Z^T * Z
R = np.cov(Z.values, rowvar=False, ddof=1)
p = len(features)

# Solve the characteristic equation: det(R - lambda * I) = 0
# R * v_j = lambda_j * v_j
eigenvalues, eigenvectors = np.linalg.eigh(R)

# Sort eigenvalues and eigenvectors in descending order
sort_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

# Total variance in standardized space equals trace(R) = p = 5
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Build explained variance summary table
pc_names = [f"PC{i+1}" for i in range(p)]
exp_var_df = pd.DataFrame({
    "Component": pc_names,
    "Eigenvalue": eigenvalues,
    "Variance_Explained_Pct": explained_variance_ratio * 100.0,
    "Cumulative_Variance_Pct": cumulative_variance_ratio * 100.0,
})

exp_var_path = OUTPUT_DIR / "explained_variance.csv"
exp_var_df.to_csv(exp_var_path, index=False)
print(f"Saved explained variance table to: {exp_var_path}")
print(exp_var_df.round(4).to_string(index=False))


# ============================================================================
# 3. DECIDING HOW MANY COMPONENTS TO RETAIN
# ============================================================================
print("\n--- Step 3: Component Retention Decision ---")

# Criteria for retention:
# 1. Kaiser Criterion (Eigenvalue > 1.0): Retains PC1 (lambda_1 = 3.1915).
#    However, lambda_2 = 0.9502 is near-Kaiser (boundary case).
# 2. Cumulative Variance Threshold (>= 80%):
#    - PC1 alone: 63.83% variance
#    - PC1 + PC2: 82.83% variance (crosses the 80% standard benchmark)
# 3. Scree Plot Elbow Rule: Clear elbow occurs after PC2; components 3 to 5
#    flatten into the residual scree.
kaiser_count = int(np.sum(eigenvalues >= 1.0))
retained_components = 2
retained_variance_pct = cumulative_variance_ratio[retained_components - 1] * 100.0

print(f"Kaiser Criterion (> 1.0): {kaiser_count} component(s)")
print(f"Selected Component Count: {retained_components} components (PC1, PC2)")
print(f"Cumulative Variance Retained: {retained_variance_pct:.2f}% of total customer variance")


# ============================================================================
# 4. COMPONENT LOADINGS (CORRELATION BETWEEN VARIABLES AND COMPONENTS)
# ============================================================================
print("\n--- Step 4: Component Loadings & Interpretation ---")

# In standardized PCA, the component loading (correlation between variable i and PC j) is:
# Loading_ij = Corr(Z_i, PC_j) = v_ij * sqrt(lambda_j)
# where v_ij is the eigenvector weight.
loadings = eigenvectors * np.sqrt(eigenvalues)

loadings_df = pd.DataFrame(
    loadings,
    index=features,
    columns=pc_names
)
loadings_path = OUTPUT_DIR / "pca_loadings.csv"
loadings_df.to_csv(loadings_path)
print(f"Saved PCA loadings table to: {loadings_path}")
print(loadings_df.round(4).to_string())

# Also save eigenvector coefficients (directional weights)
weights_df = pd.DataFrame(
    eigenvectors,
    index=features,
    columns=pc_names
)
weights_path = OUTPUT_DIR / "pca_eigenvectors.csv"
weights_df.to_csv(weights_path)


# ============================================================================
# 5. COMPUTING PRINCIPAL COMPONENT SCORES
# ============================================================================
print("\n--- Step 5: Computing Customer PC Scores ---")

# Transform customer data into principal component coordinates:
# PC_scores = Z * V (where V is the matrix of eigenvectors)
pc_scores = Z.values @ eigenvectors

# Store PC1 and PC2 in DataFrame for visualization and analysis
scores_df = pd.DataFrame({
    "CustomerID": df["CustomerID"],
    "PC1": pc_scores[:, 0],
    "PC2": pc_scores[:, 1],
    "GrossMonetary": df["GrossMonetary"],
    "Frequency": df["Frequency"],
})


# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n--- Step 6: Generating Visualizations ---")

# Plot 1: Scree Plot with Cumulative Explained Variance
fig, ax1 = plt.subplots(figsize=(8, 5))

x_ticks = np.arange(1, p + 1)

# Left y-axis: Eigenvalues (Scree)
color_bar = "#1f77b4"
ax1.bar(x_ticks, eigenvalues, color=color_bar, alpha=0.6, width=0.45, label="Eigenvalue (Variance)")
ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Kaiser Cutoff (Eigenvalue = 1.0)")
ax1.set_xlabel("Principal Component", fontsize=11)
ax1.set_ylabel("Eigenvalue", color=color_bar, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color_bar)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(pc_names)

# Right y-axis: Cumulative Explained Variance
ax2 = ax1.twinx()
color_line = "#2ca02c"
ax2.plot(x_ticks, cumulative_variance_ratio * 100.0, color=color_line, marker="o", linewidth=2, label="Cumulative Variance (%)")
ax2.axhline(80.0, color="orange", linestyle=":", linewidth=1.5, label="80% Threshold")
ax2.set_ylabel("Cumulative Explained Variance (%)", color=color_line, fontsize=11)
ax2.tick_params(axis="y", labelcolor=color_line)
ax2.set_ylim(0, 105)

# Annotate PC2 retention point
ax2.annotate(
    f"Retained (PC1+PC2): {retained_variance_pct:.1f}%",
    xy=(2, retained_variance_pct),
    xytext=(2.3, 70),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    fontsize=9,
    fontweight="bold"
)

plt.title("Figure 1: PCA Scree Plot & Cumulative Explained Variance", fontsize=12)
fig.tight_layout()
scree_path = OUTPUT_DIR / "scree_plot.png"
plt.savefig(scree_path, dpi=200)
plt.close()
print(f"Saved scree plot to: {scree_path}")


# Plot 2: Scatter Plot of First Two Principal Components (PC1 vs PC2)
fig, ax = plt.subplots(figsize=(8, 6))

# Subsample for clear visual presentation
plot_rng = np.random.default_rng(RANDOM_SEED)
sample_idx = plot_rng.choice(n_customers, size=min(1200, n_customers), replace=False)
sample_scores = scores_df.iloc[sample_idx]

scatter = ax.scatter(
    sample_scores["PC1"],
    sample_scores["PC2"],
    c=np.log1p(sample_scores["GrossMonetary"]),
    cmap="viridis",
    alpha=0.6,
    s=20,
    edgecolor="none"
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r"Customer Spend: $\ln(\mathrm{GrossMonetary} + 1)$")

ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

ax.set_title("Figure 2: Customer Projections on First Two Principal Components", fontsize=12)
ax.set_xlabel(f"PC1: Lifetime Engagement & Revenue Scale ({explained_variance_ratio[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2: Order Economics (Basket Size vs Frequency) ({explained_variance_ratio[1]*100:.1f}% var)")

# Quadrant labels for viva explanation
ax.text(0.95, 0.95, "High Engagement / Bulk Basket", transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text(0.95, 0.05, "High Engagement / Frequent Small Orders", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text(0.05, 0.50, "Low Engagement / Dormant", transform=ax.transAxes, ha="left", va="center", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
scatter_path = OUTPUT_DIR / "pca_scatter_pc1_pc2.png"
plt.savefig(scatter_path, dpi=200)
plt.close()
print(f"Saved PC1 vs PC2 scatter plot to: {scatter_path}")

print("\n" + "=" * 65)
print("UNIT 3 PCA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)
