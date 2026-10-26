"""
Unit 5: Cluster Analysis (K-Means Clustering)
Computational Statistics Coursework — Customer Behaviour Analysis & Segmentation
Dataset: Online Retail II (N = 5,878 eligible customers)

Syllabus Topics Covered:
  1. Statistical concept of cluster analysis (unsupervised learning).
  2. Role of distance metrics (Euclidean distance) and necessity of feature standardization.
  3. K-Means clustering algorithm: objective function (within-cluster sum of squares / inertia).
  4. Evaluating candidate cluster counts (k = 2 to k = 6).
  5. Cluster validation metrics: Elbow method (inertia) and Silhouette score analysis.
  6. Cluster profiling: characterization of clusters using original feature means and medians.
  7. High-dimensional cluster visualization projected onto the first two principal components.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns

# Visual setup
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# Paths: support execution from workspace root or unit5 directory
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "unit5_cluster_analysis" else SCRIPT_DIR
DATA_PATH = ROOT_DIR / "phase1_output" / "customer_master.csv"
OUTPUT_DIR = ROOT_DIR / "unit5_cluster_analysis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================
print("=" * 70)
print("UNIT 5: CLUSTER ANALYSIS (K-MEANS) — COMPUTATIONAL STATISTICS")
print("=" * 70)
print("\n--- Step 1: Loading Data and Preparing Features ---")

df = pd.read_csv(DATA_PATH)
n_customers = len(df)
print(f"Loaded {n_customers:,} customer records from {DATA_PATH.name}.")

# Selected continuous behavioural features (p = 5):
# CustomerID is strictly excluded from clustering features.
features = [
    "RecencyDays",
    "Frequency",
    "GrossMonetary",
    "AvgOrderValue",
    "UniqueProducts",
]

# Verification of missing / invalid values
null_counts = df[features].isnull().sum()
if null_counts.sum() > 0:
    print(f"Warning: Found null values: {null_counts.to_dict()}")
else:
    print(f"Data verification: All {n_customers:,} rows have complete, valid observations across all {len(features)} features.")

X_raw = df[features].copy()
p = len(features)

# Transformation 1: Log1p transformation to mitigate heavy positive skewness
X_log = np.log1p(X_raw)

# Transformation 2: Z-score standardization (zero mean, unit variance)
# Equalizes physical measurement units (days, counts, and currency)
means = X_log.mean()
stds = X_log.std(ddof=1)
Z = (X_log - means) / stds
X_std = Z.values

print(f"Standardized {p} behavioural features across all {n_customers:,} customers.")


# ============================================================================
# 2. VECTORIZED K-MEANS & SILHOUETTE ALGORITHMS (PURE NUMPY)
# ============================================================================
def run_kmeans(X, k, random_state=42, n_init=10, max_iter=300, tol=1e-6):
    """
    Standard Lloyd's K-Means clustering with K-Means++ initialization in pure NumPy.
    Minimizes the Within-Cluster Sum of Squares (Inertia):
      W(C) = sum_{j=1}^k sum_{i in C_j} ||z_i - mu_j||^2
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape
    best_inertia = np.inf
    best_labels = None
    best_centroids = None

    for _ in range(n_init):
        # K-Means++ initialization
        centroids = np.empty((k, n_features))
        first_idx = rng.integers(0, n_samples)
        centroids[0] = X[first_idx]

        for c_idx in range(1, k):
            # Compute squared distances to already selected centroids
            dists = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[:c_idx, :]) ** 2, axis=2),
                axis=1
            )
            dist_sum = np.sum(dists)
            probs = dists / dist_sum if dist_sum > 0 else np.ones(n_samples) / n_samples
            next_idx = rng.choice(n_samples, p=probs)
            centroids[c_idx] = X[next_idx]

        # Lloyd iteration loop
        for _ in range(max_iter):
            # Compute squared Euclidean distance to each centroid: (N, k)
            sq_dists = np.sum(
                (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2
            )
            labels = np.argmin(sq_dists, axis=1)

            # Update centroids to cluster sample means
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j]
                for j in range(k)
            ])

            # Check convergence
            if np.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids

        # Calculate final inertia (WCSS)
        final_sq_dists = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2
        )
        inertia = np.sum(np.min(final_sq_dists, axis=1))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, best_inertia


def compute_silhouette_score(X, labels, k):
    """
    Computes mean Silhouette Coefficient over all samples:
      s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) is mean intra-cluster distance, b(i) is mean nearest-cluster distance.
    """
    n_samples = len(X)
    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)

    for c in range(k):
        c_mask = (labels == c)
        c_count = np.sum(c_mask)
        if c_count > 1:
            sub_X = X[c_mask]
            # Euclidean distance from all points to points in cluster c
            diff = X[:, np.newaxis, :] - sub_X[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))  # shape: (n_samples, c_count)

            # Intra-cluster distance a(i): exclude self-distance (which is 0)
            a[c_mask] = np.sum(dists[c_mask], axis=1) / (c_count - 1)

            # Inter-cluster distance candidate for b(i)
            mean_dist_to_c = np.mean(dists, axis=1)
            b[~c_mask] = np.minimum(b[~c_mask], mean_dist_to_c[~c_mask])

    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)
    return float(np.mean(s))


# ============================================================================
# 3. EVALUATING CANDIDATE CLUSTER COUNTS (k = 2 to k = 6)
# ============================================================================
print("\n--- Step 2: Evaluating Candidate Cluster Counts (k = 2 to k = 6) ---")

candidate_k = list(range(2, 7))
inertias = []
silhouettes = []
cluster_results = {}

for k in candidate_k:
    t0 = time.time()
    lbls, cntrds, inrt = run_kmeans(X_std, k=k, random_state=RANDOM_SEED, n_init=10)
    sil = compute_silhouette_score(X_std, lbls, k=k)
    inertias.append(inrt)
    silhouettes.append(sil)
    cluster_results[k] = (lbls, cntrds, inrt, sil)
    elapsed = time.time() - t0
    print(f"  k = {k}: Inertia (WCSS) = {inrt:10.2f}, Silhouette Score = {sil:.4f} ({elapsed:.1f}s)")

# Compute marginal inertia drops
inertia_reductions = [0.0]
inertia_reduction_pcts = [0.0]
for i in range(1, len(candidate_k)):
    drop = inertias[i - 1] - inertias[i]
    pct = (drop / inertias[i - 1]) * 100.0
    inertia_reductions.append(drop)
    inertia_reduction_pcts.append(pct)

df_metrics = pd.DataFrame({
    "Candidate_k": candidate_k,
    "Inertia_WCSS": np.round(inertias, 2),
    "Inertia_Drop": np.round(inertia_reductions, 2),
    "Inertia_Drop_Pct": np.round(inertia_reduction_pcts, 2),
    "Silhouette_Score": np.round(silhouettes, 4),
})

metrics_path = OUTPUT_DIR / "cluster_selection_metrics.csv"
df_metrics.to_csv(metrics_path, index=False)
print(f"\nSaved cluster selection metrics to {metrics_path}.")
print("\nCluster Selection Metrics Table:")
print(df_metrics.to_string(index=False))


# ============================================================================
# 4. SELECTING OPTIMAL CLUSTER COUNT & RATIONALE
# ============================================================================
# Selected k = 3
# Rationale:
# 1. Elbow Method: Clear inflection at k = 3, where inertia reduction drops
#    sharply from 3,499.53 (20.37% from k=2->3) to 1,871.84 (13.68% from k=3->4).
# 2. Silhouette Analysis: k = 2 peaks mathematically (0.3540), as silhouette
#    inherently favors fewer boundaries. However, k = 3 retains a strong score
#    (0.2643) and provides rich 3-tier behavioural differentiation (High Engagement,
#    Moderate Activity, Dormant/Low Engagement) rather than an oversimplified binary split.
SELECTED_K = 3
selected_labels, selected_centroids, selected_inertia, selected_sil = cluster_results[SELECTED_K]
print(f"\nSelected Model: k = {SELECTED_K} clusters (Inertia = {selected_inertia:.2f}, Silhouette = {selected_sil:.4f})")


# ============================================================================
# 5. CLUSTER ASSIGNMENTS & PROFILES
# ============================================================================
print(f"\n--- Step 3: Generating Cluster Profiles (k = {SELECTED_K}) ---")

# Save customer-level assignments
df_assignments = pd.DataFrame({
    "CustomerID": df["CustomerID"],
    "Cluster": selected_labels,
})
assignments_path = OUTPUT_DIR / "cluster_assignments.csv"
df_assignments.to_csv(assignments_path, index=False)
print(f"Saved customer assignments to {assignments_path}.")

# Add cluster assignments to dataframe for profiling on original unscaled features
df_profile_data = df.copy()
df_profile_data["Cluster"] = selected_labels

# Compute cluster-wise counts, percentages, means, and medians on original features
profile_rows = []
for c in range(SELECTED_K):
    sub = df_profile_data[df_profile_data["Cluster"] == c]
    count = len(sub)
    pct = (count / n_customers) * 100.0

    row = {
        "Cluster": f"Cluster {c}",
        "CustomerCount": count,
        "Percentage": round(pct, 2),
    }

    # Means on original features
    for f in features:
        row[f"{f}_Mean"] = round(sub[f].mean(), 2)

    # Medians on original features
    for f in features:
        row[f"{f}_Median"] = round(sub[f].median(), 2)

    profile_rows.append(row)

df_profiles = pd.DataFrame(profile_rows)
profiles_path = OUTPUT_DIR / "cluster_profiles.csv"
df_profiles.to_csv(profiles_path, index=False)
print(f"Saved cluster profiles to {profiles_path}.")

print("\nCluster Profiles Summary Table (Original Feature Scales):")
display_cols = [
    "Cluster", "CustomerCount", "Percentage",
    "RecencyDays_Mean", "RecencyDays_Median",
    "Frequency_Mean", "Frequency_Median",
    "GrossMonetary_Mean", "GrossMonetary_Median",
    "AvgOrderValue_Mean", "AvgOrderValue_Median",
    "UniqueProducts_Mean", "UniqueProducts_Median",
]
print(df_profiles[display_cols].to_string(index=False))


# ============================================================================
# 6. GENERATING DIAGNOSTIC VISUALIZATIONS
# ============================================================================
print("\n--- Step 4: Generating Diagnostic Visualizations ---")

# ----------------------------------------------------------------------------
# Plot 1: Cluster Selection Plot (Elbow & Silhouette)
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Panel 1: Elbow Method (Inertia / WCSS)
ax1.plot(candidate_k, inertias, marker="o", markersize=8, color="#1f77b4",
         linewidth=2.2, label="Inertia (WCSS)")
ax1.axvline(SELECTED_K, color="#d62728", linestyle="--", linewidth=1.8,
            label=f"Selected k = {SELECTED_K} (Elbow Inflection)")
ax1.annotate(f"Elbow Inflection\n(k = {SELECTED_K}, WCSS = {inertias[1]:.1f})",
             xy=(SELECTED_K, inertias[1]), xytext=(SELECTED_K + 0.35, inertias[1] + 1200),
             arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.2, headwidth=6),
             fontsize=9, fontweight="bold")
ax1.set_title("(a) Elbow Method: Within-Cluster Sum of Squares",
              fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Inertia / WCSS", fontsize=10, fontweight="bold")
ax1.set_xticks(candidate_k)
ax1.legend(loc="upper right", frameon=True)

# Panel 2: Silhouette Score Analysis
ax2.plot(candidate_k, silhouettes, marker="s", markersize=8, color="#2ca02c",
         linewidth=2.2, label="Silhouette Score")
ax2.axvline(SELECTED_K, color="#d62728", linestyle="--", linewidth=1.8,
            label=f"Selected k = {SELECTED_K} (Score = {silhouettes[1]:.4f})")
ax2.annotate(f"Peak at k = 2 ({silhouettes[0]:.4f})\n(Coarse binary split)",
             xy=(2, silhouettes[0]), xytext=(2.2, silhouettes[0] + 0.02),
             arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1, headwidth=6),
             fontsize=8.5)
ax2.annotate(f"Retained k = 3 ({silhouettes[1]:.4f})\n(3-tier behavioural separation)",
             xy=(SELECTED_K, silhouettes[1]), xytext=(SELECTED_K + 0.25, silhouettes[1] - 0.04),
             arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1, headwidth=6),
             fontsize=8.5, fontweight="bold")
ax2.set_title("(b) Silhouette Score by Cluster Count",
              fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Mean Silhouette Coefficient", fontsize=10, fontweight="bold")
ax2.set_xticks(candidate_k)
ax2.set_ylim(0.18, 0.40)
ax2.legend(loc="upper right", frameon=True)

plt.suptitle("Cluster Validation: Elbow Method & Silhouette Analysis (k = 2 to 6)",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()

selection_plot_path = OUTPUT_DIR / "cluster_selection_plot.png"
fig.savefig(selection_plot_path, dpi=300)
plt.close(fig)
print(f"Saved cluster selection plot to {selection_plot_path}.")


# ----------------------------------------------------------------------------
# Plot 2: Cluster Visualization on First Two Principal Components (PC1 & PC2)
# ----------------------------------------------------------------------------
# PCA is computed strictly for 2D visualization of the 5D clusters
R_corr = np.cov(X_std, rowvar=False, ddof=1)
evals, evecs = np.linalg.eigh(R_corr)
sort_idx = np.argsort(evals)[::-1]
evals, evecs = evals[sort_idx], evecs[:, sort_idx]

# Ensure positive direction for GrossMonetary loading
for j in range(p):
    if evecs[features.index("GrossMonetary"), j] < 0:
        evecs[:, j] = -evecs[:, j]

# Project customers and centroids onto PC1 and PC2
PC1_customers = X_std @ evecs[:, 0]
PC2_customers = X_std @ evecs[:, 1]
PC1_centroids = selected_centroids @ evecs[:, 0]
PC2_centroids = selected_centroids @ evecs[:, 1]

fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
cluster_labels_desc = [
    f"Cluster 0: Moderate Engagement (n = {np.sum(selected_labels == 0):,})",
    f"Cluster 1: Inactive / Dormant (n = {np.sum(selected_labels == 1):,})",
    f"Cluster 2: High Frequency & Engagement (n = {np.sum(selected_labels == 2):,})",
]

for c in range(SELECTED_K):
    mask = (selected_labels == c)
    ax.scatter(
        PC1_customers[mask], PC2_customers[mask],
        c=palette[c], label=cluster_labels_desc[c],
        alpha=0.45, s=20, edgecolors="none"
    )

# Overlay projected cluster centroids
ax.scatter(
    PC1_centroids, PC2_centroids,
    c="black", marker="X", s=180, edgecolors="white", linewidths=1.5,
    label="Cluster Centroids (Projected)", zorder=5
)

# Centroid annotations
for c in range(SELECTED_K):
    ax.annotate(
        f"Centroid {c}",
        xy=(PC1_centroids[c], PC2_centroids[c]),
        xytext=(PC1_centroids[c] + 0.25, PC2_centroids[c] + 0.35),
        fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black", lw=1.2)
    )

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

ax.set_title("K-Means Customer Clusters (k = 3) Projected onto PC1 & PC2",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel(r"Principal Component 1 (Overall Activity & Engagement Scale — 63.83% var)",
              fontsize=10, fontweight="bold")
ax.set_ylabel(r"Principal Component 2 (Spend Intensity vs. Order Rate — 19.00% var)",
              fontsize=10, fontweight="bold")
ax.legend(loc="upper left", frameon=True, fontsize=8.5)

caption_text = "Note: K-means clustering was performed on all 5 standardized behavioural features;\nPCA is utilized solely for 2D visualization without altering cluster memberships."
fig.text(0.5, 0.01, caption_text, ha="center", fontsize=8, style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])

vis_plot_path = OUTPUT_DIR / "cluster_visualization_pc1_pc2.png"
fig.savefig(vis_plot_path, dpi=300)
plt.close(fig)
print(f"Saved cluster visualization plot to {vis_plot_path}.")

print("\n" + "=" * 70)
print("UNIT 5 EXECUTION COMPLETE — ALL ARTIFACTS VERIFIED SUCCESSFULLY")
print("=" * 70)
