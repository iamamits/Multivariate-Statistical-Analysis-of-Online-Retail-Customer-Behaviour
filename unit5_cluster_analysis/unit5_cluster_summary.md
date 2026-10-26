# Unit 5 Summary: Cluster Analysis (K-Means)

**Coursework:** Computational Statistics — Customer Behaviour Analysis & Segmentation  
**Dataset:** Online Retail II (`customer_master.csv`)  
**Student Level:** Undergraduate Engineering  
**Analyzed Population:** $N = 5,878$ eligible customers (complete population retained)  

---

## 1. Objective & Statistical Purpose
The primary objective of Unit 5 is to implement **Cluster Analysis** (specifically **K-Means Clustering**) to discover natural, data-driven behavioural segments among retail customers. Unlike Unit 2 (Linear Discriminant Analysis), which was supervised and required known class labels, Cluster Analysis is an **unsupervised learning method** that partitions heterogeneous customer records into homogeneous, mutually exclusive groups based purely on their geometric proximities in multivariate space.

---

## 2. Statistical Foundation & Distance Metric

### 1. Distance Metric & The Necessity of Standardization:
K-Means clustering relies on the **squared Euclidean distance** between customer observation vectors $\mathbf{z}_i$ and cluster centroids $\boldsymbol{\mu}_j$:
$$d^2(\mathbf{z}_i, \boldsymbol{\mu}_j) = \|\mathbf{z}_i - \boldsymbol{\mu}_j\|^2 = \sum_{k=1}^p (z_{ik} - \mu_{jk})^2$$

- **Why Log-Transformation is Essential:** Raw retail features exhibit severe power-law skewness (e.g., `GrossMonetary` skewness = 25.07). Applying $\ln(1 + x)$ linearizes power-law distributions and prevents extreme buyers from single-handedly pulling cluster centroids away from the bulk population.
- **Why Standardization is Essential:** Original features are measured on drastically different physical scales:
  - `RecencyDays`: 0 to 739 days
  - `Frequency`: 1 to 398 orders
  - `GrossMonetary`: £3.00 to £580,000+
  Without standardization, `GrossMonetary` (having variance $> 2 \times 10^8$) would completely overwhelm the Euclidean distance metric, reducing clustering to a 1D spend threshold. Transforming features to standard $Z$-scores ($z_{ik} = (x_{ik} - \bar{x}_k)/s_k$) equalizes each dimension's variance to 1.0, ensuring balanced geometric contributions.

### 2. K-Means Optimization Objective:
The algorithm finds a partition $C = \{C_1, C_2, \dots, C_k\}$ minimizing the **Within-Cluster Sum of Squares (WCSS / Inertia)**:
$$W(C) = \sum_{j=1}^k \sum_{i \in C_j} \|\mathbf{z}_i - \boldsymbol{\mu}_j\|^2, \quad \text{where } \boldsymbol{\mu}_j = \frac{1}{|C_j|} \sum_{i \in C_j} \mathbf{z}_i$$
Optimization is performed using Lloyd's alternating minimization with **K-Means++ initialization** over 10 independent random restarts with a fixed random seed (`42`).

---

## 3. Candidate Cluster Evaluation ($k = 2$ to $k = 6$)

We evaluated candidate cluster counts from $k = 2$ to $k = 6$ across both **Inertia (Elbow Method)** and the **Silhouette Coefficient**:

| Candidate $k$ | Inertia (WCSS) | Marginal WCSS Drop ($\Delta$) | Marginal Drop (%) | Silhouette Score |
|:---:|---:|---:|---:|---:|
| **$k = 2$** | 17,181.12 | — | — | **0.3540** |
| **$k = 3$** | **13,681.59** | **3,499.53** | **20.37%** | **0.2643** |
| **$k = 4$** | 11,809.75 | 1,871.84 | 13.68% | 0.2425 |
| **$k = 5$** | 10,535.46 | 1,274.29 | 10.79% | 0.2394 |
| **$k = 6$** | 9,576.38 | 959.08 | 9.10% | 0.2330 |

### Honest Reconciliation of Discrepant Suggestions:
- **Silhouette Score Suggestion:** Peaks at **$k = 2$ ($0.3540$)**. The silhouette coefficient mathematically penalizes points close to cluster boundaries ($s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$). Because smaller $k$ creates fewer boundaries, $k = 2$ naturally yields the highest average silhouette.
- **Elbow Method Suggestion:** Exhibits a pronounced inflection at **$k = 3$**. Moving from $k = 2$ to $k = 3$ reduces WCSS by **3,499.53** (20.37%), whereas moving from $k = 3$ to $k = 4$ yields a much smaller marginal reduction of **1,871.84** (13.68%), after which the curve flattens out into diminishing returns.
- **Selected Cluster Count ($k = 3$):**  
  While $k = 2$ has the highest mathematical silhouette score, it merely produces a trivial binary split (active vs. inactive). Selecting **$k = 3$** captures the distinct 3-tier behavioural reality of the customer base (High Engagement, Moderate Activity, and Inactive/Dormant) while maintaining a solid positive silhouette score ($0.2643$) without over-fragmenting the customer population.

---

## 4. Cluster Profiles ($k = 3$) on Original Feature Scales

*Important: Cluster numbers (0, 1, 2) are arbitrary algorithmic labels assigned by K-Means; they do not represent quality rankings or ordinal grades.*

| Cluster | Count ($n$) | Population (%) | Recency (Days) Mean (Med) | Frequency (Orders) Mean (Med) | Gross Spend (£) Mean (Med) | Avg Order Value (£) Mean (Med) | Unique Products Mean (Med) |
|:---|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **Cluster 0** | 2,551 | 43.40% | 186.4 (107.0) | 3.5 (3.0) | £1,256.92 (£964.83) | £440.01 (£324.84) | 61.1 (49.0) |
| **Cluster 1** | 1,838 | 31.27% | 353.9 (383.0) | 1.6 (1.0) | £254.52 (£224.06) | £178.42 (£157.34) | 16.4 (13.0) |
| **Cluster 2** | 1,489 | 25.33% | 38.6 (18.0) | 16.9 (11.0) | £9,201.21 (£4,275.22) | £546.47 (£363.76) | 198.8 (160.0) |

### Behavioural Interpretation of Clusters:
1. **Cluster 0 — Moderate Engagement / Occasional Purchasers ($43.4\%$):**
   - Intermediate recency (mean 186.4 days, median 107.0 days).
   - Moderate order frequency (median 3.0 orders) and steady spend (£964.83 median total spend).
   - Moderate catalog exploration (median 49 unique products).
2. **Cluster 1 — Inactive / Dormant Purchasers ($31.3\%$):**
   - High dormancy (mean 353.9 days, median 383.0 days — inactive for over a full year).
   - Predominantly single-order or few-order buyers (median 1.0 order, mean 1.59 orders).
   - Lowest total spend (£224.06 median) and narrow product trial (median 13 unique items).
3. **Cluster 2 — High Engagement / Frequent Purchasers ($25.3\%$):**
   - Extremely recent activity (mean 38.6 days, median 18.0 days).
   - Highly frequent repeat buyers (mean 16.9 orders, median 11.0 orders).
   - Substantial cumulative revenue (£4,275.22 median, £9,201.21 mean) and extensive catalog breadth (median 160.0 unique items).

---

## 5. Diagnostic Visualizations

1. **`cluster_selection_plot.png`:**
   - Panel (a) plots WCSS / Inertia across $k \in \{2, 3, 4, 5, 6\}$, visually highlighting the elbow inflection at $k = 3$.
   - Panel (b) plots average Silhouette Scores, illustrating the mathematical peak at $k = 2$ and the retained balance at $k = 3$ ($0.2643$).
2. **`cluster_visualization_pc1_pc2.png`:**
   - 2D scatter plot projecting all $5,878$ customers and cluster centroids onto Principal Components $PC_1$ (activity scale, 63.83% variance) and $PC_2$ (spend intensity vs. frequency, 19.00% variance).
   - Shows clean geometric separation along $PC_1$: Cluster 1 (dormant) occupies the negative $PC_1$ region (mean $PC_1 = -1.96$), Cluster 0 (moderate) centers around $PC_1 = +0.06$, and Cluster 2 (high engagement) occupies the high positive $PC_1$ region (mean $PC_1 = +2.31$).
   - *Crucial Methodological Note:* K-Means clustering was executed on all **5 standardized behavioural dimensions**; PCA was utilized strictly as a post-hoc dimension reduction tool for 2D visualization without altering cluster memberships.

---

## 6. Output Files (`unit5_cluster_analysis/outputs/`)

- `cluster_assignments.csv`: Customer-level table mapping all 5,878 `CustomerID`s to assigned clusters (0, 1, 2).
- `cluster_profiles.csv`: Full statistical summary containing customer counts, percentages, and original-scale means and medians for all 5 features.
- `cluster_selection_metrics.csv`: Numerical table of candidate $k$, WCSS, marginal drops, and silhouette scores.
- `cluster_selection_plot.png`: Dual-panel Elbow and Silhouette diagnostic figure.
- `cluster_visualization_pc1_pc2.png`: 2D scatter plot of customer clusters projected onto $PC_1$ and $PC_2$ with overlaid centroids.

---

## 7. Assumptions & Limitations

1. **Spherical Cluster Assumption:** K-Means assumes isotropic, spherical clusters of roughly equal variance. Because customer behaviour forms a continuous multivariate density gradient rather than isolated physical islands, boundaries represent optimal geometric partitions rather than discrete natural species.
2. **Sensitivity to Initial Seed:** K-Means can converge to local minima depending on initial centroid placement. This was mitigated by using K-Means++ initialization across 10 random restarts with a fixed seed (`42`).
3. **Scale Sensitivity:** Clustering outcomes depend heavily on preprocessing choices ($\ln(1+x)$ and z-scores); unstandardized data would yield completely different, degenerate clusters driven solely by gross spend.
