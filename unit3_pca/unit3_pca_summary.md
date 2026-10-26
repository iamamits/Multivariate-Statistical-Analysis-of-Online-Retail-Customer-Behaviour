# Unit 3 Summary: Principal Component Analysis (PCA)

**Coursework:** Computational Statistics — Customer Behaviour Analysis & Segmentation  
**Dataset:** Online Retail II (`customer_master.csv`)  
**Student Level:** Undergraduate Engineering  

---

## 1. Objective
The objective of Unit 3 is to apply **Principal Component Analysis (PCA)** to transform correlated retail customer behavioural features into an orthogonal set of linearly uncorrelated principal components. This achieves dimensionality reduction, illuminates the underlying latent dimensions of customer activity, and quantifies the proportion of total behavioural variance captured by low-dimensional representations.

---

## 2. Dataset and Selected Features
- **Population Analyzed:** $N = 5,878$ customers (all eligible customers retained; zero dropping).
- **Selected Continuous Features ($p = 5$ dimensions):**
  1. `RecencyDays`: Inactivity period (days since most recent purchase).
  2. `Frequency`: Total number of distinct purchase orders.
  3. `GrossMonetary`: Total customer purchase revenue in GBP (£).
  4. `AvgOrderValue`: Mean spend per purchase invoice (£).
  5. `UniqueProducts`: Count of distinct catalog product codes purchased.

*`CustomerID` is an identifier and is strictly excluded from PCA.*

- **Transformations & Standardization:**
  1. **Log1p Transformation:** $\ln(1 + x)$ is applied to all 5 features to mitigate severe positive skewness (e.g., raw monetary skewness = 25.07).
  2. **Standardization (z-score):** $z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$. Standardizing ensures that features measured on different physical scales (days, counts, and currency) contribute with equal weight to the variance decomposition.

---

## 3. Main Mathematical Formulas

1. **Correlation Matrix ($\mathbf{R}$):**
   Because variables are standardized, the sample covariance of $\mathbf{Z}$ equals the sample correlation matrix $\mathbf{R}$ of the log-transformed data:
   $$\mathbf{R} = \frac{1}{n-1} \mathbf{Z}^T \mathbf{Z} \in \mathbb{R}^{p \times p}$$

2. **Spectral Decomposition (Eigen-Analysis):**
   Solving the characteristic equation:
   $$\det(\mathbf{R} - \lambda \mathbf{I}) = 0 \implies \mathbf{R} \mathbf{v}_j = \lambda_j \mathbf{v}_j \quad (j = 1, \dots, p)$$
   where eigenvalues are sorted: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0$, and eigenvectors $\mathbf{v}_j$ are orthonormal ($\mathbf{v}_j^T \mathbf{v}_k = \delta_{jk}$).

3. **Total & Explained Variance:**
   $$\text{Total Variance} = \text{tr}(\mathbf{R}) = \sum_{j=1}^p \lambda_j = p = 5$$
   $$\text{Explained Variance Ratio of } PC_j = \frac{\lambda_j}{\sum_{k=1}^p \lambda_k} = \frac{\lambda_j}{p}$$
   $$\text{Cumulative Explained Variance} = \frac{\sum_{k=1}^j \lambda_k}{p}$$

4. **Component Loadings:**
   The Pearson correlation between original standardized feature $Z_i$ and principal component $PC_j$:
   $$\text{Loading}_{ij} = \text{Corr}(Z_i, PC_j) = v_{ij} \sqrt{\lambda_j}$$

5. **Principal Component Scores:**
   $$\mathbf{Y} = \mathbf{Z} \mathbf{V} \implies PC_{ij} = \sum_{k=1}^p z_{ik} v_{kj}$$

---

## 4. Actual Explained Variance Results

| Component | Eigenvalue ($\lambda_j$) | Variance Explained (%) | Cumulative Variance (%) | Strict Kaiser Criterion ($\lambda > 1.0$) | Study Retention Decision |
|:---:|---:|---:|---:|:---:|:---:|
| **PC1** | **3.1915** | **63.83%** | **63.83%** | Retained ($\lambda_1 > 1.0$) | **Retained** (Kaiser & 80% rule) |
| **PC2** | **0.9502** | **19.00%** | **82.83%** | Excluded ($\lambda_2 < 1.0$) | **Retained** ($\ge 80\%$ Cumulative Variance rule) |
| **PC3** | 0.5260 | 10.52% | 93.35% | Excluded ($\lambda_3 < 1.0$) | Excluded |
| **PC4** | 0.3292 | 6.58% | 99.94% | Excluded ($\lambda_4 < 1.0$) | Excluded |
| **PC5** | 0.0031 | 0.06% | 100.00% | Excluded ($\lambda_5 < 1.0$) | Excluded |

### Component Selection: Distinguishing Retention Criteria
Choosing the number of components requires distinguishing between two standard statistical criteria:

1. **Strict Kaiser-Guttman Criterion ($\lambda > 1.0$):**
   - Under this rule, a component is retained only if its eigenvalue exceeds 1.0 (explaining more variance than a single standardized original variable).
   - Here, $\lambda_1 = 3.1915 > 1.0$, while $\lambda_2 = 0.9502 < 1.0$.
   - **According to the strict Kaiser criterion, only PC1 is retained.**

2. **Cumulative Explained Variance Criterion ($\ge 80\%$ Threshold):**
   - In multivariate dimensionality reduction, relying solely on Kaiser's rule can be overly conservative when the first component leaves a large portion of variance unexplained. $PC_1$ alone accounts for only **63.83%** of the total customer variance.
   - By applying the standard undergraduate threshold of **$\ge 80\%$ cumulative explained variance**, retaining **2 components ($PC_1$ and $PC_2$)** captures **82.83%** of the total variance.
   - Retaining $PC_2$ preserves an additional 19.00% of variance that would otherwise be discarded.

3. **Scree Plot Elbow Criterion:**
   - The scree plot shows a steep drop from $PC_1$ (3.1915) to $PC_2$ (0.9502), after which the curve levels off smoothly through $PC_3$ (0.5260), $PC_4$ (0.3292), and $PC_5$ (0.0031), confirming that residual scree begins after $PC_2$.

**Conclusion:** We retain **2 components ($PC_1$ and $PC_2$)** based on the **80% cumulative explained variance threshold (achieving 82.83%)**, while clearly noting that strict Kaiser criterion alone selects only $PC_1$.

---

## 5. PCA Loadings & Behavioural Interpretation

| Original Feature | PC1 Loading | PC2 Loading | PC3 Loading | PC4 Loading | PC5 Loading |
|:---|---:|---:|---:|---:|---:|
| **RecencyDays** | **-0.6631** | +0.4963 | +0.5556 | +0.0729 | +0.0001 |
| **Frequency** | **+0.8658** | -0.3181 | +0.2707 | +0.2738 | -0.0302 |
| **GrossMonetary** | **+0.9627** | +0.1687 | +0.0638 | +0.1974 | +0.0416 |
| **AvgOrderValue** | **+0.5981** | **+0.7577** | -0.2598 | +0.0154 | -0.0220 |
| **UniqueProducts** | **+0.8472** | -0.0130 | +0.2691 | -0.4579 | -0.0008 |

### Behavioural Interpretation of Retained Components:
- **$PC_1$ — Overall Customer Lifetime Engagement & Activity Level (63.83% variance):**
  - Strongly positively loaded on `GrossMonetary` ($+0.963$), `Frequency` ($+0.866$), `UniqueProducts` ($+0.847$), and moderately on `AvgOrderValue` ($+0.598$).
  - Strongly negatively loaded on `RecencyDays` ($-0.663$).
  - **Mathematical Interpretation:** $PC_1$ functions as a general scale index of customer activity. High positive scores correspond to active customers with high spend, high order count, diverse product catalog exploration, and low inactivity (recent purchases). Negative scores represent inactive, low-spend, infrequent buyers.

- **$PC_2$ — Contrast: Higher Average Order Value & Recency vs. Frequency (19.00% variance):**
  - Dominated by a strong positive loading on `AvgOrderValue` ($+0.758$) and a positive loading on `RecencyDays` ($+0.496$).
  - Contrasted with a negative loading on `Frequency` ($-0.318$).
  - Minor positive loading on `GrossMonetary` ($+0.169$).
  - **Near-Zero Loading on `UniqueProducts` ($-0.0130$):** Catalog breadth has virtually no influence on $PC_2$, indicating that product variety does not differentiate customers along this axis.
  - **Mathematical Interpretation:** $PC_2$ represents a contrast between higher average spend per order accompanied by longer gaps between orders (`RecencyDays`) versus frequent repeat ordering (`Frequency`) with lower average spend per order. Unsupported customer categorization claims (such as "wholesale/bulk buyers" or "large-basket orders") are avoided; the dimension is interpreted strictly through its mathematical loadings.

---

## 6. Graphical Analysis

1. **Scree Plot & Cumulative Variance (`scree_plot.png`):**
   - Displays eigenvalues as bars on the left axis with the red dashed Kaiser cutoff ($\lambda = 1.0$).
   - Overlays the cumulative explained variance curve on the right axis, highlighting the retention point at $PC_2$ reaching **82.83%** (exceeding the 80% threshold).
2. **First Two Principal Components Scatter (`pca_scatter_pc1_pc2.png`):**
   - Plots customer coordinate projections $(PC_{i1}, PC_{i2})$ colored by log total spend.
   - Shows the continuous bivariate distribution of the customer population across $PC_1$ (overall activity scale) and $PC_2$ (contrast of higher `AvgOrderValue` and `RecencyDays` versus higher `Frequency`).

---

## 7. Useful Output Files (`unit3_pca/outputs/`)

- `explained_variance.csv`: Eigenvalues, variance explained percentages, and cumulative variance.
- `pca_loadings.csv`: $5 \times 5$ matrix of component loadings ($\text{Corr}(Z_i, PC_j)$).
- `pca_eigenvectors.csv`: $5 \times 5$ matrix of orthonormal eigenvector weights ($\mathbf{V}$).
- `scree_plot.png`: Dual-axis scree plot with eigenvalues, Kaiser threshold, and cumulative variance curve.
- `pca_scatter_pc1_pc2.png`: Scatter plot of customers projected onto $PC_1$ vs. $PC_2$.

---

## 8. Assumptions & Limitations
1. **Linearity Assumption:** PCA assumes linear relationships between features. While log-transformation linearizes the power-law tail, non-linear dependencies are not modeled by linear PCA.
2. **Sensitivity to Outliers:** Even after standardization, extreme values can exert leverage on the correlation matrix; log-transformation ($\ln(1+x)$) is essential to reduce extreme skewness.
3. **Information Loss:** Discarding components $PC_3$, $PC_4$, and $PC_5$ incurs an information loss of **17.17%**, which represents residual variation in the feature space.

---

## 9. Viva Revision & Defense Guide
1. **Q: Why did we perform PCA on the correlation matrix $\mathbf{R}$ rather than the covariance matrix $\mathbf{S}$?**  
   *A:* In our dataset, features are measured on drastically different units: `RecencyDays` (days, up to 739), `Frequency` (order counts, up to 398), and `GrossMonetary` (pounds, up to £580k). Performing PCA on the raw covariance matrix would allow `GrossMonetary` (variance $> 2 \times 10^8$) to completely dominate the first principal component simply because of its scale. Standardizing features ($Z$-scores) equalizes their variance to 1.0, making PCA equivalent to the eigen-decomposition of the correlation matrix $\mathbf{R}$.
2. **Q: Why retain 2 components if Kaiser's rule ($\lambda > 1$) only selects 1?**  
   *A:* By the strict Kaiser criterion ($\lambda > 1.0$), only $PC_1$ ($\lambda_1 = 3.1915$) is retained, because $\lambda_2 = 0.9502 < 1.0$. However, $PC_1$ alone accounts for only 63.83% of the total variance. In exploratory multivariate analysis, an 80% cumulative explained variance threshold is standard; retaining both $PC_1$ and $PC_2$ reaches **82.83% cumulative variance**, capturing the secondary dimension contrasting higher `AvgOrderValue` and `RecencyDays` against `Frequency` (with near-zero loading on `UniqueProducts`).
3. **Q: What is the mathematical difference between an eigenvector weight and a component loading?**  
   *A:* An eigenvector weight $v_{ij}$ is the directional coefficient in the linear combination defining the component ($PC_j = \sum v_{kj} Z_k$), constrained so $\sum v_{kj}^2 = 1$. A component loading is the Pearson correlation between the original variable and the component: $\text{Loading}_{ij} = v_{ij} \sqrt{\lambda_j}$.
