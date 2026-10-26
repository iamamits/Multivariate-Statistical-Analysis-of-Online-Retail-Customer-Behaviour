# Unit 4 Summary: Factor Analysis (FA)

**Coursework:** Computational Statistics — Customer Behaviour Analysis & Segmentation  
**Dataset:** Online Retail II (`customer_master.csv`)  
**Student Level:** Undergraduate Engineering  
**Analyzed Population:** $N = 5,878$ eligible customers (complete population retained)  

---

## 1. Objective & Statistical Purpose
The primary objective of Unit 4 is to fit an **Orthogonal Factor Analysis Model** to customer behavioural data. While Unit 3 (PCA) condensed overall variance via linear combinations of observed variables, Factor Analysis aims to uncover the underlying **unobserved latent factors** that generate the observed correlations, explicitly partitioning the variance of each feature into:
1. **Common Variance (Communality, $h_i^2$):** Variance shared with other features via common underlying factors.
2. **Unique / Specific Variance (Uniqueness, $u_i^2 = \psi_i$):** Variance unique to that specific feature, including measurement noise and feature-specific variation.

---

## 2. Mathematical Factor Analysis Model

### 1. Orthogonal Factor Model:
Let $\mathbf{z} = (z_1, z_2, \dots, z_p)^T$ be a standardized random vector with $E[\mathbf{z}] = \mathbf{0}$ and $\text{Cov}(\mathbf{z}) = \mathbf{R}$ ($p = 5$). An $m$-factor orthogonal factor model expresses observed variables as:
$$\mathbf{z} = \mathbf{L} \mathbf{F} + \boldsymbol{\epsilon}$$
In scalar form for variable $i$ ($i = 1, \dots, p$):
$$z_i = \ell_{i1} F_1 + \ell_{i2} F_2 + \dots + \ell_{im} F_m + \epsilon_i$$
where:
- $\mathbf{L} = [\ell_{ij}] \in \mathbb{R}^{p \times m}$ is the **factor loading matrix** ($\ell_{ij}$ is the loading of variable $i$ on common factor $j$).
- $\mathbf{F} = (F_1, \dots, F_m)^T$ are unobserved **common factors**, assumed standardized and uncorrelated: $E[\mathbf{F}] = \mathbf{0}$ and $\text{Cov}(\mathbf{F}) = \mathbf{I}_m$.
- $\boldsymbol{\epsilon} = (\epsilon_1, \dots, \epsilon_p)^T$ are **unique factors** (specific variance plus measurement error), assumed uncorrelated with each other and with the common factors: $E[\boldsymbol{\epsilon}] = \mathbf{0}$, $\text{Cov}(\boldsymbol{\epsilon}) = \boldsymbol{\Psi} = \text{diag}(\psi_1, \dots, \psi_p)$, and $\text{Cov}(\mathbf{F}, \boldsymbol{\epsilon}) = \mathbf{0}$.

### 2. Variance & Correlation Matrix Decomposition:
Because variables are standardized ($\text{Var}(z_i) = 1$):
$$\text{Var}(z_i) = \sum_{j=1}^m \ell_{ij}^2 + \psi_i = h_i^2 + u_i^2 = 1$$
- **Communality ($h_i^2$):** $h_i^2 = \sum_{j=1}^m \ell_{ij}^2$ (proportion of variance explained by common factors).
- **Uniqueness ($u_i^2$):** $u_i^2 = \psi_i = 1 - h_i^2$ (proportion of variance specific to variable $i$).
- **Matrix Formulation:** $\mathbf{R} = \mathbf{L} \mathbf{L}^T + \boldsymbol{\Psi}$.

### 3. Factor Extraction Method:
We employ the **Principal Component Factor Method** (spectral decomposition of $\mathbf{R}$):
$$\mathbf{R} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T \implies \mathbf{L} = [\sqrt{\lambda_1}\mathbf{v}_1, \dots, \sqrt{\lambda_m}\mathbf{v}_m]$$
*Why this method?* It provides a closed-form, deterministic solution, requires no distributional normality assumptions, guarantees non-negative communalities ($\le 1$), and completely avoids improper **Heywood cases** ($h_i^2 > 1$) that arise when iterative factoring is applied to near-collinear retail features.

### 4. Varimax Orthogonal Rotation:
Unrotated loadings often exhibit substantial cross-loadings. We apply **Kaiser's Varimax rotation**, an orthogonal transformation $\mathbf{L}^* = \mathbf{L} \mathbf{T}$ (where $\mathbf{T}^T \mathbf{T} = \mathbf{I}$) that maximizes the variance of squared loadings:
$$V = \sum_{j=1}^m \left[ \sum_{i=1}^p (\ell_{ij}^*)^4 - \frac{1}{p} \left( \sum_{i=1}^p (\ell_{ij}^*)^2 \right)^2 \right]$$
**Key Invariant Property:** Orthogonal rotation redistributes variance across factors to achieve Thurstone's simple structure while keeping variable communalities and uniqueness **strictly unchanged** ($\sum_j (\ell_{ij}^*)^2 = \sum_j \ell_{ij}^2 = h_i^2$).

---

## 3. Selected Features and Data Preparation
- **Population:** All $N = 5,878$ eligible customers from `phase1_output/customer_master.csv`.
- **Features ($p = 5$):** `RecencyDays`, `Frequency`, `GrossMonetary`, `AvgOrderValue`, `UniqueProducts`. `CustomerID` is strictly excluded.
- **Transformations:**
  1. **Log1p transformation:** $\ln(1 + x)$ applied to all 5 features to mitigate heavy right-skewness.
  2. **Standardization:** Z-score scaling ($z_{ij} = (x_{ij} - \bar{x}_j)/s_j$) to equalize scales into unit variance.

---

## 4. Factor Retention Decision (Eigenvalues & Scree Plot)

| Factor | Eigenvalue ($\lambda_j$) | Variance Explained (%) | Cumulative Variance (%) | Strict Kaiser Status ($\lambda > 1.0$) | Selected in Study? |
|:---:|---:|---:|---:|:---:|:---:|
| **Factor 1** | **3.1915** | **63.83%** | **63.83%** | Retained ($\lambda_1 > 1.0$) | **Yes** |
| **Factor 2** | **0.9502** | **19.00%** | **82.83%** | Excluded ($\lambda_2 < 1.0$) | **Yes** (Near-Kaiser / Communality Rule) |
| **Factor 3** | 0.5260 | 10.52% | 93.35% | Excluded | No |
| **Factor 4** | 0.3292 | 6.58% | 99.94% | Excluded | No |
| **Factor 5** | 0.0031 | 0.06% | 100.00% | Excluded | No |

### Statistical Justification for Retaining $m = 2$ Factors:
We explicitly distinguish Factor Analysis retention from PCA variance heuristics:
1. **Strict Kaiser Criterion ($\lambda > 1.0$):** Retains **only 1 factor** ($\lambda_1 = 3.1915 > 1.0$, while $\lambda_2 = 0.9502 < 1.0$).
2. **Failure of 1-Factor Solution:**
   - Under $m = 1$, `AvgOrderValue` has a communality of only **$h^2 = 0.3577$**, leaving **$64.23\%$** of its variance as unexplained uniqueness.
   - `RecencyDays` has a communality of only **$h^2 = 0.4397$** ($56.03\%$ uniqueness).
   - A single factor cannot be rotated ($m < 2$), conflating transaction frequency with order spend intensity into a single general size factor.
3. **Scree Plot Elbow Criterion (`factor_scree_plot.png`):**
   - The scree curve shows a steep descent from Factor 1 to Factor 2, with the residual elbow leveling off smoothly after Factor 2 ($\lambda_3 = 0.5260, \lambda_4 = 0.3292, \lambda_5 = 0.0031$).
4. **Communality Adequacy & Interpretability:**
   - Retaining $m = 2$ factors brings all communalities to between **$68.6\%$ and $95.5\%$** and enables Varimax rotation to separate frequency from basket expenditure.

---

## 5. Factor Loadings, Communalities, and Uniqueness

### Factor Loading Matrix ($\mathbf{L}$):
| Feature | Unrotated $F_1$ | Unrotated $F_2$ | Varimax Rotated $F_1^*$ | Varimax Rotated $F_2^*$ |
|:---|---:|---:|---:|---:|
| **RecencyDays** | -0.6631 | +0.4963 | **-0.8048** | +0.1958 |
| **Frequency** | +0.8658 | -0.3181 | **+0.9212** | +0.0477 |
| **GrossMonetary** | +0.9627 | +0.1687 | **+0.8189** | **+0.5334** |
| **AvgOrderValue** | +0.5981 | +0.7577 | +0.2522 | **+0.9318** |
| **UniqueProducts** | +0.8472 | -0.0130 | **+0.7841** | +0.3210 |

### Communalities ($h_i^2$) and Uniqueness ($u_i^2$):
| Feature | 1-Factor $h^2$ | 1-Factor $u^2$ | 2-Factor $h^2$ | 2-Factor $u^2$ |
|:---|---:|---:|---:|---:|
| **RecencyDays** | 0.4397 | 0.5603 | **0.6860** | **0.3140** |
| **Frequency** | 0.7496 | 0.2504 | **0.8508** | **0.1492** |
| **GrossMonetary** | 0.9268 | 0.0732 | **0.9552** | **0.0448** |
| **AvgOrderValue** | 0.3577 | 0.6423 | **0.9318** | **0.0682** |
| **UniqueProducts** | 0.7177 | 0.2823 | **0.7179** | **0.2821** |

*Note: Varimax rotation matrix $\mathbf{T} = \begin{bmatrix} 0.9195 & 0.3930 \\ -0.3930 & 0.9195 \end{bmatrix}$ (rotation angle $\theta \approx 23.14^\circ$). Communalities under rotated and unrotated 2-factor solutions are identical.*

---

## 6. Empirical Factor Interpretation

### Rotated Factor 1 ($F_1^*$): Customer Engagement & Volume Activity
- **Dominant Loadings:** `Frequency` ($+0.921$), `GrossMonetary` ($+0.819$), `RecencyDays` ($-0.805$), `UniqueProducts` ($+0.784$).
- **Negligible Loading:** `AvgOrderValue` ($+0.252$).
- **Interpretation:** $F_1^*$ captures repeat interaction and active commercial engagement. Customers with high positive scores purchase frequently, explore diverse catalog items, generate substantial cumulative revenue, and have ordered recently (low recency days).

### Rotated Factor 2 ($F_2^*$): Per-Order Spending Intensity
- **Dominant Loading:** `AvgOrderValue` ($+0.9318$).
- **Secondary Loading:** `GrossMonetary` ($+0.5334$).
- **Low Loading:** `Frequency` ($+0.0477$).
- **Interpretation:** Factor 2 mainly represents `AvgOrderValue` (average expenditure per purchase order), with a secondary contribution from `GrossMonetary`. The factor loading on `Frequency` is very low ($+0.0477$), indicating that order frequency contributes negligibly to this factor. However, this low loading reflects only a near-zero linear relationship with Factor 2 within this model, and must not be interpreted as proof of statistical independence between order frequency and order value.

---

## 7. Honest Reporting of Limitations

1. **Dual Loading on `GrossMonetary`:**
   In an ideal simple structure, each variable loads on exactly one common factor. However, `GrossMonetary` exhibits substantial loadings on both $F_1^*$ ($0.819$) and $F_2^*$ ($0.533$). This is mathematically inevitable because gross revenue is by definition the product of order count and spend per order ($\ln \text{GrossMonetary} \approx \ln \text{Frequency} + \ln \text{AvgOrderValue}$). Total spend naturally bridges both underlying dimensions.
2. **Kaiser Boundary Case:**
   Strict Kaiser criterion ($\lambda > 1.0$) would retain only 1 factor. Factor 2 has $\lambda_2 = 0.9502$, slightly below the 1.0 cutoff. We retain Factor 2 because stopping at 1 factor leaves `AvgOrderValue` largely unaccounted for ($64.2\%$ uniqueness) and destroys the distinction between order frequency and order value.
3. **Avoidance of Exaggerated Categorization Claims:**
   We avoid labeling factors as "wholesale vs. retail types" or speculating on buyer business models without external firmographic evidence. The factors are interpreted strictly through their empirical loadings.

---

## 8. Diagnostic Plots Description

1. **`factor_scree_plot.png`:**
   Displays eigenvalues for all 5 factors against the red dashed Kaiser cutoff ($\lambda = 1.0$), highlighting the retention boundary at Factor 2 and the scree elbow leveling off from Factor 3 onward.
2. **`factor_loadings_plot.png`:**
   A two-panel plot illustrating the effect of Varimax rotation:
   - Panel (a) shows unrotated loadings where features lie diagonally in the interior.
   - Panel (b) shows rotated loadings where `Frequency`, `RecencyDays`, and `UniqueProducts` rotate cleanly onto the $F_1^*$ axis, while `AvgOrderValue` aligns with the $F_2^*$ axis.

---

## 9. Viva Revision & Defense Guide

1. **Q: How does Factor Analysis fundamentally differ from Principal Component Analysis?**  
   *A:* PCA is a variance-oriented data reduction method: components are exact orthogonal linear combinations of observed variables ($PC_j = \mathbf{v}_j^T \mathbf{z}$) that maximize total variance, assuming zero error. Factor Analysis is a latent variable model: observed variables are modeled as linear combinations of unobserved common factors plus variable-specific unique errors ($\mathbf{z} = \mathbf{L} \mathbf{F} + \boldsymbol{\epsilon}$). FA explicitly separates common variance ($h_i^2$) from unique/error variance ($u_i^2$).

2. **Q: What is the mathematical definition of communality ($h_i^2$) and uniqueness ($u_i^2$)?**  
   *A:* For standardized variables with unit variance ($\text{Var}(z_i) = 1$), communality $h_i^2 = \sum_{j=1}^m \ell_{ij}^2$ is the proportion of variable $i$'s variance accounted for by the $m$ common factors. Uniqueness $u_i^2 = \psi_i = 1 - h_i^2$ is the remaining variance specific to variable $i$ (including measurement noise) that is not shared with any other variable.

3. **Q: Why does Varimax rotation not change the communalities?**  
   *A:* Varimax is an orthogonal rotation matrix $\mathbf{T}$ satisfying $\mathbf{T} \mathbf{T}^T = \mathbf{I}$. Because the transformation is a pure rigid rotation in factor space, the row sum of squared loadings for any variable is invariant: $\sum_{j=1}^m (\ell_{ij}^*)^2 = [\mathbf{L}^* \mathbf{L}^{*T}]_{ii} = [\mathbf{L} \mathbf{T} \mathbf{T}^T \mathbf{L}^T]_{ii} = [\mathbf{L} \mathbf{L}^T]_{ii} = \sum_{j=1}^m \ell_{ij}^2 = h_i^2$.

4. **Q: Why did we select 2 factors even though strict Kaiser's rule yields only 1?**  
   *A:* Under strict Kaiser ($\lambda > 1.0$), only Factor 1 ($\lambda_1 = 3.1915$) is retained. However, a 1-factor model leaves `AvgOrderValue` with $64.23\%$ uniqueness (unexplained) and cannot be rotated. Factor 2 has $\lambda_2 = 0.9502$ (near-cutoff) and marks the elbow of the scree plot. Retaining 2 factors raises all communalities to $\ge 68.6\%$ and enables Varimax rotation to separate frequency from order spend intensity.
