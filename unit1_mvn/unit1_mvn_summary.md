# Unit 1 Summary: Multivariate Normal Distribution (MVN)

**Coursework:** Computational Statistics — Customer Behaviour Analysis & Segmentation  
**Dataset:** Online Retail II (`customer_master.csv`)  
**Student Level:** Undergraduate Engineering  

---

## 1. Objective
The objective of Unit 1 is to study customer purchasing behaviour through the framework of the **Multivariate Normal (MVN) Distribution**. We estimate the joint mean vector and covariance matrix, assess univariate and joint normality, explore variance-stabilizing log transformations, and demonstrate the exact theoretical relationship between conditional MVN distributions and linear regression.

---

## 2. Dataset and Variables
- **Population Analyzed:** $N = 5,878$ eligible customers (100% of validated customer master; zero sampling).
- **Selected Behavioural Features ($p = 5$ continuous metrics):**
  1. `RecencyDays`: Inactivity period (days from customer's latest order to reference date).
  2. `Frequency`: Total count of distinct purchase orders/invoices.
  3. `GrossMonetary`: Total gross revenue spend in GBP (£).
  4. `AvgOrderValue`: Average spend per purchase invoice (£).
  5. `UniqueProducts`: Count of distinct catalog product codes transacted.

*Identifiers (`CustomerID`) and categoricals (`PrimaryCountry`) were excluded from continuous modelling.*

---

## 3. Main Mathematical Formulas

1. **Multivariate Normal Probability Density Function:**
   $$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$
   where $\mathbf{x}, \boldsymbol{\mu} \in \mathbb{R}^p$, and $\boldsymbol{\Sigma} \in \mathbb{R}^{p \times p}$ is symmetric positive-definite.

2. **Sample Mean Vector ($\bar{\mathbf{x}}$):**
   $$\bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i$$

3. **Sample Covariance Matrix ($\mathbf{S}$):**
   $$\mathbf{S} = \frac{1}{n-1} \sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$
   *Note: Bessel's correction uses $n-1$ for unbiasedness; the biased MLE uses $n$. For $n = 5,878$, the difference is negligible: $\frac{5,877}{5,878} \approx 0.99983$.*

4. **Sample Correlation Matrix ($\mathbf{R}$):**
   $$r_{jk} = \frac{s_{jk}}{\sqrt{s_{jj} s_{kk}}}, \quad -1 \le r_{jk} \le 1$$

5. **Conditional MVN Distribution:**
   Partitioning $\mathbf{X} = \begin{pmatrix} Y \\ \mathbf{X}_2 \end{pmatrix}$ with covariance $\mathbf{S} = \begin{pmatrix} s_{YY} & \mathbf{S}_{YX} \\ \mathbf{S}_{XY} & \mathbf{S}_{XX} \end{pmatrix}$:
   $$E[Y \mid \mathbf{X}_2 = \mathbf{x}_2] = \bar{Y} + \mathbf{S}_{YX} \mathbf{S}_{XX}^{-1} (\mathbf{x}_2 - \bar{\mathbf{X}}_2) = \beta_0 + \boldsymbol{\beta}^T \mathbf{x}_2$$
   $$\text{Var}(Y \mid \mathbf{X}_2 = \mathbf{x}_2) = s_{YY} - \mathbf{S}_{YX} \mathbf{S}_{XX}^{-1} \mathbf{S}_{XY}$$
   where conditional slopes are $\boldsymbol{\beta} = \mathbf{S}_{XX}^{-1} \mathbf{S}_{XY}$ and intercept is $\beta_0 = \bar{Y} - \boldsymbol{\beta}^T \bar{\mathbf{X}}_2$.

---

## 4. Transformations and Diagnostics
- **Raw Data Skewness:** Real retail metrics have severe power-law / Pareto tails. For example, raw `GrossMonetary` exhibits skewness of **25.07** (kurtosis 830.17) and `AvgOrderValue` exhibits skewness of **57.13** (kurtosis 3,879.66).
- **Log1p Transformation:** Applying $\ln(1 + x)$ dramatically contracts extreme commercial outliers. Transformed skewness drops to:
  - $\ln(\text{RecencyDays})$: $-0.489$
  - $\ln(\text{Frequency})$: $+1.001$
  - $\ln(\text{GrossMonetary})$: $+0.269$
  - $\ln(\text{AvgOrderValue})$: $+0.063$
  - $\ln(\text{UniqueProducts})$: $-0.281$
- **Plot Findings:**
  - *Distribution Plots:* Histograms shift from single-spike distributions to bell-shaped curves resembling the Gaussian overlay.
  - *Normal Q-Q Plots:* Raw Q-Q plots show severe upward curvature indicating heavy right tails. Log-transformed Q-Q plots align much closer to the 45-degree reference line, although slight tail flattening remains due to integer counts at $F=1$.

---

## 5. Actual Results

### Descriptive Statistics Summary ($N = 5,878$)
| Feature | Raw Mean | Raw Median | Raw Skewness | Log1p Mean | Log1p StdDev | Log1p Skewness |
|:---|---:|---:|---:|---:|---:|---:|
| **RecencyDays** | 201.33 | 96.00 | 0.887 | 4.454 | 1.559 | -0.489 |
| **Frequency** | 6.29 | 3.00 | 12.640 | 1.550 | 0.809 | 1.001 |
| **GrossMonetary** | £2,955.90 | £867.74 | 25.070 | 6.817 | 1.385 | 0.269 |
| **AvgOrderValue** | £385.18 | £279.24 | 57.133 | 5.617 | 0.736 | 0.063 |
| **UniqueProducts** | 81.99 | 45.00 | 6.224 | 3.760 | 1.219 | -0.281 |

### Parameter Estimation in Transformed Space
- **Sample Mean Vector ($\bar{\mathbf{x}}$):**  
  `[ln_RecencyDays: 4.454, ln_Frequency: 1.550, ln_GrossMonetary: 6.817, ln_AvgOrderValue: 5.617, ln_UniqueProducts: 3.760]`
- **Key Correlations ($\mathbf{R}$):**
  - $\ln(\text{Frequency})$ and $\ln(\text{GrossMonetary})$: $r = \mathbf{0.823}$ (strong positive loyalty elasticity)
  - $\ln(\text{Frequency})$ and $\ln(\text{UniqueProducts})$: $r = \mathbf{0.778}$ (frequent buyers explore broader catalog)
  - $\ln(\text{RecencyDays})$ and $\ln(\text{Frequency})$: $r = \mathbf{-0.531}$ (dormancy inversely relates to order rate)

### Conditional Regression Equivalence ($Y = \ln\text{GrossMonetary}$)
Predictors: $\ln(\text{RecencyDays})$, $\ln(\text{Frequency})$, $\ln(\text{UniqueProducts})$.

| Parameter | Theoretical MVN ($\mathbf{S}_{YX}\mathbf{S}_{XX}^{-1}$) | Empirical OLS ($\hat{\boldsymbol{\beta}}$) | Discrepancy |
|:---|---:|---:|---:|
| **Intercept ($\beta_0$)** | **3.895972** | **3.895972** | $< 10^{-14}$ |
| **Slope: ln_RecencyDays** | **-0.009801** | **-0.009801** | $< 10^{-14}$ |
| **Slope: ln_Frequency** | **+1.097112** | **+1.097112** | $< 10^{-14}$ |
| **Slope: ln_UniqueProducts**| **+0.336350** | **+0.336350** | $< 10^{-14}$ |
| **Residual Variance** | $\sigma^2_{Y|\mathbf{X}} = 0.442722$ | $s^2_{\text{OLS}} = 0.442948$ | Bessel adj. $\frac{n-1}{n-p-1}$ |
| **Model Fit ($R^2$)** | — | **0.7694** | 76.94% spend variance explained |

---

## 6. Interpretation
1. **Linearity of Conditional Expectation:** The theoretical MVN conditional slopes match empirical OLS estimates to 14 decimal places. This confirms mathematically that when variables are jointly normal, the true conditional expectation function is an affine linear equation.
2. **Economic Behaviour:** Frequency is the primary driver of customer value ($\beta = 1.097$); doubling order count roughly doubles expected spend. Catalog variety provides an additional positive boost ($\beta = 0.336$), whereas recency has a minor negative drag ($\beta = -0.010$).

---

## 7. Useful Output Files (`unit1_mvn/outputs/`)
- `feature_distributions.png`: Histograms with normal curves comparing raw versus log1p distributions.
- `normal_qq_plots.png`: Normal Q-Q plots showing departure from Gaussian linearity before and after log transform.
- `correlation_heatmap.png`: Pearson correlation matrix heatmap in log space.
- `conditional_regression_plot.png`: Scatter plot of actual vs. predicted spend with 1:1 reference line.
- `descriptive_statistics.csv`: Summary table of univariate metrics for raw and transformed variables.
- `mean_vector.csv`: Sample mean vector $\bar{\mathbf{x}}$.
- `covariance_matrix.csv`: Sample covariance matrix $\mathbf{S}$ ($5 \times 5$).
- `correlation_matrix.csv`: Sample correlation matrix $\mathbf{R}$ ($5 \times 5$).
- `conditional_regression.csv`: Comparison of theoretical MVN conditional parameters vs empirical OLS.

---

## 8. Limitations
- **Log transformation does not guarantee multivariate normality:** While $\ln(1+x)$ improves univariate symmetry and contracts extreme values, customer behaviour exhibits discrete integer counts at low frequency ($F=1$), meaning the joint distribution deviates from strict continuous Gaussian normality.
- **Constant Variance Assumption:** In true retail data, high-spending customers often exhibit greater variance (heteroscedasticity) than low-spenders, which classical MVN conditional theory assumes to be constant ($\sigma^2_{Y|\mathbf{X}}$).

---

## 9. Viva Revision & Defense Guide
1. **Q: Why does the conditional MVN expectation formula match OLS regression coefficients?**  
   *A:* Under joint multivariate normality, $E[Y \mid \mathbf{X}]$ is mathematically proven to be linear with slope vector $\boldsymbol{\Sigma}_{XX}^{-1} \boldsymbol{\Sigma}_{XY}$. The sample analogue of this expression is $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$, which is identically the Ordinary Least Squares estimator.
2. **Q: What is the difference between sample covariance and MLE covariance?**  
   *A:* Sample covariance divides by $n-1$ (Bessel's correction) to yield an unbiased estimator ($E[\mathbf{S}] = \boldsymbol{\Sigma}$), whereas Maximum Likelihood divides by $n$ and is slightly biased downwards by $\frac{n-1}{n}$.
3. **Q: Why not delete extreme outliers to force normality?**  
   *A:* High-spending wholesale customers represent genuine commercial revenue. Deleting them creates an artificial dataset. Applying a variance-stabilizing log transformation is defensible because it preserves all customers while making the distributions suitable for linear multivariate techniques.
