# Unit 2 Summary: Linear Discriminant Analysis (LDA)

**Coursework:** Computational Statistics — Customer Behaviour Analysis & Segmentation  
**Dataset:** Online Retail II (`customer_master.csv`)  
**Student Level:** Undergraduate Engineering  

---

## 1. Objective
The objective of Unit 2 is to apply **Linear Discriminant Analysis (LDA)** to classify customers into distinct behavioural spending groups. We estimate class mean vectors, compute the pooled within-class covariance matrix, derive Fisher's linear discriminant function, establish an optimal decision threshold, and evaluate out-of-sample classification performance against a baseline.

---

## 2. Dataset and Target Classes
- **Total Population:** $N = 5,878$ customers (full validated dataset; zero dropped).
- **Operational Class Definition:**  
  Because retail transaction data lacks pre-existing customer labels, we define classes operationally using the median spend:
  $$\text{Median Gross Spend Threshold} = \mathbf{£867.74}$$
  - **Class 0 (Lower-spend):** $\text{GrossMonetary} < £867.74$ &rarr; **2,939 customers (50.0%)**
  - **Class 1 (Higher-spend):** $\text{GrossMonetary} \ge £867.74$ &rarr; **2,939 customers (50.0%)**

*Note: These are operationally constructed classes designed to demonstrate supervised statistical separation, not pre-existing commercial ground truth.*

---

## 3. Predictors and Leakage Prevention
To evaluate whether non-monetary purchasing behaviour can predict spending tier, we select three predictors:
1. `RecencyDays`: Days since latest purchase (inactivity).
2. `Frequency`: Count of distinct purchase invoices (engagement).
3. `UniqueProducts`: Count of distinct product codes purchased (catalog breadth).

All three features are transformed via $\ln(1 + x)$ to stabilize variance.

> **Strict Target Leakage Prevention:**  
> Since classes are defined directly from `GrossMonetary`, **both `GrossMonetary` and `AvgOrderValue` are strictly excluded as predictors**. Using them would constitute circular target leakage.

- **Data Partitioning:** A stratified 75/25 split (Fixed Seed = 42) ensures identical class balance:
  - **Training Set (75%):** 4,408 customers (2,204 Lower, 2,204 Higher)
  - **Testing Set (25%):** 1,470 customers (735 Lower, 735 Higher)  
  *All model parameters ($\bar{\mathbf{x}}_k$, $\mathbf{S}_{\text{pool}}$, $\mathbf{w}$, $c$) are fitted exclusively on training data.*

---

## 4. Main Mathematical Formulas

1. **Class-Specific Mean Vectors:**
   $$\bar{\mathbf{x}}_k = \frac{1}{n_k} \sum_{i \in C_k} \mathbf{x}_i \in \mathbb{R}^3 \quad (k \in \{0, 1\})$$

2. **Pooled Within-Class Covariance Matrix ($\mathbf{S}_{\text{pool}}$):**
   Assuming equal covariance ($\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1$):
   $$\mathbf{S}_{\text{pool}} = \frac{(n_0 - 1)\mathbf{S}_0 + (n_1 - 1)\mathbf{S}_1}{n_0 + n_1 - 2}$$

3. **Fisher's Linear Discriminant Coefficients ($\mathbf{w}$):**
   Maximizing between-class to within-class variance ratio:
   $$\mathbf{w} = \mathbf{S}_{\text{pool}}^{-1} (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_0)$$

4. **Bayes Classification Rule & Optimal Cutoff ($c$):**
   Discriminant score: $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} = w_1 x_1 + w_2 x_2 + w_3 x_3$.  
   Optimal threshold:
   $$c = \frac{1}{2} \mathbf{w}^T (\bar{\mathbf{x}}_1 + \bar{\mathbf{x}}_0) - \ln\left(\frac{\pi_1}{\pi_0}\right)$$
   Since $\pi_0 = \pi_1 = 0.5$, $\ln(1) = 0$, so $c$ is the exact midpoint between projected centroids:
   - Predict **Class 1 (Higher-spend)** if $\mathbf{w}^T \mathbf{x} \ge c$.
   - Predict **Class 0 (Lower-spend)** if $\mathbf{w}^T \mathbf{x} < c$.

5. **Evaluation Metrics:**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N_{\text{test}}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$$

---

## 5. Actual Results

### Estimated Parameters (from Training Set, $n = 4,408$)
- **Class Mean Vectors ($\bar{\mathbf{x}}_0$ vs. $\bar{\mathbf{x}}_1$):**
  - $\ln(\text{RecencyDays})$: 5.112 (Lower) vs. 3.792 (Higher) &rarr; Higher spenders are more recent ($\Delta = -1.321$).
  - $\ln(\text{Frequency})$: 0.996 (Lower) vs. 2.120 (Higher) &rarr; Higher spenders order far more often ($\Delta = +1.124$).
  - $\ln(\text{UniqueProducts})$: 3.023 (Lower) vs. 4.522 (Higher) &rarr; Higher spenders buy wider catalog variety ($\Delta = +1.499$).
- **Linear Discriminant Weights ($\mathbf{w}$):**
  - $w_{\text{ln\_RecencyDays}} = \mathbf{-0.0466}$
  - $w_{\text{ln\_Frequency}} = \mathbf{+2.5600}$
  - $w_{\text{ln\_UniqueProducts}} = \mathbf{+0.8853}$
- **Decision Threshold ($c$):** $\mathbf{7.1203}$

### Out-of-Sample Test Set Evaluation ($n = 1,470$)

Because the test set has exactly **735 Lower-spend** and **735 Higher-spend** customers (a 50:50 balanced split), the naive majority-class baseline is a constant classifier that always predicts one fixed class. Below we compare the LDA model against the constant baseline that always predicts the positive class (**Higher-spend**):

| Metric | LDA Model | Constant Baseline (Always Higher-spend) | LDA Improvement |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **86.60%** (1,273 / 1,470) | **50.00%** (735 / 1,470) | **+36.60%** |
| **Precision (Higher-spend)** | **91.51%** (593 / 648) | **50.00%** (735 / 1,470) | **+41.51%** |
| **Recall (Higher-spend)** | **80.68%** (593 / 735) | **100.00%** (735 / 735) | -19.32% (trivial 100% via all-positive) |
| **F1-Score (Higher-spend)** | **0.8576** | **0.6667** ($2 \times \frac{0.5 \times 1.0}{0.5 + 1.0}$) | **+0.1909** (+19.09 pp) |
| **Precision (Lower-spend)** | **82.73%** (680 / 822) | **Undefined ($0/0$)** | Fully resolved by LDA |
| **Recall (Lower-spend)** | **92.52%** (680 / 735) | **0.00%** (0 / 735) | **+92.52%** |
| **F1-Score (Lower-spend)** | **0.8735** | **Undefined** (precision is $0/0$) | Fully resolved by LDA |

*Symmetric Note: If the constant baseline instead always predicted Lower-spend, the Lower-spend metrics would be Prec = 50.00%, Rec = 100.00%, F1 = 0.6667, while Higher-spend metrics would become Prec = Undefined ($0/0$), Rec = 0.00%, and F1 = Undefined. In either scenario, a constant baseline achieves only 50.00% accuracy and completely fails on the unpredicted class.*

#### Test-Set Confusion Matrix
| Actual \ Predicted | Predicted Lower-spend | Predicted Higher-spend | Total |
|:---|:---:|:---:|:---:|
| **Actual Lower-spend (0)** | **680 (TN)** | 55 (FP) | 735 |
| **Actual Higher-spend (1)** | 142 (FN) | **593 (TP)** | 735 |
| **Total** | 822 | 648 | 1,470 |

---

## 6. Interpretation
1. **High Explanatory Accuracy:** Achieving **86.60% out-of-sample accuracy** strictly from non-monetary behavioural features demonstrates that order frequency and catalog breadth are powerful predictors of customer value. This represents a substantial **+36.60 percentage point gain** over the 50.00% constant baseline.
2. **Balanced Performance across Classes:** Unlike a constant baseline that trivializes one class and leaves the other undefined, the LDA model maintains balanced performance across both groups: F1-scores of **0.8576** (Higher-spend) and **0.8735** (Lower-spend).
3. **Feature Dominance:** `Frequency` ($w = +2.5600$) is by far the strongest linear discriminator. `UniqueProducts` ($w = +0.8853$) adds substantial independent discrimination. `RecencyDays` ($w = -0.0466$) has a slight negative weight, confirming that inactivity marginally reduces the likelihood of being a higher-value customer.
4. **Asymmetric Error Profile:** The model achieves 92.52% recall on lower-spend customers (only 55 false positives), meaning it is very reliable at identifying customers who will not generate large revenue.

---

## 7. Useful Output Files (`unit2_discriminant_analysis/outputs/`)
- `discriminant_score_distribution.png`: Histogram and KDE of 1D linear discriminant scores showing class separation and decision threshold $c = 7.12$.
- `confusion_matrix.png`: Heatmap of the 2x2 test-set confusion matrix.
- `lda_class_separation.png`: 2D scatter plot of order frequency vs. unique products showing actual class clusters.
- `class_summary.csv`: Summary table of median spend cutoff (£867.74), counts, and proportions.
- `class_mean_vectors.csv`: Mean vectors for each class and pairwise differences.
- `pooled_covariance_matrix.csv`: $3 \times 3$ pooled within-class covariance matrix ($\mathbf{S}_{\text{pool}}$).
- `lda_coefficients.csv`: Discriminant weights ($w$) and decision threshold ($c = 7.1203$).
- `confusion_matrix.csv`: Tabular test confusion matrix counts.
- `test_evaluation_metrics.csv`: Final accuracy, precision, recall, F1-scores, and baseline comparison.

---

## 8. Assumptions and Limitations
- **Operational vs. Natural Classes:** The classes were created by a median split on spend. They do not represent pre-existing, natural business segments.
- **Equal Covariance Assumption:** LDA assumes both classes share an identical covariance matrix ($\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1$). In practice, higher-spend customers exhibit wider variance in order frequency and catalog breadth than lower-spend customers.
- **Normality Assumption:** Predictors were log-transformed to reduce skewness, but discrete integer heapings remain at $F=1$.
