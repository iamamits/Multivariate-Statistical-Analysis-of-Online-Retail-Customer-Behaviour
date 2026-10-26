# Customer Behaviour Analysis & Segmentation Using Computational Statistics

A complete undergraduate Computational Statistics coursework project analysing retail customer purchasing behaviour using multivariate statistical methods on the **Online Retail II** dataset.

---

## Project Overview

The project models commercial customer transaction patterns across five core syllabus units:
1. **Multivariate Normal Distribution (MVN):** Distributional analysis, parameter estimation, and conditional distribution / linear regression equivalence.
2. **Linear Discriminant Analysis (LDA):** Supervised classification of customer spend tier using non-monetary behavioural predictors ($86.60\%$ accuracy).
3. **Principal Component Analysis (PCA):** Unsupervised dimensionality reduction, variance decomposition ($82.83\%$ variance captured by 2 PCs), and behavioural axes.
4. **Factor Analysis (FA):** Latent variable modeling, common vs. unique variance decomposition, and Varimax orthogonal rotation to simple structure.
5. **Cluster Analysis (K-Means):** Unsupervised customer segmentation evaluating $k = 2$ to $k = 6$, selecting $k = 3$ via elbow inflection with 3-tier customer profiling.

---

## Dataset & Attribution

- **Dataset:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Source:** UCI Machine Learning Repository (Donated by Daqing Chen, 2019, DOI: 10.24432/C5CG6D)
- **Domain:** UK-based online retail / e-commerce (giftware & wholesale/retail merchandise)
- **Period:** December 2009 to December 2011 (both workbook sheets)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Prepared Population:** All $N = 5,878$ validated eligible customers are included in [`phase1_output/customer_master.csv`](phase1_output/customer_master.csv).

*Note on Data Safeguards:* The raw Excel file (`online_retail_II.xlsx`, 45.6 MB) and compressed transaction files (`*.csv.gz`, 32.5 MB) are excluded from version control via `.gitignore`. The lightweight customer master table (`customer_master.csv`, 990 KB) contains aggregated, anonymized behavioural features without personal identifiers, allowing all five units to run directly out of the box.

---

## Syllabus Units & Artifact Navigation

| Unit | Focus & Methodology | Canonical Script | Summary Report | Key Visualizations |
|:---:|:---|:---:|:---:|:---|
| **Unit 1** | **Multivariate Normal Distribution**<br>Parameter estimation ($\hat{\boldsymbol{\mu}}, \mathbf{S}, \mathbf{R}$), log1p skewness correction, and empirical OLS vs. conditional MVN equivalence. | [`unit1_mvn_analysis.py`](unit1_mvn/unit1_mvn_analysis.py) | [`unit1_mvn_summary.md`](unit1_mvn/unit1_mvn_summary.md) | • [Correlation Heatmap](unit1_mvn/outputs/correlation_heatmap.png)<br>• [Conditional Regression](unit1_mvn/outputs/conditional_regression_plot.png) |
| **Unit 2** | **Linear Discriminant Analysis**<br>Fisher's linear discriminant function ($w_1=-0.0466, w_2=2.5600, w_3=0.8853$), threshold $c=7.1203$, and $86.60\%$ out-of-sample test accuracy. | [`unit2_discriminant_analysis.py`](unit2_discriminant_analysis/unit2_discriminant_analysis.py) | [`unit2_lda_summary.md`](unit2_discriminant_analysis/unit2_lda_summary.md) | • [Discriminant Score Density](unit2_discriminant_analysis/outputs/discriminant_score_distribution.png)<br>• [Confusion Matrix](unit2_discriminant_analysis/outputs/confusion_matrix.png) |
| **Unit 3** | **Principal Component Analysis**<br>Spectral decomposition of correlation matrix $\mathbf{R}$, Kaiser criterion ($\lambda_1=3.1915 > 1.0$), and $82.83\%$ variance captured by 2 components. | [`unit3_pca_analysis.py`](unit3_pca/unit3_pca_analysis.py) | [`unit3_pca_summary.md`](unit3_pca/unit3_pca_summary.md) | • [Scree & Cumulative Plot](unit3_pca/outputs/scree_plot.png)<br>• [PC1 vs. PC2 Scatter](unit3_pca/outputs/pca_scatter_pc1_pc2.png) |
| **Unit 4** | **Factor Analysis**<br>Principal Component Factor method, communality adequacy ($68.6\% - 95.5\%$), and Varimax rotation to Thurstone's simple structure. | [`unit4_factor_analysis.py`](unit4_factor_analysis/unit4_factor_analysis.py) | [`unit4_fa_summary.md`](unit4_factor_analysis/unit4_fa_summary.md) | • [Factor Scree Plot](unit4_factor_analysis/outputs/factor_scree_plot.png)<br>• [Rotated Loadings Biplot](unit4_factor_analysis/outputs/factor_loadings_plot.png) |
| **Unit 5** | **K-Means Cluster Analysis**<br>Candidate evaluation ($k=2$ to $6$), selecting $k=3$ via elbow inflection, and 3-tier customer behavioural profiling. | [`unit5_cluster_analysis.py`](unit5_cluster_analysis/unit5_cluster_analysis.py) | [`unit5_cluster_summary.md`](unit5_cluster_analysis/unit5_cluster_summary.md) | • [Cluster Selection Plot](unit5_cluster_analysis/outputs/cluster_selection_plot.png)<br>• [Clusters on PC1/PC2](unit5_cluster_analysis/outputs/cluster_visualization_pc1_pc2.png) |

> [!NOTE]
> **Methodological Note on Unit 5 Clustering:**  
> Cluster numbers (0, 1, 2) are arbitrary algorithmic labels assigned by K-Means and do not imply ranking or customer quality. In cluster selection, the Silhouette score peaked mathematically at $k = 2$ ($0.3540$), while the Elbow method showed a clear inflection at $k = 3$ (WCSS drop of $3,499.53$ vs. $1,871.84$). The model retains $k = 3$ to capture the actionable 3-tier behavioural reality (High Engagement, Moderate Activity, and Inactive/Dormant) while maintaining a solid positive silhouette score ($0.2643$).

---

## Setup & Execution

### 1. Environment Installation
Install the required standard libraries:
```bash
pip install -r requirements.txt
```

### 2. Run All Five Units Sequentially
Execute the complete coursework pipeline from the project root:
```bash
python run_all_units.py
```
This runner executes Units 1 through 5 in order, prints progress, verifies exit codes, and ensures all CSV and plot artifacts are generated without modifying the underlying data.

### 3. Run Individual Units
Each unit can be run independently from the project root:
```bash
# Unit 1: Multivariate Normal Distribution
python unit1_mvn/unit1_mvn_analysis.py

# Unit 2: Linear Discriminant Analysis
python unit2_discriminant_analysis/unit2_discriminant_analysis.py

# Unit 3: Principal Component Analysis
python unit3_pca/unit3_pca_analysis.py

# Unit 4: Factor Analysis
python unit4_factor_analysis/unit4_factor_analysis.py

# Unit 5: K-Means Cluster Analysis
python unit5_cluster_analysis/unit5_cluster_analysis.py
```

### 4. Optional: Re-run Phase 1 Data Cleaning
Phase 1 data cleaning has already been executed and verified (14/14 validation checks passing). If you wish to re-generate [`phase1_output/customer_master.csv`](phase1_output/customer_master.csv) from the raw Excel workbook:
1. Download `online_retail_II.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in the project root.
2. Run:
```bash
python phase1_pipeline.py --input online_retail_II.xlsx --output phase1_output
```

---

## Project Structure

```text
CS-Customer-Behaviour-Statistics/
├── .gitignore                          # Excludes raw Excel, *.csv.gz, and bytecode
├── README.md                           # Mobile-friendly documentation & navigation
├── requirements.txt                    # Minimal third-party dependencies
├── run_all_units.py                    # Sequential runner for Units 1–5
├── phase1_pipeline.py                  # Standalone data cleaning ETL script
├── phase1_output/                      # Prepared customer data & audit reports
│   ├── customer_master.csv             # 5,878 customers with engineered features
│   ├── data_dictionary.csv             # Feature definitions and units
│   ├── data_quality_audit.csv          # Quality metrics and missingness audits
│   ├── phase1_cleaning_report.md       # Detailed cleaning methodology
│   ├── phase1_summary.json             # Reconciled row counts and classifications
│   ├── raw_audit.csv                   # Per-column sheet audit
│   └── validation_checks.json          # 14/14 automated validation checks
├── unit1_mvn/                          # Unit 1 code, report, and outputs
├── unit2_discriminant_analysis/        # Unit 2 code, report, and outputs
├── unit3_pca/                          # Unit 3 code, report, and outputs
├── unit4_factor_analysis/              # Unit 4 code, report, and outputs
└── unit5_cluster_analysis/             # Unit 5 code, report, and outputs
```

---

## Repository Authors & Coursework Context
- **Coursework:** Computational Statistics — Undergraduate Engineering
- **Domain:** Retail & E-Commerce Behavioural Analytics
- **Language:** Python (pure NumPy, Pandas, Matplotlib, Seaborn)
