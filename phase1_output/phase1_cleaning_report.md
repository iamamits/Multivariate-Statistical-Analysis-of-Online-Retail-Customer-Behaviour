# Phase 1 — Data Audit, Cleaning & Feature Engineering Report

**Project:** Customer Behaviour Analysis & Segmentation Using Computational Statistics
**Dataset:** Online Retail II (Retail / E-commerce)
**Generated:** 2026-10-26 22:43:44

## 1. Workbook Audit

- **Sheets found:** 2 — Year 2009-2010, Year 2010-2011
- **Schemas match across sheets:** Yes

| Sheet | Rows | Columns | Exact Duplicates |
|-------|------|---------|-----------------|
| Year 2009-2010 | 525,461 | 8 | 6,865 |
| Year 2010-2011 | 541,910 | 8 | 5,268 |

**Total raw rows (all sheets combined):** 1,067,371

## 2. Missing Values (Per Sheet)

### Sheet: Year 2009-2010

| Column | Missing Count | Missing % |
|--------|--------------|-----------|
| Invoice | 0 | 0.00% |
| StockCode | 0 | 0.00% |
| Description | 2,928 | 0.56% |
| Quantity | 0 | 0.00% |
| InvoiceDate | 0 | 0.00% |
| Price | 0 | 0.00% |
| Customer ID | 107,927 | 20.54% |
| Country | 0 | 0.00% |

### Sheet: Year 2010-2011

| Column | Missing Count | Missing % |
|--------|--------------|-----------|
| Invoice | 0 | 0.00% |
| StockCode | 0 | 0.00% |
| Description | 1,454 | 0.27% |
| Quantity | 0 | 0.00% |
| InvoiceDate | 0 | 0.00% |
| Price | 0 | 0.00% |
| Customer ID | 135,080 | 24.93% |
| Country | 0 | 0.00% |

## 3. Cleaning Decisions

1. Stripped whitespace from column names.
2. Converted Invoice to string and stripped whitespace.
3. Customer ID kept with NaN for missing; will convert to int-string for non-null later.
4. Parsed InvoiceDate to datetime. 0 unparseable dates set to NaT.
5. Identified 34,335 exact duplicate rows (keeping first occurrence).
6. After deduplication: 1,033,036 rows remain.
7. Classified 19,104 rows as cancellation (Invoice starts with 'C').
8. Found 22,496 rows with negative Quantity.
9. Found 6,019 rows with zero or negative Price.
10. Found 4,275 rows with missing Description.
11. Found 235,151 rows with missing Customer ID.
12. Found 0 rows with missing/invalid InvoiceDate.
13. Eligible purchase transactions: 779,425 rows (non-cancellation, Qty>0, Price>0, non-null CustomerID & date).
14. Known-customer returns/cancellations: 18,390 rows.
15. Converted non-null Customer ID to integer string (e.g., '12345').

## 4. Row Flow & Transaction Reconciliation

- Raw combined rows: **1,067,371**
- Exact duplicates removed: **34,335**
- Deduplicated transaction pool: **1,033,036**

### Mutually Exclusive Transaction Classification (100% Exhaustive)

| Category | Count | Pct of Dedup | Customer Entity | Reason / Treatment |
|:---|---:|---:|:---|:---|
| 1. Eligible Purchase Transactions | 779,425 | 75.45% | Known Customer | Commercial sales forming RFM / customer master features |
| 2. Known-Customer Returns / Cancellations | 18,390 | 1.78% | Known Customer | Separately captured into customer return descriptor features |
| 3. Known-Customer Zero-Price / Free Lines | 70 | 0.0068% | Known Customer | Zero-price lines (Price=0.0; samples/promotions/tests); £0 revenue |
| 4. Anonymous Commercial Purchases | 228,488 | 22.12% | Anonymous (NaN) | Excluded from customer master; retained in transaction audit |
| 5. Anonymous Zero-Price / Free Lines | 2,551 | 0.25% | Anonymous (NaN) | Zero-price anonymous entries (samples/unpriced items) |
| 6. Anonymous Stock Adjustments / Damages | 3,393 | 0.33% | Anonymous (NaN) | Damaged/broken/water damage/inventory write-offs (Qty<0, Price=0) |
| 7. Anonymous Cancellations | 714 | 0.07% | Anonymous (NaN) | Credit notes/cancellations without customer ID |
| 8. Accounting / Bad Debt Adjustments | 5 | 0.0005% | Anonymous (NaN) | Invoices starting with 'A' (StockCode B, 'Adjust bad debt', Price<0) |
| **Total Deduplicated Pool** | **1,033,036** | **100.00%** | **All Records** | **Mutually exclusive sum = exactly 1,033,036 (0 unexplained rows)** |

## 5. Customer Master Summary

- **Unique customers:** 5,878
- **Reference date for Recency:** 2011-12-10 12:50:00
- **Date range:** 2009-12-01 07:45:00 to 2011-12-09 12:50:00

### Feature Statistics

| Stat | RecencyDays | Frequency | GrossMonetary | AvgOrderValue | TotalQuantity | UniqueProducts | ActiveMonths |
|------|------|------|------|------|------|------|------|
| count | 5878.0 | 5878.0 | 5878.0 | 5878.0 | 5878.0 | 5878.0 | 5878.0 |
| mean | 201.33 | 6.29 | 2955.9 | 385.18 | 1788.7 | 81.99 | 4.35 |
| std | 209.34 | 13.01 | 14440.85 | 1214.29 | 8876.3 | 116.48 | 4.49 |
| min | 1.0 | 1.0 | 2.95 | 2.95 | 1.0 | 1.0 | 1.0 |
| 25% | 26.0 | 1.0 | 342.28 | 176.68 | 187.0 | 19.0 | 1.0 |
| 50% | 96.0 | 3.0 | 867.74 | 279.24 | 480.0 | 45.0 | 3.0 |
| 75% | 380.0 | 7.0 | 2248.31 | 414.9 | 1350.0 | 103.0 | 6.0 |
| max | 739.0 | 398.0 | 580987.04 | 84236.25 | 367193.0 | 2550.0 | 25.0 |

## 6. Validation Results

| Check | Status | Detail |
|-------|--------|--------|
| Unique CustomerID per master row | PASS | 5878 rows, 5878 unique IDs |
| All eligible Quantity > 0 | PASS |  |
| All eligible Price > 0 | PASS |  |
| No NaT dates in eligible transactions | PASS |  |
| No cancellation invoices in eligible | PASS |  |
| All GrossMonetary > 0 | PASS |  |
| All RecencyDays >= 0 | PASS |  |
| Row count reconciliation documented | PASS | Combined=1067371, Dedup=1033036, Eligible=779425, Returns=18390 |
| No null Customer ID in eligible | PASS |  |
| Spot-check customer 17777 | PASS | Freq=3, Monetary=647.56 |
| Spot-check customer 16719 | PASS | Freq=14, Monetary=3513.62 |
| Spot-check customer 12546 | PASS | Freq=2, Monetary=1161.50 |
| MeanPurchaseGapDays NaN for single-invoice customers | PASS |  |
| Mutually exclusive transaction reconciliation (100% exhaustion) | PASS | Sum=1,033,036 == Dedup=1,033,036. Zero-price known customer residual=70 verified. |

## 7. Generated Files

| File | Size |
|------|------|
| raw_audit.csv | 1.0 KB |
| deduplicated_transactions_classified.csv.gz | 17.7 MB |
| eligible_purchase_transactions.csv.gz | 13.4 MB |
| known_customer_returns_cancellations.csv.gz | 418.3 KB |
| customer_master.csv | 967.7 KB |
| data_quality_audit.csv | 0.8 KB |
| data_dictionary.csv | 2.5 KB |
| validation_checks.json | 3.1 KB |
| phase1_summary.json | 3.2 KB |

## 8. Limitations & Notes

- Rows with missing Customer ID are retained in the deduped transactions file but excluded from customer-level analysis.
- Returns/cancellations (Invoice starting with 'C') are separated; their descriptors are added to the customer master but do not affect purchase features.
- MeanPurchaseGapDays and MedianPurchaseGapDays are NaN for single-invoice customers (no gap to compute).
- UniqueProducts counts distinct StockCodes; some codes may represent service charges, postage, etc.
- No outlier removal was performed in Phase 1; outlier analysis is deferred to later units.
- The original workbook was not modified.

---
*Phase 1 complete. Awaiting approval before proceeding to Unit 1.*