# Data Cleaning - Churn Dataset

##  Project Overview

This project performs comprehensive data cleaning and preprocessing on the telecom customer churn dataset (`churn-bigml-80.csv`). The goal is to prepare the data for machine learning modeling by handling outliers, encoding categorical variables, and scaling numerical features.

##  Objectives

- Handle missing data (if any)
- Detect and remove outliers selectively (< 5% rule)
- Convert categorical variables into numerical format
- Normalize/standardize numerical data
- Generate a cleaning report for documentation

##  Dataset

**churn-bigml-80.csv**
- 2,666 rows, 20 columns
- Binary classification problem (Churn: True/False)
- Mixed data types (numerical + categorical)
- Imbalanced target: ~14.5% churn rate

### Original Columns:
State, Account length, Area code, International plan, Voice mail plan,
Number vmail messages, Total day minutes, Total day calls, Total day charge,
Total eve minutes, Total eve calls, Total eve charge, Total night minutes,
Total night calls, Total night charge, Total intl minutes, Total intl calls,
Total intl charge, Customer service calls, Churn

##  Tools Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Programming language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib & seaborn | Data visualization |
| scikit-learn | Label encoding, StandardScaler, train-test split |
| pickle | Save scaler for future use |

##  Data Cleaning Steps

### 1. Initial Data Exploration

The script first loads and explores the data:
- Checks data types and missing values
- Displays summary statistics
- Identifies numerical and categorical columns

### 2. Selective Outlier Removal

**Rule:** Only remove outliers if they represent less than 5% of the data.

| Column | Outliers | Percentage | Action |
|--------|----------|------------|--------|
| Number vmail messages | 2 | 0.08% | ✅ Removed |
| Total day minutes | 21 | 0.79% | ✅ Removed |
| Total day calls | 18 | 0.68% | ✅ Removed |
| Total day charge | 21 | 0.79% | ✅ Removed |
| Total eve minutes | 17 | 0.64% | ✅ Removed |
| Total eve calls | 15 | 0.56% | ✅ Removed |
| Total eve charge | 17 | 0.64% | ✅ Removed |
| Total night minutes | 22 | 0.83% | ✅ Removed |
| Total night calls | 19 | 0.71% | ✅ Removed |
| Total night charge | 22 | 0.83% | ✅ Removed |
| Total intl minutes | 37 | 1.39% | ✅ Removed |
| Total intl calls | 66 | 2.48% | ✅ Removed |
| Total intl charge | 40 | 1.50% | ✅ Removed |
| **Customer service calls** | **210** | **7.88%** | ⚠️ **KEPT** |

**Key Decision:** Customer service calls outliers (7.88%) are KEPT because they represent valuable business insights about customers who frequently call support.

### 3. Encoding Categorical Variables

| Column | Method | Result |
|--------|--------|--------|
| International plan | Label Encoding | 0 = No, 1 = Yes |
| Voice mail plan | Label Encoding | 0 = No, 1 = Yes |
| Churn | Binary Conversion | 0 = False, 1 = True |
| State | One-Hot Encoding | 50 states → 49 new columns |
| Area code | One-Hot Encoding | 3 codes → 2 new columns |

### 4. Feature Scaling (Standardization)

Applied StandardScaler to all numerical features:
- **Method:** Z-score standardization
- **Result:** All features have mean = 0, std = 1
- **Train/Test Split:** 80/20 (stratified)

##  Results

### Dataset Transformation

| Metric | Before | After |
|--------|--------|-------|
| Rows | 2,666 | 2,451 |
| Columns | 20 | 69 |
| Missing Values | 0 | 0 |
| Outliers Removed | - | 215 (8.06%) |

### Final Dataset Structure
Final Dataset: (2451, 69)
├── 14 Original Numerical Features
├── 3 Encoded Binary Columns
├── 49 One-Hot Encoded State Columns
├── 2 One-Hot Encoded Area Code Columns
└── 1 Target Variable (Churn)

text

##  Output Files

| File | Description |
|------|-------------|
| `churn80_cleaned_processed.csv` | Final cleaned dataset (69 columns, 2451 rows) |
| `churn80_scaler.pkl` | StandardScaler for future use |
| `churn_boxplots_before.png` | Boxplots before outlier removal |
| `churn_boxplots_comparison.png` | Before/after outlier removal |
| `churn_standardization_comparison.png` | Before/after standardization |
| `churn80_cleaning_report.txt` | Complete cleaning report |

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
