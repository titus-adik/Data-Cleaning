      DATA CLEANING REPORT - CHURN DATASET


1. DATASET OVERVIEW
-------------------
- Original shape: {df.shape[0]} rows, {df.shape[1]} columns
- Final shape: {df_final.shape[0]} rows, {df_final.shape[1]} columns
- Rows removed: {df.shape[0] - df_final.shape[0]} ({df.shape[0] - df_final.shape[0])/df.shape[0]*100:.2f}%)

2. OUTLIER REMOVAL
----------------
- Method: IQR (Interquartile Range)
- Threshold: 1.5
- Rule: Only removed if < 5% of data
- Columns where outliers were removed:
"""
for col, details in removed_details.items():
    report += f"  * {col}: {details['count']} outliers ({details['percentage']:.2f}%)\n"

report += f"""
- Columns with > 5% outliers (KEPT):
  * Customer service calls: 7.88% (210 outliers) - valuable business insight

3. CATEGORICAL ENCODING
---------------------
- One-Hot Encoding: State ({state_dummies.shape[1]} new columns)
- One-Hot Encoding: Area code ({area_code_dummies.shape[1]} new columns)
- Label Encoding: International plan, Voice mail plan
- Binary Conversion: Churn

4. FEATURE SCALING
----------------
- Method: Standardization (Z-score)
- Features scaled: {X_train_scaled_df.shape[1]}
- Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} ({X_train.shape[0]/len(df_final)*100:.0f}/{X_test.shape[0]/len(df_final)*100:.0f})



with open('churn_cleaning_report.txt', 'w') as f:
    f.write(report)
print("📄 Cleaning report saved as 'churn_cleaning_report.txt'")
