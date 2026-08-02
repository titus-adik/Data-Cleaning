import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)


print(" DATA CLEANING AND PREPROCESSING - CHURN DATASET")


# Load the data
df = pd.read_csv('/content/churn-bigml-80.csv')

print(f"\n📌 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n Column Names:")
print(df.columns.tolist())

print(f"\n First 5 rows:")
print(df.head())

print(f"\n Data Types:")
print(df.dtypes)

print(f"\n Basic Statistics:")
print(df.describe())

print(f"\n Missing Values:")
print(df.isnull().sum())


# 1. OUTLIER DETECTION

print("📊 1. OUTLIER DETECTION")


# Identify numerical columns (excluding identifiers)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove 'Account length' and 'Area code' as they are identifiers
numerical_cols = [col for col in numerical_cols if col not in ['Account length', 'Area code']]

print(f"\n Numerical Columns for Outlier Detection: {numerical_cols}")

# Visualize outliers using box plots
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    if i < len(axes):
        df.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f'{col}', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Hide any unused subplots
for i in range(len(numerical_cols), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Boxplots of Numerical Features (Before Outlier Removal)', fontsize=16)
plt.tight_layout()
plt.savefig('churn_boxplots_before.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Boxplot visualization saved as 'churn_boxplots_before.png'")


# 2. SELECTIVE OUTLIER REMOVAL (< 5%)

print(" 2. SELECTIVE OUTLIER REMOVAL (Only if < 5%)")


# Function to detect outliers using IQR (defined once)
def detect_outliers_iqr(df, column, threshold=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers and calculate percentages
outlier_summary = {}
for col in numerical_cols:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_pct = (len(outliers) / len(df)) * 100
    outlier_summary[col] = {
        'count': len(outliers),
        'percentage': outlier_pct,
        'lower_bound': lower,
        'upper_bound': upper,
        'action': 'Remove' if outlier_pct < 5 else 'Keep'
    }

print("\n Outlier Summary by Column:")
print("-" * 70)
print(f"{'Column':25} | {'Outliers':8} | {'Percentage':10} | {'Action':10}")
print("-" * 70)
for col, stats in outlier_summary.items():
    action = stats['action']
    if stats['count'] > 0:
        print(f"{col:25} | {stats['count']:8} | {stats['percentage']:9.2f}% | {action:10}")
    else:
        print(f"{col:25} | {stats['count']:8} | {stats['percentage']:9.2f}% | No outliers")

# Show columns that will NOT be cleaned
print("\n⚠️ Columns with > 5% outliers (will NOT be removed):")
for col, stats in outlier_summary.items():
    if stats['action'] == 'Keep':
        print(f"  - {col}: {stats['percentage']:.2f}% ({stats['count']} outliers)")

# Function to remove outliers ONLY if < 5%
def remove_outliers_selective(df, columns, threshold=1.5, max_pct=5.0):
    """
    Remove outliers only if they represent less than max_pct of the data.
    """
    df_clean = df.copy()
    removed_details = {}
    total_removed = 0

    for col in columns:
        # Detect outliers
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        outlier_pct = (outliers_count / len(df_clean)) * 100

        # Only remove if < max_pct
        if outlier_pct < max_pct and outliers_count > 0:
            df_clean = df_clean[~outliers_mask]
            removed_details[col] = {
                'count': outliers_count,
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            total_removed += outliers_count
        elif outliers_count > 0:
            print(f"   ⚠️ {col}: {outlier_pct:.2f}% outliers - KEPT (exceeds {max_pct}% threshold)")

    return df_clean, total_removed, removed_details

# Remove outliers selectively
df_clean, total_removed, removed_details = remove_outliers_selective(df, numerical_cols)

print(f"\n📌 Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📌 After outlier removal: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
print(f"📌 Total rows removed: {total_removed} ({total_removed/df.shape[0]*100:.2f}%)")

print("\n Columns where outliers were REMOVED:")
print("-" * 50)
for col, details in removed_details.items():
    print(f"{col:25} | {details['count']:3} outliers ({details['percentage']:.2f}%)")
    print(f"                      | Range: {details['lower_bound']:.2f} - {details['upper_bound']:.2f}")

# Visualize before and after
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Before removal
df[numerical_cols].boxplot(ax=axes[0])
axes[0].set_title('Before Outlier Removal', fontsize=14)
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Values')
axes[0].tick_params(axis='x', rotation=45)

# After removal
df_clean[numerical_cols].boxplot(ax=axes[1])
axes[1].set_title('After Outlier Removal (Selective - Only < 5%)', fontsize=14)
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Values')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('churn_boxplots_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Boxplot comparison saved as 'churn_boxplots_comparison.png'")


# 3. ENCODING CATEGORICAL VARIABLES

print(" 3. ENCODING CATEGORICAL VARIABLES")


# Identify categorical columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
print(f"\n Categorical Columns: {categorical_cols}")

# Check unique values
for col in categorical_cols:
    print(f"\n{col} unique values: {df_clean[col].unique()}")

# Create a copy for encoding
df_encoded = df_clean.copy()

# Label Encoding for binary columns
print("\n🔹 Label Encoding for binary columns:")
le = LabelEncoder()

binary_cols = ['International plan', 'Voice mail plan']
for col in binary_cols:
    df_encoded[col + '_encoded'] = le.fit_transform(df_clean[col])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  - {col}: {mapping}")

# Convert Churn to numeric
print("\n🔹 Converting 'Churn' to numeric:")
df_encoded['Churn'] = df_clean['Churn'].astype(int)
print(f"  - Churn distribution: {df_encoded['Churn'].value_counts().to_dict()}")

# One-Hot Encoding for State
print("\n🔹 One-Hot Encoding for 'State' column:")
state_dummies = pd.get_dummies(df_clean['State'], prefix='State', drop_first=True)
print(f"  - Created {state_dummies.shape[1]} new columns (49 dummies for 50 states)")

# One-Hot Encoding for Area Code
print("\n🔹 One-Hot Encoding for 'Area code' column:")
area_code_dummies = pd.get_dummies(df_clean['Area code'], prefix='Area', drop_first=True)
print(f"  - Created {area_code_dummies.shape[1]} new columns ({df_clean['Area code'].nunique()} unique values -> {area_code_dummies.shape[1]} dummies)")

# Combine all features
features_to_keep = numerical_cols + ['Churn', 'International plan_encoded', 'Voice mail plan_encoded']
df_final = pd.concat([
    df_encoded[features_to_keep],
    state_dummies,
    area_code_dummies
], axis=1)

print(f"\n📌 Final shape after encoding: {df_final.shape}")
print(f"📌 Features: {df_final.shape[1] - 1} predictors + 1 target (Churn)")

print("\n Encoded Dataset (first 5 rows):")
print(df_final.head())


# 4. NORMALIZATION AND STANDARDIZATION

print(" 4. NORMALIZATION AND STANDARDIZATION")


# Separate features and target
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📌 Training set size: {X_train.shape[0]} rows ({X_train.shape[0]/len(df_final)*100:.0f}%)")
print(f"📌 Testing set size: {X_test.shape[0]} rows ({X_test.shape[0]/len(df_final)*100:.0f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

print("\n Features standardized (mean=0, std=1)")
print("\n Standardized Training Data Summary:")
print(X_train_scaled_df.describe())

# Visualize before and after standardization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before standardization
X_train.iloc[:, :5].boxplot(ax=axes[0])
axes[0].set_title('Before Standardization (Original Scale)')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Values')
axes[0].tick_params(axis='x', rotation=45)

# After standardization
X_train_scaled_df.iloc[:, :5].boxplot(ax=axes[1])
axes[1].set_title('After Standardization (Standard Scale)')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Standardized Values')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('churn_standardization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Standardization visualization saved as 'churn_standardization_comparison.png'")


# 5. SAVE CLEANED DATA

print(" 5. SAVING CLEANED DATA")


# Save the cleaned dataset
df_final.to_csv('churn80_cleaned_processed.csv', index=False)
print("\n Cleaned dataset saved as 'churn80_cleaned_processed.csv'")

# Save the scaler for future use
with open('churn80_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(" Scaler saved as 'churn80_scaler.pkl'")



# 6. DATA QUALITY CHECKS 


print(" 6. DATA QUALITY CHECKS")


# Check 1: No null values
null_count = df_final.isnull().sum().sum()
print(f"1. Null values: {null_count} {'✅' if null_count == 0 else '❌'}")

# Check 2: All columns numeric
all_numeric = df_final.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()
print(f"2. All columns numeric: {all_numeric} {'✅' if all_numeric else '❌'}")

# Check 3: State encoding integrity (FIXED)
state_cols = [col for col in df_final.columns if col.startswith('State_')]
state_sums = df_final[state_cols].sum(axis=1)

# With drop_first=True, each row should have sum either 0 or 1
# 0 = dropped state, 1 = one-hot encoded state
state_check = state_sums.isin([0, 1]).all()
print(f"3. Each row has valid state encoding: {state_check} {'✅' if state_check else '❌'}")

if not state_check:
    print("   ⚠️ Rows with invalid state encoding:")
    invalid_rows = df_final[~state_sums.isin([0, 1])]
    print(f"   - {len(invalid_rows)} rows have state sum != 0 or 1")
    print(f"   - Sample invalid rows: {invalid_rows.head(2)[state_cols].index.tolist()}")

# Check 4: Area code encoding integrity
area_cols = [col for col in df_final.columns if col.startswith('Area_')]
area_sums = df_final[area_cols].sum(axis=1)
# With drop_first=True, each row should have sum either 0 or 1
area_check = area_sums.isin([0, 1]).all()
print(f"4. Each row has valid area code encoding: {area_check} {'✅' if area_check else '❌'}")

# Check 5: Churn distribution preserved
churn_rate = df_final['Churn'].mean()
print(f"5. Churn rate: {churn_rate:.2%}")

# Check 6: Check for duplicate columns (optional)
duplicate_cols = df_final.columns[df_final.columns.duplicated()].tolist()
print(f"6. Duplicate columns: {len(duplicate_cols)} {'✅' if len(duplicate_cols) == 0 else '❌'}")

# 7. GENERATE CLEANING REPORT


print(" 7. GENERATE CLEANING REPORT")


report = f"""

      DATA CLEANING REPORT - CHURN DATASET


1. DATASET OVERVIEW
-------------------
- Original shape: {df.shape[0]} rows, {df.shape[1]} columns
- After outlier removal: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns
- Final shape: {df_final.shape[0]} rows, {df_final.shape[1]} columns
- Rows removed: {df.shape[0] - df_final.shape[0]} ({((df.shape[0] - df_final.shape[0])/df.shape[0])*100:.2f}%)

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
  * Customer service calls: 7.88% (210 outliers)

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
- Stratified split: Yes (preserved churn distribution)

5. FINAL DATASET
---------------
- File saved as: churn80_cleaned_processed.csv
- All features numeric: Yes
- No missing values: Yes
- Total features: {df_final.shape[1]}
- Churn rate: {churn_rate:.2%}


Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

with open('churn80_cleaning_report.txt', 'w') as f:
    f.write(report)

print("\n Cleaning report saved as 'churn80_cleaning_report.txt'")


print(" DATA CLEANING COMPLETE")

print("\n Files Generated:")
print("  1. churn80_cleaned_processed.csv - Final cleaned dataset")
print("  2. churn80_scaler.pkl - StandardScaler for future use")
print("  3. churn_boxplots_before.png - Boxplots before outlier removal")
print("  4. churn_boxplots_comparison.png - Before/after outlier removal")
print("  5. churn_standardization_comparison.png - Before/after standardization")
print("  6. churn80_cleaning_report.txt - Complete cleaning report")
