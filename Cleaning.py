import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# ------------------ 1. Load Training Data ------------------
print("Loading data...")
df = pd.read_csv("C:/Users/Oskar Giebichenstein/Desktop/Bachelor Data/application_train.csv", na_values=["", "NA"])
print(f"Initial dataset shape: {df.shape}")
print(f"Initial columns: {len(df.columns)}")

# ------------------ 2. Initial Data Exploration ------------------
print(f"\nTarget distribution:")
print(df["TARGET"].value_counts())
print(f"Default rate: {df['TARGET'].mean():.3f}")

# Check for any obvious data quality issues
print(f"\nChecking for data quality issues...")
print(f"Duplicated rows: {df.duplicated().sum()}")

# ------------------ 3. Handle Special Values ------------------
# Some columns might have special negative values that represent missing data
# Check for suspicious values in key columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:10]:  # Check first 10 numeric columns as example
    if col != 'TARGET':
        min_val = df[col].min()
        if min_val < 0 and 'DAYS' not in col and 'FLAG' not in col:
            print(f"Warning: {col} has minimum value {min_val} - check if this represents missing data")

# ------------------ 4. Drop Columns with Too Many Missing Values ------------------
print(f"\nAnalyzing missing values...")
missing = df.isnull().mean() * 100
print(f"Columns with >40% missing values:")
to_drop = missing[missing > 40].index
for col in to_drop:
    print(f"  {col}: {missing[col]:.1f}% missing")

print(f"Dropping {len(to_drop)} columns with >40% missing values")
df.drop(columns=to_drop, inplace=True)
print(f"Shape after dropping high-missing columns: {df.shape}")

# ------------------ 5. Convert Binary Yes/No Columns ------------------
print(f"\nConverting binary Y/N columns...")
binary_map = {"Y": 1, "N": 0}

# Check if these columns exist and their values before conversion
binary_cols = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
for col in binary_cols:
    if col in df.columns:
        print(f"{col} unique values before conversion: {df[col].unique()}")
        df[col] = df[col].map(binary_map)
        print(f"{col} unique values after conversion: {df[col].unique()}")
    else:
        print(f"Warning: {col} not found in dataset")

# ------------------ 6. Handle Categorical Columns Carefully ------------------
print(f"\nProcessing categorical columns...")
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Found {len(cat_cols)} categorical columns: {cat_cols}")

# Check cardinality of categorical columns before encoding
high_cardinality_cols = []
for col in cat_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")
    if unique_count > 50:  # Flag high cardinality columns
        high_cardinality_cols.append(col)
        print(f"  Warning: {col} has high cardinality ({unique_count} values)")

# Handle high cardinality columns differently
if high_cardinality_cols:
    print(f"\nHandling high cardinality columns...")
    for col in high_cardinality_cols:
        # Keep only top categories, group rest as 'Other'
        top_categories = df[col].value_counts().head(10).index
        df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
        print(f"Reduced {col} to {df[col].nunique()} categories")

# Apply one-hot encoding
print(f"Applying one-hot encoding to categorical columns...")
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dummy_na=True)
print(f"Shape after encoding: {df.shape}")

# ------------------ 7. Impute Missing Numeric Values ------------------
print(f"\nImputing missing numeric values...")
num_cols = df.select_dtypes(include=np.number).columns.drop("TARGET")
missing_numeric = df[num_cols].isnull().sum()
cols_with_missing = missing_numeric[missing_numeric > 0]

if len(cols_with_missing) > 0:
    print(f"Columns with missing numeric values:")
    for col, count in cols_with_missing.items():
        print(f"  {col}: {count} missing ({count / len(df) * 100:.1f}%)")

    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    print(f"Imputed missing values using median strategy")
else:
    print("No missing numeric values found")

# ------------------ 8. Drop Duplicates ------------------
print(f"\nRemoving duplicates...")
initial_rows = len(df)
df.drop_duplicates(inplace=True)
duplicates_removed = initial_rows - len(df)
print(f"Removed {duplicates_removed} duplicate rows")

# ------------------ 9. Handle Outliers More Carefully ------------------
print(f"\nHandling outliers in AMT_INCOME_TOTAL...")
print(f"Income statistics before outlier removal:")
print(df["AMT_INCOME_TOTAL"].describe())

# Check for extreme outliers first
extreme_outliers = df["AMT_INCOME_TOTAL"] > df["AMT_INCOME_TOTAL"].quantile(0.99)
print(f"Extreme outliers (>99th percentile): {extreme_outliers.sum()}")

# Use IQR method but be more conservative
q1 = df["AMT_INCOME_TOTAL"].quantile(0.25)
q3 = df["AMT_INCOME_TOTAL"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers_before = len(df)
df = df[(df["AMT_INCOME_TOTAL"] >= lower) & (df["AMT_INCOME_TOTAL"] <= upper)]
outliers_removed = outliers_before - len(df)
print(f"Removed {outliers_removed} income outliers ({outliers_removed / outliers_before * 100:.1f}% of data)")
print(f"Income range after outlier removal: {df['AMT_INCOME_TOTAL'].min():.0f} - {df['AMT_INCOME_TOTAL'].max():.0f}")

# ------------------ 10. Create Derived Features ------------------
print(f"\nCreating derived features...")

# Age calculation - check if DAYS_BIRTH exists
if "DAYS_BIRTH" in df.columns:
    df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)
    print(f"Age range: {df['AGE'].min()} - {df['AGE'].max()}")

    # Create age groups with more meaningful categories
    df["age_group"] = pd.cut(df["AGE"],
                             bins=[0, 25, 35, 50, 65, 100],
                             labels=["Very Young", "Young", "Middle", "Senior", "Elderly"])
    print(f"Age group distribution:")
    print(df["age_group"].value_counts())
else:
    print("Warning: DAYS_BIRTH column not found - cannot create age features")

# Region mapping - check if column exists
if "REGION_RATING_CLIENT" in df.columns:
    region_map = {1: "Best", 2: "Middle", 3: "Worst"}
    df["region_group"] = df["REGION_RATING_CLIENT"].map(region_map)
    print(f"Region group distribution:")
    print(df["region_group"].value_counts())
else:
    print("Warning: REGION_RATING_CLIENT column not found - cannot create region groups")

# Create income-to-credit ratio if both columns exist
if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
    df["income_credit_ratio"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    print(f"Created income-to-credit ratio feature")

# ------------------ 11. Final Data Quality Checks ------------------
print(f"\nFinal data quality checks...")
print(f"Final dataset shape: {df.shape}")
print(f"Final missing values:")
final_missing = df.isnull().sum()
if final_missing.sum() > 0:
    print(final_missing[final_missing > 0])
else:
    print("No missing values remaining")

print(f"\nTarget distribution after cleaning:")
print(df["TARGET"].value_counts())
print(f"Final default rate: {df['TARGET'].mean():.3f}")

# Check for infinite or NaN values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
if inf_count > 0:
    print(f"Warning: Found {inf_count} infinite values")
    # Replace infinite values with NaN and then impute
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Re-impute if needed
    if df.isnull().sum().sum() > 0:
        num_cols = df.select_dtypes(include=np.number).columns.drop("TARGET")
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = imputer.fit_transform(df[num_cols])

# ------------------ 12. Save Cleaned Dataset ------------------
print(f"\nSaving cleaned dataset...")
output_path = "C:/Users/Oskar Giebichenstein/Desktop/Bachelor Data/cleaned_application_full.csv"
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
print(f"Final dataset info:")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

# ------------------ 13. Generate Cleaning Report ------------------
print(f"\n" + "=" * 50)
print("DATA CLEANING SUMMARY REPORT")
print(f"=" * 50)
print(f"Original dataset shape: {df.shape}")
print(f"Columns dropped (high missing): {len(to_drop)}")
print(f"Duplicates removed: {duplicates_removed}")
print(f"Outliers removed: {outliers_removed}")
print(f"Final shape: {df.shape}")
print(f"Data quality: {'✓ GOOD' if df.isnull().sum().sum() == 0 else '⚠ CHECK MISSING VALUES'}")
print(f"Ready for modeling: {'✓ YES' if df.dtypes.apply(lambda x: x.kind in 'biufc').all() else '⚠ CHECK DATA TYPES'}")

# Check if CNT_CHILDREN column exists
if "CNT_CHILDREN" in df.columns:
    # Create binary column: 1 if person has one or more children, 0 otherwise
    df["has_children"] = df["CNT_CHILDREN"].apply(lambda x: 1 if x > 0 else 0)

    # Display value counts
    print("\nChild status distribution (1 = Has children, 0 = No children):")
    print(df["has_children"].value_counts())
else:
    print("CNT_CHILDREN column not found in dataset.")
