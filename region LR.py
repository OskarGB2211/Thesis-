import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# ------------------ 1. Load Cleaned Data ------------------
df = pd.read_csv("C:/Users/Oskar Giebichenstein/Desktop/Bachelor Data/cleaned_application_full.csv")

# ------------------ 2. Create Region Groups ------------------
# Region group mapping based on REGION_RATING_CLIENT
df["region_group"] = df["REGION_RATING_CLIENT"].map({1: "Best", 2: "Middle", 3: "Worst"})

# Handle any missing values
if df["region_group"].isna().any():
    print("Warning: Some region ratings missing. Filling with 'Middle'.")
    df["region_group"] = df["region_group"].fillna("Middle")

print(f"Dataset size: {len(df):,} applications")
print(f"Default rate: {df['TARGET'].mean():.3f}")
print(f"Region groups:")
print(df['region_group'].value_counts())
print(f"Region group percentages:")
print(df['region_group'].value_counts(normalize=True).round(3))

# ------------------ 3. Define Features and Target ------------------
# Drop unused or sensitive columns - excluding region features from training
X = df.drop(columns=[
    "TARGET", "SK_ID_CURR",
    "REGION_RATING_CLIENT", "region_group"  # Remove region features from training
])
# Ensure only numeric features are used
X = X.select_dtypes(include=[np.number])
y = df["TARGET"]

print(f"\nFeature count: {X.shape[1]}")


# ------------------ 4. Pre-processing: Reweighing Implementation ------------------
def calculate_reweighing_weights(df, sensitive_attr, target_attr):
    """
    Calculate reweighing weights to ensure demographic parity across regions
    """
    print(f"\n🔄 CALCULATING REWEIGHING WEIGHTS FOR REGION GROUPS...")

    # Get unique groups and outcomes
    groups = df[sensitive_attr].unique()
    outcomes = df[target_attr].unique()

    weights = np.ones(len(df))

    # Calculate overall proportions
    total_size = len(df)

    print(f"Region group distribution analysis:")
    for group in sorted(groups):
        group_mask = df[sensitive_attr] == group
        group_size = group_mask.sum()
        group_prop = group_size / total_size

        print(f"  {group}: {group_size:,} samples ({group_prop:.3f})")

        for outcome in outcomes:
            outcome_mask = df[target_attr] == outcome
            cell_size = (group_mask & outcome_mask).sum()

            if cell_size > 0:
                # Expected count if perfectly balanced
                expected_count = group_prop * (outcome_mask.sum() / total_size) * total_size
                # Reweighing factor
                weight = expected_count / cell_size

                # Apply weights
                cell_mask = group_mask & outcome_mask
                weights[cell_mask] = weight

                print(f"    {group} & {target_attr}={outcome}: {cell_size} samples, weight={weight:.3f}")

    return weights


# Calculate reweighing weights
reweight_weights = calculate_reweighing_weights(df, "region_group", "TARGET")

# ------------------ 5. Train Models with Different Fairness Approaches ------------------
print(f"\n🤖 TRAINING MODELS WITH DIFFERENT FAIRNESS APPROACHES...")

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Get corresponding weights and groups for train/val sets
train_indices = X_train.index
val_indices = X_val.index
train_weights = reweight_weights[train_indices]
val_weights = reweight_weights[val_indices]
train_groups = df.loc[train_indices, "region_group"]
val_groups = df.loc[val_indices, "region_group"]

# Store all models and results
models = {}
results = {}

# ===== BASELINE MODEL (No Fairness Intervention) =====
print(f"\n1️⃣ BASELINE MODEL (No Fairness Intervention)")
model_baseline = LogisticRegression(
    max_iter=1000,
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)
model_baseline.fit(X_train, y_train)
models['baseline'] = model_baseline

# Evaluate baseline
train_auc_baseline = roc_auc_score(y_train, model_baseline.predict_proba(X_train)[:, 1])
val_auc_baseline = roc_auc_score(y_val, model_baseline.predict_proba(X_val)[:, 1])
print(f"  Training AUC: {train_auc_baseline:.3f}")
print(f"  Validation AUC: {val_auc_baseline:.3f}")

# Feature importance analysis for baseline
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model_baseline.coef_[0],
    'abs_coefficient': np.abs(model_baseline.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 10 Most Important Features (by coefficient magnitude):")
print(feature_importance.head(10)[['feature', 'coefficient']].to_string(index=False))

# ===== PRE-PROCESSING MODEL (Reweighing Only) =====
print(f"\n2️⃣ PRE-PROCESSING MODEL (Reweighing Only)")
model_reweigh = LogisticRegression(
    max_iter=1000,
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)
model_reweigh.fit(X_train, y_train, sample_weight=train_weights)
models['reweigh'] = model_reweigh

# Evaluate reweighing model
train_auc_reweigh = roc_auc_score(y_train, model_reweigh.predict_proba(X_train)[:, 1])
val_auc_reweigh = roc_auc_score(y_val, model_reweigh.predict_proba(X_val)[:, 1])
print(f"  Training AUC: {train_auc_reweigh:.3f}")
print(f"  Validation AUC: {val_auc_reweigh:.3f}")

# ------------------ 6. Initial Business Decisions ------------------
# Use a conservative threshold for regional lending
conservative_threshold = 0.55

print(f"\n🏦 APPLYING BUSINESS DECISIONS (Conservative Threshold: {conservative_threshold:.0%})...")

# Apply threshold to all models
for model_name, model in models.items():
    df[f"y_prob_default_{model_name}"] = model.predict_proba(X)[:, 1]
    df[f"y_pred_{model_name}"] = (df[f"y_prob_default_{model_name}"] < conservative_threshold).astype(int)

    approval_rate = df[f"y_pred_{model_name}"].mean()
    print(f"  {model_name.title()}: {approval_rate:.3f} approval rate")

# Threshold sensitivity analysis for baseline
print(f"\nThreshold Sensitivity Analysis (Baseline Model):")
thresholds = [0.08, 0.10, 0.12, 0.15, 0.20]
for thresh in thresholds:
    approval_rate = (df["y_prob_default_baseline"] < thresh).mean()
    print(f"  Threshold {thresh:.0%}: {approval_rate:.1%} approval rate")


# ------------------ 7. Post-processing Implementation ------------------
def apply_equal_opportunity_fairness(df, model_name, sensitive_col, target_tpr=None):
    """
    Apply post-processing fairness intervention to equalize True Positive Rates across groups
    """
    pred_col = f"y_pred_{model_name}"
    prob_col = f"y_prob_default_{model_name}"
    fair_col = f"y_pred_{model_name}_fair"

    df[fair_col] = df[pred_col].copy()

    if target_tpr is None:
        # Use the overall TPR as target
        target_tpr = ((df[pred_col] == 1) & (df["TARGET"] == 0)).sum() / df["TARGET"].eq(0).sum()

    print(f"  Target TPR for {model_name}: {target_tpr:.3f}")

    for group in sorted(df[sensitive_col].unique()):
        group_mask = df[sensitive_col] == group
        group_data = df[group_mask]

        # Current group TPR
        current_tpr = ((group_data[pred_col] == 1) & (group_data["TARGET"] == 0)).sum() / max(
            group_data["TARGET"].eq(0).sum(), 1)
        print(f"    {group}: Current TPR = {current_tpr:.3f}, Target = {target_tpr:.3f}")

        if current_tpr < target_tpr:
            # Need to approve more people in this group
            rejected_mask = (group_data[pred_col] == 0)
            rejected_indices = group_data[rejected_mask].index

            if len(rejected_indices) > 0:
                good_customers = (group_data["TARGET"] == 0).sum()
                additional_approvals_needed = int((target_tpr - current_tpr) * good_customers)
                additional_approvals_needed = min(additional_approvals_needed, len(rejected_indices))

                if additional_approvals_needed > 0:
                    safest_rejected = group_data.loc[rejected_indices].nsmallest(
                        additional_approvals_needed, prob_col
                    ).index
                    df.loc[safest_rejected, fair_col] = 1
                    print(f"      → Approved {additional_approvals_needed} additional applications")

        elif current_tpr > target_tpr:
            # Need to approve fewer people in this group
            approved_mask = (group_data[pred_col] == 1)
            approved_indices = group_data[approved_mask].index

            if len(approved_indices) > 0:
                good_customers = (group_data["TARGET"] == 0).sum()
                rejections_needed = int((current_tpr - target_tpr) * good_customers)
                rejections_needed = min(rejections_needed, len(approved_indices))

                if rejections_needed > 0:
                    riskiest_approved = group_data.loc[approved_indices].nlargest(
                        rejections_needed, prob_col
                    ).index
                    df.loc[riskiest_approved, fair_col] = 0
                    print(f"      → Rejected {rejections_needed} risky applications")

    return df


# ===== POST-PROCESSING ONLY (Baseline + Post-processing) =====
print(f"\n3️⃣ POST-PROCESSING ONLY (Baseline + Post-processing)")
df = apply_equal_opportunity_fairness(df, "baseline", "region_group")

# ===== COMBINED APPROACH (Pre-processing + Post-processing) =====
print(f"\n4️⃣ COMBINED APPROACH (Pre-processing + Post-processing)")
df = apply_equal_opportunity_fairness(df, "reweigh", "region_group")

# ------------------ 8. Profit Calculation Function ------------------
roi = 0.2664  # 26.64% return on investment
recovery_rate = 0.4  # 40% recovery on defaulted loans


def calculate_profit(df, pred_column):
    """Calculate profit for each loan decision"""
    return np.where(
        (df[pred_column] == 1) & (df["TARGET"] == 0),  # Approved + No Default
        df["AMT_CREDIT"] * roi,
        np.where(
            (df[pred_column] == 1) & (df["TARGET"] == 1),  # Approved + Default
            -df["AMT_CREDIT"] * (1 - recovery_rate),
            0  # Rejected loans
        )
    )


# Calculate profits for all approaches
profit_columns = []
approaches = [
    ('baseline', 'No Fairness'),
    ('baseline_fair', 'Post-processing Only'),
    ('reweigh', 'Pre-processing Only'),
    ('reweigh_fair', 'Combined Approach')
]

print(f"\n💰 PROFIT ANALYSIS FOR ALL APPROACHES:")
for pred_col, approach_name in approaches:
    profit_col = f"profit_{pred_col}"
    df[profit_col] = calculate_profit(df, f"y_pred_{pred_col}")
    total_profit = df[profit_col].sum()
    approval_rate = df[f"y_pred_{pred_col}"].mean()

    print(f"  {approach_name:20s}: €{total_profit:11,.0f} (Approval: {approval_rate:.3f})")
    profit_columns.append((profit_col, approach_name))


# ------------------ 9. Comprehensive Fairness Analysis ------------------
def analyze_fairness_comprehensive(df, group_column, approaches, title=""):
    """Analyze fairness metrics by group for all approaches"""
    print(f"\n📊 {title}")
    print(f"{'=' * 80}")

    from scipy import stats

    # Statistical significance test across regions
    region_groups = df[group_column].unique()
    print(f"Statistical significance test for default rate differences across regions:")

    # Perform ANOVA test for multiple groups
    from scipy.stats import f_oneway
    group_data = [df[df[group_column] == group]['TARGET'] for group in region_groups]
    f_stat, p_value = f_oneway(*group_data)
    print(f"  F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")

    # Region group risk analysis
    print(f"\n🌍 REGION GROUP RISK ANALYSIS:")
    region_analysis = df.groupby('region_group').agg({
        'TARGET': ['count', 'mean'],
        'y_prob_default_baseline': 'mean',
        'AMT_CREDIT': 'mean'
    }).round(3)

    region_analysis.columns = ['Count', 'Default Rate', 'Avg Default Prob', 'Avg Loan Amount']
    print(region_analysis.to_string())

    all_results = []

    for pred_col, approach_name in approaches:
        print(f"\n🔍 {approach_name.upper()}")
        print(f"-" * 50)

        pred_column = f"y_pred_{pred_col}"
        profit_column = f"profit_{pred_col}"
        prob_column = f"y_prob_default_{pred_col.split('_')[0]}"  # Use base model probabilities

        results = []
        overall_selection = df[pred_column].mean()
        overall_tpr = ((df[pred_column] == 1) & (df["TARGET"] == 0)).sum() / max(df["TARGET"].eq(0).sum(), 1)

        for group_name in sorted(df[group_column].unique()):
            group = df[df[group_column] == group_name]
            selection_rate = group[pred_column].mean()
            tpr = ((group[pred_column] == 1) & (group["TARGET"] == 0)).sum() / max(group["TARGET"].eq(0).sum(), 1)
            group_profit = group[profit_column].sum()
            avg_default_prob = group[prob_column].mean()

            results.append({
                "Region Group": group_name,
                "Size": f"{len(group):,}",
                "Default Risk": f"{avg_default_prob:.3f}",
                "Selection Rate": f"{selection_rate:.3f}",
                "TPR": f"{tpr:.3f}",
                "Profit (€M)": f"{group_profit / 1e6:.1f}",
                "DP Diff": f"{selection_rate - overall_selection:+.3f}",
                "EO Diff": f"{tpr - overall_tpr:+.3f}"
            })

        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

        # Summary metrics
        dp_diffs = [float(x) for x in df_results["DP Diff"].str.replace('+', '')]
        eo_diffs = [float(x) for x in df_results["EO Diff"].str.replace('+', '')]
        max_dp_diff = max(abs(min(dp_diffs)), abs(max(dp_diffs)))
        max_eo_diff = max(abs(min(eo_diffs)), abs(max(eo_diffs)))

        print(f"\nFairness Summary:")
        print(f"  Demographic Parity Gap: {max_dp_diff:.3f}")
        print(f"  Equal Opportunity Gap: {max_eo_diff:.3f}")

        all_results.append({
            'approach': approach_name,
            'pred_col': pred_col,
            'dp_gap': max_dp_diff,
            'eo_gap': max_eo_diff,
            'total_profit': df[profit_column].sum(),
            'approval_rate': df[pred_column].mean()
        })

    return all_results


# Run comprehensive fairness analysis
fairness_results = analyze_fairness_comprehensive(df, "region_group", approaches,
                                                  "COMPREHENSIVE REGION FAIRNESS ANALYSIS")

# ------------------ 10. Model Performance Comparison ------------------
print(f"\n📈 MODEL PERFORMANCE COMPARISON:")
print(f"{'=' * 60}")

performance_data = []
for pred_col, approach_name in approaches:
    pred_column = f"y_pred_{pred_col}"
    y_pred_binary = df[pred_column]

    perf = {
        'Approach': approach_name,
        'Accuracy': f"{accuracy_score(y, y_pred_binary):.3f}",
        'Precision': f"{precision_score(y, y_pred_binary):.3f}",
        'Recall': f"{recall_score(y, y_pred_binary):.3f}",
        'F1-Score': f"{f1_score(y, y_pred_binary):.3f}"
    }
    performance_data.append(perf)

performance_df = pd.DataFrame(performance_data)
print(performance_df.to_string(index=False))

# ------------------ 11. Summary Comparison Table ------------------
print(f"\n🎯 COMPREHENSIVE RESULTS SUMMARY:")
print(f"{'=' * 80}")

summary_data = []
baseline_profit = fairness_results[0]['total_profit']  # No fairness baseline

for result in fairness_results:
    profit_change = result['total_profit'] - baseline_profit
    profit_change_pct = (profit_change / baseline_profit) * 100 if baseline_profit != 0 else 0

    summary_data.append({
        'Approach': result['approach'],
        'AUC': f"{val_auc_baseline:.3f}" if 'baseline' in result['pred_col'] else f"{val_auc_reweigh:.3f}",
        'Approval Rate': f"{result['approval_rate']:.3f}",
        'DP Gap': f"{result['dp_gap']:.3f}",
        'EO Gap': f"{result['eo_gap']:.3f}",
        'Profit (€M)': f"{result['total_profit'] / 1e6:.1f}",
        'Profit Δ%': f"{profit_change_pct:+.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# ------------------ 12. Region-Specific Insights ------------------
print(f"\n🌍 REGION GROUP INSIGHTS:")
print(f"{'=' * 50}")
print(f"This analysis compares lending fairness across three region quality tiers:")
print(f"• Best: Regions with highest quality rating (lowest default risk)")
print(f"• Middle: Regions with average quality rating (moderate default risk)")
print(f"• Worst: Regions with lowest quality rating (highest default risk)")
print(f"The analysis ensures equal opportunity regardless of geographic location.")

print(f"\n📊 KEY FINDINGS ACROSS ALL APPROACHES:")
for region in sorted(df['region_group'].unique()):
    region_data = df[df['region_group'] == region]
    print(f"\n{region} regions ({len(region_data):,} applicants):")
    print(f"  - Default rate: {region_data['TARGET'].mean():.3f}")
    print(f"  - Avg default probability: {region_data['y_prob_default_baseline'].mean():.3f}")
    for pred_col, approach_name in approaches:
        approval_rate = region_data[f"y_pred_{pred_col}"].mean()
        print(f"  - {approach_name} approval rate: {approval_rate:.3f}")

# ------------------ 13. Business Insights and Recommendations ------------------
print(f"\n💼 BUSINESS INSIGHTS AND RECOMMENDATIONS:")
print(f"{'=' * 60}")

best_fairness = min(fairness_results, key=lambda x: x['eo_gap'])
best_profit = max(fairness_results, key=lambda x: x['total_profit'])

print(f"Key Findings:")
print(f"• Best fairness (lowest EO gap): {best_fairness['approach']} ({best_fairness['eo_gap']:.3f})")
print(f"• Best profit: {best_profit['approach']} (€{best_profit['total_profit'] / 1e6:.1f}M)")

# Compare interventions
baseline_result = fairness_results[0]
postproc_result = fairness_results[1]
preproc_result = fairness_results[2]
combined_result = fairness_results[3]

print(f"\nIntervention Effectiveness for Regional Fairness:")
print(f"• Pre-processing (reweighing):")
print(f"  - Fairness improvement: {baseline_result['eo_gap']:.3f} → {preproc_result['eo_gap']:.3f}")
print(
    f"  - Profit impact: {((preproc_result['total_profit'] - baseline_result['total_profit']) / baseline_result['total_profit'] * 100):+.1f}%")

print(f"• Post-processing:")
print(f"  - Fairness improvement: {baseline_result['eo_gap']:.3f} → {postproc_result['eo_gap']:.3f}")
print(
    f"  - Profit impact: {((postproc_result['total_profit'] - baseline_result['total_profit']) / baseline_result['total_profit'] * 100):+.1f}%")

print(f"• Combined approach:")
print(f"  - Fairness improvement: {baseline_result['eo_gap']:.3f} → {combined_result['eo_gap']:.3f}")
print(
    f"  - Profit impact: {((combined_result['total_profit'] - baseline_result['total_profit']) / baseline_result['total_profit'] * 100):+.1f}%")

print(f"\nRegional Considerations:")
print(f"• Conservative threshold ({conservative_threshold:.0%}) ensures responsible lending")
print(f"• Three-tier regional analysis provides nuanced geographic fairness insights")
print(f"• Logistic Regression offers interpretable coefficients for regulatory compliance")
print(f"• Geographic bias can reflect systemic economic inequalities")

# ------------------ 14. Final Recommendations ------------------
print(f"\n🏆 FINAL RECOMMENDATIONS FOR REGIONAL FAIRNESS:")
print(f"{'=' * 60}")

if combined_result['eo_gap'] < 0.05:
    print(f"✅ RECOMMENDED: Combined Approach")
    print(f"   - Achieves excellent fairness (EO gap: {combined_result['eo_gap']:.3f})")
    print(f"   - Profit: €{combined_result['total_profit'] / 1e6:.1f}M")
elif min(preproc_result['eo_gap'], postproc_result['eo_gap']) < 0.05:
    better_single = preproc_result if preproc_result['total_profit'] > postproc_result[
        'total_profit'] else postproc_result
    print(f"✅ RECOMMENDED: {better_single['approach']}")
    print(f"   - Good fairness (EO gap: {better_single['eo_gap']:.3f})")
    print(f"   - Better profit than combined approach")
else:
    print(f"⚠️  FURTHER TUNING NEEDED")
    print(f"   - No approach achieves satisfactory fairness (EO gap < 0.05)")
    print(f"   - Consider adjusting thresholds or additional interventions")

print(f"\nRegulatory Compliance for Regional Fairness:")
if best_fairness['eo_gap'] < 0.05:
    print(f"✅ COMPLIANT: Equal opportunity gap < 5%")
else:
    print(f"❌ NON-COMPLIANT: Equal opportunity gap ≥ 5%")
    print(f"   Additional interventions may be required for regulatory approval")

print(f"\n📋 IMPLEMENTATION PRIORITY FOR REGIONAL FAIRNESS:")
print(f"1. Start with baseline model for benchmark")
print(
    f"2. Implement {'pre-processing' if preproc_result['eo_gap'] < postproc_result['eo_gap'] else 'post-processing'} intervention first")
print(f"3. If fairness insufficient, add the other intervention")
print(f"4. Monitor regional fairness metrics continuously in production")
print(f"5. Consider economic development programs for underserved regions")
print(f"6. Ensure compliance with fair lending and community reinvestment regulations")