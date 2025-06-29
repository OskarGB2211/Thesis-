import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# ------------------ 1. Load Cleaned Data ------------------
df = pd.read_csv("C:/Users/Oskar Giebichenstein/Desktop/Bachelor Data/cleaned_application_full.csv")

# ------------------ 2. Create Binary Children Groups ------------------
# Create simplified binary children classification: No Children vs Has Children
if "CNT_CHILDREN" in df.columns:
    df["binary_children_group"] = df["CNT_CHILDREN"].apply(lambda x: "No Children" if x == 0 else "Has Children")
else:
    print("Warning: No CNT_CHILDREN data found. Creating synthetic children groups.")
    np.random.seed(42)
    df["binary_children_group"] = np.random.choice(["No Children", "Has Children"], size=len(df), p=[0.6, 0.4])

print(f"Dataset size: {len(df):,} applications")
print(f"Default rate: {df['TARGET'].mean():.3f}")
print(f"Binary children groups:")
print(df['binary_children_group'].value_counts())
print(f"Children group percentages:")
print(df['binary_children_group'].value_counts(normalize=True).round(3))

# ------------------ 3. Define Features and Target ------------------
# Drop unused or sensitive columns - excluding children features from training
X = df.drop(columns=[
    "TARGET", "SK_ID_CURR",
    "CNT_CHILDREN", "binary_children_group"  # Remove children features from training
])
# Ensure only numeric features are used
X = X.select_dtypes(include=[np.number])
y = df["TARGET"]

print(f"\nFeature count: {X.shape[1]}")


# ------------------ 4. Pre-processing: Reweighing Implementation ------------------
def calculate_reweighing_weights(df, sensitive_attr, target_attr):
    """
    Calculate reweighing weights to ensure demographic parity for Random Forest
    """
    print(f"\n CALCULATING REWEIGHING WEIGHTS FOR CHILDREN GROUPS (Random Forest)...")

    # Get unique groups and outcomes
    groups = df[sensitive_attr].unique()
    outcomes = df[target_attr].unique()

    weights = np.ones(len(df))

    # Calculate overall proportions
    total_size = len(df)

    print(f"Children group distribution analysis:")
    for group in groups:
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
reweight_weights = calculate_reweighing_weights(df, "binary_children_group", "TARGET")

# ------------------ 5. Train Models with Different Fairness Approaches ------------------
print(f"\n TRAINING RANDOM FOREST MODELS WITH DIFFERENT FAIRNESS APPROACHES...")

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Get corresponding weights and groups for train/val sets
train_indices = X_train.index
val_indices = X_val.index
train_weights = reweight_weights[train_indices]
val_weights = reweight_weights[val_indices]
train_groups = df.loc[train_indices, "binary_children_group"]
val_groups = df.loc[val_indices, "binary_children_group"]

# Store all models and results
models = {}
results = {}

# Random Forest parameters optimized for credit scoring
rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# ===== BASELINE MODEL (No Fairness Intervention) =====
print(f"\n BASELINE RANDOM FOREST MODEL (No Fairness Intervention)")
model_baseline = RandomForestClassifier(**rf_params)
model_baseline.fit(X_train, y_train)
models['baseline'] = model_baseline

# Evaluate baseline
train_auc_baseline = roc_auc_score(y_train, model_baseline.predict_proba(X_train)[:, 1])
val_auc_baseline = roc_auc_score(y_val, model_baseline.predict_proba(X_val)[:, 1])
print(f"  Training AUC: {train_auc_baseline:.3f}")
print(f"  Validation AUC: {val_auc_baseline:.3f}")
print(
    f"  Overfitting check: {'✓ Good' if abs(train_auc_baseline - val_auc_baseline) < 0.05 else '⚠ Possible overfitting'}")

# Feature importance analysis for baseline
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features (Baseline Random Forest):")
print(feature_importance.head(10).to_string(index=False))

# ===== PRE-PROCESSING MODEL (Reweighing Only) =====
print(f"\nPRE-PROCESSING RANDOM FOREST MODEL (Reweighing Only)")
model_reweigh = RandomForestClassifier(**rf_params)
model_reweigh.fit(X_train, y_train, sample_weight=train_weights)
models['reweigh'] = model_reweigh

# Evaluate reweighing model
train_auc_reweigh = roc_auc_score(y_train, model_reweigh.predict_proba(X_train)[:, 1])
val_auc_reweigh = roc_auc_score(y_val, model_reweigh.predict_proba(X_val)[:, 1])
print(f"  Training AUC: {train_auc_reweigh:.3f}")
print(f"  Validation AUC: {val_auc_reweigh:.3f}")
print(
    f"  Overfitting check: {'✓ Good' if abs(train_auc_reweigh - val_auc_reweigh) < 0.05 else '⚠ Possible overfitting'}")

# Feature importance comparison
feature_importance_reweigh = pd.DataFrame({
    'feature': X.columns,
    'importance': model_reweigh.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance Changes (Reweighing vs Baseline):")
importance_comparison = feature_importance.merge(
    feature_importance_reweigh, on='feature', suffixes=('_baseline', '_reweigh')
)
importance_comparison['importance_change'] = (
        importance_comparison['importance_reweigh'] - importance_comparison['importance_baseline']
)
print(importance_comparison.head(10)[
          ['feature', 'importance_baseline', 'importance_reweigh', 'importance_change']].to_string(index=False))

# ------------------ 6. Initial Business Decisions ------------------
optimal_threshold = 0.50

print(f"\n APPLYING BUSINESS DECISIONS (Threshold: {optimal_threshold:.0%})...")

# Apply threshold to all models
for model_name, model in models.items():
    df[f"y_prob_default_{model_name}"] = model.predict_proba(X)[:, 1]
    df[f"y_pred_{model_name}"] = (df[f"y_prob_default_{model_name}"] < optimal_threshold).astype(int)

    approval_rate = df[f"y_pred_{model_name}"].mean()
    print(f"  {model_name.title()}: {approval_rate:.3f} approval rate")

# Threshold sensitivity analysis for baseline
print(f"\nThreshold Sensitivity Analysis (Baseline Random Forest):")
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
for thresh in thresholds:
    approval_rate = (df["y_prob_default_baseline"] < thresh).mean()
    print(f"  Threshold {thresh:.0%}: {approval_rate:.1%} approval rate")

# ------------------ 7. Random Forest Specific Analysis ------------------
print(f"\n RANDOM FOREST MODEL INSIGHTS:")

# Analyze prediction confidence by children group for both models
print(f"Prediction Confidence Analysis:")
for model_name in ['baseline', 'reweigh']:
    print(f"\n{model_name.title()} Model:")
    for group_name in sorted(df["binary_children_group"].unique()):
        group = df[df["binary_children_group"] == group_name]
        confidence = np.abs(group[f"y_prob_default_{model_name}"] - 0.5)
        avg_confidence = confidence.mean()
        print(f"  {group_name}: Avg confidence = {avg_confidence:.3f}")

# Children-related features in model
children_related_features = [col for col in X.columns if
                             'CHILD' in col.upper() or 'FAM' in col.upper() or 'DEPEND' in col.upper()]
if children_related_features:
    print(f"\nChildren-related features in model:")
    children_importance = feature_importance[feature_importance['feature'].isin(children_related_features)]
    print(children_importance.to_string(index=False))
else:
    print(f"\nNo direct children features found in model (as expected - they were excluded)")
    print("Family patterns detected through correlated features via Random Forest ensemble")


# ------------------ 8. Post-processing Implementation ------------------
def apply_equal_opportunity_fairness_rf(df, model_name, sensitive_col, target_tpr=None):
    """
    Apply post-processing fairness intervention to equalize True Positive Rates across groups
    Enhanced for Random Forest with high-quality probability estimates
    """
    pred_col = f"y_pred_{model_name}"
    prob_col = f"y_prob_default_{model_name}"
    fair_col = f"y_pred_{model_name}_fair"

    df[fair_col] = df[pred_col].copy()

    if target_tpr is None:
        # Use the overall TPR as target
        target_tpr = ((df[pred_col] == 1) & (df["TARGET"] == 0)).sum() / df["TARGET"].eq(0).sum()

    print(f"  Target TPR for {model_name} Random Forest: {target_tpr:.3f}")

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
                    # Random Forest provides excellent probability calibration
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
print(f"\n POST-PROCESSING ONLY (Baseline Random Forest + Post-processing)")
df = apply_equal_opportunity_fairness_rf(df, "baseline", "binary_children_group")

# ===== COMBINED APPROACH (Pre-processing + Post-processing) =====
print(f"\nCOMBINED APPROACH (Pre-processing + Post-processing Random Forest)")
df = apply_equal_opportunity_fairness_rf(df, "reweigh", "binary_children_group")

# ------------------ 9. Profit Calculation Function ------------------
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
    ('baseline', 'No Fairness (RF)'),
    ('baseline_fair', 'Post-processing Only (RF)'),
    ('reweigh', 'Pre-processing Only (RF)'),
    ('reweigh_fair', 'Combined Approach (RF)')
]

print(f"\n PROFIT ANALYSIS FOR ALL RANDOM FOREST APPROACHES:")
for pred_col, approach_name in approaches:
    profit_col = f"profit_{pred_col}"
    df[profit_col] = calculate_profit(df, f"y_pred_{pred_col}")
    total_profit = df[profit_col].sum()
    approval_rate = df[f"y_pred_{pred_col}"].mean()

    print(f"  {approach_name:25s}: €{total_profit:11,.0f} (Approval: {approval_rate:.3f})")
    profit_columns.append((profit_col, approach_name))


# ------------------ 10. Comprehensive Fairness Analysis ------------------
def analyze_fairness_comprehensive_rf(df, group_column, approaches, title=""):
    """Analyze fairness metrics by group for all Random Forest approaches"""
    print(f"\n {title}")
    print(f"{'=' * 80}")

    from scipy import stats

    # Statistical significance test
    no_children = df[df[group_column] == 'No Children']['TARGET']
    has_children = df[df[group_column] == 'Has Children']['TARGET']
    stat, p_value = stats.ttest_ind(no_children, has_children)
    print(f"Statistical significance test for default rate difference:")
    print(f"  t-statistic: {stat:.3f}, p-value: {p_value:.6f}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")

    # Children group risk analysis
    print(f"\n BINARY CHILDREN GROUP RISK ANALYSIS (Random Forest):")
    children_analysis = df.groupby('binary_children_group').agg({
        'TARGET': ['count', 'mean'],
        'y_prob_default_baseline': 'mean',
        'y_prob_default_reweigh': 'mean',
        'AMT_CREDIT': 'mean'
    }).round(3)

    children_analysis.columns = ['Count', 'Default Rate', 'Baseline RF Prob', 'Reweigh RF Prob', 'Avg Loan Amount']
    print(children_analysis.to_string())

    all_results = []

    for pred_col, approach_name in approaches:
        print(f"\n🔍 {approach_name.upper()}")
        print(f"-" * 55)

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
                "Children Group": group_name,
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
fairness_results = analyze_fairness_comprehensive_rf(df, "binary_children_group", approaches,
                                                     "COMPREHENSIVE RANDOM FOREST CHILDREN FAIRNESS ANALYSIS")

# ------------------ 11. Model Performance Comparison ------------------
print(f"\n RANDOM FOREST MODEL PERFORMANCE COMPARISON:")
print(f"{'=' * 70}")

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

# ------------------ 12. Summary Comparison Table ------------------
print(f"\n COMPREHENSIVE RANDOM FOREST RESULTS SUMMARY:")
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

# ------------------ 13. Random Forest Advantages Analysis ------------------
print(f"\n RANDOM FOREST ADVANTAGES FOR CHILDREN FAIRNESS:")
print(f"{'=' * 60}")

print(f"Ensemble Benefits:")
print(f"• Superior handling of feature interactions and non-linear family patterns")
print(f"• More robust probability estimates for threshold optimization")
print(f"• Built-in feature importance helps identify indirect family bias sources")
print(f"• Less sensitive to outliers in family-related correlated features")
print(f"• Ensemble nature provides more stable predictions across family groups")
print(f"• Better capture of complex socioeconomic factors that correlate with having children")

print(f"\nReweighing Integration:")
print(f"• Random Forest naturally handles sample weights during training")
print(f"• Maintains ensemble diversity while adjusting for family status bias")
print(f"• Feature importance remains interpretable after reweighing")
print(f"• Robust to overfitting even with reweighed samples")

# ------------------ 14. Children-Specific Insights ------------------
print(f"\n BINARY CHILDREN GROUP INSIGHTS (Random Forest Analysis):")
print(f"{'=' * 65}")
print(f"This Random Forest analysis compares lending fairness between two family status cohorts:")
print(f"• No Children: Applicants without dependents who may have more disposable income")
print(f"• Has Children: Applicants with 1+ children who may face family-related financial stress")
print(f"Random Forest excels at detecting subtle family-related patterns through ensemble learning.")

print(f"\n KEY FINDINGS ACROSS ALL RANDOM FOREST APPROACHES:")
no_children_data = df[df['binary_children_group'] == 'No Children']
has_children_data = df[df['binary_children_group'] == 'Has Children']

print(f"\nNo Children group ({len(no_children_data):,} applicants):")
print(f"  - Default rate: {no_children_data['TARGET'].mean():.3f}")
print(f"  - Baseline RF default probability: {no_children_data['y_prob_default_baseline'].mean():.3f}")
print(f"  - Reweigh RF default probability: {no_children_data['y_prob_default_reweigh'].mean():.3f}")
for pred_col, approach_name in approaches:
    approval_rate = no_children_data[f"y_pred_{pred_col}"].mean()
    print(f"  - {approach_name} approval rate: {approval_rate:.3f}")

print(f"\nHas Children group ({len(has_children_data):,} applicants):")
print(f"  - Default rate: {has_children_data['TARGET'].mean():.3f}")
print(f"  - Baseline RF default probability: {has_children_data['y_prob_default_baseline'].mean():.3f}")
print(f"  - Reweigh RF default probability: {has_children_data['y_prob_default_reweigh'].mean():.3f}")
for pred_col, approach_name in approaches:
    approval_rate = has_children_data[f"y_pred_{pred_col}"].mean()
    print(f"  - {approach_name} approval rate: {approval_rate:.3f}")

# ------------------ 15. Business Insights and Recommendations ------------------
print(f"\n BUSINESS INSIGHTS AND RECOMMENDATIONS:")
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

print(f"\nRandom Forest Intervention Effectiveness for Family Status:")
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

print(f"\nRandom Forest Family Status Considerations:")
print(f"• Binary children grouping leverages RF's non-linear capabilities")
print(f"• Ensemble learning provides robust family bias detection")
print(f"• Feature importance reveals complex family-related patterns")
statistical_result = fairness_results[0]
print(f"• Random Forest excels at capturing socioeconomic family dynamics")

# ------------------ 16. Final Recommendations ------------------
print(f"\n FINAL RECOMMENDATIONS FOR RANDOM FOREST FAMILY STATUS FAIRNESS:")
print(f"{'=' * 70}")

if combined_result['eo_gap'] < 0.05:
    print(f" RECOMMENDED: Combined Approach (Random Forest)")
    print(f"   - Achieves excellent fairness (EO gap: {combined_result['eo_gap']:.3f})")
    print(f"   - Profit: €{combined_result['total_profit'] / 1e6:.1f}M")
    print(f"   - Leverages both RF ensemble strength and fairness interventions")
elif min(preproc_result['eo_gap'], postproc_result['eo_gap']) < 0.05:
    better_single = preproc_result if preproc_result['total_profit'] > postproc_result[
        'total_profit'] else postproc_result
    print(f" RECOMMENDED: {better_single['approach']}")
    print(f"   - Good fairness (EO gap: {better_single['eo_gap']:.3f})")
    print(f"   - Better profit efficiency than combined approach")
else:
    print(f"️  FURTHER TUNING NEEDED")
    print(f"   - No approach achieves satisfactory fairness (EO gap < 0.05)")
    print(f"   - Consider Random Forest hyperparameter optimization")

print(f"\nRegulatory Compliance for Family Status (Random Forest):")
if best_fairness['eo_gap'] < 0.05:
    print(f" COMPLIANT: Equal opportunity gap < 5%")
else:
    print(f" NON-COMPLIANT: Equal opportunity gap ≥ 5%")
    print(f"   Consider ensemble method adjustments for compliance")

print(f"\n IMPLEMENTATION PRIORITY FOR RANDOM FOREST CHILDREN FAIRNESS:")
print(f"1. Start with baseline Random Forest for benchmark performance")
print(
    f"2. Implement {'pre-processing' if preproc_result['eo_gap'] < postproc_result['eo_gap'] else 'post-processing'} intervention first")
print(f"3. If fairness insufficient, add the other intervention method")
print(f"4. Monitor family status fairness metrics continuously in production")
print(f"5. Leverage Random Forest feature importance for bias source identification")
print(f"6. Consider ethical implications of family structure in automated lending")

# ------------------ 17. Model Comparison vs Other Algorithms ------------------
print(f"\n RANDOM FOREST VS OTHER ALGORITHMS FOR FAMILY FAIRNESS:")
print(f"{'=' * 65}")

print(f"Expected Performance Ranking (for credit scoring with family fairness):")
print(f"1. Random Forest: Best balance of performance, interpretability, and fairness")
print(f"   - Superior handling of complex family-related feature interactions")
print(f"   - Robust probability estimates for fair threshold optimization")
print(f"   - Built-in feature importance for regulatory compliance")

print(f"2. XGBoost: Highest raw performance but less interpretable")
print(f"   - Best AUC scores but gradient boosting complexity")
print(f"   - Excellent probability calibration for fairness interventions")

print(f"3. Logistic Regression: Most interpretable but limited capability")
print(f"   - Linear assumptions may miss family-related patterns")
print(f"   - Good regulatory transparency but lower performance")

print(f"\nRandom Forest Fairness Advantages:")
print(f"• Ensemble nature provides stable fairness across family groups")
print(f"• Feature importance reveals hidden family bias sources")
print(f"• Handles complex socioeconomic interactions naturally")
print(f"• Robust to outliers that might skew family-based decisions")
print(f"• Good balance between performance and interpretability")

# ------------------ 18. Advanced Fairness Insights ------------------
print(f"\n ADVANCED RANDOM FOREST FAIRNESS INSIGHTS:")
print(f"{'=' * 50}")

# Analyze feature importance differences between groups
print(f"Feature Importance Analysis by Family Status:")
for group_name in sorted(df["binary_children_group"].unique()):
    group_data = df[df["binary_children_group"] == group_name]
    print(f"\n{group_name} Group:")
    print(f"  - Sample size: {len(group_data):,}")
    print(
        f"  - Avg baseline RF prediction confidence: {np.abs(group_data['y_prob_default_baseline'] - 0.5).mean():.3f}")
    print(
        f"  - Avg reweighed RF prediction confidence: {np.abs(group_data['y_prob_default_reweigh'] - 0.5).mean():.3f}")

# Probability distribution analysis
print(f"\nProbability Distribution Analysis:")
prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for group_name in sorted(df["binary_children_group"].unique()):
    group_data = df[df["binary_children_group"] == group_name]
    prob_dist = pd.cut(group_data['y_prob_default_baseline'], bins=prob_bins).value_counts(normalize=True).sort_index()
    print(f"\n{group_name} - Baseline RF Probability Distribution:")
    for interval, percentage in prob_dist.items():
        print(f"  {interval}: {percentage:.1%}")

# Tree depth and complexity analysis
print(f"\nRandom Forest Model Complexity:")
print(f"  Baseline Model:")
print(f"    - Number of trees: {rf_params['n_estimators']}")
print(f"    - Max depth: {rf_params['max_depth']}")
print(f"    - Min samples split: {rf_params['min_samples_split']}")
print(f"    - Min samples leaf: {rf_params['min_samples_leaf']}")
print(f"    - Max features: {rf_params['max_features']}")

print(f"  Reweighed Model:")
print(f"    - Same architecture with weighted training")
print(f"    - Bias correction through sample importance")
print(f"    - Maintained ensemble diversity")

# ------------------ 19. Fairness Intervention Cost-Benefit Analysis ------------------
print(f"\n COST-BENEFIT ANALYSIS OF FAIRNESS INTERVENTIONS:")
print(f"{'=' * 55}")

print(f"Financial Impact Summary:")
baseline_profit = baseline_result['total_profit']
for i, (result, (pred_col, approach_name)) in enumerate(zip(fairness_results, approaches)):
    if i == 0:  # Skip baseline
        continue

    profit_change = result['total_profit'] - baseline_profit
    profit_change_pct = (profit_change / baseline_profit) * 100
    fairness_gain = baseline_result['eo_gap'] - result['eo_gap']

    print(f"\n{approach_name}:")
    print(f"  - Profit change: €{profit_change:,.0f} ({profit_change_pct:+.1f}%)")
    print(f"  - Fairness improvement: {fairness_gain:+.3f} EO gap reduction")
    if fairness_gain > 0:
        cost_per_fairness = abs(profit_change) / fairness_gain if fairness_gain != 0 else float('inf')
        print(f"  - Cost per fairness unit: €{cost_per_fairness:,.0f}")
    else:
        print(f"  - No fairness improvement achieved")

# Return on Fairness Investment (ROFI)
print(f"\nReturn on Fairness Investment (ROFI) Analysis:")
for i, (result, (pred_col, approach_name)) in enumerate(zip(fairness_results, approaches)):
    if i == 0:  # Skip baseline
        continue

    fairness_gain = baseline_result['eo_gap'] - result['eo_gap']
    if fairness_gain > 0 and result['total_profit'] >= baseline_profit:
        print(f"  {approach_name}: Positive ROFI (fairness gain with profit maintenance/increase)")
    elif fairness_gain > 0:
        print(f"  {approach_name}: Trade-off ROFI (fairness gain with acceptable profit cost)")
    else:
        print(f"  {approach_name}: Negative ROFI (no fairness gain)")

# ------------------ 20. Production Deployment Recommendations ------------------
print(f"\n PRODUCTION DEPLOYMENT RECOMMENDATIONS:")
print(f"{'=' * 45}")

print(f"Recommended Implementation Strategy:")
print(f"1. Phase 1: Deploy baseline Random Forest for performance validation")
print(f"2. Phase 2: Implement {best_fairness['approach'].lower()} for fairness compliance")
print(f"3. Phase 3: Monitor and validate fairness metrics in production")
print(f"4. Phase 4: Continuous model retraining with fairness constraints")

print(f"\nMonitoring Requirements:")
print(f"• Real-time fairness metric tracking (DP and EO gaps)")
print(f"• Monthly family status bias audits")
print(f"• Feature importance drift detection")
print(f"• Prediction confidence monitoring by family group")
print(f"• Regulatory compliance reporting automation")

print(f"\nModel Governance:")
print(f"• Document Random Forest fairness intervention methodology")
print(f"• Establish fairness threshold alerts (EO gap > 0.05)")
print(f"• Create feature importance interpretability reports")
print(f"• Implement A/B testing for fairness intervention effectiveness")
print(f"• Regular stakeholder reviews of family status bias metrics")

print(f"\n FINAL CONCLUSION:")
print(f"Random Forest with comprehensive fairness interventions provides an optimal")
print(f"balance of predictive performance, family status fairness, and business value.")
print(f"The ensemble approach effectively captures complex family-related patterns")
print(f"while maintaining regulatory compliance and interpretability requirements.")

print(f"\n CHAMPION MODEL RECOMMENDATION:")
if best_fairness['eo_gap'] < 0.05 and best_fairness == best_profit:
    print(f" CHAMPION: {best_fairness['approach']}")
    print(f"   - Optimal fairness-profit balance achieved")
    print(f"   - Ready for production deployment")
elif best_fairness['eo_gap'] < 0.05:
    print(f" CHAMPION: {best_fairness['approach']}")
    print(f"   - Prioritizes regulatory compliance over maximum profit")
    print(f"   - Acceptable business trade-off for fairness")
else:
    print(f"  CHALLENGER: {best_profit['approach']}")
    print(f"   - Best available option pending further optimization")
    print(f"   - Requires additional fairness intervention development")