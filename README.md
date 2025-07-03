# Akhil Metukuru's Personal Portfolio

Welcome to my personal portfolio! Here you'll find all my latest work, skills, and experiences. I'm excited to share my journey and achievements with you.

## About Me

Hello! I'm Akhil Metukuru, passionate about leveraging technology to solve real-world problems and create impactful solutions. When I'm not coding or working on new projects, you might find me exploring new places, reading, or spending time with family and friends.

## About This Website

This portfolio website is a showcase of my professional journey, designed to provide a comprehensive view of my skills, projects, and experiences. Built with modern web technologies, it features:

- **Responsive Design:** Ensures optimal viewing experience across all devices.
- **Interactive Elements:** Smooth transitions and animations for an engaging user experience.
- **Project Showcases:** Detailed descriptions of my work, including the technologies used and the challenges overcome.
- **Blog Section:** Insights and thoughts on technology, coding practices, and industry trends (coming soon!).

### Technologies Used

- **Frontend:** React, Next.js
- **Styling:** Tailwind CSS
- **Hosting:** Vercel

## Skills

- **Programming Languages:** TypeScript, JavaScript, CSS
- **Frameworks:** React, Next.js, Tailwind CSS
- **Tools:** Git, Vercel, Node.js

## Contact

Feel free to reach out to me through any of the platforms below. I'm always open to networking and discussing new opportunities!

- **Email:** [akhil.metukuru2016@gmail.com](mailto:akhil.metukuru2016@gmail.com)
- **LinkedIn:** [linkedin.com/in/akmet](https://linkedin.com/in/akmet)
- **GitHub:** [github.com/akhilmet](https://github.com/akhilmet)
- **Website:** [akhilmet.com](https://akhilmet.vercel.app/)

# Feature Selection and Random Sampling Analysis for Token Predictor
# Comprehensive analysis to determine which features to keep vs remove

```python
# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from scipy import stats
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from utils import get_token_length, is_natural_language_or_code, categorize_query, extract_all_features

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("🚀 Feature Selection Analysis for Token Predictor")
print("=" * 60)
```

```python
# Cell 2: Load and Process Data with Random Noise Features
print("🔄 Loading data and adding random noise features...")

# Load datasets
training_df = pd.read_csv('../testing/training_data.csv')
print(f"✅ Loaded training data: {training_df.shape}")

# Sample dataset for faster analysis (adjust sample_size as needed)
SAMPLE_SIZE = 5000  # Use subset for faster analysis
if len(training_df) > SAMPLE_SIZE:
    training_df = training_df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"📊 Using random sample of {SAMPLE_SIZE} records for analysis")

# Extract ALL features for each query
print("🔄 Extracting comprehensive features...")
all_features_list = []

for idx, row in training_df.iterrows():
    query = row['Query']
    response_length = row['response_length']
    
    # Get ALL features
    features = extract_all_features(query)
    features['response_length'] = response_length
    features['original_index'] = idx
    
    all_features_list.append(features)

# Convert to DataFrame
enhanced_df = pd.DataFrame(all_features_list)
print(f"✅ Feature extraction complete: {enhanced_df.shape}")
```

```python
# Cell 3: Add Random Noise Features as Baselines
print("\n🎲 Adding random noise features as baselines...")

# Add various types of random features to establish noise baseline
np.random.seed(42)
n_samples = len(enhanced_df)

# Random features with different distributions
enhanced_df['random_uniform'] = np.random.uniform(0, 1, n_samples)
enhanced_df['random_normal'] = np.random.normal(0, 1, n_samples)
enhanced_df['random_exponential'] = np.random.exponential(1, n_samples)
enhanced_df['random_integer'] = np.random.randint(0, 100, n_samples)
enhanced_df['random_binary'] = np.random.choice([0, 1], n_samples)

# Random features correlated with target (mild correlation)
target = enhanced_df['response_length']
enhanced_df['random_weak_corr'] = target * 0.1 + np.random.normal(0, target.std(), n_samples)
enhanced_df['random_moderate_corr'] = target * 0.3 + np.random.normal(0, target.std() * 0.5, n_samples)

# Random categorical feature
categories = ['cat_a', 'cat_b', 'cat_c', 'cat_d', 'cat_e']
enhanced_df['random_categorical'] = np.random.choice(categories, n_samples)

print(f"✅ Added 8 random noise features")
```

```python
# Cell 4: Feature Engineering and Encoding
print("\n🔧 Engineering features for ML models...")

# Encode categorical features (including random categorical)
label_encoders = {}
categorical_features = ['question_type', 'other_subcategory', 'query_context', 'random_categorical']

for feature in categorical_features:
    if feature in enhanced_df.columns:
        le = LabelEncoder()
        enhanced_df[f'{feature}_encoded'] = le.fit_transform(enhanced_df[feature].astype(str))
        label_encoders[feature] = le
        print(f"   Encoded {feature}: {len(le.classes_)} unique values")

# Define feature sets
BASIC_FEATURES = [
    'query_token_length', 'nlp_vs_code', 
    'category0', 'category1', 'category2', 'category3', 'category4', 'category5', 'category6'
]

COMPLEXITY_FEATURES = [
    'has_numbers', 'has_punctuation', 'has_special_chars', 'has_technical_terms',
    'sentence_count', 'avg_word_length', 'complexity_score', 'has_questions',
    'has_code_indicators', 'word_count', 'char_count', 'unique_word_count',
    'punctuation_density', 'caps_ratio'
]

CATEGORICAL_ENCODED = [
    'question_type_encoded', 'other_subcategory_encoded', 'query_context_encoded', 'random_categorical_encoded'
]

RANDOM_FEATURES = [
    'random_uniform', 'random_normal', 'random_exponential', 'random_integer', 
    'random_binary', 'random_weak_corr', 'random_moderate_corr'
]

# Combine all available features
ALL_FEATURES = []
for feature_set in [BASIC_FEATURES, COMPLEXITY_FEATURES, CATEGORICAL_ENCODED, RANDOM_FEATURES]:
    for feature in feature_set:
        if feature in enhanced_df.columns:
            ALL_FEATURES.append(feature)

print(f"✅ Total features available: {len(ALL_FEATURES)}")
print(f"   Real features: {len(ALL_FEATURES) - len(RANDOM_FEATURES)}")
print(f"   Random features: {len([f for f in RANDOM_FEATURES if f in enhanced_df.columns])}")
```

```python
# Cell 5: Data Preparation and Basic Statistics
# Prepare data for modeling
X = enhanced_df[ALL_FEATURES]
y = enhanced_df['response_length']

print(f"\n📊 Dataset Summary:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(ALL_FEATURES)}")
print(f"   Target range: {y.min():.0f} - {y.max():.0f} tokens")
print(f"   Target mean: {y.mean():.1f} ± {y.std():.1f} tokens")

# Handle missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")
```

```python
# Cell 6: Correlation Analysis
print("\n📈 CORRELATION ANALYSIS")
print("=" * 40)

# Calculate correlations with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)

print("Top 15 Features by Absolute Correlation with Target:")
for i, (feature, corr) in enumerate(correlations.head(15).items()):
    feature_type = "🎲 RANDOM" if feature in RANDOM_FEATURES else "✅ REAL"
    print(f"{i+1:2d}. {feature_type} {feature:<25} {corr:.4f}")

print(f"\nRandom Feature Correlations (Baseline):")
random_corrs = correlations[correlations.index.isin(RANDOM_FEATURES)]
for feature, corr in random_corrs.items():
    print(f"   🎲 {feature:<25} {corr:.4f}")
```

```python
# Cell 7: Statistical Feature Selection
print(f"\n🔬 STATISTICAL FEATURE SELECTION")
print("=" * 45)

# Method 1: F-regression (ANOVA F-test)
print("📊 F-regression Analysis (ANOVA F-test)...")
f_selector = SelectKBest(score_func=f_regression, k='all')
f_selector.fit(X_train, y_train)

f_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': f_selector.scores_,
    'p_value': f_selector.pvalues_
}).sort_values('f_score', ascending=False)

print("Top 15 Features by F-score:")
for i, (_, row) in enumerate(f_scores.head(15).iterrows()):
    feature_type = "🎲 RANDOM" if row['feature'] in RANDOM_FEATURES else "✅ REAL"
    print(f"{i+1:2d}. {feature_type} {row['feature']:<25} F={row['f_score']:.2f} p={row['p_value']:.4f}")

# Method 2: Mutual Information
print(f"\n📊 Mutual Information Analysis...")
mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
mi_selector.fit(X_train, y_train)

mi_scores = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_selector.scores_
}).sort_values('mi_score', ascending=False)

print("Top 15 Features by Mutual Information:")
for i, (_, row) in enumerate(mi_scores.head(15).iterrows()):
    feature_type = "🎲 RANDOM" if row['feature'] in RANDOM_FEATURES else "✅ REAL"
    print(f"{i+1:2d}. {feature_type} {row['feature']:<25} MI={row['mi_score']:.4f}")
```

```python
# Cell 8: Random Forest Feature Importance with Noise Baseline
print(f"\n🌲 RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 45)

# Train Random Forest
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    oob_score=True,
    max_depth=10,
    n_jobs=-1
)

rf_regressor.fit(X_train, y_train)

# Get feature importance
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_regressor.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Features by Random Forest Importance:")
for i, (_, row) in enumerate(rf_importance.head(15).iterrows()):
    feature_type = "🎲 RANDOM" if row['feature'] in RANDOM_FEATURES else "✅ REAL"
    print(f"{i+1:2d}. {feature_type} {row['feature']:<25} {row['importance']:.6f}")

# Calculate noise baseline
random_importances = rf_importance[rf_importance['feature'].isin(RANDOM_FEATURES)]
noise_threshold = random_importances['importance'].max()
noise_mean = random_importances['importance'].mean()
noise_std = random_importances['importance'].std()

print(f"\n🎲 Random Feature Statistics:")
print(f"   Max random importance: {noise_threshold:.6f}")
print(f"   Mean random importance: {noise_mean:.6f} ± {noise_std:.6f}")
print(f"   Suggested threshold: {noise_mean + 2*noise_std:.6f} (mean + 2σ)")
```

```python
# Cell 9: Recursive Feature Elimination
print(f"\n🔄 RECURSIVE FEATURE ELIMINATION (RFE)")
print("=" * 45)

# RFE with Random Forest
estimator = RandomForestRegressor(n_estimators=50, random_state=42)
rfe = RFECV(estimator, step=1, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)

print("🔄 Running RFE analysis (this may take a moment)...")
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X.columns[rfe.support_]
eliminated_features = X.columns[~rfe.support_]

print(f"✅ RFE Results:")
print(f"   Optimal number of features: {rfe.n_features_}")
print(f"   Selected features: {len(selected_features)}")
print(f"   Eliminated features: {len(eliminated_features)}")

print(f"\n🗑️ Eliminated Features:")
for feature in eliminated_features:
    feature_type = "🎲 RANDOM" if feature in RANDOM_FEATURES else "✅ REAL"
    print(f"   {feature_type} {feature}")
```

```python
# Cell 10: Feature Sampling Analysis
print(f"\n🎯 FEATURE SAMPLING ANALYSIS")
print("=" * 40)

# Test different random subsets of features
def evaluate_feature_subset(features, X_train, X_test, y_train, y_test):
    """Evaluate a subset of features"""
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train_subset, y_train)
    
    y_pred = rf.predict(X_test_subset)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mae, r2

# Get non-random features only
real_features = [f for f in ALL_FEATURES if f not in RANDOM_FEATURES]

# Test different feature subset sizes
subset_sizes = [5, 10, 15, 20, len(real_features)]
sampling_results = []

print("🔄 Testing different feature subset sizes...")
for size in subset_sizes:
    if size <= len(real_features):
        # Sample top features by correlation
        top_features = correlations[correlations.index.isin(real_features)].head(size).index.tolist()
        mae, r2 = evaluate_feature_subset(top_features, X_train, X_test, y_train, y_test)
        
        sampling_results.append({
            'size': size,
            'mae': mae,
            'r2': r2,
            'features': top_features
        })
        
        print(f"   {size:2d} features: MAE={mae:.2f}, R²={r2:.3f}")
```

```python
# Cell 11: Feature Stability Analysis
print(f"\n🔄 FEATURE STABILITY ANALYSIS")
print("=" * 40)

# Bootstrap feature importance to check stability
n_bootstrap = 10
feature_importances_bootstrap = []

print("🔄 Running bootstrap analysis...")
for i in range(n_bootstrap):
    # Bootstrap sample
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train.iloc[indices]
    y_boot = y_train.iloc[indices]
    
    # Train model
    rf_boot = RandomForestRegressor(n_estimators=50, random_state=i)
    rf_boot.fit(X_boot, y_boot)
    
    # Store importance
    feature_importances_bootstrap.append(rf_boot.feature_importances_)

# Calculate stability metrics
importance_df = pd.DataFrame(feature_importances_bootstrap, columns=X.columns)
stability_stats = pd.DataFrame({
    'feature': X.columns,
    'mean_importance': importance_df.mean(),
    'std_importance': importance_df.std(),
    'cv_importance': importance_df.std() / importance_df.mean()  # Coefficient of variation
}).sort_values('mean_importance', ascending=False)

print("Top 15 Most Stable Features (by coefficient of variation):")
stable_features = stability_stats.sort_values('cv_importance').head(15)
for i, (_, row) in enumerate(stable_features.iterrows()):
    feature_type = "🎲 RANDOM" if row['feature'] in RANDOM_FEATURES else "✅ REAL"
    print(f"{i+1:2d}. {feature_type} {row['feature']:<25} CV={row['cv_importance']:.3f}")
```

```python
# Cell 12: Visualization of Results
print(f"\n📊 GENERATING VISUALIZATIONS")
print("=" * 35)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# Plot 1: Feature Importance vs Random Baseline
ax1 = plt.subplot(2, 3, 1)
rf_importance_plot = rf_importance.head(20)
colors = ['red' if f in RANDOM_FEATURES else 'green' for f in rf_importance_plot['feature']]
bars = ax1.barh(range(len(rf_importance_plot)), rf_importance_plot['importance'], color=colors)
ax1.set_yticks(range(len(rf_importance_plot)))
ax1.set_yticklabels(rf_importance_plot['feature'], fontsize=8)
ax1.axvline(noise_threshold, color='red', linestyle='--', alpha=0.7, label=f'Max Random: {noise_threshold:.4f}')
ax1.axvline(noise_mean + 2*noise_std, color='orange', linestyle='--', alpha=0.7, label=f'Threshold: {noise_mean + 2*noise_std:.4f}')
ax1.set_xlabel('Importance Score')
ax1.set_title('Random Forest Feature Importance\n(Red = Random Features)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Correlation Analysis
ax2 = plt.subplot(2, 3, 2)
corr_plot = correlations.head(20)
colors = ['red' if f in RANDOM_FEATURES else 'blue' for f in corr_plot.index]
bars = ax2.barh(range(len(corr_plot)), corr_plot.values, color=colors)
ax2.set_yticks(range(len(corr_plot)))
ax2.set_yticklabels(corr_plot.index, fontsize=8)
ax2.set_xlabel('Absolute Correlation')
ax2.set_title('Correlation with Target\n(Red = Random Features)')
ax2.grid(True, alpha=0.3)

# Plot 3: F-Score Analysis
ax3 = plt.subplot(2, 3, 3)
f_plot = f_scores.head(20)
colors = ['red' if f in RANDOM_FEATURES else 'purple' for f in f_plot['feature']]
bars = ax3.barh(range(len(f_plot)), f_plot['f_score'], color=colors)
ax3.set_yticks(range(len(f_plot)))
ax3.set_yticklabels(f_plot['feature'], fontsize=8)
ax3.set_xlabel('F-Score')
ax3.set_title('F-Regression Scores\n(Red = Random Features)')
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Subset Performance
ax4 = plt.subplot(2, 3, 4)
subset_df = pd.DataFrame(sampling_results)
ax4.plot(subset_df['size'], subset_df['mae'], 'o-', color='green', linewidth=2, markersize=8)
ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Mean Absolute Error')
ax4.set_title('Performance vs Feature Count')
ax4.grid(True, alpha=0.3)
for i, row in subset_df.iterrows():
    ax4.annotate(f'{row["mae"]:.1f}', (row['size'], row['mae']), 
                textcoords="offset points", xytext=(0,10), ha='center')

# Plot 5: Feature Stability
ax5 = plt.subplot(2, 3, 5)
stability_plot = stability_stats.head(15)
colors = ['red' if f in RANDOM_FEATURES else 'orange' for f in stability_plot['feature']]
bars = ax5.barh(range(len(stability_plot)), stability_plot['cv_importance'], color=colors)
ax5.set_yticks(range(len(stability_plot)))
ax5.set_yticklabels(stability_plot['feature'], fontsize=8)
ax5.set_xlabel('Coefficient of Variation')
ax5.set_title('Feature Stability\n(Lower = More Stable)')
ax5.grid(True, alpha=0.3)

# Plot 6: Random vs Real Feature Distribution
ax6 = plt.subplot(2, 3, 6)
real_importances = rf_importance[~rf_importance['feature'].isin(RANDOM_FEATURES)]['importance']
random_importances_vals = rf_importance[rf_importance['feature'].isin(RANDOM_FEATURES)]['importance']

ax6.hist(real_importances, bins=20, alpha=0.7, color='green', label='Real Features', density=True)
ax6.hist(random_importances_vals, bins=10, alpha=0.7, color='red', label='Random Features', density=True)
ax6.axvline(noise_threshold, color='red', linestyle='--', alpha=0.7, label=f'Max Random: {noise_threshold:.4f}')
ax6.set_xlabel('Importance Score')
ax6.set_ylabel('Density')
ax6.set_title('Feature Importance Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

```python
# Cell 13: Feature Recommendations
print(f"\n🎯 FEATURE SELECTION RECOMMENDATIONS")
print("=" * 50)

# Define thresholds
corr_threshold = 0.1
importance_threshold = noise_mean + 2 * noise_std
f_score_threshold = 5.0
stability_threshold = 0.8  # CV < 0.8 is considered stable

# Get recommended features based on multiple criteria
recommended_features = set()

# Criteria 1: High correlation
high_corr_features = correlations[
    (correlations > corr_threshold) & 
    (~correlations.index.isin(RANDOM_FEATURES))
].index.tolist()
recommended_features.update(high_corr_features)

# Criteria 2: High importance and above noise threshold
high_importance_features = rf_importance[
    (rf_importance['importance'] > importance_threshold) &
    (~rf_importance['feature'].isin(RANDOM_FEATURES))
]['feature'].tolist()
recommended_features.update(high_importance_features)

# Criteria 3: Significant F-score
significant_f_features = f_scores[
    (f_scores['f_score'] > f_score_threshold) &
    (f_scores['p_value'] < 0.05) &
    (~f_scores['feature'].isin(RANDOM_FEATURES))
]['feature'].tolist()
recommended_features.update(significant_f_features)

# Criteria 4: Stable features
stable_feature_list = stability_stats[
    (stability_stats['cv_importance'] < stability_threshold) &
    (~stability_stats['feature'].isin(RANDOM_FEATURES))
]['feature'].tolist()
recommended_features.update(stable_feature_list)

# Convert to sorted list
recommended_features = sorted(list(recommended_features))

print(f"📋 RECOMMENDED FEATURES TO KEEP ({len(recommended_features)} features):")
print("-" * 50)
for i, feature in enumerate(recommended_features, 1):
    # Get metrics for this feature
    corr = correlations.get(feature, 0)
    importance = rf_importance[rf_importance['feature'] == feature]['importance'].iloc[0]
    f_score = f_scores[f_scores['feature'] == feature]['f_score'].iloc[0]
    cv = stability_stats[stability_stats['feature'] == feature]['cv_importance'].iloc[0]
    
    print(f"{i:2d}. {feature:<25} Corr={corr:.3f} Imp={importance:.4f} F={f_score:.1f} CV={cv:.3f}")

# Features to potentially remove
all_real_features = [f for f in ALL_FEATURES if f not in RANDOM_FEATURES]
features_to_remove = [f for f in all_real_features if f not in recommended_features]

print(f"\n🗑️ FEATURES TO CONSIDER REMOVING ({len(features_to_remove)} features):")
print("-" * 50)
for i, feature in enumerate(features_to_remove, 1):
    corr = correlations.get(feature, 0)
    importance = rf_importance[rf_importance['feature'] == feature]['importance'].iloc[0]
    f_score = f_scores[f_scores['feature'] == feature]['f_score'].iloc[0]
    cv = stability_stats[stability_stats['feature'] == feature]['cv_importance'].iloc[0]
    
    print(f"{i:2d}. {feature:<25} Corr={corr:.3f} Imp={importance:.4f} F={f_score:.1f} CV={cv:.3f}")
```

```python
# Cell 14: Final Model Comparison
print(f"\n🏁 FINAL MODEL COMPARISON")
print("=" * 35)

# Test performance with different feature sets
feature_sets = {
    'All Features': ALL_FEATURES[:-len(RANDOM_FEATURES)],  # Exclude random features
    'Top 10 by Correlation': correlations[~correlations.index.isin(RANDOM_FEATURES)].head(10).index.tolist(),
    'Top 15 by Importance': rf_importance[~rf_importance['feature'].isin(RANDOM_FEATURES)].head(15)['feature'].tolist(),
    'Recommended Features': recommended_features,
    'RFE Selected': selected_features.tolist()
}

comparison_results = []

for name, features in feature_sets.items():
    # Filter features that actually exist in the dataset
    available_features = [f for f in features if f in X.columns]
    
    if len(available_features) > 0:
        mae, r2 = evaluate_feature_subset(available_features, X_train, X_test, y_train, y_test)
        comparison_results.append({
            'Feature Set': name,
            'Features': len(available_features),
            'MAE': mae,
            'R²': r2
        })

comparison_df = pd.DataFrame(comparison_results).sort_values('MAE')

print("Performance Comparison:")
print(comparison_df.round(3))

best_set = comparison_df.iloc[0]
print(f"\n🏆 Best Feature Set: {best_set['Feature Set']}")
print(f"   Features: {best_set['Features']}")
print(f"   MAE: {best_set['MAE']:.2f}")
print(f"   R²: {best_set['R²']:.3f}")

print(f"\n💾 Recommended feature list for production:")
print(f"SELECTED_FEATURES = {recommended_features}")

print(f"\n✅ Analysis Complete!")
print(f"💡 Use the recommended features for optimal performance with minimal complexity.")
```

### **GitHub README Ready:**
This format is optimized for copying into GitHub README.md and then easily transferring to any Jupyter environment. Each cell is clearly separated and ready for enterprise use.

## License

This project is licensed under the MIT License.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
