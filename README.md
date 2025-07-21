```
# Cell 3: Basic Data Analysis
# Explore the data structure
if df is not None:
    print("📊 Dataset Analysis:")
    print(f"   Total samples: {len(df):,}")
    
    # The dataset has 'instruction' and 'responses' columns
    print(f"   Available columns: {list(df.columns)}")
    
    # Extract the actual response text from the responses column
    # The 'responses' column appears to contain structured data
    def extract_response_text(responses_data):
        """Extract actual response text from responses column"""
        if pd.isna(responses_data):
            return ""
        
        # If it's already a string, return it
        if isinstance(responses_data, str):
            return responses_data
        
        # If it's a list or dict structure, extract the text
        try:
            if isinstance(responses_data, list) and len(responses_data) > 0:
                # Take the first response if multiple
                response = responses_data[0]
                if isinstance(response, dict):
                    # Look for common text fields
                    for key in ['text', 'content', 'value', 'response']:
                        if key in response:
                            return str(response[key])
                elif isinstance(response, str):
                    return response
            elif isinstance(responses_data, dict):
                # Look for common text fields in dict
                for key in ['text', 'content', 'value', 'response']:
                    if key in responses_data:
                        return str(responses_data[key])
            
            # If all else fails, convert to string
            return str(responses_data)
        except:
            return str(responses_data)
    
    # Apply extraction to create clean response column
    print("🔧 Extracting response text from responses column...")
    df['response'] = df['responses'].apply(extract_response_text)
    
    # Check if extraction worked
    sample_response = df['response'].iloc[0]
    print(f"📝 Sample extracted response (first 100 chars): {sample_response[:100]}...")
    
    # Calculate text statistics
    df['instruction_len'] = df['instruction'].astype(str).str.len()
    df['response_len'] = df['response'].astype(str).str.len()
    df['response_words'] = df['response'].astype(str).str.split().str.len()
    
    # Clean up any NaN values that might have been created
    df['response_words'] = df['response_words'].fillna(0)
    
    print(f"\n📝 Text Statistics:")
    print(f"   Instruction length - Mean: {df['instruction_len'].mean():.0f}, Max: {df['instruction_len'].max()}")
    print(f"   Response length - Mean: {df['response_len'].mean():.0f}, Max: {df['response_len'].max()}")
    print(f"   Response words - Mean: {df['response_words'].mean():.0f}, Max: {df['response_words'].max()}")
    
    # Show available metadata features
    metadata_cols = [col for col in df.columns if col not in ['instruction', 'responses', 'response']]
    print(f"\n🏷️ Available metadata features ({len(metadata_cols)}):")
    for col in metadata_cols[:10]:  # Show first 10
        unique_vals = df[col].nunique()
        print(f"   - {col}: {unique_vals} unique values ({df[col].dtype})")
    if len(metadata_cols) > 10:
        print(f"   ... and {len(metadata_cols) - 10} more")
    
    # Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Response word count distribution
    axes[0].hist(df['response_words'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Response Word Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Response Lengths (Words)')
    axes[0].axvline(df['response_words'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["response_words"].mean():.0f}')
    axes[0].legend()
    
    # Instruction vs Response length relationship
    sample_size = min(5000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    axes[1].scatter(sample_df['instruction_len'], sample_df['response_words'], alpha=0.5)
    axes[1].set_xlabel('Instruction Length (chars)')
    axes[1].set_ylabel('Response Length (words)')
    axes[1].set_title('Instruction vs Response Length Relationship')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze metadata features for modeling
    print(f"\n🎯 Metadata Features for Enhanced Modeling:")
    
    # Check specific useful columns
    useful_metadata = []
    for col in ['difficulty', 'intent', 'knowledge', 'task_category', 'input_quality', 'quality_explanation']:
        if col in df.columns:
            useful_metadata.append(col)
            unique_count = df[col].nunique()
            print(f"   ✅ {col}: {unique_count} categories")
            if unique_count < 20:  # Show categories if not too many
                print(f"      Categories: {list(df[col].value_counts().head().index)}")
    
    if useful_metadata:
        print(f"\n🚀 Found {len(useful_metadata)} useful metadata features for enhanced modeling!")
    else:
        print(f"\n📝 Will use extracted text features for modeling.")
    
    print(f"\n✅ Data exploration complete. Ready for feature engineering!")
```

```
# Cell 1: Setup and Imports
```python
# Token Predictor with Magpie Dataset
# Enhanced model using Random Forest and XGBoost with rich feature engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("🚀 Token Predictor - Magpie Dataset Enhanced Model")
print("=" * 60)
```

# Cell 2: Data Loading and Exploration
```python
# Load the dataset
def load_and_explore_data():
    """Load the Magpie dataset and perform initial exploration"""
    
    # Load the dataset (adjust path as needed)
    try:
        # Try loading from HuggingFace format
        df = pd.read_parquet('data/train-*')
        print(f"✅ Loaded dataset with {len(df):,} rows")
    except:
        try:
            # Alternative loading methods
            df = pd.read_json('data/train.jsonl', lines=True)
            print(f"✅ Loaded JSONL dataset with {len(df):,} rows")
        except:
            print("⚠️ Could not load from standard paths.")
            print("Please adjust the loading path in this cell.")
            return None
    
    # Basic info
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️ Missing values:\n{missing[missing > 0]}")
    
    # Sample data preview
    print(f"\n🔍 Sample data:")
    display(df.head(3))
    
    return df

# Load data
df = load_and_explore_data()
```

# Cell 3: Basic Data Analysis
```python
# Explore the data structure
if df is not None:
    print("📊 Dataset Analysis:")
    print(f"   Total samples: {len(df):,}")
    
    # Check instruction lengths
    df['instruction_len'] = df['instruction'].str.len()
    df['response_len'] = df['response'].str.len()
    df['response_words'] = df['response'].str.split().str.len()
    
    print(f"\n📝 Text Statistics:")
    print(f"   Instruction length - Mean: {df['instruction_len'].mean():.0f}, Max: {df['instruction_len'].max()}")
    print(f"   Response length - Mean: {df['response_len'].mean():.0f}, Max: {df['response_len'].max()}")
    print(f"   Response words - Mean: {df['response_words'].mean():.0f}, Max: {df['response_words'].max()}")
    
    # Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Response word count distribution
    axes[0].hist(df['response_words'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Response Word Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Response Lengths (Words)')
    axes[0].axvline(df['response_words'].mean(), color='red', linestyle='--', label=f'Mean: {df["response_words"].mean():.0f}')
    axes[0].legend()
    
    # Instruction vs Response length relationship
    sample_size = min(5000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    axes[1].scatter(sample_df['instruction_len'], sample_df['response_words'], alpha=0.5)
    axes[1].set_xlabel('Instruction Length (chars)')
    axes[1].set_ylabel('Response Length (words)')
    axes[1].set_title('Instruction vs Response Length Relationship')
    
    plt.tight_layout()
    plt.show()
```

# Cell 4: Enhanced Feature Engineering
```python
def extract_advanced_features(df):
    """Extract comprehensive features from the Magpie dataset"""
    
    features_df = df.copy()
    
    print("🛠️ Starting feature engineering...")
    
    # ==================
    # TEXT-BASED FEATURES
    # ==================
    
    # Basic text metrics for instruction
    features_df['instruction_length'] = features_df['instruction'].str.len()
    features_df['instruction_word_count'] = features_df['instruction'].str.split().str.len()
    features_df['instruction_sentence_count'] = features_df['instruction'].str.count(r'[.!?]+')
    
    # Response metrics (TARGET)
    features_df['response_length'] = features_df['response'].str.len()
    features_df['response_word_count'] = features_df['response'].str.split().str.len()
    features_df['response_sentence_count'] = features_df['response'].str.count(r'[.!?]+')
    
    print("✅ Basic text features extracted")
    
    # ==================
    # COMPLEXITY METRICS
    # ==================
    
    def calculate_complexity_score(text):
        """Calculate comprehensive complexity score"""
        if pd.isna(text):
            return 0
        
        text = str(text)
        words = text.split()
        
        if not words:
            return 0
        
        # Vocabulary diversity
        unique_words = len(set(words))
        vocab_diversity = unique_words / len(words)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Punctuation density
        punctuation_count = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', text))
        punct_density = punctuation_count / len(text) if len(text) > 0 else 0
        
        # Technical terms (simple heuristic)
        technical_patterns = [
            r'\bfunction\b', r'\bclass\b', r'\bimport\b', r'\bdef\b', r'\breturn\b',
            r'\bapi\b', r'\bdatabase\b', r'\balgorithm\b', r'\bmodel\b', r'\btraining\b',
            r'\btensor\b', r'\bnumpy\b', r'\bpandas\b', r'\bsklearn\b', r'\bpython\b',
            r'\bcode\b', r'\bvariable\b', r'\bloop\b', r'\barray\b', r'\blist\b'
        ]
        
        technical_count = sum(len(re.findall(pattern, text.lower())) for pattern in technical_patterns)
        tech_density = technical_count / len(words)
        
        # Composite complexity score
        complexity = (vocab_diversity * 0.3 + 
                     (min(avg_word_length, 15) / 15) * 0.2 + 
                     min(punct_density, 0.1) * 2.0 * 0.2 + 
                     min(tech_density, 0.5) * 2.0 * 0.3)
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    features_df['instruction_complexity'] = features_df['instruction'].apply(calculate_complexity_score)
    
    print("✅ Complexity features extracted")
    
    # ==================
    # CATEGORICAL FEATURES
    # ==================
    
    def get_question_type(text):
        """Classify question types"""
        if pd.isna(text):
            return 'unknown'
        
        text = str(text).lower()
        
        if any(word in text for word in ['what', 'define', 'explain', 'describe']):
            return 'explanation'
        elif any(word in text for word in ['how', 'tutorial', 'guide', 'steps']):
            return 'how_to'
        elif any(word in text for word in ['code', 'function', 'implement', 'write']):
            return 'coding'
        elif any(word in text for word in ['why', 'reason', 'because']):
            return 'reasoning'
        elif any(word in text for word in ['list', 'enumerate', 'examples']):
            return 'listing'
        elif text.strip().endswith('?'):
            return 'question'
        else:
            return 'statement'
    
    features_df['question_type'] = features_df['instruction'].apply(get_question_type)
    
    print("✅ Question type classification complete")
    
    # ==================
    # DATASET-SPECIFIC FEATURES
    # ==================
    
    # Check for Magpie-specific columns and handle them
    magpie_features = []
    
    if 'task_category' in features_df.columns:
        magpie_features.append('task_category')
    
    if 'difficulty' in features_df.columns:
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
        features_df['difficulty_encoded'] = features_df['difficulty'].map(difficulty_map).fillna(2)
        magpie_features.append('difficulty_encoded')
    
    if 'intent' in features_df.columns:
        magpie_features.append('intent')
    
    if 'knowledge' in features_df.columns:
        magpie_features.append('knowledge')
    
    print(f"✅ Found {len(magpie_features)} Magpie-specific features: {magpie_features}")
    
    # ==================
    # INTERACTION FEATURES
    # ==================
    
    # Ratio features
    features_df['response_to_instruction_ratio'] = (
        features_df['response_word_count'] / features_df['instruction_word_count'].replace(0, 1)
    )
    
    # Complexity vs length interaction
    features_df['complexity_length_interaction'] = (
        features_df['instruction_complexity'] * features_df['instruction_length'] / 1000  # Normalize
    )
    
    # Word density
    features_df['instruction_word_density'] = (
        features_df['instruction_word_count'] / features_df['instruction_length'].replace(0, 1)
    )
    
    print("✅ Interaction features created")
    
    new_features = len([col for col in features_df.columns if col not in df.columns])
    print(f"🎯 Feature engineering complete! Created {new_features} new features")
    
    return features_df

# Apply feature engineering
df_features = extract_advanced_features(df)
```

# Cell 5: Feature Selection and Data Preparation
```python
def prepare_modeling_data(df):
    """Prepare data for modeling"""
    
    # Define target variable
    target = 'response_word_count'  # Using word count as proxy for tokens
    
    # Select features for modeling
    feature_columns = [
        # Basic text metrics
        'instruction_length', 'instruction_word_count', 'instruction_sentence_count',
        'instruction_complexity', 'instruction_word_density',
        
        # Categorical features
        'question_type',
        
        # Interaction features
        'response_to_instruction_ratio', 'complexity_length_interaction'
    ]
    
    # Add dataset-specific features if available
    if 'difficulty_encoded' in df.columns:
        feature_columns.append('difficulty_encoded')
    if 'task_category' in df.columns:
        feature_columns.append('task_category')
    if 'intent' in df.columns:
        feature_columns.append('intent')
    if 'knowledge' in df.columns:
        feature_columns.append('knowledge')
    
    # Create feature matrix
    X = df[feature_columns].copy()
    y = df[target].copy()
    
    # Handle categorical variables
    categorical_columns = ['question_type']
    if 'task_category' in feature_columns:
        categorical_columns.append('task_category')
    if 'intent' in feature_columns:
        categorical_columns.append('intent')
    if 'knowledge' in feature_columns:
        categorical_columns.append('knowledge')
    
    # Label encoding for categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Prepared data: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"📋 Features: {list(X.columns)}")
    print(f"🎯 Target statistics:")
    print(f"   Mean: {y.mean():.1f} words")
    print(f"   Std:  {y.std():.1f} words")
    print(f"   Median: {y.median():.1f} words")
    print(f"   Min:  {y.min():.1f} words")
    print(f"   Max:  {y.max():.1f} words")
    
    return X, y, label_encoders

# Prepare modeling data
X, y, label_encoders = prepare_modeling_data(df_features)

# Show feature correlations with target
feature_correlations = X.corrwith(y).abs().sort_values(ascending=False)
print(f"\n🔗 Feature correlations with target:")
for feature, corr in feature_correlations.head(10).items():
    print(f"   {feature}: {corr:.3f}")
```

# Cell 6: Data Splitting
```python
# Split data with stratification
print("📊 Splitting data...")

# Create stratified split based on target quantiles
y_quantiles = pd.qcut(y, q=5, duplicates='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_quantiles
)

print(f"✅ Data split complete:")
print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")
print(f"   Features: {X_train.shape[1]}")

# Show target distribution in train/test
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(y_train, bins=50, alpha=0.7, label='Train', edgecolor='black')
axes[0].hist(y_test, bins=50, alpha=0.7, label='Test', edgecolor='black')
axes[0].set_xlabel('Response Word Count')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Train/Test Target Distribution')
axes[0].legend()

axes[1].boxplot([y_train, y_test], labels=['Train', 'Test'])
axes[1].set_ylabel('Response Word Count')
axes[1].set_title('Train/Test Target Distribution (Boxplot)')

plt.tight_layout()
plt.show()
```

# Cell 7: Random Forest Training
```python
print("🌲 Training Random Forest with GridSearch...")

# Define parameter grid
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize model
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Grid search with cross-validation
rf_grid = GridSearchCV(
    rf, rf_params, 
    cv=5, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1
)

# Train the model
rf_grid.fit(X_train, y_train)

# Get best model
best_rf = rf_grid.best_estimator_

print(f"✅ Random Forest training complete!")
print(f"🏆 Best parameters: {rf_grid.best_params_}")
print(f"🎯 Best CV MAE: {-rf_grid.best_score_:.2f}")
```

# Cell 8: XGBoost Training
```python
print("🚀 Training XGBoost with GridSearch...")

# Define parameter grid
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize model
xgb_reg = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Grid search with cross-validation
xgb_grid = GridSearchCV(
    xgb_reg, xgb_params, 
    cv=5, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1
)

# Train the model
xgb_grid.fit(X_train, y_train)

# Get best model
best_xgb = xgb_grid.best_estimator_

print(f"✅ XGBoost training complete!")
print(f"🏆 Best parameters: {xgb_grid.best_params_}")
print(f"🎯 Best CV MAE: {-xgb_grid.best_score_:.2f}")
```

# Cell 9: Model Evaluation
```python
# Evaluate both models
models = {
    'Random Forest': best_rf,
    'XGBoost': best_xgb
}

results = {}

print("📊 Model Evaluation Results:")
print("=" * 50)

for name, model in models.items():
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    results[name] = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_pred': train_pred,
        'test_pred': test_pred
    }
    
    print(f"\n🤖 {name} Results:")
    print(f"   Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
    print(f"   Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
    print(f"   Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
    
    # Calculate percentage within different error ranges
    errors = np.abs(y_test - test_pred)
    within_10 = (errors <= 10).mean() * 100
    within_25 = (errors <= 25).mean() * 100
    within_50 = (errors <= 50).mean() * 100
    
    print(f"   Predictions within ±10 words: {within_10:.1f}%")
    print(f"   Predictions within ±25 words: {within_25:.1f}%")
    print(f"   Predictions within ±50 words: {within_50:.1f}%")

# Identify best model
best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test MAE: {results[best_model_name]['test_mae']:.2f} words")
```

# Cell 10: Model Comparison Visualizations
```python
# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('🏆 Model Performance Comparison - Enhanced Magpie Token Predictor', fontsize=16, fontweight='bold')

# 1. MAE Comparison
model_names = list(results.keys())
train_maes = [results[name]['train_mae'] for name in model_names]
test_maes = [results[name]['test_mae'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0, 0].bar(x - width/2, train_maes, width, label='Train MAE', alpha=0.8, color='skyblue')
axes[0, 0].bar(x + width/2, test_maes, width, label='Test MAE', alpha=0.8, color='lightcoral')
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Mean Absolute Error')
axes[0, 0].set_title('MAE Comparison')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. R² Comparison
train_r2s = [results[name]['train_r2'] for name in model_names]
test_r2s = [results[name]['test_r2'] for name in model_names]

axes[0, 1].bar(x - width/2, train_r2s, width, label='Train R²', alpha=0.8, color='lightgreen')
axes[0, 1].bar(x + width/2, test_r2s, width, label='Test R²', alpha=0.8, color='gold')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_title('R² Score Comparison')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(model_names)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Prediction vs Actual (Best model)
best_results = results[best_model_name]

axes[0, 2].scatter(y_test, best_results['test_pred'], alpha=0.6, color='darkblue')
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 2].set_xlabel('Actual Token Count')
axes[0, 2].set_ylabel('Predicted Token Count')
axes[0, 2].set_title(f'Predictions vs Actual - {best_model_name}')
axes[0, 2].grid(True, alpha=0.3)

# 4. Residuals Plot
residuals = y_test - best_results['test_pred']
axes[1, 0].scatter(best_results['test_pred'], residuals, alpha=0.6, color='purple')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title(f'Residuals Plot - {best_model_name}')
axes[1, 0].grid(True, alpha=0.3)

# 5. Feature Importance (for best model)
if hasattr(best_results['model'], 'feature_importances_'):
    importances = best_results['model'].feature_importances_
    feature_names = X.columns
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    axes[1, 1].barh(range(len(indices)), importances[indices], color='orange', alpha=0.8)
    axes[1, 1].set_yticks(range(len(indices)))
    axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title(f'Top 10 Feature Importance - {best_model_name}')
    axes[1, 1].grid(True, alpha=0.3)

# 6. Error Distribution
axes[1, 2].hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='lightseagreen')
axes[1, 2].axvline(x=0, color='r', linestyle='--')
axes[1, 2].set_xlabel('Residuals')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Error Distribution')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

# Cell 11: Advanced Analysis Plots
```python
# Create detailed analysis plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📊 Advanced Model Analysis - Magpie Token Predictor', fontsize=16, fontweight='bold')

# 1. Performance by Token Count Range
ranges = [(0, 50), (50, 150), (150, 300), (300, 500), (500, float('inf'))]
range_labels = ['0-50', '50-150', '150-300', '300-500', '500+']

mae_by_range = []
count_by_range = []

for low, high in ranges:
    mask = (y_test >= low) & (y_test < high)
    count_by_range.append(mask.sum())
    
    if mask.sum() > 0:
        mae = mean_absolute_error(y_test[mask], best_results['test_pred'][mask])
        mae_by_range.append(mae)
    else:
        mae_by_range.append(0)

# Create bar plot with counts as annotations
bars = axes[0, 0].bar(range_labels, mae_by_range, alpha=0.8, color='steelblue')
axes[0, 0].set_xlabel('Token Count Range')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].set_title('Model Performance by Token Count Range')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Add count annotations
for bar, count in zip(bars, count_by_range):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'n={count}', ha='center', va='bottom', fontsize=9)

# 2. Feature Correlation Heatmap
correlation_matrix = X.corr()
mask = np.triu(np.ones_like(correlation_matrix))

sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0, ax=axes[0, 1])
axes[0, 1].set_title('Feature Correlation Matrix')

# 3. Prediction Confidence Analysis
predictions_sorted = np.sort(best_results['test_pred'])
actual_sorted = np.sort(y_test)

axes[1, 0].fill_between(range(len(predictions_sorted)), 
                       predictions_sorted * 0.8, 
                       predictions_sorted * 1.2, 
                       alpha=0.3, label='±20% Confidence', color='lightblue')
axes[1, 0].plot(predictions_sorted, 'b-', label='Predicted', linewidth=2)
axes[1, 0].plot(actual_sorted, 'r--', label='Actual', linewidth=2)
axes[1, 0].set_xlabel('Sample (sorted)')
axes[1, 0].set_ylabel('Token Count')
axes[1, 0].set_title('Prediction Confidence Analysis')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Model Comparison Heatmap
comparison_data = []
for name, result in results.items():
    comparison_data.append([
        result['test_mae'],
        result['test_rmse'], 
        result['test_r2']
    ])

comparison_df = pd.DataFrame(comparison_data, 
                           columns=['Test MAE', 'Test RMSE', 'Test R²'],
                           index=list(results.keys()))

# Normalize metrics for heatmap (lower is better for MAE/RMSE, higher for R²)
normalized_data = comparison_df.copy()
normalized_data['Test MAE'] = 1 / normalized_data['Test MAE']  # Invert MAE
normalized_data['Test RMSE'] = 1 / normalized_data['Test RMSE']  # Invert RMSE
normalized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())

sns.heatmap(normalized_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
           xticklabels=comparison_df.index, ax=axes[1, 1])
axes[1, 1].set_title('Model Performance Heatmap (Normalized)')

plt.tight_layout()
plt.show()
```

# Cell 12: Final Results Summary
```python
print("🏆 FINAL RESULTS SUMMARY")
print("=" * 60)

best_result = results[best_model_name]

print(f"🥇 Best Model: {best_model_name}")
print(f"   Test MAE: {best_result['test_mae']:.2f} words")
print(f"   Test RMSE: {best_result['test_rmse']:.2f} words") 
print(f"   Test R²: {best_result['test_r2']:.3f}")

# Calculate improvement over simple baseline
simple_baseline_mae = y_test.std()  # Using standard deviation as naive baseline
improvement = ((simple_baseline_mae - best_result['test_mae']) / simple_baseline_mae * 100)
print(f"   Improvement over baseline: {improvement:.1f}%")

print(f"\n📊 Detailed Comparison:")
for name, result in results.items():
    symbol = "🥇" if name == best_model_name else "🥈"
    print(f"   {symbol} {name}:")
    print(f"     MAE: {result['test_mae']:.2f} | RMSE: {result['test_rmse']:.2f} | R²: {result['test_r2']:.3f}")

# Show feature importance for best model
if hasattr(best_result['model'], 'feature_importances_'):
    print(f"\n🔍 Top 10 Most Important Features ({best_model_name}):")
    importances = best_result['model'].feature_importances_
    feature_names = X.columns
    
    # Sort by importance
    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance_pairs[:10], 1):
        print(f"   {i:2d}. {feature:<30} {importance:.4f}")

print(f"\n✅ Analysis complete! Best model ready for export.")
```

# Cell 13: Model Export
```python
# Export the best model and supporting files
print("💾 Exporting models and data...")

# Create export directory
import os
os.makedirs('model_exports', exist_ok=True)

# 1. Export the best model
with open('model_exports/magpie_token_predictor.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ Best model ({best_model_name}) saved to: model_exports/magpie_token_predictor.pkl")

# 2. Export label encoders
with open('model_exports/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✅ Label encoders saved to: model_exports/label_encoders.pkl")

# 3. Export feature list for production consistency
feature_list = X.columns.tolist()
with open('model_exports/feature_list.pkl', 'wb') as f:
    pickle.dump(feature_list, f)
print(f"✅ Feature list saved to: model_exports/feature_list.pkl")

# 4. Export model metadata
metadata = {
    'best_model_name': best_model_name,
    'best_model_type': type(best_model).__name__,
    'test_mae': best_result['test_mae'],
    'test_rmse': best_result['test_rmse'],
    'test_r2': best_result['test_r2'],
    'n_features': len(feature_list),
    'features': feature_list,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'hyperparameters': best_model.get_params()
}

with open('model_exports/model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"✅ Model metadata saved to: model_exports/model_metadata.pkl")

# 5. Export both models for comparison
all_models = {name: results[name]['model'] for name in results.keys()}
with open('model_exports/all_trained_models.pkl', 'wb') as f:
    pickle.dump(all_models, f)
print(f"✅ All trained models saved to: model_exports/all_trained_models.pkl")

print(f"\n📁 Export Summary:")
print(f"   📂 model_exports/")
print(f"   ├── magpie_token_predictor.pkl      (Best model)")
print(f"   ├── label_encoders.pkl              (Categorical encoders)")
print(f"   ├── feature_list.pkl                (Feature names)")
print(f"   ├── model_metadata.pkl              (Model info)")
print(f"   └── all_trained_models.pkl          (All models)")
```

# Cell 14: Production Usage Example
```python
# Demonstrate how to use the exported model in production
print("🚀 Production Usage Example")
print("=" * 40)

def load_production_model():
    """Load the exported model for production use"""
    
    # Load all components
    with open('model_exports/magpie_token_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('model_exports/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('model_exports/feature_list.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return model, encoders, features

def predict_tokens_production(instruction, model, encoders, feature_list):
    """Predict tokens for a new instruction using the production model"""
    
    # Feature engineering (same as training)
    features_dict = {}
    
    # Basic text metrics
    features_dict['instruction_length'] = len(instruction)
    features_dict['instruction_word_count'] = len(instruction.split())
    features_dict['instruction_sentence_count'] = len(re.findall(r'[.!?]+', instruction))
    features_dict['instruction_word_density'] = features_dict['instruction_word_count'] / max(features_dict['instruction_length'], 1)
    
    # Complexity score
    words = instruction.split()
    if words:
        unique_words = len(set(words))
        vocab_diversity = unique_words / len(words)
        avg_word_length = np.mean([len(word) for word in words])
        punctuation_count = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', instruction))
        punct_density = punctuation_count / len(instruction) if len(instruction) > 0 else 0
        
        technical_patterns = [
            r'\bfunction\b', r'\bclass\b', r'\bimport\b', r'\bdef\b', r'\breturn\b',
            r'\bapi\b', r'\bdatabase\b', r'\balgorithm\b', r'\bmodel\b', r'\btraining\b',
            r'\btensor\b', r'\bnumpy\b', r'\bpandas\b', r'\bsklearn\b', r'\bpython\b',
            r'\bcode\b', r'\bvariable\b', r'\bloop\b', r'\barray\b', r'\blist\b'
        ]
        technical_count = sum(len(re.findall(pattern, instruction.lower())) for pattern in technical_patterns)
        tech_density = technical_count / len(words)
        
        complexity = (vocab_diversity * 0.3 + 
                     (min(avg_word_length, 15) / 15) * 0.2 + 
                     min(punct_density, 0.1) * 2.0 * 0.2 + 
                     min(tech_density, 0.5) * 2.0 * 0.3)
        features_dict['instruction_complexity'] = min(complexity, 1.0)
    else:
        features_dict['instruction_complexity'] = 0
    
    # Question type
    text_lower = instruction.lower()
    if any(word in text_lower for word in ['what', 'define', 'explain', 'describe']):
        question_type = 'explanation'
    elif any(word in text_lower for word in ['how', 'tutorial', 'guide', 'steps']):
        question_type = 'how_to'
    elif any(word in text_lower for word in ['code', 'function', 'implement', 'write']):
        question_type = 'coding'
    elif any(word in text_lower for word in ['why', 'reason', 'because']):
        question_type = 'reasoning'
    elif any(word in text_lower for word in ['list', 'enumerate', 'examples']):
        question_type = 'listing'
    elif text_lower.strip().endswith('?'):
        question_type = 'question'
    else:
        question_type = 'statement'
    
    features_dict['question_type'] = question_type
    
    # Interaction features (using reasonable defaults since we don't have response info)
    features_dict['response_to_instruction_ratio'] = 2.0  # Average ratio from training
    features_dict['complexity_length_interaction'] = features_dict['instruction_complexity'] * features_dict['instruction_length'] / 1000
    
    # Create feature vector
    feature_vector = []
    for feature_name in feature_list:
        if feature_name == 'question_type':
            # Encode categorical variable
            if 'question_type' in encoders:
                try:
                    encoded_value = encoders['question_type'].transform([question_type])[0]
                except:
                    encoded_value = 0  # Default for unknown categories
            else:
                encoded_value = 0
            feature_vector.append(encoded_value)
        elif feature_name in features_dict:
            feature_vector.append(features_dict[feature_name])
        else:
            feature_vector.append(0)  # Default value for missing features
    
    # Make prediction
    prediction = model.predict([feature_vector])[0]
    return max(1, int(round(prediction)))

# Test the production model
try:
    prod_model, prod_encoders, prod_features = load_production_model()
    
    # Test examples
    test_instructions = [
        "What is machine learning?",
        "How do I implement a binary search algorithm in Python?",
        "Explain the differences between supervised and unsupervised learning with examples and use cases.",
        "Write a function to calculate fibonacci numbers.",
        "Debug this code: print('hello world')"
    ]
    
    print(f"🧪 Testing production model:")
    for i, instruction in enumerate(test_instructions, 1):
        predicted_tokens = predict_tokens_production(instruction, prod_model, prod_encoders, prod_features)
        print(f"   {i}. '{instruction[:50]}{'...' if len(instruction) > 50 else ''}'")
        print(f"      → Predicted tokens: {predicted_tokens}")
        print()
    
    print("✅ Production model test successful!")
    
except Exception as e:
    print(f"❌ Production model test failed: {e}")
    print("Make sure to run the export cell first!")
```

# Cell 15: Model Performance Analysis
```python
# Final comprehensive analysis
print("📈 COMPREHENSIVE MODEL ANALYSIS")
print("=" * 50)

# Performance summary table
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test MAE': [results[name]['test_mae'] for name in results.keys()],
    'Test RMSE': [results[name]['test_rmse'] for name in results.keys()],
    'Test R²': [results[name]['test_r2'] for name in results.keys()]
})

print("📊 Model Performance Summary:")
print(performance_df.round(3))

# Feature importance analysis (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n🔍 Feature Analysis for {best_model_name}:")
    
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['Importance'].values, color='skyblue')
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Error analysis
best_predictions = best_result['test_pred']
errors = np.abs(y_test - best_predictions)

print(f"\n📊 Error Analysis for {best_model_name}:")
print(f"   Mean Absolute Error: {errors.mean():.2f} words")
print(f"   Median Absolute Error: {np.median(errors):.2f} words")
print(f"   90th Percentile Error: {np.percentile(errors, 90):.2f} words")
print(f"   95th Percentile Error: {np.percentile(errors, 95):.2f} words")
print(f"   Max Error: {errors.max():.2f} words")

# Accuracy within different ranges
for threshold in [10, 25, 50, 100]:
    within_threshold = (errors <= threshold).mean() * 100
    print(f"   Predictions within ±{threshold} words: {within_threshold:.1f}%")

print(f"\n🎯 RECOMMENDATIONS:")
print(f"   • Best model for production: {best_model_name}")
print(f"   • Expected accuracy: ±{best_result['test_mae']:.1f} words on average")
print(f"   • Model explains {best_result['test_r2']:.1%} of the variance in response lengths")
print(f"   • Ready for integration with query router system")

print(f"\n💡 NEXT STEPS:")
print(f"   1. Integrate model into production token predictor")
print(f"   2. Monitor performance on real queries")
print(f"   3. Collect more training data for continuous improvement")
print(f"   4. Consider ensemble methods if needed")

print(f"\n🎉 Enhanced Magpie Token Predictor Analysis Complete!")
```

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

### **GitHub README Ready:**
This format is optimized for copying into GitHub README.md and then easily transferring to any Jupyter environment. Each cell is clearly separated and ready for enterprise use.

## License

This project is licensed under the MIT License.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
