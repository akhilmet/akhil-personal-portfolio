```
# Cell 1: Load and Clean Data
```python
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

print("🚀 Simple Embedding Token Predictor")
print("=" * 50)
print("🎯 Using raw embeddings + production-available features only")

def load_and_clean_data():
    """Load Magpie dataset and clean suspicious labels"""
    
    print("📥 LOADING AND CLEANING DATA")
    print("=" * 40)
    
    # Load data
    cwd = os.getcwd()
    base_dir = os.path.join(
        cwd,
        "query-routing-systems-datasets", 
        "Magpie-Llama-3.1-Pro-DPO-100k-v0.1",
        "data"
    )
    
    parquet_files = glob.glob(os.path.join(base_dir, "*.parquet"))
    df_list = [pd.read_parquet(fp, engine="pyarrow") for fp in parquet_files]
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"✅ Loaded {len(df)} samples from {len(parquet_files)} files")
    
    # Extract response text
    print("🔄 Extracting response text...")
    
    response_texts = []
    for idx, responses_data in enumerate(df['responses']):
        try:
            if responses_data is None:
                response_texts.append("")
                continue
            
            if hasattr(responses_data, 'dtype'):  # It's a numpy array
                if len(responses_data) == 0:
                    response_texts.append("")
                    continue
                first_item = responses_data[0]
            elif isinstance(responses_data, list):
                if len(responses_data) == 0:
                    response_texts.append("")
                    continue
                first_item = responses_data[0]
            elif isinstance(responses_data, str):
                response_texts.append(responses_data)
                continue
            else:
                response_texts.append(str(responses_data))
                continue
            
            # Handle the first item
            if isinstance(first_item, dict):
                for key in ('text', 'content', 'value', 'response'):
                    if key in first_item:
                        response_texts.append(str(first_item[key]))
                        break
                else:
                    response_texts.append(str(first_item))
            else:
                response_texts.append(str(first_item))
                
        except Exception as e:
            if idx < 5:
                print(f"Error processing response {idx}: {e}")
            response_texts.append("")
    
    df['response'] = response_texts
    
    # Calculate tokens
    print("🔥 Loading Mistral tokenizer...")
    tokenizer = MistralTokenizer.v3()
    
    def count_tokens(text):
        try:
            if not text or pd.isna(text):
                return 0
            text_str = str(text).strip()
            if len(text_str) == 0:
                return 0
            req = ChatCompletionRequest(messages=[UserMessage(content=text_str)])
            enc = tokenizer.encode_chat_completion(req)
            return len(enc.tokens)
        except Exception:
            return 0
    
    print("🎯 Calculating tokens...")
    df['input_tokens_mistral'] = [count_tokens(inst) for inst in df['instruction']]
    df['actual_output_tokens'] = [count_tokens(resp) for resp in df['response']]
    
    # Convert to proper types
    df['input_tokens_mistral'] = pd.Series(df['input_tokens_mistral'], dtype='int64')
    df['actual_output_tokens'] = pd.Series(df['actual_output_tokens'], dtype='int64')
    
    # Clean data
    initial_count = len(df)
    df = df[
        (df['input_tokens_mistral'] > 0) & 
        (df['actual_output_tokens'] > 0)
    ].copy()
    
    # Remove suspicious labels
    suspicious_mask = (df['actual_output_tokens'] < 10) & (df['input_tokens_mistral'] > 100)
    df_clean = df[~suspicious_mask].copy()
    
    # Remove extreme outliers
    if len(df_clean) > 0:
        q1_out = df_clean['actual_output_tokens'].quantile(0.01)
        q99_out = df_clean['actual_output_tokens'].quantile(0.99)
        df_clean = df_clean[
            (df_clean['actual_output_tokens'] >= q1_out) & 
            (df_clean['actual_output_tokens'] <= q99_out)
        ].copy()
    
    final_count = len(df_clean)
    print(f"🧹 Removed {initial_count - final_count} suspicious/outlier samples")
    print(f"✅ Clean dataset: {final_count:,} samples")
    print(f"📊 Token range: {df_clean['actual_output_tokens'].min()}-{df_clean['actual_output_tokens'].max()}")
    
    return df_clean

# Load the data
df_clean = load_and_clean_data()
```

# Cell 2: Generate Raw Embeddings (No PCA, No Clustering)
```python
from sentence_transformers import SentenceTransformer

def generate_raw_embeddings(df):
    """Generate raw sentence embeddings without dimensionality reduction"""
    
    print("🤖 GENERATING RAW SENTENCE EMBEDDINGS")
    print("=" * 45)
    
    # Load sentence transformer model
    print("📥 Loading all-MiniLM-L6-v2 model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded!")
    
    # Prepare instructions
    instructions = df['instruction'].astype(str).fillna("").tolist()
    print(f"📊 Processing {len(instructions)} instructions")
    
    # Generate embeddings in batches
    print("🔄 Generating embeddings...")
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(instructions), batch_size):
        batch_end = min(i + batch_size, len(instructions))
        batch = instructions[i:batch_end]
        
        print(f"   Batch {i//batch_size + 1}: {i+1}-{batch_end} ({(batch_end/len(instructions)*100):.1f}%)")
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
    
    embeddings_matrix = np.array(all_embeddings)
    print(f"✅ Generated embeddings: {embeddings_matrix.shape}")
    print(f"📊 Shape: {embeddings_matrix.shape[0]} samples × {embeddings_matrix.shape[1]} dimensions")
    
    return embeddings_matrix, model

# Generate embeddings
print("🚀 Starting embedding generation...")
embeddings_matrix, embedding_model = generate_raw_embeddings(df_clean)
```

# Cell 3: Production-Ready Feature Engineering
```python
def create_production_features(df, embeddings_matrix):
    """Create features that are available in production (query-only)"""
    
    print("🛠️ PRODUCTION-READY FEATURE ENGINEERING")
    print("=" * 50)
    print("🎯 Using only features available from input query")
    
    features_df = df.copy()
    
    # 1. Raw embeddings as features (384 dimensions)
    print("📊 Adding raw embedding features...")
    for i in range(embeddings_matrix.shape[1]):
        features_df[f'embed_{i}'] = embeddings_matrix[:, i]
    print(f"   Added {embeddings_matrix.shape[1]} embedding dimensions")
    
    # 2. Basic text statistics (available from query)
    print("📝 Adding basic text features...")
    
    def extract_basic_features(text):
        """Extract features available from input query only"""
        if pd.isna(text) or text == "":
            return [0] * 12
        
        txt = str(text)
        txt_lower = txt.lower()
        words = txt_lower.split()
        
        if not words:
            return [0] * 12
        
        features = []
        
        # Length features
        features.append(len(txt))  # character length
        features.append(len(words))  # word count
        features.append(np.mean([len(w) for w in words]))  # average word length
        
        # Vocabulary diversity
        features.append(len(set(words)) / len(words))  # unique word ratio
        
        # Punctuation counts
        features.append(txt.count('?'))  # questions
        features.append(txt.count('.'))  # statements
        features.append(txt.count('!'))  # exclamations
        features.append(txt.count(','))  # commas
        
        # Question type indicators
        has_what = int(any(word in txt_lower for word in ['what', 'define']))
        has_how = int(any(word in txt_lower for word in ['how', 'tutorial', 'guide']))
        has_implement = int(any(word in txt_lower for word in ['implement', 'build', 'create', 'develop']))
        has_explain = int(any(word in txt_lower for word in ['explain', 'describe', 'elaborate']))
        
        features.extend([has_what, has_how, has_implement, has_explain])
        
        return features
    
    # Extract basic features for all instructions
    basic_features = []
    for instruction in features_df['instruction']:
        basic_features.append(extract_basic_features(instruction))
    
    # Add basic features to dataframe
    basic_feature_names = [
        'char_length', 'word_count', 'avg_word_length', 'unique_word_ratio',
        'question_marks', 'periods', 'exclamations', 'commas',
        'has_what_questions', 'has_how_questions', 'has_implement_requests', 'has_explain_requests'
    ]
    
    for i, name in enumerate(basic_feature_names):
        features_df[name] = [f[i] for f in basic_features]
    
    print(f"   Added {len(basic_feature_names)} basic text features")
    
    # 3. Code detection (available from query)
    print("💻 Adding code detection features...")
    
    def detect_code_features(text):
        if pd.isna(text) or text == "":
            return [0] * 5
        
        txt = str(text)
        txt_lower = txt.lower()
        
        features = []
        
        # Code block indicators
        features.append(int('```' in txt))  # code blocks
        features.append(int(bool(re.search(r'`[^`]+`', txt))))  # inline code
        
        # Programming language mentions
        prog_langs = ['python', 'javascript', 'java', 'sql', 'html', 'css', 'react', 'node']
        features.append(int(any(lang in txt_lower for lang in prog_langs)))
        
        # Technical terms
        tech_terms = ['function', 'class', 'method', 'variable', 'array', 'object', 'api', 'database']
        features.append(int(any(term in txt_lower for term in tech_terms)))
        
        # Action words for coding
        code_actions = ['implement', 'debug', 'fix', 'optimize', 'refactor', 'deploy']
        features.append(int(any(action in txt_lower for action in code_actions)))
        
        return features
    
    code_features = []
    for instruction in features_df['instruction']:
        code_features.append(detect_code_features(instruction))
    
    code_feature_names = [
        'has_code_blocks', 'has_inline_code', 'mentions_programming', 
        'has_tech_terms', 'has_code_actions'
    ]
    
    for i, name in enumerate(code_feature_names):
        features_df[name] = [f[i] for f in code_features]
    
    print(f"   Added {len(code_feature_names)} code detection features")
    
    # 4. Input tokens (available in production)
    print("🎯 Adding input token count...")
    # Use existing input_tokens_mistral as it's production-available
    
    print(f"✅ Production features complete!")
    print(f"📊 Total features: {features_df.shape[1]}")
    
    # Define feature columns for model training
    embedding_cols = [f'embed_{i}' for i in range(embeddings_matrix.shape[1])]
    feature_cols = embedding_cols + basic_feature_names + code_feature_names + ['input_tokens_mistral']
    
    return features_df, feature_cols

# Create production features
print("🚀 Starting production feature engineering...")
features_df, feature_columns = create_production_features(df_clean, embeddings_matrix)

print(f"\n🔍 Feature summary:")
print(f"   Embedding features: {len([col for col in feature_columns if col.startswith('embed_')])}")
print(f"   Text features: {len([col for col in feature_columns if not col.startswith('embed_') and col != 'input_tokens_mistral'])}")
print(f"   Total features: {len(feature_columns)}")
```

# Cell 4: Train Models
```python
def train_production_models(features_df, feature_columns, target_col='actual_output_tokens'):
    """Train models using production-ready features"""
    
    print("🎯 TRAINING PRODUCTION MODELS")
    print("=" * 40)
    
    # Prepare data
    X = features_df[feature_columns].copy()
    y = features_df[target_col].copy()
    
    # Handle any missing values
    X = X.fillna(0)
    
    # Remove samples with missing target
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    print(f"📊 Training data: {len(X)} samples × {len(feature_columns)} features")
    print(f"🎯 Target range: {y.min():.0f} - {y.max():.0f} tokens (mean: {y.mean():.1f})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for XGBoost (Random Forest doesn't need scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    results = {}
    
    # Random Forest
    print("\n🌳 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # XGBoost
    print("🚀 Training XGBoost...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_reg.fit(X_train, y_train)
    models['XGBoost'] = xgb_reg
    
    # Evaluate models
    for name, model in models.items():
        print(f"\n📊 Evaluating {name}...")
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        # Accuracy metrics
        errors = np.abs(y_test - test_pred)
        within_5 = (errors <= 5).mean() * 100
        within_10 = (errors <= 10).mean() * 100
        within_20 = (errors <= 20).mean() * 100
        within_50 = (errors <= 50).mean() * 100
        
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'within_5': within_5,
            'within_10': within_10,
            'within_20': within_20,
            'within_50': within_50,
            'test_pred': test_pred,
            'feature_importance': None
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            results[name]['feature_importance'] = importance_df
        
        print(f"   Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
        print(f"   Test RMSE: {test_rmse:.2f} | Test R²: {test_r2:.3f}")
        print(f"   Within ±5: {within_5:.1f}% | ±10: {within_10:.1f}% | ±20: {within_20:.1f}% | ±50: {within_50:.1f}%")
    
    # Best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test MAE: {results[best_model_name]['test_mae']:.2f} tokens")
    
    return {
        'results': results,
        'best_model_name': best_model_name,
        'best_model': results[best_model_name]['model'],
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }

# Train models
print("🚀 Starting model training...")
training_results = train_production_models(features_df, feature_columns)

print(f"\n🎉 Training complete!")
print(f"Best model: {training_results['best_model_name']}")
print(f"Best MAE: {training_results['results'][training_results['best_model_name']]['test_mae']:.2f}")
```

# Cell 5: Feature Importance Analysis
```python
def analyze_feature_importance(training_results, feature_columns):
    """Analyze which features are most important"""
    
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    for model_name, result in training_results['results'].items():
        if result['feature_importance'] is not None:
            print(f"\n🔍 {model_name} - Top 20 Features:")
            top_features = result['feature_importance'].head(20)
            
            for idx, row in top_features.iterrows():
                feature_type = ""
                if row['feature'].startswith('embed_'):
                    feature_type = "📊 Embedding"
                elif row['feature'] == 'input_tokens_mistral':
                    feature_type = "🎯 Input Tokens"
                elif row['feature'] in ['char_length', 'word_count', 'avg_word_length', 'unique_word_ratio']:
                    feature_type = "📝 Text Stats"
                elif row['feature'].startswith('has_'):
                    feature_type = "🔍 Pattern"
                else:
                    feature_type = "📋 Other"
                
                print(f"   {feature_type:15} {row['feature']:25}: {row['importance']:.4f}")
            
            # Feature type summary
            embedding_importance = top_features[top_features['feature'].str.startswith('embed_')]['importance'].sum()
            text_importance = top_features[~top_features['feature'].str.startswith('embed_')]['importance'].sum()
            
            print(f"\n📊 {model_name} Feature Type Summary:")
            print(f"   Embedding features total importance: {embedding_importance:.4f}")
            print(f"   Non-embedding features total importance: {text_importance:.4f}")
            
            # Visualize top 10 features
            plt.figure(figsize=(10, 6))
            top_10 = top_features.head(10)
            colors = ['red' if feat.startswith('embed_') else 'blue' for feat in top_10['feature']]
            
            plt.barh(range(len(top_10)), top_10['importance'], color=colors)
            plt.yticks(range(len(top_10)), top_10['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top 10 Feature Importance\n(Red=Embeddings, Blue=Other)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

# Analyze feature importance
analyze_feature_importance(training_results, feature_columns)
```

# Cell 6: Model Performance Visualization
```python
def visualize_model_performance(training_results):
    """Visualize model performance and predictions"""
    
    print("📊 MODEL PERFORMANCE VISUALIZATION")
    print("=" * 45)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MAE Comparison
    models = list(training_results['results'].keys())
    maes = [training_results['results'][m]['test_mae'] for m in models]
    
    axes[0, 0].bar(models, maes, color=['blue', 'orange'])
    axes[0, 0].set_title('Test MAE Comparison')
    axes[0, 0].set_ylabel('MAE (tokens)')
    
    # Add baseline comparison
    baseline_mae = 165  # Previous result
    axes[0, 0].axhline(y=baseline_mae, color='red', linestyle='--', label=f'Previous Best ({baseline_mae})')
    axes[0, 0].legend()
    
    # 2. R² Comparison
    r2s = [training_results['results'][m]['test_r2'] for m in models]
    axes[0, 1].bar(models, r2s, color=['blue', 'orange'])
    axes[0, 1].set_title('Test R² Comparison')
    axes[0, 1].set_ylabel('R² Score')
    
    # 3. Accuracy within ±10 tokens
    within_10s = [training_results['results'][m]['within_10'] for m in models]
    axes[1, 0].bar(models, within_10s, color=['blue', 'orange'])
    axes[1, 0].set_title('Accuracy Within ±10 Tokens')
    axes[1, 0].set_ylabel('Percentage (%)')
    
    # 4. Predictions vs Actual (best model)
    best_model_name = training_results['best_model_name']
    best_pred = training_results['results'][best_model_name]['test_pred']
    y_test = training_results['y_test']
    
    axes[1, 1].scatter(y_test, best_pred, alpha=0.5, color='green')
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Tokens')
    axes[1, 1].set_ylabel('Predicted Tokens')
    axes[1, 1].set_title(f'{best_model_name} - Predictions vs Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    baseline_mae = 165
    for model_name, results in training_results['results'].items():
        improvement = ((baseline_mae - results['test_mae']) / baseline_mae) * 100
        print(f"   {model_name:15}: {results['test_mae']:6.2f} MAE ({improvement:+5.1f}% vs baseline)")

# Visualize performance
visualize_model_performance(training_results)
```

# Cell 7: Production Prediction Function with Testing
```python
def create_production_predictor(best_model, embedding_model, scaler, feature_columns):
    """Create production-ready prediction function"""
    
    def predict_tokens(query):
        """
        Predict output tokens for a given query
        
        Args:
            query (str): Input query string
            
        Returns:
            int: Predicted number of output tokens
        """
        
        if not query or pd.isna(query):
            return 1
        
        query = str(query).strip()
        if len(query) == 0:
            return 1
        
        # 1. Generate embedding
        embedding = embedding_model.encode([query])[0]  # Shape: (384,)
        
        # 2. Extract basic text features
        def extract_basic_features(text):
            txt_lower = text.lower()
            words = txt_lower.split()
            
            if not words:
                return [0] * 12
            
            features = []
            features.append(len(text))  # char_length
            features.append(len(words))  # word_count
            features.append(np.mean([len(w) for w in words]))  # avg_word_length
            features.append(len(set(words)) / len(words))  # unique_word_ratio
            features.append(text.count('?'))  # question_marks
            features.append(text.count('.'))  # periods
            features.append(text.count('!'))  # exclamations
            features.append(text.count(','))  # commas
            features.append(int(any(word in txt_lower for word in ['what', 'define'])))  # has_what_questions
            features.append(int(any(word in txt_lower for word in ['how', 'tutorial', 'guide'])))  # has_how_questions
            features.append(int(any(word in txt_lower for word in ['implement', 'build', 'create', 'develop'])))  # has_implement_requests
            features.append(int(any(word in txt_lower for word in ['explain', 'describe', 'elaborate'])))  # has_explain_requests
            
            return features
        
        # 3. Extract code detection features
        def detect_code_features(text):
            txt_lower = text.lower()
            
            features = []
            features.append(int('```' in text))  # has_code_blocks
            features.append(int(bool(re.search(r'`[^`]+`', text))))  # has_inline_code
            
            prog_langs = ['python', 'javascript', 'java', 'sql', 'html', 'css', 'react', 'node']
            features.append(int(any(lang in txt_lower for lang in prog_langs)))  # mentions_programming
            
            tech_terms = ['function', 'class', 'method', 'variable', 'array', 'object', 'api', 'database']
            features.append(int(any(term in txt_lower for term in tech_terms)))  # has_tech_terms
            
            code_actions = ['implement', 'debug', 'fix', 'optimize', 'refactor', 'deploy']
            features.append(int(any(action in txt_lower for action in code_actions)))  # has_code_actions
            
            return features
        
        # 4. Calculate input tokens (rough approximation for production)
        input_tokens = len(query.split()) * 1.3  # Rough approximation
        
        # 5. Combine all features
        basic_features = extract_basic_features(query)
        code_features = detect_code_features(query)
        
        # Create feature vector in the same order as training
        feature_vector = []
        
        # Add embeddings (384 dimensions)
        feature_vector.extend(embedding)
        
        # Add basic features
        feature_vector.extend(basic_features)
        
        # Add code features
        feature_vector.extend(code_features)
        
        # Add input tokens
        feature_vector.append(input_tokens)
        
        # Convert to numpy array and reshape
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction = best_model.predict(feature_array)[0]
        
        # Return as positive integer
        return max(1, int(round(prediction)))
    
    return predict_tokens

# Create production predictor
print("🚀 Creating production predictor...")
predict_tokens = create_production_predictor(
    training_results['best_model'],
    embedding_model,
    training_results['scaler'],
    feature_columns
)

print("✅ Production predictor ready!")

# Test the prediction function
print("\n🧪 TESTING PRODUCTION PREDICTOR")
print("=" * 40)

test_queries = [
    "What is Python?",
    "How do I sort a list in Python?",
    "Implement a binary search algorithm in Python with error handling",
    "Explain the mathematical foundations of transformer architecture",
    "Debug this error: AttributeError: 'NoneType' object has no attribute 'split'",
    "Hi there!",
    "Create a comprehensive machine learning pipeline for image classification"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n📝 Test {i}: '{query}'")
    predicted_tokens = predict_tokens(query)
    print(f"🎯 Predicted tokens: {predicted_tokens}")
    
    # Add some context about expected range
    if predicted_tokens < 50:
        print("   💡 Expected: Short response")
    elif predicted_tokens < 200:
        print("   💡 Expected: Medium response")
    else:
        print("   💡 Expected: Long response")

print(f"\n🎉 PRODUCTION PREDICTOR TESTING COMPLETE!")
print(f"=" * 50)
print(f"✅ Model ready for deployment")
print(f"🎯 Best model: {training_results['best_model_name']}")
print(f"📊 Test MAE: {training_results['results'][training_results['best_model_name']]['test_mae']:.2f} tokens")
print(f"🚀 Use predict_tokens(query) function for predictions")

# Example usage for production
print(f"\n📋 PRODUCTION USAGE EXAMPLE:")
print(f">>> predicted_tokens = predict_tokens('How do I implement binary search?')")
print(f">>> print(predicted_tokens)")
print(f">>> # Output: {predict_tokens('How do I implement binary search?')}")
```
        print("
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
