```
# Cell 1: Load and Clean Data
import os
import glob
import pandas as pd
import numpy as np
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

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
    
    # Extract response text - fixed version
    def extract_response_text(responses_data):
        try:
            if responses_data is None:
                return ""
            if pd.isna(responses_data):
                return ""
            
            # Handle different data types
            if isinstance(responses_data, str):
                return responses_data
            
            if isinstance(responses_data, list):
                if len(responses_data) == 0:
                    return ""
                first = responses_data[0]
                if isinstance(first, dict):
                    for key in ('text', 'content', 'value', 'response'):
                        if key in first:
                            return str(first[key])
                    return str(first)
                else:
                    return str(first)
            
            if isinstance(responses_data, dict):
                for key in ('text', 'content', 'value', 'response'):
                    if key in responses_data:
                        return str(responses_data[key])
                return str(responses_data)
            
            # Fallback
            return str(responses_data)
            
        except Exception as e:
            print(f"Error processing response: {e}")
            return ""
    
    # Apply extraction safely
    print("🔄 Extracting response text...")
    df['response'] = df['responses'].apply(extract_response_text)
    
    # Calculate tokens
    print("🔥 Loading Mistral tokenizer...")
    tokenizer = MistralTokenizer.v3()
    
    def count_tokens(text):
        try:
            if pd.isna(text) or text == "" or text is None:
                return 0
            text_str = str(text)
            if len(text_str.strip()) == 0:
                return 0
            req = ChatCompletionRequest(messages=[UserMessage(content=text_str)])
            enc = tokenizer.encode_chat_completion(req)
            return len(enc.tokens)
        except Exception as e:
            print(f"Error tokenizing: {e}")
            return 0
    
    print("🎯 Calculating input tokens...")
    df['input_tokens_mistral'] = df['instruction'].apply(count_tokens)
    
    print("🎯 Calculating output tokens...")
    df['actual_output_tokens'] = df['response'].apply(count_tokens)
    
    # Remove missing or zero tokens
    initial_count = len(df)
    df = df[
        (df['input_tokens_mistral'] > 0) & 
        (df['actual_output_tokens'] > 0)
    ].copy()
    
    print(f"🧹 Removed {initial_count - len(df)} samples with zero tokens")
    
    # Clean suspicious labels
    print("🧹 Cleaning suspicious labels...")
    suspicious_mask = (df['actual_output_tokens'] < 10) & (df['input_tokens_mistral'] > 100)
    suspicious_count = suspicious_mask.sum()
    df_clean = df[~suspicious_mask].copy()
    
    # Remove extreme outliers
    q1_out = df_clean['actual_output_tokens'].quantile(0.01)
    q99_out = df_clean['actual_output_tokens'].quantile(0.99)
    df_clean = df_clean[
        (df_clean['actual_output_tokens'] >= q1_out) & 
        (df_clean['actual_output_tokens'] <= q99_out)
    ].copy()
    
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    
    print(f"🧹 Removed {suspicious_count} suspicious labels")
    print(f"🧹 Removed {removed_count} total samples")
    print(f"✅ Clean dataset: {final_count:,} samples")
    
    if final_count > 0:
        print(f"📊 Token stats - Mean: {df_clean['actual_output_tokens'].mean():.1f}, "
              f"Range: {df_clean['actual_output_tokens'].min()}-{df_clean['actual_output_tokens'].max()}")
    
    return df_clean

# Load the data
df_clean = load_and_clean_data()
```
```
# Enhanced Token Predictor with Embeddings and Data Cleaning
# Adding semantic embeddings to improve prediction accuracy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# For embeddings
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

print("🚀 Enhanced Token Predictor with Embeddings")
print("=" * 60)
print("🎯 Adding semantic embeddings + data cleaning")

# Cell 1: Load and Clean Data
def load_and_clean_data():
    """Load data and identify suspicious labels"""
    
    # Assuming df_with_tokens is already loaded from previous notebook
    # If not, you'll need to run the data loading cells first
    
    print("🧹 CLEANING DATA - Identifying Suspicious Labels")
    print("=" * 50)
    
    # Load your existing data (replace with actual loading if needed)
    # df = df_with_tokens.copy()
    
    # For now, let's create a synthetic check - replace with your actual data
    print("⚠️ Replace this section with your actual df_with_tokens")
    print("   This is placeholder code showing the cleaning approach")
    
    # Identify suspicious short responses (high input, very low output)
    def identify_suspicious_labels(df):
        suspicious_short = df[
            (df['actual_output_tokens'] < 10) & 
            (df['input_tokens_mistral'] > 100)
        ]
        
        suspicious_ratios = df[
            df['actual_output_tokens'] / df['input_tokens_mistral'] < 0.01
        ]
        
        print(f"🚨 Found {len(suspicious_short)} suspicious short labels")
        print(f"🚨 Found {len(suspicious_ratios)} suspicious ratio labels")
        
        return suspicious_short, suspicious_ratios
    
    # Remove or flag suspicious data
    def clean_dataset(df):
        print("🧹 Cleaning dataset...")
        
        # Remove extreme outliers
        initial_count = len(df)
        
        # Remove suspiciously low output tokens for long inputs
        df_clean = df[~(
            (df['actual_output_tokens'] < 10) & 
            (df['input_tokens_mistral'] > 100)
        )].copy()
        
        # Remove extreme outliers in general
        q1_out = df_clean['actual_output_tokens'].quantile(0.01)
        q99_out = df_clean['actual_output_tokens'].quantile(0.99)
        df_clean = df_clean[
            (df_clean['actual_output_tokens'] >= q1_out) & 
            (df_clean['actual_output_tokens'] <= q99_out)
        ].copy()
        
        removed_count = initial_count - len(df_clean)
        print(f"✅ Removed {removed_count} suspicious/outlier samples")
        print(f"📊 Clean dataset: {len(df_clean)} samples")
        
        return df_clean
    
    return None  # Placeholder - replace with actual cleaned data

# Cell 2: Generate Embeddings
def generate_embeddings(df):
    """Generate semantic embeddings for instructions"""
    
    print("🤖 GENERATING SEMANTIC EMBEDDINGS")
    print("=" * 40)
    
    # Load sentence transformer model
    print("📥 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model
    print("✅ Model loaded!")
    
    # Generate embeddings for instructions
    print("🔄 Generating embeddings for instructions...")
    instructions = df['instruction'].astype(str).tolist()
    
    # Generate embeddings in batches to avoid memory issues
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        all_embeddings.extend(batch_embeddings)
        print(f"   Processed {min(i + batch_size, len(instructions))}/{len(instructions)} instructions")
    
    embeddings_matrix = np.array(all_embeddings)
    print(f"✅ Generated embeddings: {embeddings_matrix.shape}")
    
    return embeddings_matrix, model

# Cell 3: Dimensionality Reduction and Clustering
def process_embeddings(embeddings_matrix, n_components=50, n_clusters=20):
    """Reduce dimensionality and add clustering features"""
    
    print("🔄 PROCESSING EMBEDDINGS")
    print("=" * 30)
    
    # PCA for dimensionality reduction
    print(f"📉 Applying PCA: {embeddings_matrix.shape[1]} → {n_components} dimensions")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_matrix)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA complete - explained variance: {explained_variance:.1%}")
    
    # K-means clustering for semantic grouping
    print(f"🎯 K-means clustering: {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_pca)
    
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print(f"✅ Clustering complete - cluster sizes: {cluster_counts.min()}-{cluster_counts.max()}")
    
    return embeddings_pca, clusters, pca, kmeans

# Cell 4: Enhanced Feature Engineering
def create_enhanced_features(df, embeddings_pca, clusters):
    """Create enhanced features combining embeddings with existing features"""
    
    print("🛠️ ENHANCED FEATURE ENGINEERING")
    print("=" * 40)
    
    features_df = df.copy()
    
    # Add PCA embeddings as features
    print("📊 Adding PCA embedding features...")
    for i in range(embeddings_pca.shape[1]):
        features_df[f'embed_pca_{i}'] = embeddings_pca[:, i]
    
    # Add cluster assignments
    print("🎯 Adding cluster features...")
    features_df['semantic_cluster'] = clusters
    
    # Enhanced text analysis features
    print("📝 Enhanced text analysis...")
    
    # Advanced complexity metrics
    def advanced_complexity(text):
        if pd.isna(text) or text == "":
            return 0, 0, 0, 0
        
        txt = str(text).lower()
        words = txt.split()
        
        if not words:
            return 0, 0, 0, 0
        
        # Lexical diversity
        unique_ratio = len(set(words)) / len(words)
        
        # Technical density
        tech_patterns = [
            r'\b(function|class|import|def|return|api|database|algorithm|model|training)\b',
            r'\b(tensor|numpy|pandas|sklearn|python|javascript|react|sql)\b',
            r'\b(implement|optimize|debug|refactor|deploy|configure)\b',
            r'\b(analysis|statistics|machine learning|neural network|regression)\b'
        ]
        tech_score = sum(len(re.findall(p, txt)) for p in tech_patterns) / len(words)
        
        # Question complexity
        question_patterns = [
            r'\bhow to\b', r'\bwhy does\b', r'\bwhat is the difference\b',
            r'\bcompare\b', r'\bexplain\b', r'\bimplement\b'
        ]
        question_complexity = sum(len(re.findall(p, txt)) for p in question_patterns)
        
        # Instruction complexity
        instruction_patterns = [
            r'\bstep by step\b', r'\bdetailed\b', r'\bcomprehensive\b',
            r'\btutorial\b', r'\bguide\b', r'\bexample\b'
        ]
        instruction_complexity = sum(len(re.findall(p, txt)) for p in instruction_patterns)
        
        return unique_ratio, tech_score, question_complexity, instruction_complexity
    
    complexity_features = features_df['instruction'].apply(lambda x: pd.Series(advanced_complexity(x)))
    features_df['lexical_diversity'] = complexity_features[0]
    features_df['technical_density'] = complexity_features[1]
    features_df['question_complexity'] = complexity_features[2]
    features_df['instruction_complexity_enhanced'] = complexity_features[3]
    
    # Semantic similarity to known patterns
    print("🔍 Pattern matching features...")
    
    # Common query patterns and their typical response lengths
    pattern_features = {}
    patterns = {
        'simple_what': (r'\bwhat is\b', 'short_expected'),
        'how_to_basic': (r'\bhow to .{1,20}\?', 'medium_expected'),
        'implement_complex': (r'\bimplement .{10,}', 'long_expected'),
        'explain_detailed': (r'\bexplain .{20,}', 'long_expected'),
        'debug_help': (r'\bdebug|error|fix\b', 'medium_expected'),
        'code_example': (r'\bexample|sample code\b', 'medium_expected'),
        'comparison': (r'\bcompare|difference|vs\b', 'medium_expected'),
        'tutorial_request': (r'\btutorial|guide|walkthrough\b', 'long_expected')
    }
    
    for pattern_name, (regex, expected_length) in patterns.items():
        features_df[f'pattern_{pattern_name}'] = features_df['instruction'].str.contains(
            regex, case=False, na=False
        ).astype(int)
    
    # Original features (production-safe)
    print("📋 Adding original production-safe features...")
    
    # Basic features
    features_df['input_tokens_mistral'] = features_df.get('input_tokens_mistral', 0)
    features_df['instruction_len'] = features_df['instruction'].astype(str).str.len()
    features_df['instruction_word_count'] = features_df['instruction'].astype(str).str.split().str.len()
    
    # Metadata features (encode if present)
    categorical_features = ['task_category', 'intent', 'knowledge', 'question_type']
    label_encoders = {}
    
    for col in categorical_features:
        if col in features_df.columns:
            le = LabelEncoder()
            features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
            label_encoders[col] = le
        else:
            # Create default values if metadata missing
            features_df[f'{col}_encoded'] = 0
    
    # Difficulty encoding
    if 'difficulty' in features_df.columns:
        diff_map = {'easy': 1, 'medium': 2, 'hard': 3}
        features_df['difficulty_encoded'] = features_df['difficulty'].map(diff_map).fillna(2)
    else:
        features_df['difficulty_encoded'] = 2  # Default to medium
    
    print(f"✅ Enhanced features created: {features_df.shape[1]} total features")
    
    return features_df, label_encoders

# Cell 5: Enhanced Model Training
def train_enhanced_models(features_df, target_col='actual_output_tokens'):
    """Train models with enhanced features including embeddings"""
    
    print("🎯 TRAINING ENHANCED MODELS")
    print("=" * 40)
    
    # Define feature columns
    embedding_cols = [col for col in features_df.columns if col.startswith('embed_pca_')]
    pattern_cols = [col for col in features_df.columns if col.startswith('pattern_')]
    
    feature_columns = (
        ['input_tokens_mistral', 'instruction_len', 'instruction_word_count'] +
        ['lexical_diversity', 'technical_density', 'question_complexity', 'instruction_complexity_enhanced'] +
        ['semantic_cluster', 'difficulty_encoded'] +
        [col for col in features_df.columns if col.endswith('_encoded')] +
        embedding_cols +
        pattern_cols
    )
    
    # Remove duplicates and non-existent columns
    feature_columns = [col for col in feature_columns if col in features_df.columns]
    feature_columns = list(dict.fromkeys(feature_columns))  # Remove duplicates
    
    print(f"📊 Using {len(feature_columns)} features:")
    print(f"   Embedding features: {len(embedding_cols)}")
    print(f"   Pattern features: {len(pattern_cols)}")
    print(f"   Traditional features: {len(feature_columns) - len(embedding_cols) - len(pattern_cols)}")
    
    # Prepare data
    X = features_df[feature_columns].copy()
    y = features_df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Remove samples with missing target
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    print(f"📊 Training data: {len(X)} samples × {len(feature_columns)} features")
    print(f"🎯 Target range: {y.min():.0f} - {y.max():.0f} tokens (mean: {y.mean():.1f})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    results = {}
    
    # Random Forest (handles mixed features well)
    print("\n🌳 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300, 
        max_depth=20, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)  # RF doesn't need scaling
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
        n_jobs=-1
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
        'feature_columns': feature_columns,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }

# Cell 6: Feature Importance Analysis
def analyze_feature_importance(training_results):
    """Analyze which features are most important"""
    
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    for model_name, result in training_results['results'].items():
        if result['feature_importance'] is not None:
            print(f"\n🔍 {model_name} - Top 20 Features:")
            top_features = result['feature_importance'].head(20)
            
            for idx, row in top_features.iterrows():
                feature_type = ""
                if row['feature'].startswith('embed_pca_'):
                    feature_type = "📊 Embedding"
                elif row['feature'].startswith('pattern_'):
                    feature_type = "🎯 Pattern"
                elif row['feature'] in ['input_tokens_mistral', 'instruction_len', 'instruction_word_count']:
                    feature_type = "📝 Basic"
                else:
                    feature_type = "🔧 Enhanced"
                
                print(f"   {feature_type} {row['feature']}: {row['importance']:.4f}")
            
            # Visualize top features
            plt.figure(figsize=(10, 8))
            top_10 = top_features.head(10)
            plt.barh(range(len(top_10)), top_10['importance'])
            plt.yticks(range(len(top_10)), top_10['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

# Cell 7: Enhanced Query Testing
def create_enhanced_query_tester(training_results, embeddings_model, pca, kmeans, label_encoders):
    """Create query tester with embedding features"""
    
    def predict_with_embeddings(query, difficulty='medium', task_category='Information seeking', 
                               intent='informational', knowledge='intermediate'):
        """Predict tokens using enhanced features including embeddings"""
        
        print(f"🧪 ENHANCED PREDICTION (with embeddings)")
        print(f"   Query: '{query[:60]}{'...' if len(query) > 60 else ''}'")
        print()
        
        # Generate embedding for query
        query_embedding = embeddings_model.encode([query])
        query_embedding_pca = pca.transform(query_embedding)
        query_cluster = kmeans.predict(query_embedding_pca)[0]
        
        # Extract all features
        features = {}
        
        # Basic features
        features['input_tokens_mistral'] = len(query.split()) * 1.3  # Rough approximation
        features['instruction_len'] = len(query)
        features['instruction_word_count'] = len(query.split())
        
        # Enhanced complexity
        words = query.lower().split()
        if words:
            features['lexical_diversity'] = len(set(words)) / len(words)
            tech_patterns = [r'\b(function|class|import|def|return|api|database|algorithm|model)\b']
            features['technical_density'] = sum(len(re.findall(p, query.lower())) for p in tech_patterns) / len(words)
        else:
            features['lexical_diversity'] = 0
            features['technical_density'] = 0
        
        features['question_complexity'] = len(re.findall(r'\bhow to\b|\bwhy does\b|\bexplain\b', query.lower()))
        features['instruction_complexity_enhanced'] = len(re.findall(r'\bstep by step\b|\bdetailed\b|\btutorial\b', query.lower()))
        
        # Embedding features
        for i in range(query_embedding_pca.shape[1]):
            features[f'embed_pca_{i}'] = query_embedding_pca[0, i]
        
        # Cluster
        features['semantic_cluster'] = query_cluster
        
        # Pattern features
        patterns = {
            'simple_what': r'\bwhat is\b',
            'how_to_basic': r'\bhow to .{1,20}\?',
            'implement_complex': r'\bimplement .{10,}',
            'explain_detailed': r'\bexplain .{20,}',
            'debug_help': r'\bdebug|error|fix\b',
            'code_example': r'\bexample|sample code\b',
            'comparison': r'\bcompare|difference|vs\b',
            'tutorial_request': r'\btutorial|guide|walkthrough\b'
        }
        
        for pattern_name, regex in patterns.items():
            features[f'pattern_{pattern_name}'] = int(bool(re.search(regex, query.lower())))
        
        # Metadata
        features['difficulty_encoded'] = {'easy': 1, 'medium': 2, 'hard': 3}.get(difficulty, 2)
        
        # Encode categorical features
        categorical_map = {
            'task_category': task_category,
            'intent': intent,
            'knowledge': knowledge,
            'question_type': 'explanation'  # Default
        }
        
        for col, value in categorical_map.items():
            if col in label_encoders:
                try:
                    features[f'{col}_encoded'] = label_encoders[col].transform([value])[0]
                except:
                    features[f'{col}_encoded'] = 0
            else:
                features[f'{col}_encoded'] = 0
        
        # Create feature vector
        feature_vector = []
        for col in training_results['feature_columns']:
            feature_vector.append(features.get(col, 0))
        
        # Make prediction
        best_model = training_results['results'][training_results['best_model_name']]['model']
        prediction = best_model.predict([feature_vector])[0]
        
        mae = training_results['results'][training_results['best_model_name']]['test_mae']
        
        print(f"🎯 Prediction: {prediction:.0f} tokens (±{mae:.1f} MAE)")
        print(f"🎯 Semantic cluster: {query_cluster}")
        print(f"🧠 Using {len(feature_vector)} enhanced features including embeddings")
        
        return prediction
    
    return predict_with_embeddings

print("✅ Enhanced Token Predictor with Embeddings - Ready!")
print("🚀 Run these functions in order:")
print("   1. load_and_clean_data()")
print("   2. generate_embeddings(df)")
print("   3. process_embeddings(embeddings)")
print("   4. create_enhanced_features(df, embeddings_pca, clusters)")
print("   5. train_enhanced_models(features_df)")
print("   6. analyze_feature_importance(results)")
print("   7. create_enhanced_query_tester(...)")
```
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
