# Cell 3: Process Embeddings - PCA and Clustering
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
    
    # Show explained variance per component (first 10)
    print(f"📊 Top 10 components explained variance:")
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        print(f"   Component {i+1}: {pca.explained_variance_ratio_[i]:.3f}")
    
    # K-means clustering for semantic grouping
    print(f"🎯 K-means clustering: {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_pca)
    
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print(f"✅ Clustering complete - cluster sizes: {cluster_counts.min()}-{cluster_counts.max()}")
    print(f"📊 Cluster distribution:")
    for i in range(min(10, n_clusters)):
        print(f"   Cluster {i}: {cluster_counts[i]} samples")
    
    return embeddings_pca, clusters, pca, kmeans

# Actually call the function
print("🚀 Starting embedding processing...")
embeddings_pca, clusters, pca_model, kmeans_model = process_embeddings(embeddings_matrix)

print(f"\n🔍 Results summary:")
print(f"   PCA shape: {embeddings_pca.shape}")
print(f"   Cluster assignments: {len(clusters)} samples")
print(f"   Unique clusters: {len(np.unique(clusters))}")
```

# Cell 4: Enhanced Feature Engineering
```python
import re

def create_enhanced_features(df, embeddings_pca, clusters):
    """Create enhanced features combining embeddings with existing features"""
    
    print("🛠️ ENHANCED FEATURE ENGINEERING")
    print("=" * 40)
    
    features_df = df.copy()
    
    # Add PCA embeddings as features
    print("📊 Adding PCA embedding features...")
    for i in range(embeddings_pca.shape[1]):
        features_df[f'embed_pca_{i}'] = embeddings_pca[:, i]
    print(f"   Added {embeddings_pca.shape[1]} embedding features")
    
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
    
    print("   Calculating advanced complexity metrics...")
    complexity_features = []
    for instruction in features_df['instruction']:
        complexity_features.append(advanced_complexity(instruction))
    
    complexity_df = pd.DataFrame(complexity_features, columns=[
        'lexical_diversity', 'technical_density', 'question_complexity', 'instruction_complexity_enhanced'
    ])
    
    for col in complexity_df.columns:
        features_df[col] = complexity_df[col].values
    
    # Pattern matching features
    print("🔍 Pattern matching features...")
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
    
    print(f"   Added {len(patterns)} pattern features")
    
    # Basic features
    print("📋 Adding basic features...")
    features_df['instruction_len'] = features_df['instruction'].astype(str).str.len()
    features_df['instruction_word_count'] = features_df['instruction'].astype(str).str.split().str.len()
    
    # Metadata features (encode if present)
    print("🏷️ Processing metadata features...")
    categorical_features = ['task_category', 'intent', 'knowledge']
    label_encoders = {}
    
    for col in categorical_features:
        if col in features_df.columns:
            print(f"   Encoding {col}")
            le = LabelEncoder()
            features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
            label_encoders[col] = le
        else:
            print(f"   {col} not found, using default")
            features_df[f'{col}_encoded'] = 0
    
    # Difficulty encoding
    if 'difficulty' in features_df.columns:
        diff_map = {'easy': 1, 'medium': 2, 'hard': 3}
        features_df['difficulty_encoded'] = features_df['difficulty'].map(diff_map).fillna(2)
        print("   Encoded difficulty from metadata")
    else:
        features_df['difficulty_encoded'] = 2  # Default to medium
        print("   Using default difficulty encoding")
    
    print(f"✅ Enhanced features created: {features_df.shape[1]} total features")
    
    return features_df, label_encoders

# Actually call the function
print("🚀 Starting enhanced feature engineering...")
features_df, label_encoders = create_enhanced_features(df_clean, embeddings_pca, clusters)

print(f"\n🔍 Feature summary:")
print(f"   Total features: {features_df.shape[1]}")
print(f"   Total samples: {features_df.shape[0]}")
print(f"   Embedding features: {len([col for col in features_df.columns if col.startswith('embed_pca_')])}")
print(f"   Pattern features: {len([col for col in features_df.columns if col.startswith('pattern_')])}")
```

# Cell 5: Enhanced Model Training
```python
from sklearn.preprocessing import StandardScaler

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

# Actually call the function
print("🚀 Starting enhanced model training...")
training_results = train_enhanced_models(features_df)

print(f"\n🎉 Training complete!")
print(f"Best model: {training_results['best_model_name']}")
print(f"Best MAE: {training_results['results'][training_results['best_model_name']]['test_mae']:.2f}")
```

# Cell 6: Feature Importance Analysis
```python
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
            
            # Summary by feature type
            embedding_importance = top_features[top_features['feature'].str.startswith('embed_pca_')]['importance'].sum()
            pattern_importance = top_features[top_features['feature'].str.startswith('pattern_')]['importance'].sum()
            basic_importance = top_features[top_features['feature'].isin(['input_tokens_mistral', 'instruction_len', 'instruction_word_count'])]['importance'].sum()
            
            print(f"\n📊 {model_name} Feature Type Summary:")
            print(f"   Embedding features importance: {embedding_importance:.4f}")
            print(f"   Pattern features importance: {pattern_importance:.4f}")
            print(f"   Basic features importance: {basic_importance:.4f}")

# Actually call the function
print("🚀 Starting feature importance analysis...")
analyze_feature_importance(training_results)
```

# Cell 7: Enhanced Query Testing Interface
```python
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
            'knowledge': knowledge
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

# Actually create the query tester
print("🚀 Creating enhanced query tester...")
query_tester = create_enhanced_query_tester(training_results, embedding_model, pca_model, kmeans_model, label_encoders)

# Test with sample queries
print("\n🧪 Testing sample queries:")

test_queries = [
    "What is Python?",
    "How do I sort a list in Python?", 
    "Implement a binary search algorithm with error handling",
    "Debug this error: AttributeError"
]

for query in test_queries:
    print(f"\n" + "="*50)
    prediction = query_tester(query)
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
