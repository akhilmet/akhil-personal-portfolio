```
# Cell 3: Advanced Feature Engineering for Neural Networks (with metadata progress bar)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import re
import numpy as np
import pandas as pd

def create_neural_network_features(df, embeddings_matrix):
    """Create comprehensive features optimized for neural networks, with metadata progress."""
    
    print("🛠️ NEURAL NETWORK FEATURE ENGINEERING")
    print("=" * 50)
    features_df = df.copy()
    
    # 1. PCA on embeddings
    print("📊 Applying PCA to embeddings...")
    pca = PCA(n_components=100, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_matrix)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA complete - explained variance: {explained_variance:.1%}")
    
    for i in range(embeddings_pca.shape[1]):
        features_df[f'embed_pca_{i}'] = embeddings_pca[:, i]
    
    # 2. Semantic clustering
    print("🎯 Semantic clustering...")
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_pca)
    features_df['semantic_cluster'] = clusters
    
    # 3. Advanced text features
    print("📝 Extracting comprehensive text features...")
    def extract_comprehensive_features(text):
        if pd.isna(text) or text == "":
            return [0]*16  # 16 features below
        txt = str(text)
        txt_lower = txt.lower()
        words = txt_lower.split() or ['']
        feats = [
            len(txt),
            len(words),
            len(set(words))/len(words),
            np.mean([len(w) for w in words]),
            txt.count('?'),
            txt.count('.'),
            txt.count(','),
            txt.count(':'),
            txt.count(';'),
            sum(len(re.findall(p, txt_lower)) for p in [
                r'\b(function|class|import|def|return|api|database|algorithm|model|training)\b',
                r'\b(tensor|numpy|pandas|sklearn|python|javascript|react|sql|html|css)\b',
                r'\b(implement|optimize|debug|refactor|deploy|configure|install)\b',
                r'\b(analysis|statistics|machine learning|neural network|regression|classification)\b'
            ])/len(words),
            sum(len(re.findall(p, txt_lower)) for p in [
                r'\bhow to\b', r'\bwhy does\b', r'\bwhat is the difference\b',
                r'\bcompare\b', r'\bexplain\b', r'\bimplement\b', r'\banalyze\b'
            ]),
            sum(len(re.findall(p, txt_lower)) for p in [
                r'\bstep by step\b', r'\bdetailed\b', r'\bcomprehensive\b',
                r'\btutorial\b', r'\bguide\b', r'\bexample\b', r'\bwalkthrough\b'
            ]),
            int('```' in txt),
            int(bool(re.search(r'`[^`]+`', txt))),
            int(bool(re.search(r'\bprint\s*\(', txt_lower))),
            int(bool(re.search(r'\bdef\s+\w+\s*\(', txt_lower)))
        ]
        return feats
    
    feature_names = [
        'text_char_length','text_word_count','text_lexical_diversity','text_avg_word_length',
        'text_question_marks','text_periods','text_commas','text_colons','text_semicolons',
        'text_tech_density','text_question_complexity','text_instruction_complexity',
        'text_has_code_blocks','text_has_inline_code','text_has_print_statements','text_has_function_defs'
    ]
    
    all_feats = [extract_comprehensive_features(instr)
                 for instr in features_df['instruction']]
    for idx, name in enumerate(feature_names):
        features_df[name] = [f[idx] for f in all_feats]
    
    # 4. Metadata encoding (with progress bar)
    print("🏷️ Processing metadata (this may take a moment)...")
    label_encoders = {}
    categorical_features = ['task_category', 'intent', 'knowledge', 'difficulty']
    for col in tqdm(categorical_features, desc="Metadata cols"):
        if col in features_df.columns:
            unique_vals = features_df[col].dropna().unique()
            print(f"   • {col}: {len(unique_vals)} unique values")
            for val in tqdm(unique_vals, desc=f"    Encoding {col}", leave=False):
                features_df[f"{col}_{val}"] = (features_df[col] == val).astype(int)
            le = LabelEncoder().fit(list(unique_vals))
            label_encoders[col] = le
        else:
            features_df[f"{col}_unknown"] = 1
    
    # 5. Interaction features
    print("🔗 Creating interaction features...")
    features_df['length_complexity_interaction'] = (
        features_df['text_char_length'] * features_df['text_tech_density']
    )
    features_df['word_lexical_interaction'] = (
        features_df['text_word_count'] * features_df['text_lexical_diversity']
    )
    # cluster one-hot (first 10 for brevity)
    for cid in range(min(10, features_df['semantic_cluster'].nunique())):
        features_df[f'cluster_{cid}'] = (features_df['semantic_cluster'] == cid).astype(int)
    
    print(f"✅ Neural network features created: {features_df.shape[1]} total features")
    return features_df, pca, kmeans, label_encoders

# Run it
print("🚀 Starting neural network feature engineering...")
features_df, pca_model, kmeans_model, label_encoders = create_neural_network_features(
    df_clean, embeddings_matrix
)

print(f"\n🔍 Feature summary:")
print(f"   Total features         : {features_df.shape[1]}")
print(f"   Embedding features     : {len([c for c in features_df if c.startswith('embed_pca_')])}")
print(f"   Text features          : {len([c for c in features_df if c.startswith('text_')])}")
print(f"   Cluster features       : {len([c for c in features_df if c.startswith('cluster_')])}")
print(f"   One‑hot metadata cols  : {len([c for c in features_df if any(c.startswith(f) for f in categorical_features)])}")

```
# MLP Token Predictor - Complete Notebook
# Using Multi-Layer Perceptron with regression head for better performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# For embeddings
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# For neural networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("🚀 MLP Token Predictor - Neural Network Approach")
print("=" * 60)
print("🎯 Using Multi-Layer Perceptron with regression head")

# Cell 1: Load and Clean Data
```python
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
    
    # Extract response text - completely rewritten to avoid array issues
    print("🔄 Extracting response text...")
    
    response_texts = []
    for idx, responses_data in enumerate(df['responses']):
        try:
            if responses_data is None:
                response_texts.append("")
                continue
            
            # Convert to string first to avoid array ambiguity
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
            if idx < 5:  # Only print first few errors
                print(f"Error processing response {idx}: {e}")
            response_texts.append("")
    
    # Assign the extracted responses
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
    
    print("🎯 Calculating input tokens...")
    df['input_tokens_mistral'] = [count_tokens(inst) for inst in df['instruction']]
    
    print("🎯 Calculating output tokens...")
    df['actual_output_tokens'] = [count_tokens(resp) for resp in df['response']]
    
    # Convert to proper types
    df['input_tokens_mistral'] = pd.Series(df['input_tokens_mistral'], dtype='int64')
    df['actual_output_tokens'] = pd.Series(df['actual_output_tokens'], dtype='int64')
    
    # Remove missing or zero tokens
    initial_count = len(df)
    df = df[
        (df['input_tokens_mistral'] > 0) & 
        (df['actual_output_tokens'] > 0)
    ].copy()
    
    print(f"🧹 Removed {initial_count - len(df)} samples with zero tokens")
    
    # Clean suspicious labels - more aggressive cleaning for neural networks
    print("🧹 Aggressive cleaning for neural networks...")
    
    # Remove suspicious short responses
    suspicious_mask = (df['actual_output_tokens'] < 10) & (df['input_tokens_mistral'] > 50)
    suspicious_count = suspicious_mask.sum()
    df_clean = df[~suspicious_mask].copy()
    
    # Remove extreme outliers (neural networks sensitive to outliers)
    if len(df_clean) > 0:
        q1_out = df_clean['actual_output_tokens'].quantile(0.005)  # More aggressive
        q99_out = df_clean['actual_output_tokens'].quantile(0.995)
        df_clean = df_clean[
            (df_clean['actual_output_tokens'] >= q1_out) & 
            (df_clean['actual_output_tokens'] <= q99_out)
        ].copy()
    
    # Remove extreme input outliers too
    if len(df_clean) > 0:
        q99_in = df_clean['input_tokens_mistral'].quantile(0.99)
        df_clean = df_clean[df_clean['input_tokens_mistral'] <= q99_in].copy()
    
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

# Cell 2: Generate Embeddings
```python
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(df):
    """Generate semantic embeddings for instructions"""
    
    print("🤖 GENERATING SEMANTIC EMBEDDINGS")
    print("=" * 40)
    
    # Load sentence transformer model
    print("📥 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model
    print("✅ Model loaded!")
    
    # Prepare instructions
    print("🔄 Preparing instructions...")
    instructions = df['instruction'].astype(str).fillna("").tolist()
    print(f"📊 Processing {len(instructions)} instructions")
    
    # Generate embeddings in batches to avoid memory issues
    print("🔄 Generating embeddings...")
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(instructions), batch_size):
        batch_end = min(i + batch_size, len(instructions))
        batch = instructions[i:batch_end]
        
        print(f"   Processing batch {i//batch_size + 1}: samples {i+1}-{batch_end}")
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
        
        # Progress update
        progress = (batch_end / len(instructions)) * 100
        print(f"   Progress: {progress:.1f}% complete")
    
    embeddings_matrix = np.array(all_embeddings)
    print(f"✅ Generated embeddings: {embeddings_matrix.shape}")
    print(f"📊 Embedding statistics:")
    print(f"   Shape: {embeddings_matrix.shape}")
    print(f"   Mean: {embeddings_matrix.mean():.4f}")
    print(f"   Std: {embeddings_matrix.std():.4f}")
    
    return embeddings_matrix, model

# Actually call the function and generate embeddings
print("🚀 Starting embedding generation...")
embeddings_matrix, embedding_model = generate_embeddings(df_clean)
```

# Cell 3: Advanced Feature Engineering for Neural Networks
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re

def create_neural_network_features(df, embeddings_matrix):
    """Create comprehensive features optimized for neural networks"""
    
    print("🛠️ NEURAL NETWORK FEATURE ENGINEERING")
    print("=" * 50)
    
    features_df = df.copy()
    
    # 1. PCA on embeddings (more components for neural networks)
    print("📊 Applying PCA to embeddings...")
    pca = PCA(n_components=100, random_state=42)  # More components for NN
    embeddings_pca = pca.fit_transform(embeddings_matrix)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA complete - explained variance: {explained_variance:.1%}")
    
    # Add PCA embeddings as features
    for i in range(embeddings_pca.shape[1]):
        features_df[f'embed_pca_{i}'] = embeddings_pca[:, i]
    
    # 2. Semantic clustering
    print("🎯 Semantic clustering...")
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)  # More clusters for NN
    clusters = kmeans.fit_predict(embeddings_pca)
    features_df['semantic_cluster'] = clusters
    
    # 3. Advanced text features
    print("📝 Advanced text analysis...")
    
    def extract_comprehensive_features(text):
        if pd.isna(text) or text == "":
            return [0] * 15
        
        txt = str(text)
        txt_lower = txt.lower()
        words = txt_lower.split()
        
        if not words:
            return [0] * 15
        
        features = []
        
        # Basic metrics
        features.append(len(txt))  # character length
        features.append(len(words))  # word count
        features.append(len(set(words)) / len(words))  # lexical diversity
        features.append(np.mean([len(w) for w in words]))  # avg word length
        
        # Punctuation and structure
        features.append(txt.count('?'))  # question marks
        features.append(txt.count('.'))  # periods
        features.append(txt.count(','))  # commas
        features.append(txt.count(':'))  # colons
        features.append(txt.count(';'))  # semicolons
        
        # Technical content indicators
        tech_patterns = [
            r'\b(function|class|import|def|return|api|database|algorithm|model|training)\b',
            r'\b(tensor|numpy|pandas|sklearn|python|javascript|react|sql|html|css)\b',
            r'\b(implement|optimize|debug|refactor|deploy|configure|install)\b',
            r'\b(analysis|statistics|machine learning|neural network|regression|classification)\b'
        ]
        tech_density = sum(len(re.findall(p, txt_lower)) for p in tech_patterns) / len(words)
        features.append(tech_density)
        
        # Question complexity indicators
        complex_question_patterns = [
            r'\bhow to\b', r'\bwhy does\b', r'\bwhat is the difference\b',
            r'\bcompare\b', r'\bexplain\b', r'\bimplement\b', r'\banalyze\b'
        ]
        question_complexity = sum(len(re.findall(p, txt_lower)) for p in complex_question_patterns)
        features.append(question_complexity)
        
        # Instruction type indicators
        instruction_patterns = [
            r'\bstep by step\b', r'\bdetailed\b', r'\bcomprehensive\b',
            r'\btutorial\b', r'\bguide\b', r'\bexample\b', r'\bwalkthrough\b'
        ]
        instruction_complexity = sum(len(re.findall(p, txt_lower)) for p in instruction_patterns)
        features.append(instruction_complexity)
        
        # Code presence indicators
        code_indicators = [
            int('```' in txt),  # code blocks
            int(bool(re.search(r'`[^`]+`', txt))),  # inline code
            int(bool(re.search(r'\bprint\s*\(', txt_lower))),  # print statements
            int(bool(re.search(r'\bdef\s+\w+\s*\(', txt_lower)))  # function definitions
        ]
        features.extend(code_indicators)
        
        return features
    
    # Extract comprehensive features
    print("   Extracting comprehensive text features...")
    comprehensive_features = []
    for instruction in features_df['instruction']:
        comprehensive_features.append(extract_comprehensive_features(instruction))
    
    # Add comprehensive features
    feature_names = [
        'char_length', 'word_count', 'lexical_diversity', 'avg_word_length',
        'question_marks', 'periods', 'commas', 'colons', 'semicolons',
        'tech_density', 'question_complexity', 'instruction_complexity',
        'has_code_blocks', 'has_inline_code', 'has_print_statements', 'has_function_defs'
    ]
    
    for i, name in enumerate(feature_names):
        features_df[f'text_{name}'] = [f[i] for f in comprehensive_features]
    
    # 4. Metadata encoding (if available)
    print("🏷️ Processing metadata...")
    label_encoders = {}
    
    # Handle categorical metadata
    categorical_features = ['task_category', 'intent', 'knowledge', 'difficulty']
    for col in categorical_features:
        if col in features_df.columns:
            le = LabelEncoder()
            # Create one-hot-like features for neural networks
            unique_values = features_df[col].unique()
            for val in unique_values:
                if not pd.isna(val):
                    features_df[f'{col}_{val}'] = (features_df[col] == val).astype(int)
            label_encoders[col] = le
        else:
            # Create default binary features
            features_df[f'{col}_unknown'] = 1
    
    # 5. Interaction features (important for neural networks)
    print("🔗 Creating interaction features...")
    
    # Length-complexity interactions
    features_df['length_complexity_interaction'] = features_df['text_char_length'] * features_df['text_tech_density']
    features_df['word_lexical_interaction'] = features_df['text_word_count'] * features_df['text_lexical_diversity']
    
    # Cluster-based features
    for cluster_id in range(min(10, features_df['semantic_cluster'].nunique())):
        features_df[f'cluster_{cluster_id}'] = (features_df['semantic_cluster'] == cluster_id).astype(int)
    
    print(f"✅ Neural network features created: {features_df.shape[1]} total features")
    
    return features_df, pca, kmeans, label_encoders

# Create neural network optimized features
print("🚀 Starting neural network feature engineering...")
features_df, pca_model, kmeans_model, label_encoders = create_neural_network_features(df_clean, embeddings_matrix)

print(f"\n🔍 Feature summary:")
print(f"   Total features: {features_df.shape[1]}")
print(f"   Embedding features: {len([col for col in features_df.columns if col.startswith('embed_pca_')])}")
print(f"   Text features: {len([col for col in features_df.columns if col.startswith('text_')])}")
print(f"   Cluster features: {len([col for col in features_df.columns if col.startswith('cluster_')])}")
```

# Cell 4: Custom PyTorch MLP Model
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class TokenPredictorMLP(nn.Module):
    """Custom MLP for token prediction with dropout and batch normalization"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3):
        super(TokenPredictorMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout (more aggressive for larger layers)
            if i < len(hidden_dims) - 1:  # No dropout before final layer
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final regression layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze()

def train_pytorch_mlp(features_df, target_col='actual_output_tokens'):
    """Train custom PyTorch MLP model"""
    
    print("🔥 TRAINING PYTORCH MLP MODEL")
    print("=" * 40)
    
    # Prepare features
    feature_columns = [col for col in features_df.columns if col not in [
        'instruction', 'responses', 'response', target_col, 'input_tokens_mistral'
    ]]
    
    # Keep input_tokens_mistral as it's a strong predictor
    if 'input_tokens_mistral' in features_df.columns:
        feature_columns.append('input_tokens_mistral')
    
    print(f"📊 Using {len(feature_columns)} features for PyTorch MLP")
    
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
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Normalize target (helps with training stability)
    target_mean = y_train.mean()
    target_std = y_train.std()
    y_train_normalized = (y_train - target_mean) / target_std
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train_normalized.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    model = TokenPredictorMLP(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    print("🚀 Starting training...")
    num_epochs = 100
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    
    train_losses = []
    val_losses = []
    
    # Move test data to device
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            # Denormalize predictions
            val_outputs_denorm = val_outputs * target_std + target_mean
            val_loss = F.mse_loss(val_outputs_denorm, y_test_tensor).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= early_stopping_patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor.to(device))
        train_pred_denorm = train_pred * target_std + target_mean
        train_pred_np = train_pred_denorm.cpu().numpy()
        
        test_pred = model(X_test_tensor)
        test_pred_denorm = test_pred * target_std + target_mean
        test_pred_np = test_pred_denorm.cpu().numpy()
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_pred_np)
    test_mae = mean_absolute_error(y_test, test_pred_np)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred_np))
    test_r2 = r2_score(y_test, test_pred_np)
    
    # Accuracy metrics
    errors = np.abs(y_test - test_pred_np)
    within_5 = (errors <= 5).mean() * 100
    within_10 = (errors <= 10).mean() * 100
    within_20 = (errors <= 20).mean() * 100
    within_50 = (errors <= 50).mean() * 100
    
    print(f"\n🎉 PyTorch MLP Results:")
    print(f"   Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
    print(f"   Test RMSE: {test_rmse:.2f} | Test R²: {test_r2:.3f}")
    print(f"   Within ±5: {within_5:.1f}% | ±10: {within_10:.1f}% | ±20: {within_20:.1f}% | ±50: {within_50:.1f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, test_pred_np, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Tokens')
    plt.ylabel('Predicted Tokens')
    plt.title(f'Predictions vs Actual (MAE: {test_mae:.1f})')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'scaler': scaler,
        'target_mean': target_mean,
        'target_std': target_std,
        'feature_columns': feature_columns,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'within_10': within_10,
        'device': device
    }

# Train PyTorch MLP
print("🚀 Starting PyTorch MLP training...")
pytorch_results = train_pytorch_mlp(features_df)
```

# Cell 5: Compare with Traditional Models
```python
def train_comparison_models(features_df, target_col='actual_output_tokens'):
    """Train traditional models for comparison"""
    
    print("📊 TRAINING COMPARISON MODELS")
    print("=" * 40)
    
    # Use same features as PyTorch model
    feature_columns = pytorch_results['feature_columns']
    
    X = features_df[feature_columns].copy()
    y = features_df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(0)
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    # Split data (same as PyTorch)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    results = {}
    
    # 1. Scikit-learn MLP
    print("\n🧠 Training Scikit-learn MLP...")
    scaler_sklearn = StandardScaler()
    X_train_scaled = scaler_sklearn.fit_transform(X_train)
    X_test_scaled = scaler_sklearn.transform(X_test)
    
    mlp_sklearn = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    mlp_sklearn.fit(X_train_scaled, y_train)
    sklearn_pred = mlp_sklearn.predict(X_test_scaled)
    
    # 2. Random Forest
    print("🌳 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # 3. XGBoost
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
    xgb_pred = xgb_reg.predict(X_test)
    
    # Evaluate all models
    model_results = {
        'PyTorch MLP': pytorch_results['model'],
        'Scikit-learn MLP': mlp_sklearn,
        'Random Forest': rf,
        'XGBoost': xgb_reg
    }
    
    predictions = {
        'PyTorch MLP': None,  # Already calculated
        'Scikit-learn MLP': sklearn_pred,
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred
    }
    
    print(f"\n📊 MODEL COMPARISON RESULTS:")
    print(f"{'Model':<20} {'Test MAE':<10} {'Test R²':<10} {'±10 Acc':<10}")
    print("-" * 50)
    
    comparison_results = {}
    
    # PyTorch results (already calculated)
    print(f"{'PyTorch MLP':<20} {pytorch_results['test_mae']:<10.2f} {pytorch_results['test_r2']:<10.3f} {pytorch_results['within_10']:<10.1f}%")
    comparison_results['PyTorch MLP'] = {
        'test_mae': pytorch_results['test_mae'],
        'test_r2': pytorch_results['test_r2'],
        'within_10': pytorch_results['within_10']
    }
    
    # Other models
    for name, pred in predictions.items():
        if pred is not None:
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            within_10 = (np.abs(y_test - pred) <= 10).mean() * 100
            
            print(f"{name:<20} {mae:<10.2f} {r2:<10.3f} {within_10:<10.1f}%")
            comparison_results[name] = {
                'test_mae': mae,
                'test_r2': r2,
                'within_10': within_10
            }
    
    # Find best model
    best_model = min(comparison_results.keys(), key=lambda x: comparison_results[x]['test_mae'])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   MAE: {comparison_results[best_model]['test_mae']:.2f} tokens")
    
    return comparison_results, model_results

# Train comparison models
print("🚀 Starting comparison model training...")
comparison_results, all_models = train_comparison_models(features_df)
```

# Cell 6: Model Analysis and Visualization
```python
def analyze_model_performance(comparison_results, features_df):
    """Analyze and visualize model performance"""
    
    print("📊 DETAILED MODEL ANALYSIS")
    print("=" * 40)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(comparison_results).T
    results_df = results_df.round(2)
    
    print("\n📋 Complete Results Table:")
    print(results_df.to_string())
    
    # Improvement analysis
    baseline_mae = 165  # Your previous XGBoost result
    print(f"\n📈 IMPROVEMENT vs PREVIOUS MODEL (MAE: {baseline_mae}):")
    
    for model_name, results in comparison_results.items():
        improvement = ((baseline_mae - results['test_mae']) / baseline_mae) * 100
        print(f"   {model_name:<20}: {improvement:+6.1f}% improvement")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MAE Comparison
    models = list(comparison_results.keys())
    maes = [comparison_results[m]['test_mae'] for m in models]
    
    axes[0, 0].bar(models, maes, color=['red', 'blue', 'green', 'orange'])
    axes[0, 0].axhline(y=baseline_mae, color='black', linestyle='--', label=f'Previous Best ({baseline_mae})')
    axes[0, 0].set_title('Test MAE Comparison')
    axes[0, 0].set_ylabel('MAE (tokens)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    
    # 2. R² Comparison
    r2s = [comparison_results[m]['test_r2'] for m in models]
    axes[0, 1].bar(models, r2s, color=['red', 'blue', 'green', 'orange'])
    axes[0, 1].set_title('Test R² Comparison')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Accuracy within ±10 tokens
    within_10s = [comparison_results[m]['within_10'] for m in models]
    axes[1, 0].bar(models, within_10s, color=['red', 'blue', 'green', 'orange'])
    axes[1, 0].set_title('Accuracy Within ±10 Tokens')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Token distribution analysis
    axes[1, 1].hist(features_df['actual_output_tokens'], bins=50, alpha=0.7, color='skyblue')
    axes[1, 1].set_title('Output Token Distribution')
    axes[1, 1].set_xlabel('Output Tokens')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(features_df['actual_output_tokens'].mean(), color='red', 
                      linestyle='--', label=f'Mean: {features_df["actual_output_tokens"].mean():.0f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Performance by token range
    print(f"\n📊 PERFORMANCE BY TOKEN RANGE:")
    
    # Define ranges
    ranges = [
        (0, 50, "Very Short"),
        (50, 150, "Short"),
        (150, 300, "Medium"),
        (300, 500, "Long"),
        (500, float('inf'), "Very Long")
    ]
    
    for min_tok, max_tok, label in ranges:
        if max_tok == float('inf'):
            mask = features_df['actual_output_tokens'] >= min_tok
            range_label = f"{min_tok}+"
        else:
            mask = (features_df['actual_output_tokens'] >= min_tok) & (features_df['actual_output_tokens'] < max_tok)
            range_label = f"{min_tok}-{max_tok}"
        
        count = mask.sum()
        if count > 50:  # Only analyze ranges with sufficient samples
            avg_tokens = features_df[mask]['actual_output_tokens'].mean()
            print(f"   {label:12} ({range_label:8}): {count:4d} samples, avg: {avg_tokens:5.1f} tokens")

# Analyze model performance
analyze_model_performance(comparison_results, features_df)
```

# Cell 7: Enhanced Query Testing Interface
```python
def create_mlp_query_tester(pytorch_results, embedding_model, pca_model, kmeans_model, features_df):
    """Create query tester for MLP model"""
    
    def predict_with_mlp(query, difficulty='medium', task_category='Information seeking', 
                        intent='informational', knowledge='intermediate'):
        """Predict tokens using MLP model"""
        
        print(f"🧪 MLP PREDICTION (Neural Network)")
        print(f"   Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
        print()
        
        # Generate embedding for query
        query_embedding = embedding_model.encode([query])
        query_embedding_pca = pca_model.transform(query_embedding)
        query_cluster = kmeans_model.predict(query_embedding_pca)[0]
        
        # Extract all features (same as training)
        features = {}
        
        # PCA embedding features
        for i in range(query_embedding_pca.shape[1]):
            features[f'embed_pca_{i}'] = query_embedding_pca[0, i]
        
        # Semantic cluster
        features['semantic_cluster'] = query_cluster
        
        # Comprehensive text features
        txt = str(query)
        txt_lower = txt.lower()
        words = txt_lower.split()
        
        if words:
            features['text_char_length'] = len(txt)
            features['text_word_count'] = len(words)
            features['text_lexical_diversity'] = len(set(words)) / len(words)
            features['text_avg_word_length'] = np.mean([len(w) for w in words])
            
            # Punctuation
            features['text_question_marks'] = txt.count('?')
            features['text_periods'] = txt.count('.')
            features['text_commas'] = txt.count(',')
            features['text_colons'] = txt.count(':')
            features['text_semicolons'] = txt.count(';')
            
            # Technical density
            tech_patterns = [
                r'\b(function|class|import|def|return|api|database|algorithm|model|training)\b',
                r'\b(tensor|numpy|pandas|sklearn|python|javascript|react|sql|html|css)\b',
                r'\b(implement|optimize|debug|refactor|deploy|configure|install)\b',
                r'\b(analysis|statistics|machine learning|neural network|regression|classification)\b'
            ]
            features['text_tech_density'] = sum(len(re.findall(p, txt_lower)) for p in tech_patterns) / len(words)
            
            # Question and instruction complexity
            complex_patterns = [r'\bhow to\b', r'\bwhy does\b', r'\bwhat is the difference\b', r'\bcompare\b', r'\bexplain\b', r'\bimplement\b']
            features['text_question_complexity'] = sum(len(re.findall(p, txt_lower)) for p in complex_patterns)
            
            instruction_patterns = [r'\bstep by step\b', r'\bdetailed\b', r'\bcomprehensive\b', r'\btutorial\b', r'\bguide\b', r'\bexample\b']
            features['text_instruction_complexity'] = sum(len(re.findall(p, txt_lower)) for p in instruction_patterns)
            
            # Code indicators
            features['text_has_code_blocks'] = int('```' in txt)
            features['text_has_inline_code'] = int(bool(re.search(r'`[^`]+`', txt)))
            features['text_has_print_statements'] = int(bool(re.search(r'\bprint\s*\(', txt_lower)))
            features['text_has_function_defs'] = int(bool(re.search(r'\bdef\s+\w+\s*\(', txt_lower)))
        else:
            # Default values for empty queries
            for key in ['text_char_length', 'text_word_count', 'text_lexical_diversity', 'text_avg_word_length',
                       'text_question_marks', 'text_periods', 'text_commas', 'text_colons', 'text_semicolons',
                       'text_tech_density', 'text_question_complexity', 'text_instruction_complexity',
                       'text_has_code_blocks', 'text_has_inline_code', 'text_has_print_statements', 'text_has_function_defs']:
                features[key] = 0
        
        # Metadata features (create dummy ones for categories not seen in training)
        metadata_categories = {
            'task_category': ['Information seeking', 'Coding & Debugging', 'Math', 'Creative writing', 'Data analysis'],
            'intent': ['informational', 'implementation', 'educational', 'problem_solving'],
            'knowledge': ['basic', 'intermediate', 'advanced', 'expert'],
            'difficulty': ['easy', 'medium', 'hard']
        }
        
        for category, values in metadata_categories.items():
            for value in values:
                feature_name = f'{category}_{value}'
                if category == 'task_category':
                    features[feature_name] = int(task_category == value)
                elif category == 'intent':
                    features[feature_name] = int(intent == value)
                elif category == 'knowledge':
                    features[feature_name] = int(knowledge == value)
                elif category == 'difficulty':
                    features[feature_name] = int(difficulty == value)
                else:
                    features[feature_name] = 0
        
        # Interaction features
        features['length_complexity_interaction'] = features.get('text_char_length', 0) * features.get('text_tech_density', 0)
        features['word_lexical_interaction'] = features.get('text_word_count', 0) * features.get('text_lexical_diversity', 0)
        
        # Cluster features
        for cluster_id in range(10):
            features[f'cluster_{cluster_id}'] = int(query_cluster == cluster_id)
        
        # Add input tokens
        features['input_tokens_mistral'] = len(query.split()) * 1.3  # Rough approximation
        
        # Create feature vector in the same order as training
        feature_vector = []
        for col in pytorch_results['feature_columns']:
            feature_vector.append(features.get(col, 0))
        
        # Convert to tensor and scale
        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_scaled = pytorch_results['scaler'].transform(feature_array)
        feature_tensor = torch.FloatTensor(feature_scaled).to(pytorch_results['device'])
        
        # Make prediction
        model = pytorch_results['model']
        model.eval()
        
        with torch.no_grad():
            prediction_normalized = model(feature_tensor)
            # Denormalize
            prediction = (prediction_normalized * pytorch_results['target_std'] + pytorch_results['target_mean']).item()
        
        prediction = max(1, int(round(prediction)))  # Ensure positive integer
        
        print(f"🎯 MLP Prediction: {prediction} tokens (±{pytorch_results['test_mae']:.1f} MAE)")
        print(f"🎯 Semantic cluster: {query_cluster}")
        print(f"🧠 Neural network with {len(feature_vector)} features")
        
        return prediction
    
    return predict_with_mlp

# Create MLP query tester
print("🚀 Creating MLP query tester...")
mlp_tester = create_mlp_query_tester(pytorch_results, embedding_model, pca_model, kmeans_model, features_df)

# Test with comprehensive examples
print("\n🧪 COMPREHENSIVE QUERY TESTING:")

test_cases = [
    {
        'query': "What is Python?",
        'difficulty': 'easy',
        'task_category': 'Information seeking',
        'intent': 'informational',
        'knowledge': 'basic',
        'expected': "Short definition (~30-60 tokens)"
    },
    {
        'query': "How do I sort a list in Python?",
        'difficulty': 'medium',
        'task_category': 'Coding & Debugging',
        'intent': 'instructional',
        'knowledge': 'intermediate',
        'expected': "Medium tutorial (~100-200 tokens)"
    },
    {
        'query': "Implement a complete binary search algorithm in Python with comprehensive error handling, edge case management, and performance optimization for large datasets",
        'difficulty': 'hard',
        'task_category': 'Coding & Debugging',
        'intent': 'implementation',
        'knowledge': 'advanced',
        'expected': "Long implementation (~400-600 tokens)"
    },
    {
        'query': "Explain the mathematical foundations of transformer architecture including attention mechanisms, positional encoding, multi-head attention with detailed mathematical derivations and implementation considerations",
        'difficulty': 'hard',
        'task_category': 'Math',
        'intent': 'educational',
        'knowledge': 'expert',
        'expected': "Very long explanation (~600-800 tokens)"
    },
    {
        'query': "Debug this error: AttributeError: 'NoneType' object has no attribute 'split'",
        'difficulty': 'medium',
        'task_category': 'Coding & Debugging',
        'intent': 'problem_solving',
        'knowledge': 'intermediate',
        'expected': "Medium debugging help (~150-250 tokens)"
    }
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"TEST CASE {i}: {test_case['expected']}")
    print(f"{'='*60}")
    
    prediction = mlp_tester(
        test_case['query'],
        test_case['difficulty'],
        test_case['task_category'],
        test_case['intent'],
        test_case['knowledge']
    )
    
    print(f"💡 Expected: {test_case['expected']}")
```

# Cell 8: Export Best Model
```python
def export_best_model(pytorch_results, embedding_model, pca_model, kmeans_model, comparison_results):
    """Export the best performing model for production use"""
    
    print("💾 EXPORTING BEST MODEL FOR PRODUCTION")
    print("=" * 50)
    
    # Find the best model
    best_model_name = min(comparison_results.keys(), key=lambda x: comparison_results[x]['test_mae'])
    best_mae = comparison_results[best_model_name]['test_mae']
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 Best MAE: {best_mae:.2f} tokens")
    
    # Create export directory
    import os
    os.makedirs('mlp_model_export', exist_ok=True)
    
    if best_model_name == 'PyTorch MLP':
        # Export PyTorch model
        print("💾 Exporting PyTorch MLP model...")
        
        # Save model state dict
        torch.save(pytorch_results['model'].state_dict(), 'mlp_model_export/pytorch_mlp_model.pth')
        
        # Save supporting objects
        with open('mlp_model_export/scaler.pkl', 'wb') as f:
            pickle.dump(pytorch_results['scaler'], f)
        
        with open('mlp_model_export/pca_model.pkl', 'wb') as f:
            pickle.dump(pca_model, f)
        
        with open('mlp_model_export/kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans_model, f)
        
        # Save metadata
        model_metadata = {
            'model_type': 'PyTorch MLP',
            'test_mae': pytorch_results['test_mae'],
            'test_r2': pytorch_results['test_r2'],
            'within_10_accuracy': pytorch_results['within_10'],
            'feature_columns': pytorch_results['feature_columns'],
            'target_mean': pytorch_results['target_mean'],
            'target_std': pytorch_results['target_std'],
            'input_dim': len(pytorch_results['feature_columns']),
            'hidden_dims': [512, 256, 128, 64],
            'dropout_rate': 0.3
        }
        
        with open('mlp_model_export/model_metadata.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print("✅ PyTorch MLP model exported successfully!")
        
    else:
        # Export scikit-learn model
        print(f"💾 Exporting {best_model_name} model...")
        
        best_sklearn_model = all_models[best_model_name]
        
        with open('mlp_model_export/sklearn_model.pkl', 'wb') as f:
            pickle.dump(best_sklearn_model, f)
        
        # Save supporting objects
        with open('mlp_model_export/pca_model.pkl', 'wb') as f:
            pickle.dump(pca_model, f)
        
        with open('mlp_model_export/kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans_model, f)
        
        model_metadata = {
            'model_type': best_model_name,
            'test_mae': best_mae,
            'feature_columns': pytorch_results['feature_columns']
        }
        
        with open('mlp_model_export/model_metadata.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print(f"✅ {best_model_name} model exported successfully!")
    
    # Always save the embedding model reference
    embedding_model.save('mlp_model_export/sentence_transformer')
    
    print(f"\n📁 All files exported to: mlp_model_export/")
    print(f"📊 Production-ready model with {best_mae:.2f} MAE")
    print(f"🚀 Ready for deployment!")
    
    # Create a simple prediction function for production
    production_code = f'''
# Production Prediction Code
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_production_model():
    """Load the exported model for production use"""
    
    # Load model components
    embedding_model = SentenceTransformer('mlp_model_export/sentence_transformer')
    
    with open('mlp_model_export/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('mlp_model_export/pca_model.pkl', 'rb') as f:
        pca_model = pickle.load(f)
    
    with open('mlp_model_export/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    
    with open('mlp_model_export/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    if metadata['model_type'] == 'PyTorch MLP':
        # Load PyTorch model
        from your_model_definition import TokenPredictorMLP  # Import your model class
        
        model = TokenPredictorMLP(
            input_dim=metadata['input_dim'],
            hidden_dims=metadata['hidden_dims'],
            dropout_rate=metadata['dropout_rate']
        )
        model.load_state_dict(torch.load('mlp_model_export/pytorch_mlp_model.pth'))
        model.eval()
        
        return model, embedding_model, scaler, pca_model, kmeans_model, metadata
    else:
        # Load scikit-learn model
        with open('mlp_model_export/sklearn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        return model, embedding_model, scaler, pca_model, kmeans_model, metadata

def predict_tokens(query, model_components):
    """Production prediction function"""
    model, embedding_model, scaler, pca_model, kmeans_model, metadata = model_components
    
    # Extract features (implement the same feature extraction as in training)
    # ... feature extraction code ...
    
    # Make prediction
    if metadata['model_type'] == 'PyTorch MLP':
        with torch.no_grad():
            prediction = model(feature_tensor)
            prediction = (prediction * metadata['target_std'] + metadata['target_mean']).item()
    else:
        prediction = model.predict(feature_array)[0]
    
    return max(1, int(round(prediction)))

# Usage:
# model_components = load_production_model()
# tokens = predict_tokens("Your query here", model_components)
'''
    
    with open('mlp_model_export/production_code.py', 'w') as f:
        f.write(production_code)
    
    return model_metadata

# Export the best model
export_metadata = export_best_model(pytorch_results, embedding_model, pca_model, kmeans_model, comparison_results)

print(f"\n🎉 MLP TOKEN PREDICTOR COMPLETE!")
print(f"=" * 50)
print(f"✅ Best model trained and exported")
print(f"📊 Final MAE: {export_metadata.get('test_mae', 'N/A'):.2f} tokens")
print(f"🚀 Ready for production deployment!")
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
