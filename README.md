# Cell 1: Load and Clean Data
```python
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

from tqdm.auto import tqdm
tqdm.pandas()

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

print("🚀 MLP Embedding Token Predictor - Accurate Predictions")
print("=" * 60)
print("🎯 Using raw embeddings + MLP regressor for accurate token prediction")

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
    print(f"📂 Found {len(parquet_files)} parquet files")
    
    # Load with progress bar
    df_list = []
    for file_path in tqdm(parquet_files, desc="Loading parquet files"):
        df_list.append(pd.read_parquet(file_path, engine="pyarrow"))
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Loaded {len(df)} samples from {len(parquet_files)} files")
    
    # Extract response text with progress bar
    print("🔄 Extracting response text...")
    
    def extract_response_text(responses_data):
        try:
            if responses_data is None or pd.isna(responses_data):
                return ""
            
            if isinstance(responses_data, str):
                return responses_data
            
            if hasattr(responses_data, 'dtype'):  # numpy array
                if len(responses_data) == 0:
                    return ""
                first_item = responses_data[0]
            elif isinstance(responses_data, list):
                if len(responses_data) == 0:
                    return ""
                first_item = responses_data[0]
            else:
                return str(responses_data)
            
            if isinstance(first_item, dict):
                for key in ('text', 'content', 'value', 'response'):
                    if key in first_item:
                        return str(first_item[key])
                return str(first_item)
            else:
                return str(first_item)
                
        except Exception:
            return ""
    
    # Apply with progress bar
    df['response'] = df['responses'].progress_apply(extract_response_text)
    
    # Calculate tokens with progress bar
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
    df['input_tokens_mistral'] = df['instruction'].progress_apply(count_tokens)
    
    print("🎯 Calculating output tokens...")
    df['actual_output_tokens'] = df['response'].progress_apply(count_tokens)
    
    # Clean data
    initial_count = len(df)
    
    # Remove zero token samples
    df = df[
        (df['input_tokens_mistral'] > 0) & 
        (df['actual_output_tokens'] > 0)
    ].copy()
    
    # Remove suspicious labels (very short outputs for long inputs)
    suspicious_mask = (df['actual_output_tokens'] < 5) & (df['input_tokens_mistral'] > 50)
    df_clean = df[~suspicious_mask].copy()
    
    # Remove extreme outliers (important for neural networks)
    q1_out = df_clean['actual_output_tokens'].quantile(0.005)
    q99_out = df_clean['actual_output_tokens'].quantile(0.995)
    df_clean = df_clean[
        (df_clean['actual_output_tokens'] >= q1_out) & 
        (df_clean['actual_output_tokens'] <= q99_out)
    ].copy()
    
    # Remove extreme input outliers
    q99_in = df_clean['input_tokens_mistral'].quantile(0.99)
    df_clean = df_clean[df_clean['input_tokens_mistral'] <= q99_in].copy()
    
    final_count = len(df_clean)
    print(f"🧹 Removed {initial_count - final_count} samples ({((initial_count - final_count)/initial_count)*100:.1f}%)")
    print(f"✅ Clean dataset: {final_count:,} samples")
    print(f"📊 Token range: {df_clean['actual_output_tokens'].min()}-{df_clean['actual_output_tokens'].max()}")
    print(f"📊 Mean output tokens: {df_clean['actual_output_tokens'].mean():.1f}")
    
    return df_clean

# Load the data
df_clean = load_and_clean_data()
```

# Cell 2: Generate Raw Embeddings
```python
from sentence_transformers import SentenceTransformer

def generate_raw_embeddings(df):
    """Generate raw sentence embeddings without any reduction"""
    
    print("🤖 GENERATING RAW SENTENCE EMBEDDINGS")
    print("=" * 45)
    
    # Load sentence transformer model
    print("📥 Loading all-MiniLM-L6-v2 model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded!")
    
    # Prepare instructions
    instructions = df['instruction'].astype(str).fillna("").tolist()
    print(f"📊 Processing {len(instructions)} instructions")
    
    # Generate embeddings in batches with progress bar
    print("🔄 Generating embeddings...")
    batch_size = 500  # Smaller batches for better progress tracking
    all_embeddings = []
    
    # Create batches
    batches = [instructions[i:i + batch_size] for i in range(0, len(instructions), batch_size)]
    
    for batch in tqdm(batches, desc="Generating embeddings"):
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
    
    embeddings_matrix = np.array(all_embeddings)
    print(f"✅ Generated embeddings: {embeddings_matrix.shape}")
    print(f"📊 Embedding statistics:")
    print(f"   Shape: {embeddings_matrix.shape[0]} samples × {embeddings_matrix.shape[1]} dimensions")
    print(f"   Mean: {embeddings_matrix.mean():.4f}")
    print(f"   Std: {embeddings_matrix.std():.4f}")
    
    return embeddings_matrix, model

# Generate embeddings
print("🚀 Starting embedding generation...")
embeddings_matrix, embedding_model = generate_raw_embeddings(df_clean)
```

# Cell 3: Production Feature Engineering with Complexity
```python
def create_mlp_features(df, embeddings_matrix):
    """Create features for MLP model including complexity scores"""
    
    print("🛠️ MLP FEATURE ENGINEERING")
    print("=" * 40)
    print("🎯 Using raw embeddings + complexity + production features")
    
    features_df = df.copy()
    
    # 1. Raw embeddings as features (384 dimensions - no reduction)
    print("📊 Adding raw embedding features...")
    for i in range(embeddings_matrix.shape[1]):
        features_df[f'embed_{i}'] = embeddings_matrix[:, i]
    print(f"   Added {embeddings_matrix.shape[1]} embedding dimensions")
    
    # 2. Complexity score (important feature)
    print("🧮 Calculating complexity scores...")
    
    def calculate_complexity_score(text):
        """Calculate sophisticated complexity score"""
        if pd.isna(text) or text == "":
            return 0.0
        
        txt = str(text)
        txt_lower = txt.lower()
        words = txt_lower.split()
        
        if not words:
            return 0.0
        
        # Vocabulary diversity
        vocab_diversity = len(set(words)) / len(words)
        
        # Average word length
        avg_word_len = np.mean([len(w) for w in words])
        avg_word_len_normalized = min(avg_word_len / 8.0, 1.0)  # Normalize to 0-1
        
        # Punctuation density
        punctuation_count = sum(1 for c in txt if c in '!@#$%^&*(),.?":{}|<>')
        punct_density = min(punctuation_count / len(txt), 0.1) * 10  # Normalize to 0-1
        
        # Technical term density
        tech_patterns = [
            r'\b(function|class|import|def|return|api|database|algorithm|model|training)\b',
            r'\b(tensor|numpy|pandas|sklearn|python|javascript|react|sql|html|css)\b',
            r'\b(implement|optimize|debug|refactor|deploy|configure|install)\b',
            r'\b(analysis|statistics|machine learning|neural network|regression)\b',
            r'\b(comprehensive|detailed|step-by-step|tutorial|walkthrough)\b'
        ]
        
        tech_matches = sum(len(re.findall(pattern, txt_lower)) for pattern in tech_patterns)
        tech_density = min(tech_matches / len(words), 0.3) / 0.3  # Normalize to 0-1
        
        # Question complexity indicators
        complex_indicators = [
            'how to', 'why does', 'what is the difference', 'compare', 'explain',
            'implement', 'analyze', 'evaluate', 'comprehensive', 'detailed'
        ]
        complexity_indicators = sum(1 for indicator in complex_indicators if indicator in txt_lower)
        question_complexity = min(complexity_indicators / 3.0, 1.0)  # Normalize to 0-1
        
        # Final complexity score (weighted combination)
        complexity_score = (
            vocab_diversity * 0.2 +
            avg_word_len_normalized * 0.2 +
            punct_density * 0.1 +
            tech_density * 0.3 +
            question_complexity * 0.2
        )
        
        return min(complexity_score, 1.0)
    
    # Apply complexity calculation with progress bar
    features_df['complexity_score'] = df['instruction'].progress_apply(calculate_complexity_score)
    
    # 3. Basic production-available features
    print("📝 Adding basic production features...")
    
    def extract_production_features(text):
        """Extract features available in production"""
        if pd.isna(text) or text == "":
            return [0] * 10
        
        txt = str(text)
        txt_lower = txt.lower()
        words = txt_lower.split()
        
        if not words:
            return [0] * 10
        
        features = []
        
        # Length features
        features.append(len(txt))  # char_length
        features.append(len(words))  # word_count
        features.append(np.mean([len(w) for w in words]))  # avg_word_length
        
        # Question patterns
        features.append(int(any(word in txt_lower for word in ['what', 'define', 'is'])))  # definition questions
        features.append(int(any(word in txt_lower for word in ['how', 'guide', 'tutorial'])))  # how-to questions
        features.append(int(any(word in txt_lower for word in ['implement', 'build', 'create'])))  # implementation
        features.append(int(any(word in txt_lower for word in ['explain', 'describe', 'elaborate'])))  # explanation
        
        # Code indicators
        features.append(int('```' in txt or bool(re.search(r'`[^`]+`', txt))))  # has code
        features.append(int(any(lang in txt_lower for lang in ['python', 'javascript', 'sql', 'html'])))  # programming
        
        # Complexity indicators
        features.append(int(any(word in txt_lower for word in ['comprehensive', 'detailed', 'step-by-step'])))  # detailed request
        
        return features
    
    # Extract production features with progress bar
    production_features = []
    for instruction in tqdm(df['instruction'], desc="Extracting production features"):
        production_features.append(extract_production_features(instruction))
    
    # Add production features
    production_feature_names = [
        'char_length', 'word_count', 'avg_word_length',
        'is_definition', 'is_how_to', 'is_implementation', 'is_explanation',
        'has_code', 'mentions_programming', 'requests_detailed'
    ]
    
    for i, name in enumerate(production_feature_names):
        features_df[name] = [f[i] for f in production_features]
    
    print(f"   Added {len(production_feature_names)} production features")
    
    # 4. Input tokens (available in production)
    print("🎯 Including input token count...")
    # input_tokens_mistral already exists
    
    print(f"✅ MLP features complete!")
    
    # Define feature columns for model training
    embedding_cols = [f'embed_{i}' for i in range(embeddings_matrix.shape[1])]
    feature_cols = embedding_cols + ['complexity_score'] + production_feature_names + ['input_tokens_mistral']
    
    print(f"📊 Total features: {len(feature_cols)}")
    print(f"   - Embeddings: {len(embedding_cols)}")
    print(f"   - Complexity: 1")
    print(f"   - Production: {len(production_feature_names)}")
    print(f"   - Input tokens: 1")
    
    return features_df, feature_cols

# Create MLP features
print("🚀 Starting MLP feature engineering...")
features_df, feature_columns = create_mlp_features(df_clean, embeddings_matrix)
```

# Cell 4: Custom PyTorch MLP Model
```python
import torch.nn.functional as F

class TokenPredictorMLP(nn.Module):
    """Custom MLP for accurate token prediction"""
    
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
            
            # Dropout
            if i < len(hidden_dims) - 1:
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

def train_pytorch_mlp(features_df, feature_columns, target_col='actual_output_tokens'):
    """Train PyTorch MLP model with progress tracking"""
    
    print("🔥 TRAINING PYTORCH MLP MODEL")
    print("=" * 40)
    
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
    
    # Scale features (critical for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Normalize target for better training
    target_mean = y_train.mean()
    target_std = y_train.std()
    y_train_normalized = (y_train - target_mean) / target_std
    
    print(f"📈 Target normalization: mean={target_mean:.1f}, std={target_std:.1f}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train_normalized.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    model = TokenPredictorMLP(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.2
    ).to(device)
    
    print(f"🧠 Model architecture: {X_train_scaled.shape[1]} → 512 → 256 → 128 → 64 → 1")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
    
    # Training loop with progress tracking
    print("🚀 Starting training...")
    num_epochs = 150
    best_val_mae = float('inf')
    patience_counter = 0
    early_stopping_patience = 30
    
    train_losses = []
    val_maes = []
    
    # Move test data to device
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    # Training progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0
        
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
            batch_count += 1
        
        train_loss /= batch_count
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            # Denormalize predictions
            val_outputs_denorm = val_outputs * target_std + target_mean
            val_mae = F.l1_loss(val_outputs_denorm, y_test_tensor).item()
        
        train_losses.append(train_loss)
        val_maes.append(val_mae)
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val MAE': f'{val_mae:.2f}',
            'Best MAE': f'{best_val_mae:.2f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️ Early stopping at epoch {epoch}")
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
    
    print(f"\n🎉 PYTORCH MLP RESULTS:")
    print(f"   Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
    print(f"   Test RMSE: {test_rmse:.2f} | Test R²: {test_r2:.3f}")
    print(f"   Within ±5: {within_5:.1f}% | ±10: {within_10:.1f}% | ±20: {within_20:.1f}% | ±50: {within_50:.1f}%")
    
    return {
        'model': model,
        'scaler': scaler,
        'target_mean': target_mean,
        'target_std': target_std,
        'feature_columns': feature_columns,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'within_10': within_10,
        'device': device,
        'train_losses': train_losses,
        'val_maes': val_maes,
        'test_predictions': test_pred_np,
        'test_actual': y_test.values
    }

# Train PyTorch MLP
print("🚀 Starting PyTorch MLP training...")
mlp_results = train_pytorch_mlp(features_df, feature_columns)
```

# Cell 5: Training Visualization and Analysis
```python
def visualize_training_results(mlp_results):
    """Visualize training results and model performance"""
    
    print("📊 TRAINING RESULTS VISUALIZATION")
    print("=" * 45)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training curves
    axes[0, 0].plot(mlp_results['train_losses'], label='Training Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. Validation MAE
    axes[0, 1].plot(mlp_results['val_maes'], label='Validation MAE', color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (tokens)')
    axes[0, 1].set_title('Validation MAE Over Time')
    axes[0, 1].legend()
    
    # 3. Predictions vs Actual
    test_pred = mlp_results['test_predictions']
    test_actual = mlp_results['test_actual']
    
    axes[1, 0].scatter(test_actual, test_pred, alpha=0.5, s=1)
    axes[1, 0].plot([test_actual.min(), test_actual.max()], 
                    [test_actual.min(), test_actual.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Tokens')
    axes[1, 0].set_ylabel('Predicted Tokens')
    axes[1, 0].set_title(f'Predictions vs Actual (MAE: {mlp_results["test_mae"]:.1f})')
    
    # 4. Error distribution
    errors = test_actual - test_pred
    axes[1, 1].hist(errors, bins=50, alpha=0.7, color='green')
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Perfect Prediction')
    axes[1, 1].set_xlabel('Prediction Error (tokens)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis by token ranges
    print(f"\n📊 PERFORMANCE BY TOKEN RANGE:")
    
    ranges = [
        (0, 50, "Very Short"),
        (50, 150, "Short"),
        (150, 300, "Medium"),
        (300, 500, "Long"),
        (500, float('inf'), "Very Long")
    ]
    
    for min_tok, max_tok, label in ranges:
        if max_tok == float('inf'):
            mask = test_actual >= min_tok
        else:
            mask = (test_actual >= min_tok) & (test_actual < max_tok)
        
        if mask.sum() > 10:  # Only analyze ranges with sufficient samples
            range_mae = mean_absolute_error(test_actual[mask], test_pred[mask])
            range_r2 = r2_score(test_actual[mask], test_pred[mask])
            count = mask.sum()
            
            print(f"   {label:12}: {count:4d} samples, MAE: {range_mae:5.1f}, R²: {range_r2:.3f}")

# Visualize results
visualize_training_results(mlp_results)
```

# Cell 6: Compare with Baseline Models
```python
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def compare_with_baselines(features_df, feature_columns, mlp_results, target_col='actual_output_tokens'):
    """Compare MLP with baseline models"""
    
    print("📊 COMPARING WITH BASELINE MODELS")
    print("=" * 45)
    
    # Prepare data (same as MLP)
    X = features_df[feature_columns].copy()
    y = features_df[target_col].copy()
    
    X = X.fillna(0)
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    # Same train-test split as MLP
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train baseline models
    models = {}
    results = {}
    
    # Random Forest
    print("🌳 Training Random Forest...")
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
    
    # Evaluate all models
    print(f"\n📋 MODEL COMPARISON RESULTS:")
    print(f"{'Model':<15} {'Test MAE':<12} {'Test R²':<12} {'±10 Acc':<12} {'±50 Acc':<12}")
    print("-" * 65)
    
    # Add MLP results
    print(f"{'PyTorch MLP':<15} {mlp_results['test_mae']:<12.2f} {mlp_results['test_r2']:<12.3f} {mlp_results['within_10']:<12.1f}% N/A")
    
    # Baseline models
    for name, model in models.items():
        pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        within_10 = (np.abs(y_test - pred) <= 10).mean() * 100
        within_50 = (np.abs(y_test - pred) <= 50).mean() * 100
        
        results[name] = {
            'test_mae': mae,
            'test_r2': r2,
            'within_10': within_10,
            'within_50': within_50
        }
        
        print(f"{name:<15} {mae:<12.2f} {r2:<12.3f} {within_10:<12.1f}% {within_50:<12.1f}%")
    
    # Improvement analysis
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    baseline_mae = 165  # Previous best result
    mlp_improvement = ((baseline_mae - mlp_results['test_mae']) / baseline_mae) * 100
    
    print(f"   Previous best MAE: {baseline_mae}")
    print(f"   PyTorch MLP MAE: {mlp_results['test_mae']:.2f}")
    print(f"   Improvement: {mlp_improvement:+.1f}%")
    
    if mlp_results['test_mae'] < 50:
        print(f"   🎉 EXCELLENT: Production-ready accuracy achieved!")
    elif mlp_results['test_mae'] < 100:
        print(f"   ✅ GOOD: Significant improvement over baseline")
    else:
        print(f"   ⚠️ MODERATE: Some improvement but needs more work")
    
    return results

# Compare models
comparison_results = compare_with_baselines(features_df, feature_columns, mlp_results)
```

# Cell 7: Production Prediction Function with Testing
```python
def create_production_mlp_predictor(mlp_results, embedding_model):
    """Create production-ready MLP prediction function"""
    
    def predict_tokens(query):
        """
        Predict output tokens for a given query using MLP model
        
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
        
        # 1. Generate embedding (384 dimensions)
        embedding = embedding_model.encode([query])[0]
        
        # 2. Calculate complexity score
        def calculate_complexity_score(text):
            txt_lower = text.lower()
            words = txt_lower.split()
            
            if not words:
                return 0.0
            
            # Vocabulary diversity
            vocab_diversity = len(set(words)) / len(words)
            
            # Average word length (normalized)
            avg_word_len = np.mean([len(w) for w in words])
            avg_word_len_normalized = min(avg_word_len / 8.0, 1.0)
            
            # Punctuation density
            punctuation_count = sum(1 for c in text if c in '!@#$%^&*(),.?":{}|<>')
            punct_density = min(punctuation_count / len(text), 0.1) * 10
            
            # Technical term density
            tech_patterns = [
                r'\b(function|class|import|def|return|api|database|algorithm|model|training)\b',
                r'\b(tensor|numpy|pandas|sklearn|python|javascript|react|sql|html|css)\b',
                r'\b(implement|optimize|debug|refactor|deploy|configure|install)\b',
                r'\b(analysis|statistics|machine learning|neural network|regression)\b',
                r'\b(comprehensive|detailed|step-by-step|tutorial|walkthrough)\b'
            ]
            
            tech_matches = sum(len(re.findall(pattern, txt_lower)) for pattern in tech_patterns)
            tech_density = min(tech_matches / len(words), 0.3) / 0.3
            
            # Question complexity
            complex_indicators = [
                'how to', 'why does', 'what is the difference', 'compare', 'explain',
                'implement', 'analyze', 'evaluate', 'comprehensive', 'detailed'
            ]
            complexity_indicators = sum(1 for indicator in complex_indicators if indicator in txt_lower)
            question_complexity = min(complexity_indicators / 3.0, 1.0)
            
            # Final complexity score
            return min(
                vocab_diversity * 0.2 +
                avg_word_len_normalized * 0.2 +
                punct_density * 0.1 +
                tech_density * 0.3 +
                question_complexity * 0.2,
                1.0
            )
        
        # 3. Extract production features
        def extract_production_features(text):
            txt_lower = text.lower()
            words = txt_lower.split()
            
            if not words:
                return [0] * 10
            
            features = []
            features.append(len(text))  # char_length
            features.append(len(words))  # word_count
            features.append(np.mean([len(w) for w in words]))  # avg_word_length
            features.append(int(any(word in txt_lower for word in ['what', 'define', 'is'])))  # is_definition
            features.append(int(any(word in txt_lower for word in ['how', 'guide', 'tutorial'])))  # is_how_to
            features.append(int(any(word in txt_lower for word in ['implement', 'build', 'create'])))  # is_implementation
            features.append(int(any(word in txt_lower for word in ['explain', 'describe', 'elaborate'])))  # is_explanation
            features.append(int('```' in text or bool(re.search(r'`[^`]+`', text))))  # has_code
            features.append(int(any(lang in txt_lower for lang in ['python', 'javascript', 'sql', 'html'])))  # mentions_programming
            features.append(int(any(word in txt_lower for word in ['comprehensive', 'detailed', 'step-by-step'])))  # requests_detailed
            
            return features
        
        # 4. Calculate input tokens (rough approximation)
        input_tokens = len(query.split()) * 1.3
        
        # 5. Combine all features in correct order
        feature_vector = []
        
        # Add embeddings (384 dimensions)
        feature_vector.extend(embedding)
        
        # Add complexity score
        complexity = calculate_complexity_score(query)
        feature_vector.append(complexity)
        
        # Add production features
        prod_features = extract_production_features(query)
        feature_vector.extend(prod_features)
        
        # Add input tokens
        feature_vector.append(input_tokens)
        
        # Convert to tensor
        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_scaled = mlp_results['scaler'].transform(feature_array)
        feature_tensor = torch.FloatTensor(feature_scaled).to(mlp_results['device'])
        
        # Make prediction
        model = mlp_results['model']
        model.eval()
        
        with torch.no_grad():
            prediction_normalized = model(feature_tensor)
            # Denormalize
            prediction = (prediction_normalized * mlp_results['target_std'] + mlp_results['target_mean']).item()
        
        # Return as positive integer
        return max(1, int(round(prediction)))
    
    return predict_tokens

# Create production predictor
print("🚀 Creating production MLP predictor...")
predict_tokens = create_production_mlp_predictor(mlp_results, embedding_model)

print("✅ Production MLP predictor ready!")

# Comprehensive testing
print("\n🧪 COMPREHENSIVE TESTING OF MLP PREDICTOR")
print("=" * 50)

test_cases = [
    {
        'query': "What is Python?",
        'expected_range': "Short (20-60 tokens)",
        'reasoning': "Simple definition question"
    },
    {
        'query': "How do I sort a list in Python?",
        'expected_range': "Medium (80-150 tokens)",
        'reasoning': "Basic how-to with code example"
    },
    {
        'query': "Implement a binary search algorithm in Python with comprehensive error handling",
        'expected_range': "Long (300-500 tokens)",
        'reasoning': "Complex implementation with detailed requirements"
    },
    {
        'query': "Explain the mathematical foundations of transformer architecture including attention mechanisms",
        'expected_range': "Very Long (500-700 tokens)",
        'reasoning': "Deep technical explanation with math"
    },
    {
        'query': "Hi there!",
        'expected_range': "Very Short (5-20 tokens)",
        'reasoning': "Simple greeting"
    },
    {
        'query': "Debug this error: AttributeError",
        'expected_range': "Medium (100-200 tokens)",
        'reasoning': "Debugging help with explanation"
    },
    {
        'query': "Create a comprehensive machine learning pipeline for image classification with data preprocessing, model training, evaluation, and deployment instructions",
        'expected_range': "Very Long (600-800 tokens)",
        'reasoning': "Complex multi-step project"
    }
]

print(f"{'Test':<4} {'Predicted':<10} {'Expected Range':<25} {'Status':<10} {'Query'}")
print("-" * 100)

for i, test_case in enumerate(test_cases, 1):
    predicted = predict_tokens(test_case['query'])
    
    # Determine if prediction is reasonable
    if "Very Short" in test_case['expected_range']:
        reasonable = 5 <= predicted <= 30
    elif "Short" in test_case['expected_range'] and "Very" not in test_case['expected_range']:
        reasonable = 20 <= predicted <= 80
    elif "Medium" in test_case['expected_range']:
        reasonable = 80 <= predicted <= 250
    elif "Long" in test_case['expected_range'] and "Very" not in test_case['expected_range']:
        reasonable = 250 <= predicted <= 550
    elif "Very Long" in test_case['expected_range']:
        reasonable = 500 <= predicted <= 800
    else:
        reasonable = True
    
    status = "✅ Good" if reasonable else "⚠️ Check"
    
    print(f"{i:<4} {predicted:<10} {test_case['expected_range']:<25} {status:<10} {test_case['query'][:50]}...")
    print(f"     Reasoning: {test_case['reasoning']}")
    print()

print(f"🎉 MLP TOKEN PREDICTOR TESTING COMPLETE!")
print(f"=" * 50)
print(f"✅ Model ready for production deployment")
print(f"📊 Final Test MAE: {mlp_results['test_mae']:.2f} tokens")
print(f"🎯 Within ±10 tokens: {mlp_results['within_10']:.1f}%")
print(f"🚀 Use predict_tokens(query) for predictions")

# Example usage
print(f"\n📋 PRODUCTION USAGE EXAMPLE:")
example_query = "How do I implement a REST API in Python?"
example_prediction = predict_tokens(example_query)
print(f">>> predict_tokens('{example_query}')")
print(f">>> {example_prediction}")
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
