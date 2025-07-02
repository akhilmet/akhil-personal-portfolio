# Enhanced utils.py with ALL EDA features
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import re
import numpy as np
import pandas as pd

def get_token_length(text: str) -> int:
    """
    Returns the length of tokens for a given text using the specified tokenizer.
    Args:
        text (str): The text to tokenize.
        tokenizer: The tokenizer to use for encoding the text.
    Returns:
        int: The length of the tokens.
    """
    
    tokenizer = MistralTokenizer.v1()
    return len(tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=text)])).tokens)

def is_natural_language_or_code(text: str) -> int:
    """Determine if query is natural language or code request"""
    if pd.isna(text):
        return 0
    
    text_lower = str(text).lower()
    words = text_lower.split()
    
    # ENHANCED: Comprehensive code keywords as complete words
    CODE_KEYWORDS = {
        # Core Programming & Language Names
        "python", "javascript", "java", "c++", "c#", "typescript", "php", "swift", "kotlin", "ruby", "golang", "go",
        "rust", "scala", "perl", "lua", "sql", "css", "sass", "less", "bash", "shell", "powershell", "objective-c",
        "assembly", "matlab", "fortran", "cobol", "lisp", "haskell", "r", "dart", "elixir", "lua", "zig",
        
        # Programming Constructs
        "function", "def", "class", "import", "return", "print", "console", "log", "var", "let", "const", "if",
        "else", "elif", "for", "while", "do", "switch", "case", "break", "continue", "try", "catch", "finally",
        "throw", "async", "await", "yield", "lambda", "arrow", "callback", "promise", "thread",
        
        # Data Structures & Types
        "array", "list", "dict", "dictionary", "tuple", "set", "map", "hash", "stack", "queue", "tree",
        "linked", "heap", "graph", "node", "pointer", "reference", "struct", "enum", "union", "interface",
        
        # Object-Oriented Programming
        "class", "object", "method", "property", "attribute", "inheritance", "polymorphism", "encapsulation",
        "constructor", "destructor", "getter", "setter", "static", "final", "abstract", "virtual", "override",
        
        # Web Technologies
        "html", "xml", "json", "ajax", "dom", "css", "javascript", "jquery", "react", "angular", "vue",
        "node", "express", "flask", "django", "spring", "laravel", "rails", "mvc", "api", "rest", "graphql",
        
        # Databases
        "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra", "elasticsearch", "nosql",
        "database", "db", "table", "schema", "index", "join", "select", "insert", "update", "delete",
        "transaction", "migration", "orm", "sqlalchemy", "prisma",
        
        # DevOps & Tools
        "git", "github", "gitlab", "bitbucket", "version", "commit", "branch", "merge", "pull", "rebase",
        "docker", "container", "kubernetes", "k8s", "jenkins", "circleci", "travis", "ci/cd", "pipeline", "devops",
        "terraform", "ansible", "puppet", "chef", "vagrant", "aws", "azure", "gcp", "lambda", "ec2", "s3", "cloud",
        "serverless", "microservices", "virtual", "ssh", "bashrc", "zshrc", "cron", "crontab",
        
        # Commands & Syntax
        "install", "pip", "npm", "yarn", "apt-get", "brew", "import", "require", "include", "def", "const", "let", "var",
        "execute", "print", "console", "log", "return", "new", "self", "this", "try", "catch", "finally", "raise", "throw",
        "assert", "yield", "pass", "continue", "break",
        
        # Errors & Debugging
        "error", "bug", "debug", "debugger", "exception", "traceback", "stack", "segfault",
        "null", "undefined", "not", "found", "404", "500", "timeout", "leak", "memory", "test", "unittest",
        "pytest", "jest", "junit", "assertion", "breakpoint", "logging", "fix", "troubleshoot", "resolve"
    }
    
    if any(word in CODE_KEYWORDS for word in words):
        return 1
    else:
        return 0

def categorize_query(query):
    """
    Categorize a query into useful categories for LLM response length prediction.
    Returns a one-hot encoded numpy array: [open_ended, list_request, code_request, summary, yes_no, instruction, other]
    """
    query_lower = query.lower().strip()
    categories = {
        "open_ended": 0,
        "list_request": 0,
        "code_request": 0,
        "summary": 0,
        "yes_no": 0,
        "instruction": 0,
        "other": 0
    }
    
    # Open-ended: starts with what, how, why, explain, describe, tell me
    if re.match(r"^(what|how|why|explain|describe|tell me)\b", query_lower):
        categories["open_ended"] = 1
    
    # List request: contains "list", "give me N", "show me N", "top N"
    elif re.search(r"\b(list\b|\bgive me \d+|\bshow me \d+|\btop \d+)", query_lower):
        categories["list_request"] = 1
    
    # Code request: contains "code", "python", "write a function", "implement"
    elif re.search(r"\b(code\b|\bpython\b|\bwrite a function\b|\bimplement\b)", query_lower):
        categories["code_request"] = 1
    
    # Summary: contains "summarize", "summary", "in short", "briefly"
    elif re.search(r"\b(summarize\b|\bsummary\b|\bin short\b|\bbriefly\b)", query_lower):
        categories["summary"] = 1
    
    # Yes/No: starts with is, are, do, does, can, will, should, could
    elif re.match(r"^(is|are|do|does|can|will|should|could)\b", query_lower):
        categories["yes_no"] = 1
    
    # Instruction: contains "step by step", "instructions", "how to", "guide"
    elif re.search(r"\b(step by step\b|\binstructions\b|\bhow to\b|\bguide\b)", query_lower):
        categories["instruction"] = 1
    
    # If none matched, mark as other
    if not any(categories.values()):
        categories["other"] = 1
    
    return np.array(list(categories.values()))

def categorize_question_type_improved(text):
    """IMPROVED: Single category per prompt with priority order"""
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower()
    
    # Priority order: who > what > how > where > when > why > other
    if any(word in text_lower for word in ['who', 'whom', 'whose']):
        return "who"
    elif any(word in text_lower for word in ['what', "what's", 'which']):
        return "what"
    elif any(word in text_lower for word in ['how', "how's", 'how to', 'how do', 'how can']):
        return "how"
    elif any(word in text_lower for word in ['where', "where's", 'where is', 'where are']):
        return "where"
    elif any(word in text_lower for word in ['when', "when's", 'when is', 'when do']):
        return "when"
    elif any(word in text_lower for word in ['why', "why's", 'why do', 'why is']):
        return "why"
    else:
        return "other"

def analyze_other_category(text):
    """Refined analysis of 'other' category queries"""
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower().strip()
    
    # Imperative commands
    if any(text_lower.startswith(word) for word in ['list', 'describe', 'explain', 'tell me', 'show me', 'give me']):
        return "imperative_request"
    
    # Yes/No questions  
    if any(text_lower.startswith(word) for word in ['is', 'are', 'do', 'does', 'can', 'will', 'would', 'should']):
        return "yes_no_question"
    
    # Comparative questions
    if any(word in text_lower for word in ['compare', 'difference', 'better', 'worse', 'vs', 'versus']):
        return "comparative"
    
    # Help/Support requests
    if any(word in text_lower for word in ['help', 'support', 'assist', 'trouble', 'problem', 'issue']):
        return "help_request"
    
    # Creative/Generation requests
    if any(word in text_lower for word in ['write', 'create', 'generate', 'make', 'build', 'design']):
        return "creative_generation"
    
    # Analysis/Evaluation requests
    if any(word in text_lower for word in ['analyze', 'evaluate', 'assess', 'review', 'examine']):
        return "analysis_request"
    
    # Opinion/Recommendation requests
    if any(word in text_lower for word in ['recommend', 'suggest', 'opinion', 'think', 'believe']):
        return "opinion_request"
    
    # Calculation/Math requests
    if any(word in text_lower for word in ['calculate', 'compute', 'solve', '%', '$', 'equation']):
        return "calculation"
    
    return "unclassified_other"

def is_independent_or_continuation(text, response_length=None):
    """Determine if query is independent or multi-thought"""
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower()
    
    # Multi-thought indicators
    continuation_patterns = [
        'also', 'and', 'but', 'however', 'additionally', 'furthermore',
        'what about', 'how about', 'can you also', 'tell me more',
        'continue', 'next', 'then', 'after that', 'moreover', 'besides',
        'in addition', 'similarly', 'likewise', 'on the other hand'
    ]
    
    # Independent indicators
    independent_patterns = [
        'i need help', 'can you help', 'i want to', 'how do i',
        'what is', 'please explain', 'tell me about', 'help me with',
        'i would like', 'could you please'
    ]
    
    if any(pattern in text_lower for pattern in continuation_patterns):
        return "continuation"
    elif any(pattern in text_lower for pattern in independent_patterns):
        return "independent"
    else:
        # Use text length as indicator
        word_count = len(text.split())
        if word_count > 20:  # Longer queries often multi-thought
            return "continuation"
        else:
            return "independent"

def extract_text_complexity_features(text):
    """Extract comprehensive complexity features from text"""
    if pd.isna(text):
        return {
            'has_numbers': False,
            'has_punctuation': False,
            'has_special_chars': False,
            'has_technical_terms': False,
            'sentence_count': 0,
            'avg_word_length': 0,
            'complexity_score': 0,
            'has_questions': False,
            'has_code_indicators': False,
            'word_count': 0,
            'char_count': 0,
            'unique_word_count': 0,
            'punctuation_density': 0,
            'caps_ratio': 0
        }
    
    text_str = str(text)
    words = text_str.split()
    
    # Basic features
    has_numbers = bool(re.search(r'\d+', text_str))
    has_punctuation = bool(re.search(r'[.,!?;:(){}[\]"\'`]', text_str))
    has_special_chars = bool(re.search(r'[@#$%^&*+=<>~/\\|]', text_str))
    has_questions = bool('?' in text_str)
    
    # Technical terms
    technical_terms = [
        'algorithm', 'function', 'variable', 'database', 'api', 'framework',
        'library', 'module', 'class', 'object', 'method', 'parameter',
        'syntax', 'compile', 'debug', 'optimize', 'interface', 'server',
        'client', 'protocol', 'encryption', 'authentication', 'deployment'
    ]
    has_technical_terms = any(term in text_str.lower() for term in technical_terms)
    
    # Code indicators (different from technical terms)
    code_indicators = ['def ', 'function(', 'import ', 'return ', 'print(', 'console.log', 'if(', 'for(', 'while(']
    has_code_indicators = any(indicator in text_str.lower() for indicator in code_indicators)
    
    # Text metrics
    sentence_count = len(re.split(r'[.!?]+', text_str.strip()))
    word_count = len(words)
    char_count = len(text_str)
    unique_words = set(word.lower().strip('.,!?;:"()[]{}') for word in words)
    unique_word_count = len(unique_words)
    
    # Ratios and densities
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    punctuation_count = len(re.findall(r'[.,!?;:(){}[\]"\'`]', text_str))
    punctuation_density = punctuation_count / len(text_str) if text_str else 0
    caps_count = sum(1 for c in text_str if c.isupper())
    caps_ratio = caps_count / len(text_str) if text_str else 0
    
    # Enhanced complexity score
    complexity_score = (
        word_count * 0.1 +
        sentence_count * 0.2 +
        avg_word_length * 0.2 +
        unique_word_count * 0.1 +
        has_numbers * 0.1 +
        has_punctuation * 0.1 +
        has_special_chars * 0.1 +
        has_technical_terms * 0.1 +
        punctuation_density * 10 +
        caps_ratio * 5
    )
    
    return {
        'has_numbers': has_numbers,
        'has_punctuation': has_punctuation,
        'has_special_chars': has_special_chars,
        'has_technical_terms': has_technical_terms,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'complexity_score': round(complexity_score, 2),
        'has_questions': has_questions,
        'has_code_indicators': has_code_indicators,
        'word_count': word_count,
        'char_count': char_count,
        'unique_word_count': unique_word_count,
        'punctuation_density': round(punctuation_density, 4),
        'caps_ratio': round(caps_ratio, 4)
    }

def extract_all_features(text):
    """Extract ALL features for comprehensive analysis"""
    if pd.isna(text):
        return {}
    
    # Get basic token count
    token_count = get_token_length(text)
    
    # Get complexity features
    complexity_features = extract_text_complexity_features(text)
    
    # Get categorization features
    question_type = categorize_question_type_improved(text)
    other_subcategory = analyze_other_category(text)
    query_context = is_independent_or_continuation(text)
    
    # Get content type
    nlp_vs_code = is_natural_language_or_code(text)
    
    # Original category features (one-hot encoded)
    category_features = categorize_query(text)
    
    # Combine everything
    all_features = {
        'query_token_length': token_count,
        'nlp_vs_code': nlp_vs_code,
        'question_type': question_type,
        'other_subcategory': other_subcategory,
        'query_context': query_context,
        # Category one-hot features
        'category0': category_features[0],  # open_ended
        'category1': category_features[1],  # list_request
        'category2': category_features[2],  # code_request
        'category3': category_features[3],  # summary
        'category4': category_features[4],  # yes_no
        'category5': category_features[5],  # instruction
        'category6': category_features[6],  # other
        # Complexity features
        **complexity_features
    }
    
    return all_features


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

# Token Predictor EDA - Complete Jupyter Notebook

Copy and paste each cell below into separate Jupyter notebook cells:


---
# Complete Token Predictor EDA - Updated with Improvements

# Enhanced Token Predictor Model Testing

```python
# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
from utils import get_token_length, is_natural_language_or_code, categorize_query, extract_all_features
```

```python
# Cell 2: Load and Process Data with ALL Features
print("🔄 Loading and processing data with ALL EDA features...")
# Load datasets
training_df = pd.read_csv('../testing/training_data.csv')
print(f"✅ Loaded training data: {training_df.shape}")

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

    if idx % 1000 == 0:
        print(f"   Processed {idx} queries...")

# Convert to DataFrame
enhanced_df = pd.DataFrame(all_features_list)
print(f"✅ Feature extraction complete: {enhanced_df.shape}")
```

```python
# Cell 3: Feature Engineering and Encoding
print("\n🔧 Engineering features for ML models...")
# Encode categorical features
label_encoders = {}
categorical_features = ['question_type', 'other_subcategory', 'query_context']
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
    'question_type_encoded', 'other_subcategory_encoded', 'query_context_encoded'
]

# Combine all available features
ALL_FEATURES = []
for feature_set in [BASIC_FEATURES, COMPLEXITY_FEATURES, CATEGORICAL_ENCODED]:
    for feature in feature_set:
        if feature in enhanced_df.columns:
            ALL_FEATURES.append(feature)

print(f"✅ Total features available: {len(ALL_FEATURES)}")
print(f"   Basic features: {len([f for f in BASIC_FEATURES if f in enhanced_df.columns])}")
print(f"   Complexity features: {len([f for f in COMPLEXITY_FEATURES if f in enhanced_df.columns])}")
print(f"   Categorical features: {len([f for f in CATEGORICAL_ENCODED if f in enhanced_df.columns])}")
```

```python
# Cell 4: Data Preparation
# Prepare data for modeling
X = enhanced_df[ALL_FEATURES]
y = enhanced_df['response_length']

print(f"\n📊 Dataset Summary:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(ALL_FEATURES)}")
print(f"   Target range: {y.min():.0f} - {y.max():.0f} tokens")
print(f"   Target mean: {y.mean():.1f} tokens")

# Handle any missing values
print(f"\n🔍 Missing values check:")
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print("   Missing values found:")
    for feature, count in missing_counts[missing_counts > 0].items():
        print(f"     {feature}: {count}")
    X = X.fillna(0)  # Fill missing values with 0
else:
    print("   ✅ No missing values found")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")
```

```python
# Cell 5: Random Forest with ALL Features
print("\n🌲 RANDOM FOREST MODEL - ALL FEATURES")
print("=" * 50)

# Train Random Forest with enhanced parameters
rf_regressor = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    oob_score=True,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1
)

print("🔄 Training Random Forest...")
rf_regressor.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_regressor.predict(X_test)

# Calculate metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
oob_rf = rf_regressor.oob_score_ if hasattr(rf_regressor, 'oob_score_') else "N/A"

print(f"\n--- Random Forest Model Evaluation (ALL FEATURES) ---")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"R-squared (R²): {r2_rf:.2f}")
print(f"Out-of-Bag (OOB) Score: {oob_rf:.2f}" if oob_rf != "N/A" else f"Out-of-Bag (OOB) Score: {oob_rf}")
print("=" * 40)

# Feature Importance Analysis
print("\n📊 FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)
importances = rf_regressor.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Top 15 Most Important Features:")
for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.6f}")
```

```python
# Cell 6: XGBoost with ALL Features
print(f"\n🚀 XGBOOST MODEL - ALL FEATURES")
print("=" * 50)

import xgboost as xgb

# Enhanced XGBoost parameters
xgb_regressor = xgb.XGBRegressor(
    objective='reg\:squarederror',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='mae'
)

print("🔄 Training XGBoost...")
# Split training data for early stopping
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

xgb_regressor.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)

# Predictions
y_pred_xgb = xgb_regressor.predict(X_test)

# Calculate metrics
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"\n--- XGBoost Model Evaluation (ALL FEATURES) ---")
print(f"Mean Absolute Error (MAE): {mae_xgb:.2f}")
print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f"R-squared (R²): {r2_xgb:.2f}")
print(f"Best Iteration: {xgb_regressor.best_iteration}")
print("=" * 40)
```

```python
# Cell 7: Model Comparison and Visualization
print(f"\n📈 MODEL COMPARISON")
print("=" * 30)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': ['Random Forest (All Features)', 'XGBoost (All Features)'],
    'MAE': [mae_rf, mae_xgb],
    'MSE': [mse_rf, mse_xgb],
    'R²': [r2_rf, r2_xgb]
})

print(comparison_df.round(2))

# Determine best model
best_model_idx = comparison_df['MAE'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_mae = comparison_df.loc[best_model_idx, 'MAE']
print(f"\n🏆 Best Model: {best_model_name} (MAE: {best_mae:.2f})")
```

```python
# Cell 8: Feature Importance Visualization
plt.style.use('seaborn-v0_8-whitegrid')

# Random Forest Feature Importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Feature Importance Analysis - All Features', fontsize=16)

# Plot 1: Random Forest Feature Importance (Top 15)
top_features_rf = feature_importance_df.head(15)
ax1.barh(range(len(top_features_rf)), top_features_rf['importance'], color='forestgreen')
ax1.set_yticks(range(len(top_features_rf)))
ax1.set_yticklabels(top_features_rf['feature'])
ax1.set_xlabel('Importance Score')
ax1.set_title('Random Forest - Top 15 Features')
ax1.grid(True, alpha=0.3)

# Plot 2: XGBoost Feature Importance (Top 15)
xgb_importances = xgb_regressor.feature_importances_
xgb_feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_importances
}).sort_values(by='importance', ascending=False)

top_features_xgb = xgb_feature_importance_df.head(15)
ax2.barh(range(len(top_features_xgb)), top_features_xgb['importance'], color='darkorange')
ax2.set_yticks(range(len(top_features_xgb)))
ax2.set_yticklabels(top_features_xgb['feature'])
ax2.set_xlabel('Importance Score')
ax2.set_title('XGBoost - Top 15 Features')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```


### **GitHub README Ready:**
This format is optimized for copying into GitHub README.md and then easily transferring to any Jupyter environment. Each cell is clearly separated and ready for enterprise use.

## License

This project is licensed under the MIT License.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
