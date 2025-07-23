```
# Cell 5: Enhanced Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣  Prepare feature matrix X and target vector y
if 'actual_output_tokens' not in features_df.columns:
    raise KeyError(
        "Column 'actual_output_tokens' not found in features_df. "
        "Make sure you computed it in an earlier cell."
    )

X = features_df.drop(columns=['actual_output_tokens'])
y = features_df['actual_output_tokens'].astype(float)

print(f"🚀  Data ready: {X.shape[0]} samples, {X.shape[1]} features")

# 2️⃣  Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"📊  Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

# 3️⃣  Train Random Forest (with chosen hyperparameters)
print("\n🌳  Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 4️⃣  Train XGBoost
print("🚀  Training XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# 5️⃣  Evaluate both models
def evaluate(name, model):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2   = r2_score(y_test, preds)
    print(f"\n📊  {name} evaluation:")
    print(f"    • MAE:  {mae:.2f}")
    print(f"    • RMSE: {rmse:.2f}")
    print(f"    • R²:   {r2:.3f}")

evaluate("Random Forest", rf)
evaluate("XGBoost", xgb_model)

```
```
# Cell 3: Generate Sentence Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

print("🚀 Loading sentence embedding model...")
# Note the exact model name and correct casing:
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("📡 Generating embeddings for instructions...")
# Turn your instructions column into a list of strings
instruction_texts = df['instruction'].astype(str).tolist()

# Encode with progress bar and return a NumPy array
embeddings_matrix = embedder.encode(
    instruction_texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"✅ Embeddings shape: {embeddings_matrix.shape}")  # Should be (n_samples, 384)


# Cell 4: Enhanced Feature Engineering with Sentence Embeddings
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_enhanced_features(df, embeddings_matrix):
    """Create enhanced features combining sentence embeddings with existing features"""
    print("🛠️ ENHANCED FEATURE ENGINEERING")
    print("=" * 30)
    features_df = df.copy()
    
    # 1️⃣ Add raw embedding dimensions as features
    print("📊 Adding embedding features...")
    n_dim = embeddings_matrix.shape[1]
    for i in range(n_dim):
        features_df[f'embed_{i}'] = embeddings_matrix[:, i]
    print(f"   Added {n_dim} embedding features")
    
    # 2️⃣ Advanced text‐complexity metrics
    def advanced_complexity(text):
        if pd.isna(text) or text == "":
            return 0, 0, 0, 0
        txt = str(text).lower()
        words = txt.split()
        if not words:
            return 0, 0, 0, 0
        # lexical diversity
        lex_div = len(set(words)) / len(words)
        # technical density
        tech_pat = r'\b(function|class|import|def|return|api|database|algorithm|model|training|tensor|numpy|pandas|sklearn|python)\b'
        tech_den = len(re.findall(tech_pat, txt)) / len(words)
        # question complexity
        ques_pat = r'\bhow to\b|\bwhy does\b|\bwhat is\b|\bcompare\b|\bexplain\b|\bimplement\b'
        ques_comp = len(re.findall(ques_pat, txt))
        # instruction complexity
        instr_pat = r'\bstep by step\b|\bdetailed\b|\btutorial\b|\bguide\b|\bexample\b|\bcomprehensive\b'
        instr_comp = len(re.findall(instr_pat, txt))
        return lex_div, tech_den, ques_comp, instr_comp
    
    print("   Calculating advanced complexity metrics…")
    comp_feats = [advanced_complexity(t) for t in features_df['instruction']]
    comp_df = pd.DataFrame(comp_feats, columns=[
        'lexical_diversity',
        'technical_density',
        'question_complexity',
        'instruction_complexity_enhanced'
    ])
    for c in comp_df.columns:
        features_df[c] = comp_df[c].values
    
    # 3️⃣ Pattern‐matching features
    print("🔍 Adding pattern‐matching features…")
    patterns = {
        'pattern_what_is':       r'\bwhat is\b',
        'pattern_how_to':        r'\bhow to .{1,20}\?',
        'pattern_implement':     r'\bimplement .{5,}',
        'pattern_explain':       r'\bexplain .{20,}',
        'pattern_debug':         r'\b(debug|error|fix)\b',
        'pattern_example_code':  r'\b(example|sample code)\b',
        'pattern_compare':       r'\b(compare|difference|vs)\b',
        'pattern_tutorial':      r'\b(tutorial|guide|walkthrough)\b'
    }
    for name, rx in patterns.items():
        features_df[name] = features_df['instruction'].str.contains(rx, case=False, na=False).astype(int)
    print(f"   Added {len(patterns)} pattern features")
    
    # 4️⃣ Basic instruction stats
    print("📋 Adding basic text features…")
    features_df['instruction_len'] = features_df['instruction'].astype(str).str.len()
    features_df['instruction_word_count'] = features_df['instruction'].astype(str).str.split().str.len().fillna(0)
    
    # 5️⃣ Metadata encoding
    print("🏷️ Encoding metadata features…")
    label_encoders = {}
    for col in ['task_category', 'intent', 'knowledge']:
        if col in features_df.columns:
            le = LabelEncoder()
            features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
            label_encoders[col] = le
            print(f"   Encoded {col} ({len(le.classes_)} classes)")
        else:
            features_df[f'{col}_encoded'] = 0
            print(f"   {col} missing → default 0")
    # difficulty
    if 'difficulty' in features_df.columns:
        dm = {'easy':1,'medium':2,'hard':3}
        features_df['difficulty_encoded'] = features_df['difficulty'].map(dm).fillna(2).astype(int)
        print("   Encoded difficulty from metadata")
    else:
        features_df['difficulty_encoded'] = 2
        print("   Default difficulty=2")
    
    print(f"✅ Enhanced features created: {features_df.shape[1]} total columns")
    return features_df, label_encoders

# Apply
print("🚀 Starting enhanced feature engineering…")
features_df, label_encoders = create_enhanced_features(df, embeddings_matrix)
python
Copy
Edit
# Cell 5: Enhanced Model Training
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_enhanced_models(features_df, target_col='actual_output_tokens'):
    print("🎯 TRAINING ENHANCED MODELS")
    print("=" * 30)
    # select features
    embed_cols   = [c for c in features_df.columns if c.startswith('embed_')]
    pattern_cols = [c for c in features_df.columns if c.startswith('pattern_')]
    basic_cols   = ['input_tokens_mistral','instruction_len','instruction_word_count',
                    'lexical_diversity','technical_density',
                    'question_complexity','instruction_complexity_enhanced',
                    'difficulty_encoded'] + list(label_encoders.keys())
    # actually encoded cols have suffix "_encoded"
    enc_cols = [f"{c}_encoded" for c in label_encoders.keys()]
    feature_cols = embed_cols + pattern_cols + basic_cols + enc_cols
    # dedupe & filter
    feature_cols = [c for i,c in enumerate(feature_cols) if c in features_df.columns and c not in feature_cols[:i]]
    print(f"📊 Using {len(feature_cols)} features")
    X = features_df[feature_cols].fillna(0)
    y = features_df[target_col]
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # scale for XGB
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    results = {}
    # Random Forest
    print("🌳 Training Random Forest…")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['Random Forest'] = rf
    # XGBoost
    print("🚀 Training XGBoost…")
    xgb_reg = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train_sc, y_train)
    results['XGBoost'] = xgb_reg
    # evaluate
    for name, model in results.items():
        print(f"\n📊 Evaluating {name}")
        if name=='XGBoost':
            pred_train = model.predict(X_train_sc)
            pred_test  = model.predict(X_test_sc)
        else:
            pred_train = model.predict(X_train)
            pred_test  = model.predict(X_test)
        mae_train = mean_absolute_error(y_train, pred_train)
        mae_test  = mean_absolute_error(y_test, pred_test)
        rmse_test = mean_squared_error(y_test, pred_test, squared=False)
        r2_test   = r2_score(y_test, pred_test)
        err = abs(y_test - pred_test)
        print(f"   Train MAE: {mae_train:.2f} | Test MAE: {mae_test:.2f}")
        print(f"   Test RMSE: {rmse_test:.2f} | Test R²: {r2_test:.3f}")
        print(f"   ±5 tok: {(err<=5).mean()*100:.1f}% | ±10 tok: {(err<=10).mean()*100:.1f}%")
    # choose best
    best = min(results, key=lambda n: mean_absolute_error(y_test, 
                     (results[n].predict(X_test_sc) if n=='XGBoost' else results[n].predict(X_test))))
    print(f"\n🏆 Best model: {best}")
    return {
        'models': results,
        'best_model': results[best],
        'feature_columns': feature_cols,
        'scaler': scaler,
        'X_test': X_test,
        'X_test_sc': X_test_sc,
        'y_test': y_test
    }

# Apply
print("🚀 Starting model training…")
training_results = train_enhanced_models(features_df)
python
Copy
Edit
# Cell 6: Feature Importance Analysis
import matplotlib.pyplot as plt

def analyze_feature_importance(training_results):
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 30)
    for name, model in training_results['models'].items():
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            cols = training_results['feature_columns']
            imp_df = pd.DataFrame({'feature': cols, 'importance': fi})
            top20 = imp_df.nlargest(20, 'importance')
            print(f"\n🔍 {name} - Top 20 features:")
            for _, row in top20.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            plt.figure(figsize=(8,6))
            plt.barh(top20['feature'], top20['importance'])
            plt.title(f"{name} Top 20 Feature Importance")
            plt.gca().invert_yaxis()
            plt.show()

# Apply
print("🚀 Analyzing feature importance…")
analyze_feature_importance(training_results)
python
Copy
Edit
# Cell 7: Enhanced Query Testing Interface
def create_query_tester(training_results, embedder, label_encoders):
    def predict(query, difficulty='medium', task_category='Information seeking',
                intent='informational', knowledge='intermediate'):
        print(f"🧪 Predicting for: {query[:60]}{'...' if len(query)>60 else ''}")
        # embed
        q_emb = embedder.encode([query], convert_to_numpy=True)[0]
        feat = {}
        # embedding dims
        for i, val in enumerate(q_emb):
            feat[f'embed_{i}'] = val
        # complexity
        lex_div, tech_den, qc, ic = advanced_complexity(query)
        feat.update({
            'lexical_diversity': lex_div,
            'technical_density': tech_den,
            'question_complexity': qc,
            'instruction_complexity_enhanced': ic,
            'instruction_len': len(query),
            'instruction_word_count': len(query.split())
        })
        # patterns
        for name, rx in patterns.items():
            feat[name] = int(bool(re.search(rx, query, re.IGNORECASE)))
        # metadata
        feat['difficulty_encoded'] = {'easy':1,'medium':2,'hard':3}.get(difficulty,2)
        for col, value in [('task_category', task_category),
                           ('intent', intent),
                           ('knowledge', knowledge)]:
            if col in label_encoders:
                try:
                    feat[f'{col}_encoded'] = label_encoders[col].transform([value])[0]
                except:
                    feat[f'{col}_encoded'] = 0
            else:
                feat[f'{col}_encoded'] = 0
        # build vector
        vec = [feat.get(c, 0) for c in training_results['feature_columns']]
        model = training_results['best_model']
        # scale if XGBoost
        if isinstance(model, xgb.XGBRegressor):
            vec = training_results['scaler'].transform([vec])
        pred = model.predict([vec])[0]
        print(f"🎯 Predicted tokens: {pred:.0f}")
        return pred
    return predict

# Create tester
print("🚀 Creating query tester…")
patterns = {
    'pattern_what_is':       r'\bwhat is\b',
    'pattern_how_to':        r'\bhow to .{1,20}\?',
    'pattern_implement':     r'\bimplement .{5,}',
    'pattern_explain':       r'\bexplain .{20,}',
    'pattern_debug':         r'\b(debug|error|fix)\b',
    'pattern_example_code':  r'\b(example|sample code)\b',
    'pattern_compare':       r'\b(compare|difference|vs)\b',
    'pattern_tutorial':      r'\b(tutorial|guide|walkthrough)\b'
}
def advanced_complexity(text):
    txt = str(text).lower(); words = txt.split()
    if not words: return (0,0,0,0)
    lex_div = len(set(words))/len(words)
    tech_den = len(re.findall(r'\b(function|class|import|def|return|api|database|algorithm|model)\b', txt))/len(words)
    ques_comp = len(re.findall(r'\bhow to\b|\bwhy does\b|\bwhat is\b', txt))
    instr_comp = len(re.findall(r'\bstep by step\b|\bdetailed\b|\btutorial\b', txt))
    return lex_div, tech_den, ques_comp, instr_comp

query_tester = create_query_tester(training_results, embedder, label_encoders)

# Test
for q in ["What is Python?", "How to sort a list in Python?"]:
    print("\n" + "-"*40)
    query_tester(q)
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
