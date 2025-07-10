# Model Retraining Notebook - Random Forest with Optimized Features
# Retrain the Random Forest model using the optimized token_estimator_feature_collection.py

```python
# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our optimized feature extraction
from token_estimator_feature_collection import extract_optimized_features, get_label_encoders

print("🚀 Starting Random Forest Model Retraining")
print("Using optimized features from token_estimator_feature_collection.py")
```

```python
# Cell 2: Load Training Data
print("📂 Loading training data...")

# Load the original training dataset
training_df = pd.read_csv('../testing/training_data.csv')
print(f"✅ Loaded training data: {training_df.shape}")
print(f"   Columns: {list(training_df.columns)}")
print(f"   Target range: {training_df['response_length'].min():.0f} - {training_df['response_length'].max():.0f} tokens")
print(f"   Target mean: {training_df['response_length'].mean():.1f} ± {training_df['response_length'].std():.1f}")
```

```python
# Cell 3: Extract Optimized Features
print("\n🔧 Extracting optimized features for all samples...")

# Extract features using the optimized feature collection
all_features_list = []
failed_extractions = 0

for idx, row in training_df.iterrows():
    try:
        query = row['Query']
        response_length = row['response_length']
        
        # Extract optimized features
        features = extract_optimized_features(query)
        features['response_length'] = response_length
        features['original_index'] = idx
        
        all_features_list.append(features)
        
        if idx % 1000 == 0:
            print(f"   Processed {idx:,} queries...")
            
    except Exception as e:
        print(f"   ⚠️ Failed to extract features for query {idx}: {str(e)}")
        failed_extractions += 1
        continue

# Convert to DataFrame
features_df = pd.DataFrame(all_features_list)
print(f"✅ Feature extraction complete: {features_df.shape}")
print(f"   Successful extractions: {len(all_features_list):,}")
print(f"   Failed extractions: {failed_extractions}")

# Display feature summary
print(f"\n📊 Feature Summary:")
feature_cols = [col for col in features_df.columns if col not in ['response_length', 'original_index']]
print(f"   Total features extracted: {len(feature_cols)}")
print(f"   Feature names: {feature_cols}")
```

```python
# Cell 4: Data Preparation and Feature Selection
print("\n🎯 Preparing data for model training...")

# Define the features to use (based on random sampling analysis)
SELECTED_FEATURES = [
    # Core numerical features (above random threshold)
    'complexity_score',
    'word_count', 
    'has_questions',
    'query_token_length',
    'char_count',
    'caps_ratio',
    'punctuation_density',
    'avg_word_length',
    
    # Category features (above threshold only)
    'category0', 'category1', 'category2', 'category3', 
    'category4', 'category5', 'category6',
    
    # Encoded categorical features
    'question_type_encoded',
    'other_subcategory_encoded', 
    'query_context_encoded'
]

# Prepare feature matrix and target
X = features_df[SELECTED_FEATURES]
y = features_df['response_length']

print(f"   Features selected: {len(SELECTED_FEATURES)}")
print(f"   Samples: {len(X):,}")
print(f"   Target range: {y.min():.0f} - {y.max():.0f} tokens")

# Check for missing values
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(f"\n⚠️ Missing values found:")
    for feature, count in missing_counts[missing_counts > 0].items():
        print(f"     {feature}: {count}")
    X = X.fillna(0)
    print("   ✅ Missing values filled with 0")
else:
    print("   ✅ No missing values found")

# Display feature statistics
print(f"\n📈 Feature Statistics:")
for feature in SELECTED_FEATURES[:10]:  # Show first 10 for brevity
    if feature in X.columns:
        print(f"   {feature:<25} min={X[feature].min():.3f} max={X[feature].max():.3f} mean={X[feature].mean():.3f}")
```

```python
# Cell 5: Train-Test Split
print("\n🔀 Splitting data into train/test sets...")

# Split data with stratification to maintain distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")
print(f"   Train target mean: {y_train.mean():.1f} ± {y_train.std():.1f}")
print(f"   Test target mean: {y_test.mean():.1f} ± {y_test.std():.1f}")
```

```python
# Cell 6: Train Random Forest Model
print("\n🌲 Training Random Forest Model...")

# Initialize Random Forest with optimized parameters
rf_model = RandomForestRegressor(
    n_estimators=200,           # Good balance of performance vs speed
    max_depth=15,              # Prevent overfitting
    min_samples_split=5,       # Minimum samples to split
    min_samples_leaf=2,        # Minimum samples in leaf
    max_features='sqrt',       # Feature sampling strategy
    random_state=42,           # Reproducibility
    oob_score=True,           # Out-of-bag scoring
    n_jobs=-1,                # Use all CPU cores
    verbose=1                 # Show progress
)

print("🔄 Training Random Forest...")
print(f"   Parameters: n_estimators={rf_model.n_estimators}, max_depth={rf_model.max_depth}")

# Train the model
rf_model.fit(X_train, y_train)

print("✅ Training complete!")
print(f"   OOB Score: {rf_model.oob_score_:.4f}")
```

```python
# Cell 7: Model Evaluation
print("\n📊 Evaluating Model Performance...")

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("=" * 50)
print("🎯 MODEL PERFORMANCE RESULTS")
print("=" * 50)
print(f"Training Set:")
print(f"   MAE:  {train_mae:.2f} tokens")
print(f"   RMSE: {train_rmse:.2f} tokens") 
print(f"   R²:   {train_r2:.4f}")
print(f"\nTest Set:")
print(f"   MAE:  {test_mae:.2f} tokens")
print(f"   RMSE: {test_rmse:.2f} tokens")
print(f"   R²:   {test_r2:.4f}")
print(f"\nOut-of-Bag Score: {rf_model.oob_score_:.4f}")
print("=" * 50)

# Check for overfitting
overfit_mae = train_mae - test_mae
overfit_r2 = train_r2 - test_r2

print(f"\n🔍 Overfitting Check:")
print(f"   MAE difference (train - test): {overfit_mae:.2f}")
print(f"   R² difference (train - test): {overfit_r2:.4f}")

if abs(overfit_mae) < 2.0 and abs(overfit_r2) < 0.05:
    print("   ✅ Good generalization - minimal overfitting")
elif abs(overfit_mae) < 5.0 and abs(overfit_r2) < 0.1:
    print("   ⚠️ Moderate overfitting - acceptable")
else:
    print("   ❌ High overfitting - consider regularization")
```

```python
# Cell 8: Feature Importance Analysis
print("\n📈 FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

# Get feature importances
importances = rf_model.feature_importances_
feature_names = X.columns

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.6f}")

# Calculate cumulative importance
importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
features_for_90_percent = len(importance_df[importance_df['cumulative_importance'] <= 0.9])

print(f"\n📊 Feature Importance Summary:")
print(f"   Features needed for 90% importance: {features_for_90_percent}")
print(f"   Total features used: {len(feature_names)}")
print(f"   Top feature accounts for: {importance_df.iloc[0]['importance']:.1%} of importance")
```

```python
# Cell 9: Save the Trained Model
print("\n💾 Saving trained model and encoders...")

# Create model artifacts directory
import os
os.makedirs('model_artifacts', exist_ok=True)

# Save the Random Forest model
model_path = 'model_artifacts/random_forest_token_predictor.pkl'
joblib.dump(rf_model, model_path)
print(f"✅ Model saved to: {model_path}")

# Save the label encoders
encoders_path = 'model_artifacts/label_encoders.pkl'
label_encoders = get_label_encoders()
joblib.dump(label_encoders, encoders_path)
print(f"✅ Label encoders saved to: {encoders_path}")

# Save feature names for consistency
features_path = 'model_artifacts/selected_features.pkl'
joblib.dump(SELECTED_FEATURES, features_path)
print(f"✅ Feature names saved to: {features_path}")

# Save model metadata
metadata = {
    'model_type': 'RandomForestRegressor',
    'n_estimators': rf_model.n_estimators,
    'max_depth': rf_model.max_depth,
    'features_used': SELECTED_FEATURES,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'test_mae': test_mae,
    'test_r2': test_r2,
    'oob_score': rf_model.oob_score_,
    'trained_date': pd.Timestamp.now().isoformat()
}

metadata_path = 'model_artifacts/model_metadata.pkl'
joblib.dump(metadata, metadata_path)
print(f"✅ Model metadata saved to: {metadata_path}")

print(f"\n🎉 Model training complete!")
print(f"   Final Test MAE: {test_mae:.2f} tokens")
print(f"   Final Test R²: {test_r2:.4f}")
print(f"   Model ready for production use!")
```

```python
# Cell 10: Test Model Loading and Prediction
print("\n🧪 Testing model loading and prediction...")

# Test loading the saved model
try:
    loaded_model = joblib.load(model_path)
    loaded_encoders = joblib.load(encoders_path)
    loaded_features = joblib.load(features_path)
    
    print("✅ Successfully loaded all model artifacts")
    
    # Test prediction with a sample query
    test_query = "How do I debug Python code that's throwing an error?"
    
    # Extract features
    test_features = extract_optimized_features(test_query)
    
    # Prepare feature vector
    feature_vector = []
    for feature_name in loaded_features:
        if feature_name in test_features:
            feature_vector.append(test_features[feature_name])
        else:
            feature_vector.append(0)  # Default value
    
    # Make prediction
    prediction = loaded_model.predict([feature_vector])[0]
    
    print(f"\n🔮 Test Prediction:")
    print(f"   Query: '{test_query}'")
    print(f"   Predicted tokens: {prediction:.0f}")
    print(f"   Features extracted: {len(test_features)}")
    
    print("\n✅ Model loading and prediction test successful!")
    
except Exception as e:
    print(f"❌ Error during model loading test: {str(e)}")

print("\n🏁 Retraining process complete!")
```

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

### **GitHub README Ready:**
This format is optimized for copying into GitHub README.md and then easily transferring to any Jupyter environment. Each cell is clearly separated and ready for enterprise use.

## License

This project is licensed under the MIT License.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
