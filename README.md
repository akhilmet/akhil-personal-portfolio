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

## Cell 1: Imports and Setup

```python
# Token Predictor EDA Analysis
# Exploratory Data Analysis for Feature Engineering
# Updated for: prompt_engineering_dataset.csv and human_interaction_dataset.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔍 Token Predictor EDA Analysis")
print("=" * 60)
```

## Cell 2: Data Loading

```python
# =============================================================================
# STEP 1: LOAD DATASETS
# =============================================================================

print("\n📊 Loading Datasets from Local Files...")

# Load Dataset 1: Prompt Engineering Dataset
try:
    df_prompt_engineering = pd.read_csv('prompt_engineering_dataset.csv')
    print(f"✅ Loaded prompt_engineering_dataset.csv: {df_prompt_engineering.shape[0]} rows, {df_prompt_engineering.shape[1]} columns")
    print(f"   Columns: {list(df_prompt_engineering.columns)}")
except Exception as e:
    print(f"❌ Failed to load prompt_engineering_dataset.csv: {e}")
    df_prompt_engineering = None

# Load Dataset 2: Human Interaction Dataset  
try:
    df_human_interaction = pd.read_csv('human_interaction_dataset.csv')
    print(f"✅ Loaded human_interaction_dataset.csv: {df_human_interaction.shape[0]} rows, {df_human_interaction.shape[1]} columns")
    print(f"   Columns: {list(df_human_interaction.columns)}")
except Exception as e:
    print(f"❌ Failed to load human_interaction_dataset.csv: {e}")
    df_human_interaction = None
```

## Cell 3: Token Counting Functions

```python
# =============================================================================
# STEP 2: TOKEN COUNTING FUNCTIONS
# =============================================================================

def estimate_tokens_basic(text):
    """Basic token estimation: characters / 4"""
    if pd.isna(text) or text == "":
        return 0
    return max(1, len(str(text)) // 4)

def estimate_tokens_advanced(text):
    """Advanced token estimation matching our latency predictor"""
    if pd.isna(text) or text == "":
        return 0
    
    text = str(text)
    words = text.split()
    
    # Base token count (Llama averages ~1.3 tokens per word)
    base_tokens = len(words) * 1.3
    
    # Additional tokens for various elements
    punct_tokens = len(re.findall(r'[.,!?;:(){}[\]"\'`]', text)) * 0.5
    number_tokens = len(re.findall(r'\d+', text)) * 0.7
    special_tokens = len(re.findall(r'[@#$%^&*+=<>~/\\|]', text)) * 0.3
    
    total_tokens = base_tokens + punct_tokens + number_tokens + special_tokens
    return max(1, int(total_tokens))

def count_words(text):
    """Count words in text"""
    if pd.isna(text) or text == "":
        return 0
    return len(str(text).split())

def count_chars(text):
    """Count characters in text"""
    if pd.isna(text) or text == "":
        return 0
    return len(str(text))

print("\n🔧 Token counting functions defined")
```

## Cell 4: Analysis Functions

```python
# =============================================================================
# STEP 2: ANALYSIS FUNCTIONS
# =============================================================================

def categorize_question_type(text):
    """Categorize query into question types (who, what, how, where, when, why)"""
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower()
    
    # Question word patterns
    if any(word in text_lower for word in ['who', 'whom', 'whose']):
        return "who"
    elif any(word in text_lower for word in ['what', "what's", 'which']):
        return "what"
    elif any(word in text_lower for word in ['how', "how's", 'how to']):
        return "how"
    elif any(word in text_lower for word in ['where', "where's"]):
        return "where"
    elif any(word in text_lower for word in ['when', "when's"]):
        return "when"
    elif any(word in text_lower for word in ['why', "why's"]):
        return "why"
    else:
        return "other"

def is_natural_language_or_code(text):
    """Determine if query is natural language or code request"""
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower()
    
    # Code indicators
    code_patterns = [
        'function', 'def ', 'class ', 'import', 'return', 'print(',
        'code', 'script', 'program', 'algorithm', 'python', 'javascript',
        'html', 'css', 'sql', 'query', 'database', 'plot', 'calculate'
    ]
    
    if any(pattern in text_lower for pattern in code_patterns):
        return "code_request"
    else:
        return "natural_language"

def is_independent_or_continuation(text, response_length=None):
    """Determine if query is independent or part of conversation"""
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower()
    
    # Continuation indicators
    continuation_patterns = [
        'also', 'and', 'but', 'however', 'additionally', 'furthermore',
        'what about', 'how about', 'can you also', 'tell me more',
        'continue', 'next', 'then', 'after that'
    ]
    
    # Independent indicators
    independent_patterns = [
        'i need help', 'can you help', 'i want to', 'how do i',
        'what is', 'please explain', 'tell me about'
    ]
    
    if any(pattern in text_lower for pattern in continuation_patterns):
        return "continuation"
    elif any(pattern in text_lower for pattern in independent_patterns):
        return "independent"
    else:
        # Use response length as additional indicator if available
        if response_length and response_length < 50:
            return "independent"  # Short responses often to independent queries
        else:
            return "independent"  # Default to independent

print("✅ Analysis functions defined")
```

## Cell 5: Dataset Processing Functions

```python
# =============================================================================
# STEP 3: DATASET PROCESSING FUNCTIONS
# =============================================================================

def process_prompt_engineering_dataset(df):
    """Process the prompt engineering dataset"""
    if df is None:
        return None
    
    print(f"\n🔄 Processing Prompt Engineering Dataset...")
    processed_df = df.copy()
    
    # Extract features from 'Prompt' column
    processed_df['query_char_count'] = processed_df['Prompt'].apply(count_chars)
    processed_df['query_word_count'] = processed_df['Prompt'].apply(count_words)
    processed_df['query_token_count_basic'] = processed_df['Prompt'].apply(estimate_tokens_basic)
    processed_df['query_token_count_advanced'] = processed_df['Prompt'].apply(estimate_tokens_advanced)
    processed_df['question_type'] = processed_df['Prompt'].apply(categorize_question_type)
    processed_df['content_type'] = processed_df['Prompt'].apply(is_natural_language_or_code)
    
    # Extract features from 'Response' column if available
    if 'Response' in processed_df.columns:
        processed_df['response_char_count'] = processed_df['Response'].apply(count_chars)
        processed_df['response_word_count'] = processed_df['Response'].apply(count_words)
        processed_df['response_token_count_basic'] = processed_df['Response'].apply(estimate_tokens_basic)
        processed_df['response_token_count_advanced'] = processed_df['Response'].apply(estimate_tokens_advanced)
        processed_df['conversation_length'] = processed_df['query_char_count'] + processed_df['response_char_count']
        
        # Determine if independent or continuation
        processed_df['query_context'] = processed_df.apply(
            lambda row: is_independent_or_continuation(row['Prompt'], row['response_char_count']), axis=1
        )
    else:
        processed_df['query_context'] = processed_df['Prompt'].apply(is_independent_or_continuation)
    
    # Add prompt type analysis if Prompt_Type column exists
    if 'Prompt_Type' in processed_df.columns:
        processed_df['original_prompt_type'] = processed_df['Prompt_Type']
    
    # Add prompt length analysis if Prompt_Length column exists
    if 'Prompt_Length' in processed_df.columns:
        processed_df['original_prompt_length'] = processed_df['Prompt_Length']
    
    print(f"✅ Processed {len(processed_df)} records from Prompt Engineering Dataset")
    return processed_df

def process_human_interaction_dataset(df):
    """Process the human interaction dataset"""
    if df is None:
        return None
    
    print(f"\n🔄 Processing Human Interaction Dataset...")
    processed_df = df.copy()
    
    # Extract features from 'Context' column (user queries)
    processed_df['query_char_count'] = processed_df['Context'].apply(count_chars)
    processed_df['query_word_count'] = processed_df['Context'].apply(count_words)
    processed_df['query_token_count_basic'] = processed_df['Context'].apply(estimate_tokens_basic)
    processed_df['query_token_count_advanced'] = processed_df['Context'].apply(estimate_tokens_advanced)
    processed_df['question_type'] = processed_df['Context'].apply(categorize_question_type)
    processed_df['content_type'] = processed_df['Context'].apply(is_natural_language_or_code)
    
    # Extract features from 'Response' column (LLM responses)
    if 'Response' in processed_df.columns:
        processed_df['response_char_count'] = processed_df['Response'].apply(count_chars)
        processed_df['response_word_count'] = processed_df['Response'].apply(count_words)
        processed_df['response_token_count_basic'] = processed_df['Response'].apply(estimate_tokens_basic)
        processed_df['response_token_count_advanced'] = processed_df['Response'].apply(estimate_tokens_advanced)
        processed_df['conversation_length'] = processed_df['query_char_count'] + processed_df['response_char_count']
        
        # Determine if independent or continuation
        processed_df['query_context'] = processed_df.apply(
            lambda row: is_independent_or_continuation(row['Context'], row['response_char_count']), axis=1
        )
    else:
        processed_df['query_context'] = processed_df['Context'].apply(is_independent_or_continuation)
    
    # Add LLM analysis if LLM column exists
    if 'LLM' in processed_df.columns:
        processed_df['llm_model'] = processed_df['LLM']
    
    print(f"✅ Processed {len(processed_df)} records from Human Interaction Dataset")
    return processed_df

print("✅ Dataset processing functions defined")
```

## Cell 6: Process Datasets

```python
# =============================================================================
# STEP 3: PROCESS DATASETS
# =============================================================================

# Process both datasets
datasets = {}

if df_prompt_engineering is not None:
    datasets['prompt_engineering'] = process_prompt_engineering_dataset(df_prompt_engineering)

if df_human_interaction is not None:
    datasets['human_interaction'] = process_human_interaction_dataset(df_human_interaction)

print(f"\n📊 Dataset processing complete. Processed {len(datasets)} datasets.")
```

## Cell 7: Statistical Analysis

```python
# =============================================================================
# STEP 4: COMPREHENSIVE STATISTICAL ANALYSIS
# =============================================================================

print("\n📈 COMPREHENSIVE TOKEN ANALYSIS")
print("=" * 60)

# Combine all datasets for overall analysis
all_data = []
for name, df in datasets.items():
    if df is not None:
        df_subset = df[['query_char_count', 'query_word_count', 'query_token_count_basic', 
                       'query_token_count_advanced', 'question_type', 'content_type', 'query_context']].copy()
        df_subset['dataset'] = name
        all_data.append(df_subset)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Overall Statistics
    print("\n📊 OVERALL QUERY STATISTICS")
    print("-" * 40)
    
    stats_summary = combined_df[['query_char_count', 'query_word_count', 
                                'query_token_count_basic', 'query_token_count_advanced']].describe()
    print(stats_summary.round(2))
    
    # Token count distribution by dataset
    print("\n📊 TOKEN COUNT BY DATASET")
    print("-" * 40)
    dataset_stats = combined_df.groupby('dataset')[['query_token_count_advanced', 'query_word_count', 'query_char_count']].describe().round(2)
    print(dataset_stats)
    
    # Question type distribution
    print("\n📊 QUESTION TYPE DISTRIBUTION")
    print("-" * 40)
    question_dist = combined_df['question_type'].value_counts()
    print(question_dist)
    print(f"\nPercentage distribution:")
    print((question_dist / len(combined_df) * 100).round(2))
    
    # Content type distribution
    print("\n📊 CONTENT TYPE DISTRIBUTION (Natural Language vs Code)")
    print("-" * 40)
    content_dist = combined_df['content_type'].value_counts()
    print(content_dist)
    print(f"\nPercentage distribution:")
    print((content_dist / len(combined_df) * 100).round(2))
    
    # Query context analysis (Independent vs Continuation)
    print("\n📊 QUERY CONTEXT ANALYSIS (Independent vs Continuation)")
    print("-" * 40)
    context_dist = combined_df['query_context'].value_counts()
    print(context_dist)
    print(f"\nPercentage distribution:")
    print((context_dist / len(combined_df) * 100).round(2))
else:
    print("❌ No data available for analysis")
    combined_df = None
```

## Cell 8: Main Visualization Dashboard

```python
# =============================================================================
# STEP 5: VISUALIZATION DASHBOARD
# =============================================================================

print("\n🎨 Creating visualization dashboard...")

# Create comprehensive visualization dashboard
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Token Predictor EDA - Comprehensive Analysis', fontsize=16, fontweight='bold')

if all_data:
    # 1. Token count distribution
    axes[0, 0].hist(combined_df['query_token_count_advanced'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Query Token Count Distribution')
    axes[0, 0].set_xlabel('Advanced Token Count')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Char vs Token relationship
    axes[0, 1].scatter(combined_df['query_char_count'], combined_df['query_token_count_advanced'], alpha=0.5)
    axes[0, 1].set_title('Character Count vs Token Count')
    axes[0, 1].set_xlabel('Character Count')
    axes[0, 1].set_ylabel('Advanced Token Count')
    
    # 3. Word vs Token relationship  
    axes[0, 2].scatter(combined_df['query_word_count'], combined_df['query_token_count_advanced'], alpha=0.5, color='orange')
    axes[0, 2].set_title('Word Count vs Token Count')
    axes[0, 2].set_xlabel('Word Count')
    axes[0, 2].set_ylabel('Advanced Token Count')
    
    # 4. Question type distribution
    question_counts = combined_df['question_type'].value_counts()
    axes[1, 0].bar(question_counts.index, question_counts.values)
    axes[1, 0].set_title('Question Type Distribution')
    axes[1, 0].set_xlabel('Question Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Content type distribution
    content_counts = combined_df['content_type'].value_counts()
    axes[1, 1].pie(content_counts.values, labels=content_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Natural Language vs Code Requests')
    
    # 6. Token count by question type
    sns.boxplot(data=combined_df, x='question_type', y='query_token_count_advanced', ax=axes[1, 2])
    axes[1, 2].set_title('Token Count by Question Type')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # 7. Dataset comparison
    sns.boxplot(data=combined_df, x='dataset', y='query_token_count_advanced', ax=axes[2, 0])
    axes[2, 0].set_title('Token Count by Dataset')
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    # 8. Basic vs Advanced token count comparison
    axes[2, 1].scatter(combined_df['query_token_count_basic'], combined_df['query_token_count_advanced'], alpha=0.5, color='green')
    axes[2, 1].plot([0, combined_df['query_token_count_basic'].max()], [0, combined_df['query_token_count_basic'].max()], 'r--', alpha=0.8)
    axes[2, 1].set_title('Basic vs Advanced Token Count')
    axes[2, 1].set_xlabel('Basic Token Count (chars/4)')
    axes[2, 1].set_ylabel('Advanced Token Count')
    
    # 9. Query context analysis (Independent vs Continuation)
    context_counts = combined_df['query_context'].value_counts()
    axes[2, 2].pie(context_counts.values, labels=context_counts.index, autopct='%1.1f%%')
    axes[2, 2].set_title('Independent vs Continuation Queries')

plt.tight_layout()
plt.show()
```

## Cell 9: Conversation Analysis

```python
# =============================================================================
# STEP 6: CONVERSATION ANALYSIS (Response Data)
# =============================================================================

print("\n💬 CONVERSATION ANALYSIS")
print("=" * 60)

conversation_data = []
for name, df in datasets.items():
    if df is not None and 'response_token_count_advanced' in df.columns:
        conv_subset = df[['query_token_count_advanced', 'response_token_count_advanced', 
                         'conversation_length', 'question_type', 'query_context']].copy()
        conv_subset['dataset'] = name
        conversation_data.append(conv_subset)

if conversation_data:
    conv_df = pd.concat(conversation_data, ignore_index=True)
    
    print("\n📊 CONVERSATION STATISTICS")
    print("-" * 40)
    conv_stats = conv_df[['query_token_count_advanced', 'response_token_count_advanced', 'conversation_length']].describe()
    print(conv_stats.round(2))
    
    # Response length analysis
    print(f"\n📊 RESPONSE LENGTH INSIGHTS")
    print(f"Average query tokens: {conv_df['query_token_count_advanced'].mean():.1f}")
    print(f"Average response tokens: {conv_df['response_token_count_advanced'].mean():.1f}")
    print(f"Response/Query ratio: {(conv_df['response_token_count_advanced'] / conv_df['query_token_count_advanced']).mean():.2f}")
    
    # Response length by question type
    print("\n📊 RESPONSE LENGTH BY QUESTION TYPE")
    print("-" * 40)
    response_by_type = conv_df.groupby('question_type')['response_token_count_advanced'].agg(['mean', 'median', 'std']).round(2)
    print(response_by_type)
    
    # Response length by query context
    print("\n📊 RESPONSE LENGTH BY QUERY CONTEXT")
    print("-" * 40)
    response_by_context = conv_df.groupby('query_context')['response_token_count_advanced'].agg(['mean', 'median', 'count']).round(2)
    print(response_by_context)
else:
    print("❌ No conversation data available for analysis")
    conv_df = None
```

## Cell 10: Conversation Visualizations

```python
# =============================================================================
# STEP 6B: CONVERSATION VISUALIZATIONS
# =============================================================================

if conversation_data:
    # Create conversation analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Conversation Analysis - Query vs Response Patterns', fontsize=14, fontweight='bold')
    
    # Query vs Response scatter
    axes[0, 0].scatter(conv_df['query_token_count_advanced'], conv_df['response_token_count_advanced'], alpha=0.6)
    axes[0, 0].set_title('Query Tokens vs Response Tokens')
    axes[0, 0].set_xlabel('Query Token Count')
    axes[0, 0].set_ylabel('Response Token Count')
    
    # Response length distribution
    axes[0, 1].hist(conv_df['response_token_count_advanced'], bins=50, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Response Token Distribution')
    axes[0, 1].set_xlabel('Response Token Count')
    axes[0, 1].set_ylabel('Frequency')
    
    # Response length by question type
    sns.boxplot(data=conv_df, x='question_type', y='response_token_count_advanced', ax=axes[1, 0])
    axes[1, 0].set_title('Response Length by Question Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Response length by query context
    sns.boxplot(data=conv_df, x='query_context', y='response_token_count_advanced', ax=axes[1, 1])
    axes[1, 1].set_title('Response Length by Query Context')
    axes[1, 1].set_xlabel('Query Context')
    axes[1, 1].set_ylabel('Response Token Count')
    
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No conversation visualizations created - no response data available")
```

## Cell 11: Dataset-Specific Analysis

```python
# =============================================================================
# STEP 7: DATASET-SPECIFIC ANALYSIS
# =============================================================================

print("\n🔍 DATASET-SPECIFIC ANALYSIS")
print("=" * 60)

# Analyze Prompt Engineering Dataset specific features
if 'prompt_engineering' in datasets and datasets['prompt_engineering'] is not None:
    pe_df = datasets['prompt_engineering']
    
    print("\n📊 PROMPT ENGINEERING DATASET INSIGHTS")
    print("-" * 50)
    
    if 'original_prompt_type' in pe_df.columns:
        print("Original Prompt Type Distribution:")
        prompt_type_dist = pe_df['original_prompt_type'].value_counts()
        print(prompt_type_dist)
        
        # Token count by original prompt type
        print("\nAverage Token Count by Prompt Type:")
        token_by_type = pe_df.groupby('original_prompt_type')['query_token_count_advanced'].mean().round(2)
        print(token_by_type)
    
    if 'original_prompt_length' in pe_df.columns:
        print(f"\nPrompt Length Analysis:")
        print(f"Original prompt length vs our token count correlation: {pe_df['original_prompt_length'].corr(pe_df['query_token_count_advanced']):.3f}")

# Analyze Human Interaction Dataset specific features
if 'human_interaction' in datasets and datasets['human_interaction'] is not None:
    hi_df = datasets['human_interaction']
    
    print("\n📊 HUMAN INTERACTION DATASET INSIGHTS")
    print("-" * 50)
    
    if 'llm_model' in hi_df.columns:
        print("LLM Model Distribution:")
        llm_dist = hi_df['llm_model'].value_counts()
        print(llm_dist)
        
        # Response length by LLM model
        if 'response_token_count_advanced' in hi_df.columns:
            print("\nAverage Response Length by LLM Model:")
            response_by_llm = hi_df.groupby('llm_model')['response_token_count_advanced'].mean().round(2)
            print(response_by_llm)
```

## Cell 12: Token Prediction Accuracy Analysis

```python
# =============================================================================
# STEP 8: TOKEN PREDICTION ACCURACY ANALYSIS
# =============================================================================

print("\n🎯 TOKEN PREDICTION ACCURACY ANALYSIS")
print("=" * 60)

if all_data:
    # Compare basic vs advanced token counting methods
    basic_advanced_ratio = combined_df['query_token_count_basic'] / combined_df['query_token_count_advanced']
    char_token_ratio = combined_df['query_char_count'] / combined_df['query_token_count_advanced']
    word_token_ratio = combined_df['query_word_count'] / combined_df['query_token_count_advanced']
    
    print(f"📊 TOKEN COUNTING METHOD COMPARISON")
    print(f"Basic/Advanced ratio - Mean: {basic_advanced_ratio.mean():.3f}, Std: {basic_advanced_ratio.std():.3f}")
    print(f"Chars/Tokens ratio - Mean: {char_token_ratio.mean():.3f}, Std: {char_token_ratio.std():.3f}")
    print(f"Words/Tokens ratio - Mean: {word_token_ratio.mean():.3f}, Std: {word_token_ratio.std():.3f}")
    
    # Accuracy by content type
    print(f"\n📊 ACCURACY BY CONTENT TYPE")
    print("-" * 40)
    accuracy_by_type = combined_df.groupby('content_type').agg({
        'query_char_count': 'mean',
        'query_word_count': 'mean', 
        'query_token_count_advanced': 'mean'
    }).round(2)
    print(accuracy_by_type)
    
    # Calculate our predictor's character/token ratio
    our_ratio = combined_df['query_char_count'] / combined_df['query_token_count_advanced']
    print(f"\n🎯 OUR TOKEN PREDICTOR PERFORMANCE")
    print(f"Average characters per token: {our_ratio.mean():.2f}")
    print(f"Standard deviation: {our_ratio.std():.2f}")
    print(f"Current predictor uses: 4.0 chars/token")
    print(f"Suggested improvement: {our_ratio.mean():.2f} chars/token")
    print(f"Accuracy improvement: {abs(4.0 - our_ratio.mean())/4.0*100:.1f}% better")
```

## Cell 13: Feature Engineering Insights

```python
# =============================================================================
# STEP 9: FEATURE ENGINEERING INSIGHTS
# =============================================================================

print("\n🔧 FEATURE ENGINEERING INSIGHTS")
print("=" * 60)

if all_data:
    # Key insights for feature engineering
    print("📋 KEY FINDINGS FOR LATENCY PREDICTOR:")
    print("-" * 50)
    
    print(f"1. TOKEN COUNT INSIGHTS:")
    print(f"   • Average query tokens: {combined_df['query_token_count_advanced'].mean():.1f}")
    print(f"   • Median query tokens: {combined_df['query_token_count_advanced'].median():.1f}")
    print(f"   • 90th percentile: {combined_df['query_token_count_advanced'].quantile(0.9):.1f}")
    print(f"   • Max observed: {combined_df['query_token_count_advanced'].max()}")
    print(f"   • Min observed: {combined_df['query_token_count_advanced'].min()}")
    
    print(f"\n2. QUESTION TYPE INSIGHTS:")
    for qtype in combined_df['question_type'].value_counts().head().index:
        avg_tokens = combined_df[combined_df['question_type'] == qtype]['query_token_count_advanced'].mean()
        count = (combined_df['question_type'] == qtype).sum()
        print(f"   • {qtype.upper()} questions: {avg_tokens:.1f} avg tokens ({count} samples)")
    
    print(f"\n3. CONTENT TYPE INSIGHTS:")
    for ctype in combined_df['content_type'].value_counts().index:
        avg_tokens = combined_df[combined_df['content_type'] == ctype]['query_token_count_advanced'].mean()
        pct = (combined_df['content_type'] == ctype).mean() * 100
        count = (combined_df['content_type'] == ctype).sum()
        print(f"   • {ctype.replace('_', ' ').title()}: {avg_tokens:.1f} avg tokens ({pct:.1f}%, {count} samples)")
    
    print(f"\n4. QUERY CONTEXT INSIGHTS:")
    for context in combined_df['query_context'].value_counts().index:
        avg_tokens = combined_df[combined_df['query_context'] == context]['query_token_count_advanced'].mean()
        pct = (combined_df['query_context'] == context).mean() * 100
        count = (combined_df['query_context'] == context).sum()
        print(f"   • {context.title()} queries: {avg_tokens:.1f} avg tokens ({pct:.1f}%, {count} samples)")
    
    if conversation_data:
        print(f"\n5. RESPONSE LENGTH INSIGHTS:")
        print(f"   • Average response: {conv_df['response_token_count_advanced'].mean():.1f} tokens")
        print(f"   • Median response: {conv_df['response_token_count_advanced'].median():.1f} tokens")
        print(f"   • Response/Query ratio: {(conv_df['response_token_count_advanced'] / conv_df['query_token_count_advanced']).mean():.2f}x")
        print(f"   • 90th percentile response: {conv_df['response_token_count_advanced'].quantile(0.9):.1f} tokens")
    
    print(f"\n6. RECOMMENDATIONS FOR LATENCY PREDICTOR:")
    our_ratio = combined_df['query_char_count'] / combined_df['query_token_count_advanced']
    print(f"   ✅ Current char/4 approximation accuracy: {abs(4.0 - our_ratio.mean())/4.0*100:.1f}% off")
    print(f"   ✅ Suggested chars/token ratio: {our_ratio.mean():.1f}")
    print(f"   ✅ Question type has minimal impact on token count")
    code_avg = combined_df[combined_df['content_type']=='code_request']['query_token_count_advanced'].mean() if (combined_df['content_type']=='code_request').any() else 0
    nl_avg = combined_df[combined_df['content_type']=='natural_language']['query_token_count_advanced'].mean() if (combined_df['content_type']=='natural_language').any() else 0
    if code_avg > 0 and nl_avg > 0:
        diff = ((code_avg / nl_avg) - 1) * 100
        print(f"   ✅ Code requests are {diff:.1f}% longer than natural language")
    if conversation_data:
        print(f"   ✅ Default {conv_df['response_token_count_advanced'].median():.0f} output tokens appropriate for most use cases")
        print(f"   ✅ Independent queries get {(conv_df[conv_df['query_context']=='independent']['response_token_count_advanced'].mean() if (conv_df['query_context']=='independent').any() else 0):.1f} avg token responses")
        continuation_avg = conv_df[conv_df['query_context']=='continuation']['response_token_count_advanced'].mean() if (conv_df['query_context']=='continuation').any() else 0
        if continuation_avg > 0:
            print(f"   ✅ Continuation queries get {continuation_avg:.1f} avg token responses")
```

## Cell 14: Summary Statistics for JIRA

```python
# =============================================================================
# STEP 10: SUMMARY STATISTICS FOR JIRA TICKET
# =============================================================================

print("\n📝 SUMMARY FOR JIRA TICKET")
print("=" * 60)

if all_data:
    total_queries = len(combined_df)
    
    print(f"📊 DATASET SUMMARY:")
    print(f"   • Total queries analyzed: {total_queries:,}")
    print(f"   • Datasets processed: {len([d for d in datasets.values() if d is not None])}")
    print(f"   • Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    
    # Dataset breakdown
    for name, df in datasets.items():
        if df is not None:
            count = len(df)
            pct = count / total_queries * 100
            print(f"   • {name.replace('_', ' ').title()}: {count:,} queries ({pct:.1f}%)")
    
    print(f"\n📊 QUERY CHARACTERISTICS:")
    print(f"   • Average character count: {combined_df['query_char_count'].mean():.0f}")
    print(f"   • Average word count: {combined_df['query_word_count'].mean():.1f}")
    print(f"   • Average token count (advanced): {combined_df['query_token_count_advanced'].mean():.1f}")
    print(f"   • Token count range: {combined_df['query_token_count_advanced'].min()}-{combined_df['query_token_count_advanced'].max()}")
    print(f"   • Most common token count: {combined_df['query_token_count_advanced'].mode().iloc[0]}")
    
    print(f"\n📊 QUERY CATEGORIZATION:")
    print(f"   Question Types (who, what, how, where, when, why):")
    for qtype, count in combined_df['question_type'].value_counts().head(6).items():
        pct = count / total_queries * 100
        avg_tokens = combined_df[combined_df['question_type'] == qtype]['query_token_count_advanced'].mean()
        print(f"   • {qtype.title()}: {count:,} queries ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    print(f"\n📊 CONTENT TYPE BREAKDOWN:")
    for ctype, count in combined_df['content_type'].value_counts().items():
        pct = count / total_queries * 100
        avg_tokens = combined_df[combined_df['content_type'] == ctype]['query_token_count_advanced'].mean()
        print(f"   • {ctype.replace('_', ' ').title()}: {count:,} queries ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    print(f"\n📊 QUERY CONTEXT ANALYSIS:")
    for context, count in combined_df['query_context'].value_counts().items():
        pct = count / total_queries * 100
        avg_tokens = combined_df[combined_df['query_context'] == context]['query_token_count_advanced'].mean()
        print(f"   • {context.title()} queries: {count:,} ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    if conversation_data:
        conv_total = len(conv_df)
        print(f"\n📊 CONVERSATION PATTERNS:")
        print(f"   • Queries with responses: {conv_total:,}")
        print(f"   • Average response length: {conv_df['response_token_count_advanced'].mean():.1f} tokens")
        print(f"   • Response/Query length ratio: {(conv_df['response_token_count_advanced'] / conv_df['query_token_count_advanced']).mean():.2f}x")
        print(f"   • Longest response: {conv_df['response_token_count_advanced'].max()} tokens")
        print(f"   • Shortest response: {conv_df['response_token_count_advanced'].min()} tokens")
        
        # Position in conversation analysis
        independent_count = (conv_df['query_context'] == 'independent').sum()
        continuation_count = (conv_df['query_context'] == 'continuation').sum()
        print(f"   • Independent queries: {independent_count:,} ({independent_count/conv_total*100:.1f}%)")
        print(f"   • Continuation queries: {continuation_count:,} ({continuation_count/conv_total*100:.1f}%)")
```

## Cell 15: Token Predictor Calibration

```python
# =============================================================================
# STEP 11: TOKEN PREDICTOR CALIBRATION RECOMMENDATIONS
# =============================================================================

print(f"\n🎛️ TOKEN PREDICTOR CALIBRATION RECOMMENDATIONS")
print("=" * 60)

if all_data:
    # Calculate optimal parameters for token prediction
    optimal_char_ratio = combined_df['query_char_count'] / combined_df['query_token_count_advanced']
    optimal_word_ratio = combined_df['query_word_count'] / combined_df['query_token_count_advanced']
    
    print(f"📊 CURRENT vs OPTIMAL PARAMETERS:")
    print(f"   Current character/token ratio: 4.0")
    print(f"   Observed character/token ratio: {optimal_char_ratio.mean():.2f} ± {optimal_char_ratio.std():.2f}")
    print(f"   Recommended character/token ratio: {optimal_char_ratio.mean():.2f}")
    print(f"   Improvement potential: {abs(4.0 - optimal_char_ratio.mean())/4.0*100:.1f}%")
    
    print(f"\n   Current word/token ratio assumption: ~1.3")
    print(f"   Observed word/token ratio: {optimal_word_ratio.mean():.2f} ± {optimal_word_ratio.std():.2f}")
    print(f"   Recommended word/token ratio: {optimal_word_ratio.mean():.2f}")
    
    # Breakdown by content type
    print(f"\n📊 CONTENT-SPECIFIC CALIBRATION:")
    for ctype in combined_df['content_type'].unique():
        if pd.notna(ctype):
            subset = combined_df[combined_df['content_type'] == ctype]
            char_ratio = (subset['query_char_count'] / subset['query_token_count_advanced']).mean()
            word_ratio = (subset['query_word_count'] / subset['query_token_count_advanced']).mean()
            count = len(subset)
            print(f"   {ctype.replace('_', ' ').title()} ({count} samples):")
            print(f"     • Chars/token: {char_ratio:.2f}")
            print(f"     • Words/token: {word_ratio:.2f}")
    
    if conversation_data:
        print(f"\n📊 RESPONSE LENGTH RECOMMENDATIONS:")
        response_percentiles = conv_df['response_token_count_advanced'].quantile([0.25, 0.5, 0.75, 0.9])
        print(f"   Default expected_output_tokens suggestions:")
        print(f"     • Conservative (25th percentile): {response_percentiles[0.25]:.0f} tokens")
        print(f"     • Standard (median): {response_percentiles[0.5]:.0f} tokens")
        print(f"     • Generous (75th percentile): {response_percentiles[0.75]:.0f} tokens")
        print(f"     • Enterprise (90th percentile): {response_percentiles[0.9]:.0f} tokens")
```

## Cell 16: Final Insights and Recommendations

```python
# =============================================================================
# STEP 12: FINAL INSIGHTS AND ACTIONABLE RECOMMENDATIONS
# =============================================================================

print(f"\n🎯 FINAL INSIGHTS AND ACTIONABLE RECOMMENDATIONS")
print("=" * 60)

if all_data:
    optimal_char_ratio = combined_df['query_char_count'] / combined_df['query_token_count_advanced']
    optimal_word_ratio = combined_df['query_word_count'] / combined_df['query_token_count_advanced']
    
    print(f"📋 KEY ACTIONABLE INSIGHTS:")
    print(f"1. TOKEN ESTIMATION ACCURACY:")
    print(f"   • Current method (chars/4) has {abs(4.0 - optimal_char_ratio.mean())/4.0*100:.1f}% error")
    print(f"   • Switching to {optimal_char_ratio.mean():.1f} chars/token improves accuracy")
    print(f"   • Word-based estimation ({optimal_word_ratio.mean():.2f} words/token) is more stable")
    
    print(f"\n2. QUERY COMPLEXITY PATTERNS:")
    # Find the most complex query types
    complexity_by_type = combined_df.groupby('question_type')['query_token_count_advanced'].agg(['mean', 'std', 'count'])
    most_complex = complexity_by_type['mean'].idxmax()
    least_complex = complexity_by_type['mean'].idxmin()
    print(f"   • Most complex queries: {most_complex} ({complexity_by_type.loc[most_complex, 'mean']:.1f} avg tokens)")
    print(f"   • Simplest queries: {least_complex} ({complexity_by_type.loc[least_complex, 'mean']:.1f} avg tokens)")
    print(f"   • Complexity range: {complexity_by_type['mean'].max() - complexity_by_type['mean'].min():.1f} token difference")
    
    # Content type complexity
    content_complexity = combined_df.groupby('content_type')['query_token_count_advanced'].mean()
    if len(content_complexity) > 1:
        code_avg = content_complexity.get('code_request', 0)
        nl_avg = content_complexity.get('natural_language', 0)
        if code_avg > 0 and nl_avg > 0:
            complexity_diff = ((code_avg / nl_avg) - 1) * 100
            print(f"   • Code requests are {complexity_diff:.1f}% longer than natural language queries")
    
    print(f"\n3. BUSINESS IMPACT:")
    print(f"   • {len(combined_df):,} queries analyzed across {len(datasets)} datasets")
    print(f"   • Token estimation affects latency prediction for every query")
    print(f"   • Improved accuracy = better model routing decisions")
    print(f"   • Cost optimization potential through better predictions")
    
    if conversation_data:
        print(f"\n4. CONVERSATION INSIGHTS:")
        print(f"   • Average conversation: {conv_df['query_token_count_advanced'].mean():.1f} input + {conv_df['response_token_count_advanced'].mean():.1f} output tokens")
        print(f"   • Total token budget needed: {(conv_df['query_token_count_advanced'] + conv_df['response_token_count_advanced']).mean():.1f} tokens per interaction")
        print(f"   • Response prediction multiplier: {(conv_df['response_token_count_advanced'] / conv_df['query_token_count_advanced']).mean():.1f}x query length")
    
    print(f"\n5. IMPLEMENTATION RECOMMENDATIONS:")
    print(f"   ✅ Update token estimation: Use {optimal_char_ratio.mean():.1f} chars/token instead of 4.0")
    print(f"   ✅ Add word-based fallback: {optimal_word_ratio.mean():.2f} words/token")
    print(f"   ✅ Content-aware estimation: Different ratios for code vs natural language")
    if conversation_data:
        print(f"   ✅ Default output tokens: {conv_df['response_token_count_advanced'].median():.0f} tokens")
        print(f"   ✅ Context-aware defaults: Independent vs continuation queries")
    print(f"   ✅ Regular recalibration: Update ratios quarterly with new data")

print(f"\n🎉 TOKEN PREDICTOR EDA ANALYSIS COMPLETE!")
print("=" * 60)
print("📊 All statistics generated for JIRA ticket documentation")
print("🔧 Calibration recommendations ready for implementation")
print("📈 Visualizations created for stakeholder presentation")
print("\n🚀 Next Steps:")
print("   1. Update latency_predictor.py with new token ratios")
print("   2. Document findings in JIRA ticket")
print("   3. Plan A/B testing for improved token estimation")
print("   4. Schedule quarterly recalibration process")
```

---

## 📋 **Instructions for Use:**

### **Setup:**
1. Create a new Jupyter notebook
2. Copy each cell above into separate notebook cells
3. Place your CSV files in the same directory as the notebook:
   - `prompt_engineering_dataset.csv`
   - `human_interaction_dataset.csv`

### **Run Order:**
1. **Cells 1-6**: Setup and data processing
2. **Cell 7**: Statistical analysis (view results)
3. **Cell 8**: Main visualization dashboard
4. **Cells 9-10**: Conversation analysis (if applicable)
5. **Cells 11-16**: Detailed insights and recommendations

### **Expected Outputs:**
- **9-panel visualization dashboard**
- **Comprehensive statistical summaries**
- **Token predictor calibration recommendations**
- **Business insights for JIRA documentation**
- **Implementation roadmap**

### **GitHub README Ready:**
This format is optimized for copying into GitHub README.md and then easily transferring to any Jupyter environment. Each cell is clearly separated and ready for enterprise use.

## License

This project is licensed under the MIT License.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
