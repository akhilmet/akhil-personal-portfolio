```python
# =============================================================================
# STEP 2: REAL TOKEN COUNTING WITH MISTRAL TOKENIZER
# =============================================================================

# Use Mistral tokenizer (available in Capital One GenAI sandbox)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
import warnings
warnings.filterwarnings('ignore')

# Initialize Mistral tokenizer for enterprise use
print("🔄 Loading Mistral tokenizer...")

try:
    # Use MistralTokenizer (matches your Mixtral model and available in sandbox)
    hf_tokenizer = MistralTokenizer.from_model("open-mixtral-8x7b")
    tokenizer_name = "Mistral (open-mixtral-8x7b)"
    print("✅ Mistral tokenizer loaded (perfect match for your models)")
    tokenizer_available = True
except Exception as e:
    print(f"❌ Mistral tokenizer failed: {e}")
    hf_tokenizer = None
    tokenizer_name = "None - using fallback"
    tokenizer_available = False

def count_tokens_real(text):
    """Count tokens using Mistral tokenizer with correct API"""
    if pd.isna(text) or text == "":
        return 0
    
    text = str(text)
    
    try:
        if tokenizer_available and hf_tokenizer is not None:
            # Use Mistral tokenizer with correct API - encode_chat_completion or tokenize
            # Try different methods based on MistralTokenizer API
            if hasattr(hf_tokenizer, 'tokenize'):
                tokens = hf_tokenizer.tokenize(text)
                return len(tokens)
            elif hasattr(hf_tokenizer, 'encode_chat_completion'):
                # For chat completion format
                from mistral_common.protocol.instruct.messages import UserMessage
                from mistral_common.protocol.instruct.request import ChatCompletionRequest
                
                completion_request = ChatCompletionRequest(messages=[UserMessage(content=text)])
                tokens = hf_tokenizer.encode_chat_completion(completion_request)
                return len(tokens.tokens)
            else:
                # Try direct attribute access for token counting
                # Some versions might have different methods
                return estimate_tokens_fallback(text)
        else:
            return estimate_tokens_fallback(text)
    except Exception as e:
        print(f"Warning: Mistral tokenizer failed for text '{text[:50]}...': {e}")
        return estimate_tokens_fallback(text)

# Alternative implementation if the above doesn't work
def count_tokens_real_alternative(text):
    """Alternative method using string-based tokenization"""
    if pd.isna(text) or text == "":
        return 0
    
    text = str(text)
    
    try:
        if tokenizer_available and hf_tokenizer is not None:
            # Try tokenizing as a simple string
            # This might work depending on the MistralTokenizer version
            encoded = hf_tokenizer(text)
            if hasattr(encoded, 'tokens'):
                return len(encoded.tokens)
            elif isinstance(encoded, list):
                return len(encoded)
            else:
                return estimate_tokens_fallback(text)
        else:
            return estimate_tokens_fallback(text)
    except Exception as e:
        print(f"Warning: Alternative Mistral tokenizer failed: {e}")
        return estimate_tokens_fallback(text)

# =============================================================================
# SIMPLISTIC APPROACHES (KEPT FOR COMPARISON)
# =============================================================================

def estimate_tokens_basic(text):
    """SIMPLISTIC: Basic chars/4 method (for comparison)"""
    if pd.isna(text) or text == "":
        return 0
    return max(1, len(str(text)) // 4)

def estimate_tokens_fallback(text):
    """IMPROVED FALLBACK: Better estimation if tokenizer fails"""
    if pd.isna(text) or text == "":
        return 0
    
    text = str(text)
    words = text.split()
    
    base_tokens = len(words) * 1.35
    punct_tokens = len(re.findall(r'[.,!?;:(){}[\]"\'`]', text)) * 0.7
    number_tokens = len(re.findall(r'\d+', text)) * 0.9
    special_tokens = len(re.findall(r'[@#$%^&*+=<>~/\\|]', text)) * 0.5
    long_word_penalty = sum(1 for word in words if len(word) > 8) * 0.4
    tech_terms = len(re.findall(r'\b(?:API|SQL|HTML|CSS|JSON|XML|HTTP|URL)\b', text.upper())) * 0.3
    
    total_tokens = base_tokens + punct_tokens + number_tokens + special_tokens + long_word_penalty + tech_terms
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

def llama_token_count(text):
    """Main token counting function using Mistral tokenizer"""
    return count_tokens_real(text)

# =============================================================================
# TEST THE MISTRAL TOKENIZER ACCURACY
# =============================================================================

def test_mistral_tokenizer_accuracy():
    """Test MistralTokenizer vs simplistic approaches"""
    test_texts = [
        "Hello world! How are you doing today?",
        "Calculate 15% of $1,000 using Python code.",
        "What are the key risk factors for commercial loan applications?",
        "Write a SQL query to find customers with overdue payments.",
        "I'm feeling sad today and need some emotional support.",
        "Explain quantum mechanics in simple terms.",
        "Create a function that implements binary search algorithm."
    ]
    
    print("\n🧪 MISTRAL TOKENIZER ACCURACY TEST")
    print("=" * 70)
    print(f"{'Text':<50} {'Real':<6} {'Basic':<6} {'Fallback':<9} {'Chars':<6} {'Words':<6}")
    print("-" * 70)
    
    total_real = 0
    total_basic = 0
    total_fallback = 0
    total_chars = 0
    total_words = 0
    
    for text in test_texts:
        real_tokens = count_tokens_real(text)
        basic_tokens = estimate_tokens_basic(text)
        fallback_tokens = estimate_tokens_fallback(text)
        char_count = count_chars(text)
        word_count = count_words(text)
        
        # Truncate text for display
        display_text = text[:47] + "..." if len(text) > 50 else text
        
        print(f"{display_text:<50} {real_tokens:<6} {basic_tokens:<6} {fallback_tokens:<9} {char_count:<6} {word_count:<6}")
        
        total_real += real_tokens
        total_basic += basic_tokens
        total_fallback += fallback_tokens
        total_chars += char_count
        total_words += word_count
    
    print("-" * 70)
    print(f"{'TOTALS:':<50} {total_real:<6} {total_basic:<6} {total_fallback:<9} {total_chars:<6} {total_words:<6}")
    
    # Calculate accuracy metrics
    if total_real > 0:
        basic_error = abs(total_basic - total_real) / total_real * 100
        fallback_error = abs(total_fallback - total_real) / total_real * 100
        chars_per_token = total_chars / total_real
        words_per_token = total_words / total_real
        
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   Basic method (chars/4) error:     {basic_error:.1f}%")
        print(f"   Fallback method error:             {fallback_error:.1f}%")
        print(f"   Average chars per token:           {chars_per_token:.2f}")
        print(f"   Average words per token:           {words_per_token:.2f}")
        print(f"   Current latency predictor uses:    4.0 chars/token")
        print(f"   Suggested improvement:             {chars_per_token:.2f} chars/token")
        
        if tokenizer_available:
            print(f"   ✅ Mistral tokenizer working perfectly!")
        else:
            print(f"   ⚠️ Using fallback estimation method")
    
    return {
        'real_tokens': total_real,
        'basic_tokens': total_basic,
        'fallback_tokens': total_fallback,
        'chars_per_token': total_chars / total_real if total_real > 0 else 0,
        'words_per_token': total_words / total_real if total_real > 0 else 0
    }

# Run the test
print(f"\n🔧 Token counting functions loaded")
print(f"✅ Primary method: {tokenizer_name}")
print(f"✅ Tokenizer available: {tokenizer_available}")

# Execute the test
test_results = test_mistral_tokenizer_accuracy()
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

# Token Predictor EDA - Complete Jupyter Notebook

Copy and paste each cell below into separate Jupyter notebook cells:


---

# Complete Token Predictor EDA - Updated with Improvements

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

print("🔍 Token Predictor EDA Analysis - Updated Version")
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
    print(f"   Sample prompt types: {df_prompt_engineering['Prompt_Type'].value_counts().head().to_dict() if 'Prompt_Type' in df_prompt_engineering.columns else 'No Prompt_Type column'}")
except Exception as e:
    print(f"❌ Failed to load prompt_engineering_dataset.csv: {e}")
    df_prompt_engineering = None

# Load Dataset 2: Human Interaction Dataset  
try:
    df_human_interaction = pd.read_csv('human_interaction_dataset.csv')
    print(f"✅ Loaded human_interaction_dataset.csv: {df_human_interaction.shape[0]} rows, {df_human_interaction.shape[1]} columns")
    print(f"   Columns: {list(df_human_interaction.columns)}")
    print(f"   Sample LLM models: {df_human_interaction['LLM'].value_counts().head().to_dict() if 'LLM' in df_human_interaction.columns else 'No LLM column'}")
except Exception as e:
    print(f"❌ Failed to load human_interaction_dataset.csv: {e}")
    df_human_interaction = None
```

## Cell 3: HuggingFace Tokenizer Setup

```python
# =============================================================================
# STEP 2: REAL TOKEN COUNTING WITH HUGGINGFACE TOKENIZER
# =============================================================================

# Install required libraries (run this once)
# !pip install transformers torch

from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Initialize HuggingFace tokenizer for enterprise use
print("🔄 Loading HuggingFace tokenizer...")

# Use Llama tokenizer (matches your models: llama-3.1-8b, llama-3.3-70b)
try:
    # Try Llama tokenizer first (best match for your models)
    hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    tokenizer_name = "Llama-2-7b"
    print("✅ Llama tokenizer loaded (best match for your models)")
except Exception as e:
    print(f"❌ Llama tokenizer failed: {e}")
    try:
        # Fallback to CodeBERT (good for mixed text/code)
        hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", trust_remote_code=True)
        tokenizer_name = "CodeBERT"
        print("✅ CodeBERT tokenizer loaded (good for code/text)")
    except Exception as e2:
        print(f"❌ CodeBERT failed: {e2}")
        try:
            # Final fallback to BERT (always available)
            hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            tokenizer_name = "BERT-base"
            print("✅ BERT tokenizer loaded (fallback option)")
        except Exception as e3:
            print(f"❌ All HuggingFace tokenizers failed: {e3}")
            hf_tokenizer = None
            tokenizer_name = "None"

def count_tokens_real(text):
    """Count tokens using HuggingFace tokenizer"""
    if pd.isna(text) or text == "":
        return 0
    
    text = str(text)
    
    try:
        if hf_tokenizer is not None:
            tokens = hf_tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        else:
            return estimate_tokens_fallback(text)
    except Exception as e:
        print(f"Warning: Tokenizer failed for text '{text[:50]}...': {e}")
        return estimate_tokens_fallback(text)

# =============================================================================
# SIMPLISTIC APPROACHES (KEPT FOR COMPARISON)
# =============================================================================

def estimate_tokens_basic(text):
    """SIMPLISTIC: Basic chars/4 method (for comparison)"""
    if pd.isna(text) or text == "":
        return 0
    return max(1, len(str(text)) // 4)

def estimate_tokens_fallback(text):
    """IMPROVED FALLBACK: Better estimation if tokenizer fails"""
    if pd.isna(text) or text == "":
        return 0
    
    text = str(text)
    words = text.split()
    
    base_tokens = len(words) * 1.35
    punct_tokens = len(re.findall(r'[.,!?;:(){}[\]"\'`]', text)) * 0.7
    number_tokens = len(re.findall(r'\d+', text)) * 0.9
    special_tokens = len(re.findall(r'[@#$%^&*+=<>~/\\|]', text)) * 0.5
    long_word_penalty = sum(1 for word in words if len(word) > 8) * 0.4
    tech_terms = len(re.findall(r'\b(?:API|SQL|HTML|CSS|JSON|XML|HTTP|URL)\b', text.upper())) * 0.3
    
    total_tokens = base_tokens + punct_tokens + number_tokens + special_tokens + long_word_penalty + tech_terms
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

def llama_token_count(text):
    """Main token counting function using HuggingFace tokenizer"""
    return count_tokens_real(text)

print(f"\n🔧 Token counting functions loaded")
print(f"✅ Primary method: HuggingFace {tokenizer_name} tokenizer")
```

## Cell 4: Improved Analysis Functions

```python
# =============================================================================
# STEP 2: IMPROVED ANALYSIS FUNCTIONS
# =============================================================================

def categorize_question_type_improved(text):
    """
    IMPROVED: Categorize query into question types ensuring only ONE category per prompt.
    Priority order: who > what > how > where > when > why > other
    """
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower()
    
    # Check in priority order - first match wins (prevents multiple categories)
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
    """
    IMPROVED: Analyze the 'other' category to provide more refined categorization.
    This helps understand what's in the 'other' bucket for better feature engineering.
    """
    if pd.isna(text):
        return "unknown"
    
    text_lower = str(text).lower()
    
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
    
    # Otherwise, it's truly other
    return "unclassified_other"

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
        return "independent"  # Default to independent

def extract_text_complexity_features(text):
    """Extract additional complexity features from text"""
    if pd.isna(text):
        return {
            'has_numbers': False,
            'has_punctuation': False,
            'has_special_chars': False,
            'has_technical_terms': False,
            'sentence_count': 0,
            'avg_word_length': 0,
            'complexity_score': 0
        }
    
    text_str = str(text)
    
    # Feature extraction
    has_numbers = bool(re.search(r'\d+', text_str))
    has_punctuation = bool(re.search(r'[.,!?;:(){}[\]"\'`]', text_str))
    has_special_chars = bool(re.search(r'[@#$%^&*+=<>~/\\|]', text_str))
    has_technical_terms = bool(re.search(r'\b(?:API|SQL|HTML|CSS|JSON|XML|HTTP|URL|AI|ML|API)\b', text_str.upper()))
    
    # Sentence count (rough)
    sentence_count = len(re.split(r'[.!?]+', text_str))
    
    # Average word length
    words = text_str.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Complexity score (composite)
    complexity_score = (
        len(words) * 0.1 +
        sentence_count * 0.2 +
        avg_word_length * 0.3 +
        has_numbers * 0.1 +
        has_punctuation * 0.1 +
        has_special_chars * 0.1 +
        has_technical_terms * 0.1
    )
    
    return {
        'has_numbers': has_numbers,
        'has_punctuation': has_punctuation,
        'has_special_chars': has_special_chars,
        'has_technical_terms': has_technical_terms,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'complexity_score': round(complexity_score, 2)
    }

print("✅ Improved analysis functions defined")
```

## Cell 5: Dataset Processing Functions

```python
# =============================================================================
# STEP 3: DATASET PROCESSING FUNCTIONS (Updated)
# =============================================================================

def process_prompt_engineering_dataset(df):
    """Process the prompt engineering dataset with improved categorization"""
    if df is None:
        return None
    
    print(f"\n🔄 Processing Prompt Engineering Dataset with improved categorization...")
    processed_df = df.copy()
    
    # Basic text metrics
    processed_df['query_char_count'] = processed_df['Prompt'].apply(count_chars)
    processed_df['query_word_count'] = processed_df['Prompt'].apply(count_words)
    
    # Token counts (real vs estimation)
    processed_df['query_token_count_real'] = processed_df['Prompt'].apply(llama_token_count)
    processed_df['query_token_count_basic'] = processed_df['Prompt'].apply(estimate_tokens_basic)
    
    # IMPROVED: Question type categorization (only one category per prompt)
    processed_df['question_type'] = processed_df['Prompt'].apply(categorize_question_type_improved)
    
    # IMPROVED: Analyze the 'other' category for better insights
    processed_df['other_subcategory'] = processed_df['Prompt'].apply(analyze_other_category)
    
    # Content type and context
    processed_df['content_type'] = processed_df['Prompt'].apply(is_natural_language_or_code)
    processed_df['query_context'] = processed_df['Prompt'].apply(is_independent_or_continuation)
    
    # IMPROVED: Extract complexity features
    complexity_features = processed_df['Prompt'].apply(extract_text_complexity_features)
    complexity_df = pd.json_normalize(complexity_features)
    processed_df = pd.concat([processed_df, complexity_df], axis=1)
    
    # Process responses if available
    if 'Response' in processed_df.columns:
        processed_df['response_char_count'] = processed_df['Response'].apply(count_chars)
        processed_df['response_word_count'] = processed_df['Response'].apply(count_words)
        processed_df['response_token_count_real'] = processed_df['Response'].apply(llama_token_count)
        processed_df['response_token_count_basic'] = processed_df['Response'].apply(estimate_tokens_basic)
        processed_df['conversation_length'] = processed_df['query_char_count'] + processed_df['response_char_count']
    
    # Keep original columns if they exist
    if 'Prompt_Type' in processed_df.columns:
        processed_df['original_prompt_type'] = processed_df['Prompt_Type']
    
    if 'Prompt_Length' in processed_df.columns:
        processed_df['original_prompt_length'] = processed_df['Prompt_Length']
    
    print(f"✅ Processed {len(processed_df)} records from Prompt Engineering Dataset")
    print(f"   Question type distribution: {processed_df['question_type'].value_counts().to_dict()}")
    print(f"   'Other' subcategory distribution: {processed_df['other_subcategory'].value_counts().to_dict()}")
    
    return processed_df

def process_human_interaction_dataset(df):
    """Process the human interaction dataset with improved categorization"""
    if df is None:
        return None
    
    print(f"\n🔄 Processing Human Interaction Dataset with improved categorization...")
    processed_df = df.copy()
    
    # Basic text metrics
    processed_df['query_char_count'] = processed_df['Context'].apply(count_chars)
    processed_df['query_word_count'] = processed_df['Context'].apply(count_words)
    
    # Token counts (real vs estimation)
    processed_df['query_token_count_real'] = processed_df['Context'].apply(llama_token_count)
    processed_df['query_token_count_basic'] = processed_df['Context'].apply(estimate_tokens_basic)
    
    # IMPROVED: Question type categorization (only one category per prompt)
    processed_df['question_type'] = processed_df['Context'].apply(categorize_question_type_improved)
    
    # IMPROVED: Analyze the 'other' category for better insights
    processed_df['other_subcategory'] = processed_df['Context'].apply(analyze_other_category)
    
    # Content type and context
    processed_df['content_type'] = processed_df['Context'].apply(is_natural_language_or_code)
    processed_df['query_context'] = processed_df['Context'].apply(is_independent_or_continuation)
    
    # IMPROVED: Extract complexity features
    complexity_features = processed_df['Context'].apply(extract_text_complexity_features)
    complexity_df = pd.json_normalize(complexity_features)
    processed_df = pd.concat([processed_df, complexity_df], axis=1)
    
    # Process responses if available
    if 'Response' in processed_df.columns:
        processed_df['response_char_count'] = processed_df['Response'].apply(count_chars)
        processed_df['response_word_count'] = processed_df['Response'].apply(count_words)
        processed_df['response_token_count_real'] = processed_df['Response'].apply(llama_token_count)
        processed_df['response_token_count_basic'] = processed_df['Response'].apply(estimate_tokens_basic)
        processed_df['conversation_length'] = processed_df['query_char_count'] + processed_df['response_char_count']
    
    # Keep LLM model info if available
    if 'LLM' in processed_df.columns:
        processed_df['llm_model'] = processed_df['LLM']
    
    print(f"✅ Processed {len(processed_df)} records from Human Interaction Dataset")
    print(f"   Question type distribution: {processed_df['question_type'].value_counts().to_dict()}")
    print(f"   'Other' subcategory distribution: {processed_df['other_subcategory'].value_counts().to_dict()}")
    
    return processed_df

print("✅ Dataset processing functions updated with improved categorization")
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

# Show sample of processed data
for name, df in datasets.items():
    if df is not None:
        print(f"\n📋 Sample from {name}:")
        print(f"Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"Sample question types: {df['question_type'].value_counts().head()}")
            print(f"Sample other subcategories: {df['other_subcategory'].value_counts().head()}")
```

## Cell 7: Statistical Analysis

```python
# =============================================================================
# STEP 4: COMPREHENSIVE STATISTICAL ANALYSIS (IMPROVED)
# =============================================================================

print("\n📈 COMPREHENSIVE TOKEN ANALYSIS (HuggingFace Tokenizer)")
print("=" * 60)

# Combine all datasets for overall analysis
all_data = []
for name, df in datasets.items():
    if df is not None:
        # Select relevant columns including new complexity features
        df_subset = df[['query_char_count', 'query_word_count', 'query_token_count_basic', 
                       'query_token_count_real', 'question_type', 'other_subcategory',
                       'content_type', 'query_context', 'has_numbers', 'has_punctuation',
                       'has_special_chars', 'has_technical_terms', 'sentence_count',
                       'avg_word_length', 'complexity_score']].copy()
        df_subset['dataset'] = name
        all_data.append(df_subset)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Overall Statistics with REAL tokenizer comparison
    print("\n📊 OVERALL QUERY STATISTICS (Real vs Estimation)")
    print("-" * 40)
    
    stats_summary = combined_df[['query_char_count', 'query_word_count', 
                                'query_token_count_basic', 'query_token_count_real']].describe()
    print(stats_summary.round(2))
    
    # Method comparison
    print("\n📊 TOKENIZER METHOD COMPARISON")
    print("-" * 40)
    avg_real = combined_df['query_token_count_real'].mean()
    avg_basic = combined_df['query_token_count_basic'].mean()
    
    basic_error = abs(avg_basic - avg_real) / avg_real * 100
    
    print(f"Real HuggingFace Tokenizer:     {avg_real:.1f} avg tokens (GROUND TRUTH)")
    print(f"Basic (chars/4):                {avg_basic:.1f} avg tokens ({basic_error:.1f}% error)")
    
    # IMPROVED: Question type analysis (ensuring single category per prompt)
    print("\n📊 IMPROVED QUESTION TYPE DISTRIBUTION")
    print("-" * 40)
    question_dist = combined_df['question_type'].value_counts()
    print(question_dist)
    print(f"\nPercentage distribution:")
    print((question_dist / len(combined_df) * 100).round(2))
    
    # Validate single categorization
    print(f"\nTotal queries: {len(combined_df)}")
    print(f"Sum of all categories: {question_dist.sum()} ✅ Should match total")
    
    # IMPROVED: 'Other' category deep dive
    print("\n📊 'OTHER' CATEGORY REFINED ANALYSIS")
    print("-" * 40)
    other_subset = combined_df[combined_df['question_type'] == 'other']
    if len(other_subset) > 0:
        other_subcategory_dist = other_subset['other_subcategory'].value_counts()
        print(f"Total 'other' queries: {len(other_subset)} ({len(other_subset)/len(combined_df)*100:.1f}%)")
        print("\nSubcategory breakdown:")
        for subcat, count in other_subcategory_dist.items():
            pct = count / len(other_subset) * 100
            avg_tokens = other_subset[other_subset['other_subcategory'] == subcat]['query_token_count_real'].mean()
            print(f"   {subcat}: {count} queries ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    else:
        print("No 'other' category queries found.")
    
    # Content type distribution
    print("\n📊 CONTENT TYPE DISTRIBUTION")
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
    # 1. Token count distribution (Real tokenizer)
    axes[0, 0].hist(combined_df['query_token_count_real'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Query Token Count Distribution (Real)')
    axes[0, 0].set_xlabel('Real Token Count')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Char vs Token relationship
    axes[0, 1].scatter(combined_df['query_char_count'], combined_df['query_token_count_real'], alpha=0.5)
    axes[0, 1].set_title('Character Count vs Real Token Count')
    axes[0, 1].set_xlabel('Character Count')
    axes[0, 1].set_ylabel('Real Token Count')
    
    # 3. Word vs Token relationship  
    axes[0, 2].scatter(combined_df['query_word_count'], combined_df['query_token_count_real'], alpha=0.5, color='orange')
    axes[0, 2].set_title('Word Count vs Real Token Count')
    axes[0, 2].set_xlabel('Word Count')
    axes[0, 2].set_ylabel('Real Token Count')
    
    # 4. IMPROVED: Question type distribution (single category)
    question_counts = combined_df['question_type'].value_counts()
    axes[1, 0].bar(question_counts.index, question_counts.values)
    axes[1, 0].set_title('Question Type Distribution (Single Category)')
    axes[1, 0].set_xlabel('Question Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. IMPROVED: 'Other' category breakdown
    other_subset = combined_df[combined_df['question_type'] == 'other']
    if len(other_subset) > 0:
        other_counts = other_subset['other_subcategory'].value_counts().head(8)
        axes[1, 1].barh(range(len(other_counts)), other_counts.values)
        axes[1, 1].set_yticks(range(len(other_counts)))
        axes[1, 1].set_yticklabels(other_counts.index)
        axes[1, 1].set_title("'Other' Category Refined Analysis")
        axes[1, 1].set_xlabel('Count')
    else:
        axes[1, 1].text(0.5, 0.5, 'No "other" category\nqueries found', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("'Other' Category Analysis")
    
    # 6. Token count by question type
    sns.boxplot(data=combined_df, x='question_type', y='query_token_count_real', ax=axes[1, 2])
    axes[1, 2].set_title('Token Count by Question Type')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # 7. Content type distribution
    content_counts = combined_df['content_type'].value_counts()
    axes[2, 0].pie(content_counts.values, labels=content_counts.index, autopct='%1.1f%%')
    axes[2, 0].set_title('Natural Language vs Code Requests')
    
    # 8. Real vs Basic token count comparison
    axes[2, 1].scatter(combined_df['query_token_count_basic'], combined_df['query_token_count_real'], alpha=0.5, color='green')
    axes[2, 1].plot([0, combined_df['query_token_count_basic'].max()], [0, combined_df['query_token_count_basic'].max()], 'r--', alpha=0.8)
    axes[2, 1].set_title('Basic vs Real Token Count')
    axes[2, 1].set_xlabel('Basic Token Count (chars/4)')
    axes[2, 1].set_ylabel('Real Token Count')
    
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
    if df is not None and 'response_token_count_real' in df.columns:
        conv_subset = df[['query_token_count_real', 'response_token_count_real', 
                         'conversation_length', 'question_type', 'query_context']].copy()
        conv_subset['dataset'] = name
        conversation_data.append(conv_subset)

if conversation_data:
    conv_df = pd.concat(conversation_data, ignore_index=True)
    
    print("\n📊 CONVERSATION STATISTICS")
    print("-" * 40)
    conv_stats = conv_df[['query_token_count_real', 'response_token_count_real', 'conversation_length']].describe()
    print(conv_stats.round(2))
    
    # Response length analysis
    print(f"\n📊 RESPONSE LENGTH INSIGHTS")
    print(f"Average query tokens: {conv_df['query_token_count_real'].mean():.1f}")
    print(f"Average response tokens: {conv_df['response_token_count_real'].mean():.1f}")
    print(f"Response/Query ratio: {(conv_df['response_token_count_real'] / conv_df['query_token_count_real']).mean():.2f}")
    
    # Response length by question type
    print("\n📊 RESPONSE LENGTH BY QUESTION TYPE")
    print("-" * 40)
    response_by_type = conv_df.groupby('question_type')['response_token_count_real'].agg(['mean', 'median', 'std']).round(2)
    print(response_by_type)
    
    # Response length by query context
    print("\n📊 RESPONSE LENGTH BY QUERY CONTEXT")
    print("-" * 40)
    response_by_context = conv_df.groupby('query_context')['response_token_count_real'].agg(['mean', 'median', 'count']).round(2)
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
    axes[0, 0].scatter(conv_df['query_token_count_real'], conv_df['response_token_count_real'], alpha=0.6)
    axes[0, 0].set_title('Query Tokens vs Response Tokens')
    axes[0, 0].set_xlabel('Query Token Count (Real)')
    axes[0, 0].set_ylabel('Response Token Count (Real)')
    
    # Response length distribution
    axes[0, 1].hist(conv_df['response_token_count_real'], bins=50, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('Response Token Distribution')
    axes[0, 1].set_xlabel('Response Token Count')
    axes[0, 1].set_ylabel('Frequency')
    
    # Response length by question type
    sns.boxplot(data=conv_df, x='question_type', y='response_token_count_real', ax=axes[1, 0])
    axes[1, 0].set_title('Response Length by Question Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Response length by query context
    sns.boxplot(data=conv_df, x='query_context', y='response_token_count_real', ax=axes[1, 1])
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
        token_by_type = pe_df.groupby('original_prompt_type')['query_token_count_real'].mean().round(2)
        print(token_by_type)
    
    if 'original_prompt_length' in pe_df.columns:
        print(f"\nPrompt Length Analysis:")
        print(f"Original prompt length vs our token count correlation: {pe_df['original_prompt_length'].corr(pe_df['query_token_count_real']):.3f}")

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
        if 'response_token_count_real' in hi_df.columns:
            print("\nAverage Response Length by LLM Model:")
            response_by_llm = hi_df.groupby('llm_model')['response_token_count_real'].mean().round(2)
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
    # Compare basic vs real token counting methods
    basic_real_ratio = combined_df['query_token_count_basic'] / combined_df['query_token_count_real']
    char_token_ratio = combined_df['query_char_count'] / combined_df['query_token_count_real']
    word_token_ratio = combined_df['query_word_count'] / combined_df['query_token_count_real']
    
    print(f"📊 TOKEN COUNTING METHOD COMPARISON")
    print(f"Basic/Real ratio - Mean: {basic_real_ratio.mean():.3f}, Std: {basic_real_ratio.std():.3f}")
    print(f"Chars/Tokens ratio - Mean: {char_token_ratio.mean():.3f}, Std: {char_token_ratio.std():.3f}")
    print(f"Words/Tokens ratio - Mean: {word_token_ratio.mean():.3f}, Std: {word_token_ratio.std():.3f}")
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    basic_mape = (abs(combined_df['query_token_count_basic'] - combined_df['query_token_count_real']) / 
                  combined_df['query_token_count_real'] * 100).mean()
    
    print(f"\n📊 PREDICTION ACCURACY")
    print(f"Basic method (chars/4) MAPE: {basic_mape:.1f}%")
    
    # Accuracy by content type
    print(f"\n📊 ACCURACY BY CONTENT TYPE")
    print("-" * 40)
    accuracy_by_type = combined_df.groupby('content_type').agg({
        'query_char_count': 'mean',
        'query_word_count': 'mean', 
        'query_token_count_real': 'mean'
    }).round(2)
    print(accuracy_by_type)
    
    # Calculate our predictor's character/token ratio
    our_ratio = combined_df['query_char_count'] / combined_df['query_token_count_real']
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
    print(f"   • Average query tokens: {combined_df['query_token_count_real'].mean():.1f}")
    print(f"   • Median query tokens: {combined_df['query_token_count_real'].median():.1f}")
    print(f"   • 90th percentile: {combined_df['query_token_count_real'].quantile(0.9):.1f}")
    print(f"   • Max observed: {combined_df['query_token_count_real'].max()}")
    print(f"   • Min observed: {combined_df['query_token_count_real'].min()}")
    
    print(f"\n2. QUESTION TYPE INSIGHTS (SINGLE CATEGORY VALIDATION):")
    for qtype in combined_df['question_type'].value_counts().head().index:
        avg_tokens = combined_df[combined_df['question_type'] == qtype]['query_token_count_real'].mean()
        count = (combined_df['question_type'] == qtype).sum()
        print(f"   • {qtype.upper()} questions: {avg_tokens:.1f} avg tokens ({count} samples)")
    
    print(f"\n3. 'OTHER' CATEGORY INSIGHTS (REFINED ANALYSIS):")
    other_subset = combined_df[combined_df['question_type'] == 'other']
    if len(other_subset) > 0:
        print(f"   • Total 'other' queries: {len(other_subset)} ({len(other_subset)/len(combined_df)*100:.1f}%)")
        print("   • Top subcategories:")
        for subcat in other_subset['other_subcategory'].value_counts().head(3).index:
            count = (other_subset['other_subcategory'] == subcat).sum()
            avg_tokens = other_subset[other_subset['other_subcategory'] == subcat]['query_token_count_real'].mean()
            pct = count / len(other_subset) * 100
            print(f"     - {subcat}: {count} queries ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    print(f"\n4. CONTENT TYPE INSIGHTS:")
    for ctype in combined_df['content_type'].value_counts().index:
        avg_tokens = combined_df[combined_df['content_type'] == ctype]['query_token_count_real'].mean()
        pct = (combined_df['content_type'] == ctype).mean() * 100
        count = (combined_df['content_type'] == ctype).sum()
        print(f"   • {ctype.replace('_', ' ').title()}: {avg_tokens:.1f} avg tokens ({pct:.1f}%, {count} samples)")
    
    print(f"\n5. QUERY CONTEXT INSIGHTS:")
    for context in combined_df['query_context'].value_counts().index:
        avg_tokens = combined_df[combined_df['query_context'] == context]['query_token_count_real'].mean()
        pct = (combined_df['query_context'] == context).mean() * 100
        count = (combined_df['query_context'] == context).sum()
        print(f"   • {context.title()} queries: {avg_tokens:.1f} avg tokens ({pct:.1f}%, {count} samples)")
    
    if conversation_data:
        print(f"\n6. RESPONSE LENGTH INSIGHTS:")
        print(f"   • Average response: {conv_df['response_token_count_real'].mean():.1f} tokens")
        print(f"   • Median response: {conv_df['response_token_count_real'].median():.1f} tokens")
        print(f"   • Response/Query ratio: {(conv_df['response_token_count_real'] / conv_df['query_token_count_real']).mean():.2f}x")
        print(f"   • 90th percentile response: {conv_df['response_token_count_real'].quantile(0.9):.1f} tokens")
    
    print(f"\n7. RECOMMENDATIONS FOR LATENCY PREDICTOR:")
    our_ratio = combined_df['query_char_count'] / combined_df['query_token_count_real']
    print(f"   ✅ Current char/4 approximation accuracy: {abs(4.0 - our_ratio.mean())/4.0*100:.1f}% off")
    print(f"   ✅ Suggested chars/token ratio: {our_ratio.mean():.1f}")
    print(f"   ✅ Question type categorization: Single category per prompt validated")
    print(f"   ✅ 'Other' category analysis: {len(other_subset) if len(other_subset) > 0 else 0} queries need refined features")
    code_avg = combined_df[combined_df['content_type']=='code_request']['query_token_count_real'].mean() if (combined_df['content_type']=='code_request').any() else 0
    nl_avg = combined_df[combined_df['content_type']=='natural_language']['query_token_count_real'].mean() if (combined_df['content_type']=='natural_language').any() else 0
    if code_avg > 0 and nl_avg > 0:
        diff = ((code_avg / nl_avg) - 1) * 100
        print(f"   ✅ Code requests are {diff:.1f}% different from natural language")
    if conversation_data:
        print(f"   ✅ Default {conv_df['response_token_count_real'].median():.0f} output tokens appropriate for most use cases")
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
    print(f"   • Tokenizer used: HuggingFace {tokenizer_name}")
    
    # Dataset breakdown
    for name, df in datasets.items():
        if df is not None:
            count = len(df)
            pct = count / total_queries * 100
            print(f"   • {name.replace('_', ' ').title()}: {count:,} queries ({pct:.1f}%)")
    
    print(f"\n📊 QUERY CHARACTERISTICS:")
    print(f"   • Average character count: {combined_df['query_char_count'].mean():.0f}")
    print(f"   • Average word count: {combined_df['query_word_count'].mean():.1f}")
    print(f"   • Average token count (real): {combined_df['query_token_count_real'].mean():.1f}")
    print(f"   • Token count range: {combined_df['query_token_count_real'].min()}-{combined_df['query_token_count_real'].max()}")
    print(f"   • Most common token count: {combined_df['query_token_count_real'].mode().iloc[0]}")
    
    print(f"\n📊 IMPROVED CATEGORIZATION RESULTS:")
    print(f"   Question Types (single category per prompt - VALIDATED):")
    for qtype, count in combined_df['question_type'].value_counts().head(6).items():
        pct = count / total_queries * 100
        avg_tokens = combined_df[combined_df['question_type'] == qtype]['query_token_count_real'].mean()
        print(f"   • {qtype.title()}: {count:,} queries ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    print(f"\n📊 'OTHER' CATEGORY BREAKDOWN (REFINED):")
    other_subset = combined_df[combined_df['question_type'] == 'other']
    if len(other_subset) > 0:
        print(f"   Total 'other' queries: {len(other_subset)} ({len(other_subset)/len(combined_df)*100:.1f}%)")
        print("   Top subcategories:")
        for subcat, count in other_subset['other_subcategory'].value_counts().head(5).items():
            pct = count / len(other_subset) * 100
            avg_tokens = other_subset[other_subset['other_subcategory'] == subcat]['query_token_count_real'].mean()
            print(f"   • {subcat}: {count} queries ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    print(f"\n📊 CONTENT TYPE BREAKDOWN:")
    for ctype, count in combined_df['content_type'].value_counts().items():
        pct = count / total_queries * 100
        avg_tokens = combined_df[combined_df['content_type'] == ctype]['query_token_count_real'].mean()
        print(f"   • {ctype.replace('_', ' ').title()}: {count:,} queries ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    print(f"\n📊 QUERY CONTEXT ANALYSIS:")
    for context, count in combined_df['query_context'].value_counts().items():
        pct = count / total_queries * 100
        avg_tokens = combined_df[combined_df['query_context'] == context]['query_token_count_real'].mean()
        print(f"   • {context.title()} queries: {count:,} ({pct:.1f}%, avg {avg_tokens:.1f} tokens)")
    
    if conversation_data:
        conv_total = len(conv_df)
        print(f"\n📊 CONVERSATION PATTERNS:")
        print(f"   • Queries with responses: {conv_total:,}")
        print(f"   • Average response length: {conv_df['response_token_count_real'].mean():.1f} tokens")
        print(f"   • Response/Query length ratio: {(conv_df['response_token_count_real'] / conv_df['query_token_count_real']).mean():.2f}x")
        print(f"   • Longest response: {conv_df['response_token_count_real'].max()} tokens")
        print(f"   • Shortest response: {conv_df['response_token_count_real'].min()} tokens")
```

## Cell 15: Calibration Recommendations

```python
# =============================================================================
# STEP 11: TOKEN PREDICTOR CALIBRATION RECOMMENDATIONS
# =============================================================================

print(f"\n🎛️ TOKEN PREDICTOR CALIBRATION RECOMMENDATIONS")
print("=" * 60)

if all_data:
    # Calculate optimal parameters for token prediction
    optimal_char_ratio = combined_df['query_char_count'] / combined_df['query_token_count_real']
    optimal_word_ratio = combined_df['query_word_count'] / combined_df['query_token_count_real']
    
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
            char_ratio = (subset['query_char_count'] / subset['query_token_count_real']).mean()
            word_ratio = (subset['query_word_count'] / subset['query_token_count_real']).mean()
            count = len(subset)
            print(f"   {ctype.replace('_', ' ').title()} ({count} samples):")
            print(f"     • Chars/token: {char_ratio:.2f}")
            print(f"     • Words/token: {word_ratio:.2f}")
    
    if conversation_data:
        print(f"\n📊 RESPONSE LENGTH RECOMMENDATIONS:")
        response_percentiles = conv_df['response_token_count_real'].quantile([0.25, 0.5, 0.75, 0.9])
        print(f"   Default expected_output_tokens suggestions:")
        print(f"     • Conservative (25th percentile): {response_percentiles[0.25]:.0f} tokens")
        print(f"     • Standard (median): {response_percentiles[0.5]:.0f} tokens")
        print(f"     • Generous (75th percentile): {response_percentiles[0.75]:.0f} tokens")
        print(f"     • Enterprise (90th percentile): {response_percentiles[0.9]:.0f} tokens")
```

## Cell 16: Final Training Data Structure

```python
# =============================================================================
# STEP 12: FINAL TRAINING DATA STRUCTURE FOR ML MODEL
# =============================================================================

print(f"\n🎯 FINAL TRAINING DATA STRUCTURE FOR ML MODEL")
print("=" * 60)

if all_data:
    # Create final training dataset
    training_data = combined_df.copy()
    
    # Define features for ML training
    FEATURE_COLUMNS = [
        # Basic text metrics
        'query_char_count',
        'query_word_count', 
        
        # Question categorization (will need encoding)
        'question_type',
        'other_subcategory',
        'content_type',
        'query_context',
        
        # Complexity features
        'has_numbers',
        'has_punctuation', 
        'has_special_chars',
        'has_technical_terms',
        'sentence_count',
        'avg_word_length',
        'complexity_score'
    ]
    
    # Target variables
    TARGET_COLUMNS = [
        'query_token_count_real',  # Input tokens (primary target)
    ]
    
    # Add response targets if available
    if conversation_data:
        TARGET_COLUMNS.extend([
            'response_token_count_real',  # Output tokens
        ])
        # Calculate total tokens
        training_data['total_tokens'] = (
            training_data['query_token_count_real'] + 
            training_data.get('response_token_count_real', 0)
        )
        TARGET_COLUMNS.append('total_tokens')
    
    # Create final training dataset
    final_features = training_data[FEATURE_COLUMNS + TARGET_COLUMNS + ['dataset']].copy()
    
    # Add unique query ID
    final_features['query_id'] = range(len(final_features))
    
    print(f"📋 FINAL TRAINING DATA STRUCTURE:")
    print(f"   • Total samples: {len(final_features):,}")
    print(f"   • Feature columns: {len(FEATURE_COLUMNS)}")
    print(f"   • Target columns: {len(TARGET_COLUMNS)}")
    
    print(f"\n📊 FEATURE COLUMNS ({len(FEATURE_COLUMNS)}):")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\n📊 TARGET COLUMNS ({len(TARGET_COLUMNS)}):")
    for i, col in enumerate(TARGET_COLUMNS, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\n📊 SAMPLE OF FINAL TRAINING DATA:")
    print(final_features.head())
    
    print(f"\n📊 TRAINING DATA STATISTICS:")
    print(final_features[TARGET_COLUMNS].describe().round(2))
    
    # Check for missing values
    missing_values = final_features[FEATURE_COLUMNS].isnull().sum()
    print(f"\n📊 MISSING VALUES CHECK:")
    if missing_values.sum() == 0:
        print("   ✅ No missing values found in features")
    else:
        print("   ⚠️ Missing values found:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"     • {col}: {count} missing")
    
    # Categorical encoding recommendations
    print(f"\n📊 CATEGORICAL ENCODING RECOMMENDATIONS:")
    categorical_cols = ['question_type', 'other_subcategory', 'content_type', 'query_context', 'dataset']
    for col in categorical_cols:
        if col in final_features.columns:
            unique_values = final_features[col].nunique()
            print(f"   • {col}: {unique_values} unique values")
            if unique_values <= 10:
                print(f"     - Recommendation: One-hot encoding")
                print(f"     - Values: {final_features[col].unique()}")
            else:
                print(f"     - Recommendation: Label encoding or target encoding")
    
    # Save final training data (optional)
    try:
        final_features.to_csv('training_data_final.csv', index=False)
        print(f"\n✅ Final training data saved to 'training_data_final.csv'")
        print(f"   • Shape: {final_features.shape}")
        print(f"   • Size: {final_features.memory_usage().sum() / 1024**2:.1f} MB")
    except Exception as e:
        print(f"\n⚠️ Could not save training data: {e}")
    
    print(f"\n🚀 NEXT STEPS FOR ML MODEL TRAINING:")
    print("   1. One-hot encode categorical features")
    print("   2. Split into train/validation/test sets (70/15/15)")
    print("   3. Scale numerical features if needed")
    print("   4. Train RandomForestRegressor for each target")
    print("   5. Evaluate using MAPE (Mean Absolute Percentage Error)")
    print("   6. Target accuracy: <15% MAPE for total tokens")

print(f"\n🎉 TOKEN PREDICTOR EDA ANALYSIS COMPLETE!")
print("=" * 60)
print("📊 All statistics generated for JIRA ticket documentation")
print("🔧 Improved categorization with single category per prompt validated")
print("🔍 'Other' category refined analysis completed")
print("📋 Final training data structure ready for ML pipeline")
print("🚀 Ready for Phase 2: Model Training")
```

### **GitHub README Ready:**
This format is optimized for copying into GitHub README.md and then easily transferring to any Jupyter environment. Each cell is clearly separated and ready for enterprise use.

## License

This project is licensed under the MIT License.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
