```

# utils.py - Optimized Feature Extraction for Token Predictor
# Only includes features that performed above random baseline

import re
import string
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Initialize tokenizer
tokenizer = MistralTokenizer.v3()

def get_token_length(text):
    """Get accurate token count using MistralTokenizer"""
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback to character-based estimation
        return max(1, len(text) // 4)

def is_natural_language_or_code(text):
    """Determine if text is natural language (1) or code (0)"""
    code_indicators = [
        'def ', 'function', 'import ', 'from ', '#!/', 
        'class ', 'SELECT', 'INSERT', 'UPDATE', 'DELETE',
        '<?php', '<html>', '<script>', '$(', 'console.log',
        'print(', 'println(', 'System.out', 'cout <<'
    ]
    
    text_lower = text.lower()
    code_score = sum(1 for indicator in code_indicators if indicator.lower() in text_lower)
    
    # If multiple code indicators or text is very short, classify as code
    if code_score >= 2 or (len(text.split()) < 5 and code_score > 0):
        return 0
    return 1

def categorize_query(text):
    """Categorize query into one of 7 categories with priority logic"""
    text_lower = text.lower()
    
    # Category 3: Summary/Analysis requests (HIGHEST PRIORITY)
    summary_keywords = ['summary', 'summarize', 'tldr', 'key points', 'main points', 
                       'overview', 'brief', 'analyze', 'analysis', 'review']
    if any(keyword in text_lower for keyword in summary_keywords):
        return 3
    
    # Category 2: Code requests (HIGH PRIORITY) 
    code_keywords = ['code', 'function', 'script', 'programming', 'algorithm',
                    'implement', 'write a', 'create a function', 'debug', 'fix']
    if any(keyword in text_lower for keyword in code_keywords):
        return 2
    
    # Category 1: Questions (MEDIUM PRIORITY)
    if text.strip().endswith('?') or text_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
        return 1
    
    # Category 0: Explanations/Instructions (DEFAULT)
    explain_keywords = ['explain', 'tell me', 'describe', 'what is', 'how to']
    if any(keyword in text_lower for keyword in explain_keywords):
        return 0
    
    # Category 4: Creative requests
    creative_keywords = ['write a story', 'poem', 'creative', 'imagine', 'fiction']
    if any(keyword in text_lower for keyword in creative_keywords):
        return 4
    
    # Category 5: Data/Research requests  
    data_keywords = ['data', 'research', 'statistics', 'numbers', 'calculate']
    if any(keyword in text_lower for keyword in data_keywords):
        return 5
    
    # Category 6: Other/General
    return 6

def extract_text_complexity_features(text):
    """Extract text complexity and linguistic features"""
    # Basic text metrics
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    unique_words = set(word.lower().strip(string.punctuation) for word in words)
    unique_word_count = len(unique_words)
    
    # Sentence analysis
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
    # Punctuation density
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    punctuation_density = punctuation_count / max(1, char_count)
    
    # Capitalization ratio
    caps_count = sum(1 for char in text if char.isupper())
    caps_ratio = caps_count / max(1, char_count)
    
    # Question detection
    has_questions = 1 if '?' in text else 0
    
    # Complexity score (weighted combination)
    complexity_score = (
        (unique_word_count / max(1, word_count)) * 0.3 +  # Vocabulary diversity
        (avg_word_length / 10) * 0.2 +                    # Word complexity
        (sentence_count / max(1, word_count / 15)) * 0.2 + # Sentence structure
        punctuation_density * 0.3                          # Punctuation complexity
    )
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'unique_word_count': unique_word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'punctuation_density': punctuation_density,
        'caps_ratio': caps_ratio,
        'has_questions': has_questions,
        'complexity_score': complexity_score
    }

def categorize_question_type(text):
    """Enhanced question type categorization"""
    text_lower = text.lower().strip()
    
    if text_lower.startswith('how'):
        return 'how'
    elif text_lower.startswith('what'):
        return 'what'
    elif text_lower.startswith('why'):
        return 'why'
    elif text_lower.startswith('when'):
        return 'when'
    elif text_lower.startswith('where'):
        return 'where'
    elif text_lower.startswith('who'):
        return 'who'
    elif text_lower.startswith(('can', 'could', 'would', 'should')):
        return 'modal'
    elif text_lower.startswith(('is', 'are', 'was', 'were', 'do', 'does', 'did')):
        return 'yes_no'
    else:
        return 'other'

def analyze_other_category(text):
    """Analyze subcategories within 'other' category"""
    text_lower = text.lower()
    
    # Creative writing requests
    if any(word in text_lower for word in ['story', 'poem', 'creative', 'write a', 'fiction']):
        return 'creative'
    
    # Technical/coding requests
    if any(word in text_lower for word in ['code', 'function', 'debug', 'programming', 'script']):
        return 'technical'
    
    # Analysis/summary requests
    if any(word in text_lower for word in ['analyze', 'summary', 'review', 'evaluate']):
        return 'analysis'
    
    # Math/calculation requests
    if any(word in text_lower for word in ['calculate', 'math', 'solve', 'equation']):
        return 'math'
    
    # Research/data requests
    if any(word in text_lower for word in ['research', 'data', 'statistics', 'information']):
        return 'research'
    
    # Translation requests
    if any(word in text_lower for word in ['translate', 'translation', 'language']):
        return 'translation'
    
    # Explanation requests
    if any(word in text_lower for word in ['explain', 'describe', 'tell me about']):
        return 'explanation'
    
    # Instructions/how-to
    if any(word in text_lower for word in ['how to', 'steps', 'guide', 'tutorial']):
        return 'instructions'
    
    return 'general'

def get_query_context(text):
    """Determine query context/domain"""
    text_lower = text.lower()
    
    # Technical domains
    if any(word in text_lower for word in ['programming', 'code', 'software', 'algorithm']):
        return 'technical'
    
    # Academic domains
    if any(word in text_lower for word in ['research', 'study', 'academic', 'paper']):
        return 'academic'
    
    # Business domains
    if any(word in text_lower for word in ['business', 'marketing', 'sales', 'company']):
        return 'business'
    
    # Creative domains
    if any(word in text_lower for word in ['creative', 'art', 'design', 'story']):
        return 'creative'
    
    return 'general'

def extract_optimized_features(text):
    """
    Extract only the features that performed above random baseline.
    Based on Random Forest Feature Importance analysis.
    
    Features to keep (in order of importance):
    1. complexity_score
    2. word_count  
    3. has_questions
    4. query_token_length
    5. char_count
    6. caps_ratio
    7. punctuation_density
    8. avg_word_length
    9. sentence_count
    10. unique_word_count
    Plus categorical features that were above threshold
    """
    
    # Get basic metrics
    query_token_length = get_token_length(text)
    nlp_vs_code = is_natural_language_or_code(text)
    
    # Get category (single category with priority logic)
    primary_category = categorize_query(text)
    
    # Create category features (one-hot style)
    category_features = {f'category{i}': 0 for i in range(7)}
    category_features[f'category{primary_category}'] = 1
    
    # Get complexity features
    complexity_features = extract_text_complexity_features(text)
    
    # Get categorical features
    question_type = categorize_question_type(text)
    other_subcategory = analyze_other_category(text)
    query_context = get_query_context(text)
    
    # Combine all optimized features
    features = {
        # Core features (highest importance)
        'complexity_score': complexity_features['complexity_score'],
        'word_count': complexity_features['word_count'],
        'has_questions': complexity_features['has_questions'],
        'query_token_length': query_token_length,
        'char_count': complexity_features['char_count'],
        'caps_ratio': complexity_features['caps_ratio'],
        'punctuation_density': complexity_features['punctuation_density'],
        'avg_word_length': complexity_features['avg_word_length'],
        'sentence_count': complexity_features['sentence_count'],
        'unique_word_count': complexity_features['unique_word_count'],
        
        # Basic features
        'nlp_vs_code': nlp_vs_code,
        
        # Category features (above threshold only)
        **category_features,
        
        # Categorical features for encoding
        'question_type': question_type,
        'other_subcategory': other_subcategory,
        'query_context': query_context
    }
    
    return features

# Main extraction function (use this in production)
def extract_all_features(text):
    """Extract all features - maintained for backward compatibility"""
    return extract_optimized_features(text)

# Legacy function aliases for backward compatibility
def categorize_question_type_improved(text):
    """Legacy alias"""
    return categorize_question_type(text)
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
