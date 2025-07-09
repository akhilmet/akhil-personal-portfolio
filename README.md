```
# token_estimator_feature_collection.py - Optimized Feature Extraction for Token Predictor
# Only includes features that performed above random baseline
# Uses EXACT categorical functions from original training for encoding consistency

import re
import string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Initialize tokenizer
tokenizer = MistralTokenizer.v3()

# Global label encoders for categorical features (matching training exactly)
_label_encoders = {}

def _get_or_create_encoder(feature_name, values):
    """Get or create a label encoder for a categorical feature"""
    if feature_name not in _label_encoders:
        _label_encoders[feature_name] = LabelEncoder()
        # Fit with all possible values to ensure consistency with training
        _label_encoders[feature_name].fit(values)
    return _label_encoders[feature_name]

# QueryCategorizationRule pattern implementation
class QueryCategorizationRule:
    """Abstract base class for query categorization rules"""
    def __init__(self, category):
        self.category = category
    
    def execute(self, query_text):
        """Return True/False if the category applies"""
        raise NotImplementedError()

class SummaryCategory(QueryCategorizationRule):
    """Category 3: Summary/Analysis requests (HIGHEST PRIORITY)"""
    def execute(self, query_text):
        text_lower = query_text.lower()
        summary_keywords = ['summary', 'summarize', 'tldr', 'key points', 'main points', 
                           'overview', 'brief', 'analyze', 'analysis', 'review']
        return any(keyword in text_lower for keyword in summary_keywords)

class CodeCategory(QueryCategorizationRule):
    """Category 2: Code requests (HIGH PRIORITY)"""
    def execute(self, query_text):
        text_lower = query_text.lower()
        code_keywords = ['code', 'function', 'script', 'programming', 'algorithm',
                        'implement', 'write a', 'create a function', 'debug', 'fix']
        return any(keyword in text_lower for keyword in code_keywords)

class QuestionCategory(QueryCategorizationRule):
    """Category 1: Questions (MEDIUM PRIORITY)"""
    def execute(self, query_text):
        text_lower = query_text.lower().strip()
        first_word = text_lower.split()[0] if text_lower.split() else ''
        
        question_starters = ['how', 'what', 'why', 'when', 'where', 'who', 
                           'can', 'could', 'would', 'should', 'is', 'are', 
                           'was', 'were', 'do', 'does', 'did']
        
        return first_word in question_starters or query_text.strip().endswith('?')

class ExplanationCategory(QueryCategorizationRule):
    """Category 0: Explanations/Instructions (DEFAULT)"""
    def execute(self, query_text):
        text_lower = query_text.lower()
        explain_keywords = ['explain', 'tell me', 'describe', 'what is', 'how to']
        return any(keyword in text_lower for keyword in explain_keywords)

class CreativeCategory(QueryCategorizationRule):
    """Category 4: Creative requests"""
    def execute(self, query_text):
        text_lower = query_text.lower()
        creative_keywords = ['write a story', 'poem', 'creative', 'imagine', 'fiction']
        return any(keyword in text_lower for keyword in creative_keywords)

class DataCategory(QueryCategorizationRule):
    """Category 5: Data/Research requests"""
    def execute(self, query_text):
        text_lower = query_text.lower()
        data_keywords = ['data', 'research', 'statistics', 'numbers', 'calculate']
        return any(keyword in text_lower for keyword in data_keywords)

class OtherCategory(QueryCategorizationRule):
    """Category 6: Other/General (FALLBACK)"""
    def execute(self, query_text):
        return True  # Always matches as fallback

class TokenEstimatorFeatureCollection:
    """
    Feature collection class for token predictor.
    Can be instantiated per request or used as singleton.
    Uses EXACT categorical functions from original training.
    """
    
    def __init__(self, feature_config=None):
        """Initialize with optional feature configuration"""
        self.config = feature_config or {}
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize label encoders for categorical features with EXACT training values"""
        # EXACT categorical values from original training
        question_types = ["who", "what", "how", "where", "when", "why", "other"]
        other_subcategories = ["imperative_request", "yes_no_question", "comparative", 
                              "help_request", "creative_generation", "analysis_request", 
                              "opinion_request", "calculation", "unclassified_other"]
        query_contexts = ["continuation", "independent", "unknown"]
        
        # Initialize encoders with exact training values
        _get_or_create_encoder('question_type', question_types)
        _get_or_create_encoder('other_subcategory', other_subcategories)
        _get_or_create_encoder('query_context', query_contexts)
    
    def preprocess(self, query):
        """Any preprocessing on the input query"""
        return query
    
    def engineer_features(self, processed_query):
        """Extract features from the processed query"""
        return self.extract_optimized_features(processed_query)
    
    def get_features(self, query):
        """Exposed method to the router ensemble"""
        processed_query = self.preprocess(query)
        feature_vector = self.engineer_features(processed_query)
        return feature_vector
    
    def get_token_length(self, text):
        """Get accurate token count using MistralTokenizer"""
        try:
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            # Fallback to character-based estimation
            return max(1, len(text) // 4)

    def categorize_query(self, text):
        """
        Categorize query into one of 7 categories using QueryCategorizationRule pattern.
        Rules are evaluated in sequence with priority logic.
        """
        # Initialize rule classes and evaluate in sequence
        rules = [
            SummaryCategory(3),
            CodeCategory(2), 
            QuestionCategory(1),
            ExplanationCategory(0),
            CreativeCategory(4),
            DataCategory(5),
            OtherCategory(6)
        ]
        
        # Run through rules in sequence and return first match
        for rule in rules:
            if rule.execute(text):
                return rule.category
        
        # Default fallback
        return 6

    def extract_text_complexity_features(self, text):
        """Extract text complexity and linguistic features"""
        # Basic text metrics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
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
            (len(set(word.lower().strip(string.punctuation) for word in words)) / max(1, word_count)) * 0.3 +  # Vocabulary diversity
            (avg_word_length / 10) * 0.2 +                    # Word complexity
            (len(re.split(r'[.!?]+', text.strip())) / max(1, word_count / 15)) * 0.2 + # Sentence structure
            punctuation_density * 0.3                          # Punctuation complexity
        )
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'punctuation_density': punctuation_density,
            'caps_ratio': caps_ratio,
            'has_questions': has_questions,
            'complexity_score': complexity_score
        }

    def categorize_question_type_improved(self, text):
        """
        EXACT function from original training - Single category per prompt with priority order
        """
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

    def analyze_other_category(self, text):
        """
        EXACT function from original training - Refined analysis of 'other' category queries
        """
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

    def is_independent_or_continuation(self, text):
        """
        EXACT function from original training - Determine if query is independent or multi-thought
        """
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

    def extract_optimized_features(self, text):
        """
        Extract only the features that performed above random baseline.
        Uses EXACT categorical functions from original training for encoding consistency.
        
        Removed features based on peer review:
        - nlp_vs_code (removed per feedback #2)
        - sentence_count and unique_word_count (removed per feedback #4)
        
        Features kept (in order of importance):
        1. complexity_score
        2. word_count  
        3. has_questions
        4. query_token_length
        5. char_count
        6. caps_ratio
        7. punctuation_density
        8. avg_word_length
        Plus categorical features that were above threshold (ENCODED with training consistency)
        """
        
        # Get basic metrics
        query_token_length = self.get_token_length(text)
        
        # Get category (single category with priority logic)
        primary_category = self.categorize_query(text)
        
        # Create category features (above threshold only)
        category_features = {}
        # Only include categories that were above random threshold
        for i in range(7):
            category_features[f'category{i}'] = 1 if i == primary_category else 0
        
        # Get complexity features
        complexity_features = self.extract_text_complexity_features(text)
        
        # Get categorical features using EXACT original training functions
        question_type = self.categorize_question_type_improved(text)
        other_subcategory = self.analyze_other_category(text)
        query_context = self.is_independent_or_continuation(text)
        
        # ENCODE categorical features for ML model using EXACT training encodings
        question_type_encoded = _get_or_create_encoder('question_type', []).transform([question_type])[0]
        other_subcategory_encoded = _get_or_create_encoder('other_subcategory', []).transform([other_subcategory])[0]
        query_context_encoded = _get_or_create_encoder('query_context', []).transform([query_context])[0]
        
        # Combine all optimized features (removed nlp_vs_code, sentence_count, unique_word_count)
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
            
            # Category features (above threshold only)
            **category_features,
            
            # Categorical features (ENCODED for ML model with training consistency)
            'question_type_encoded': question_type_encoded,
            'other_subcategory_encoded': other_subcategory_encoded,
            'query_context_encoded': query_context_encoded,
            
            # Raw categorical features (for reference)
            'question_type': question_type,
            'other_subcategory': other_subcategory,
            'query_context': query_context
        }
        
        return features

# Standalone functions for backward compatibility
def get_token_length(text):
    """Get accurate token count using MistralTokenizer"""
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback to character-based estimation
        return max(1, len(text) // 4)

def categorize_query(text):
    """Categorize query into one of 7 categories with priority logic"""
    feature_collector = TokenEstimatorFeatureCollection()
    return feature_collector.categorize_query(text)

def extract_all_features(text):
    """Extract all features - main entry point"""
    feature_collector = TokenEstimatorFeatureCollection()
    return feature_collector.extract_optimized_features(text)

def extract_optimized_features(text):
    """Extract optimized features - main entry point"""
    feature_collector = TokenEstimatorFeatureCollection()
    return feature_collector.extract_optimized_features(text)

# Original training functions for exact compatibility
def categorize_question_type_improved(text):
    """EXACT function from original training"""
    feature_collector = TokenEstimatorFeatureCollection()
    return feature_collector.categorize_question_type_improved(text)

def analyze_other_category(text):
    """EXACT function from original training"""
    feature_collector = TokenEstimatorFeatureCollection()
    return feature_collector.analyze_other_category(text)

def is_independent_or_continuation(text):
    """EXACT function from original training"""
    feature_collector = TokenEstimatorFeatureCollection()
    return feature_collector.is_independent_or_continuation(text)

# Utility functions for encoder management
def get_label_encoders():
    """Get the current label encoders for saving/loading"""
    return _label_encoders

def set_label_encoders(encoders):
    """Set the label encoders (for loading from saved state)"""
    global _label_encoders
    _label_encoders = encoders
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
