Latency = Time to read the entire model from GPU memory × Number of tokens to process
Step-by-Step Process:
1. Calculate Base Reading Time
base_time_per_token = model_size_gb / (gpu_bandwidth × number_of_shards)

# Example: llama-3.3-70b with 8 A10s
base_time_per_token = 140 GB / (240 GB/s × 8) = 72.9 ms
2. Calculate Input Processing (Prefill)
python# Process all input tokens in parallel (one pass through model)
prefill_time = input_tokens × base_time_per_token

# Example: 8 input tokens
prefill_time = 8 × 72.9 ms = 583 ms
3. Calculate Output Generation (Decode)
python# Generate each output token one-by-one (sequential)
decode_time = output_tokens × base_time_per_token

# Example: 180 output tokens  
decode_time = 180 × 72.9 ms = 13,122 ms
4. Total Latency
python# TTFT (Time to First Token)
ttft = prefill_time + base_time_per_token
ttft = 583 + 72.9 = 656 ms

# Full Completion Time
total_time = prefill_time + decode_time
total_time = 583 + 13,122 = 13,705 ms = 13.7 seconds

Real-World Example:
Query: "What is Python?" → Predict 118 tokens output
llama-3.1-8b (1 A10):
Base time: 16GB / 240GB/s = 66.7ms per token
Input: 3 tokens × 66.7ms = 200ms
Output: 118 tokens × 66.7ms = 7,870ms
Total: 200 + 7,870 = 8,070ms = 8.1 seconds


```
"""
Latency Predictor for Query Router

Hardware-based latency prediction for intelligent LLM routing using the
trained token predictor model and Mistral tokenizer for accurate estimates.

This module provides latency estimation based on GPU memory bandwidth,
model specifications, and predicted response token counts to enable optimal
model selection in the query router.
"""

import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import the trained token predictor
try:
    from token_predictor import TokenPredictorNode, predict_tokens
    TOKEN_PREDICTOR_AVAILABLE = True
except ImportError:
    print("⚠️ Token predictor not available, using fallback estimation")
    TOKEN_PREDICTOR_AVAILABLE = False

# Import Mistral tokenizer for accurate token counting
try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    MISTRAL_AVAILABLE = True
    tokenizer = MistralTokenizer.v3()
except ImportError:
    print("⚠️ Mistral tokenizer not available, using approximation")
    MISTRAL_AVAILABLE = False
    tokenizer = None


@dataclass
class ModelSpec:
    """Model specification for latency calculations."""
    name: str
    size_gb: float
    shards: int
    client_model_name: str


class LatencyPredictor:
    """
    Hardware-based latency predictor for LLM routing decisions.
    
    Predicts Time-to-First-Token (TTFT) and full completion latency
    based on model size, GPU memory bandwidth, and accurate token counts
    using the trained token predictor model.
    """
    
    # Default model configurations for Capital One AI Sandbox
    DEFAULT_MODELS = [
        ModelSpec("llama-3.1-8b", size_gb=16, shards=1, client_model_name="llama-3.1-8b"),
        ModelSpec("llama-3.3-70b", size_gb=140, shards=8, client_model_name="llama-3.3-70b"),
        ModelSpec("mixtral-8x7b", size_gb=87, shards=4, client_model_name="mixtral-8x7b"),
    ]
    
    def __init__(self, models: Optional[List[ModelSpec]] = None, bandwidth_gbps: float = 600.0):
        """
        Initialize the latency predictor.
        
        Args:
            models: List of ModelSpec objects. Uses defaults if None.
            bandwidth_gbps: GPU memory bandwidth in GB/s (default: 600.0)
        """
        self.models = models or self.DEFAULT_MODELS
        self.bandwidth_gbps = bandwidth_gbps
        self._model_lookup = {model.name: model for model in self.models}
        
        # Initialize token predictor
        self.token_predictor = None
        if TOKEN_PREDICTOR_AVAILABLE:
            try:
                self.token_predictor = TokenPredictorNode()
                print("✅ Token predictor loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load token predictor: {e}")
        
    def estimate_input_tokens(self, text: str) -> int:
        """
        Estimate token count for input text using Mistral tokenizer.
        
        Uses accurate Mistral tokenization when available, falls back to
        Llama approximation patterns:
        - ~1.3 tokens per word for English
        - Additional tokens for punctuation and special characters
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Estimated input token count
        """
        if not text or not text.strip():
            return 1
            
        # Use Mistral tokenizer if available (most accurate)
        if MISTRAL_AVAILABLE and tokenizer:
            try:
                tokens = tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                print(f"⚠️ Mistral tokenization failed: {e}")
        
        # Fallback to approximation
        words = text.split()
        
        # Base token count (Llama averages ~1.3 tokens per word)
        base_tokens = len(words) * 1.3
        
        # Additional tokens for various other elements
        punct_tokens = len(re.findall(r'[.,!?;:()\[\]"\']', text)) * 0.5
        number_tokens = len(re.findall(r'\d+', text)) * 0.7
        special_tokens = len(re.findall(r'[@#$%^&*+=<>~/\\|]', text)) * 0.3
        
        total_tokens = base_tokens + punct_tokens + number_tokens + special_tokens
        return max(1, int(total_tokens))
    
    def estimate_output_tokens(self, text: str) -> int:
        """
        Estimate expected output token count using the trained token predictor.
        
        Uses the Random Forest model trained on query-response patterns
        to predict response length with high accuracy (~13.75 MAE).
        
        Args:
            text: Input prompt text
            
        Returns:
            Predicted output token count
        """
        if not text or not text.strip():
            return 50  # Default fallback
        
        # Use trained token predictor if available
        if TOKEN_PREDICTOR_AVAILABLE and self.token_predictor:
            try:
                predicted_tokens = self.token_predictor._execute(text)
                return max(1, predicted_tokens)
            except Exception as e:
                print(f"⚠️ Token prediction failed: {e}")
        
        # Fallback to heuristic-based estimation
        input_tokens = self.estimate_input_tokens(text)
        text_lower = text.lower()
        
        # Base multiplier based on input length
        base_multiplier = 2.0
        
        # Adjust based on query type
        if any(keyword in text_lower for keyword in ['explain', 'describe', 'tell me about']):
            base_multiplier = 3.5  # Explanations tend to be longer
        elif any(keyword in text_lower for keyword in ['code', 'function', 'implement']):
            base_multiplier = 4.0  # Code responses are typically longer
        elif any(keyword in text_lower for keyword in ['list', 'steps', 'how to']):
            base_multiplier = 3.0  # Structured responses
        elif text.strip().endswith('?') and len(text.split()) < 10:
            base_multiplier = 1.5  # Short questions
        elif any(keyword in text_lower for keyword in ['summary', 'summarize', 'brief']):
            base_multiplier = 1.2  # Summaries are shorter
        
        estimated_output = int(input_tokens * base_multiplier)
        return max(10, estimated_output)
    
    def estimate_ttft(self, prompt_tokens: int, model_name: str) -> float:
        """
        Estimate Time-to-First-Token (TTFT) in milliseconds.
        
        TTFT = Prefill latency + First token decode latency
        Based on GPU memory bandwidth bottleneck model.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            model_name: Name of the model to use
            
        Returns:
            Estimated TTFT in milliseconds
            
        Raises:
            ValueError: If model_name is not found
        """
        model = self._get_model(model_name)
        
        # Effective bandwidth scales with GPU shards
        effective_bandwidth = self.bandwidth_gbps * model.shards
        
        # Base per-token latency (ms) - accounts for memory access patterns
        base_latency_ms = (model.size_gb / effective_bandwidth) * 1000.0
        
        # Prefill cost (process entire prompt)
        prefill_cost = prompt_tokens * base_latency_ms
        
        # First token decode cost (single token generation)
        decode_cost = base_latency_ms
        
        return prefill_cost + decode_cost
    
    def estimate_full_latency(
        self,
        prompt_tokens: int,
        expected_output_tokens: int,
        model_name: str
    ) -> float:
        """
        Estimate end-to-end completion latency in milliseconds.
        
        Full latency = Prefill latency + (Output tokens × Decode latency)
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            expected_output_tokens: Expected number of output tokens
            model_name: Name of the model to use
            
        Returns:
            Estimated full completion latency in milliseconds
            
        Raises:
            ValueError: If model_name is not found
        """
        model = self._get_model(model_name)
        
        # Effective bandwidth scales with GPU shards
        effective_bandwidth = self.bandwidth_gbps * model.shards
        base_latency_ms = (model.size_gb / effective_bandwidth) * 1000.0
        
        # Prefill phase (process entire prompt once)
        prefill_cost = prompt_tokens * base_latency_ms
        
        # Decode phase (generate each output token sequentially)
        decode_cost = expected_output_tokens * base_latency_ms
        
        return prefill_cost + decode_cost
    
    def predict_latencies(
        self, 
        prompt: str, 
        expected_output_tokens: int = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict latencies for all available models using trained token predictor.
        
        Args:
            prompt: Input prompt text
            expected_output_tokens: Expected number of output tokens.
                                  If None, uses token predictor estimation.
                                  
        Returns:
            Dictionary mapping model names to latency predictions:
            {
                "model_name": {
                    "ttft_ms": float,
                    "full_latency_ms": float,
                    "prompt_tokens": int,
                    "expected_output_tokens": int,
                    "model_size_gb": float,
                    "model_shards": int
                }
            }
        """
        # Get accurate token counts
        prompt_tokens = self.estimate_input_tokens(prompt)
        
        if expected_output_tokens is None:
            expected_output_tokens = self.estimate_output_tokens(prompt)
        
        results = {}
        
        for model in self.models:
            ttft = self.estimate_ttft(prompt_tokens, model.name)
            full_latency = self.estimate_full_latency(
                prompt_tokens, expected_output_tokens, model.name
            )
            
            results[model.name] = {
                "ttft_ms": round(ttft, 2),
                "full_latency_ms": round(full_latency, 2),
                "prompt_tokens": prompt_tokens,
                "expected_output_tokens": expected_output_tokens,
                "model_size_gb": model.size_gb,
                "model_shards": model.shards
            }
        
        return results
    
    def recommend_models_for_budget(
        self, 
        prompt: str, 
        max_ttft_ms: float
    ) -> List[Dict[str, any]]:
        """
        Recommend models that meet a TTFT budget requirement.
        
        Args:
            prompt: Input prompt text
            max_ttft_ms: Maximum acceptable TTFT in milliseconds
            
        Returns:
            List of model recommendations sorted by TTFT (fastest first):
            [
                {
                    "model_name": str,
                    "client_model_name": str,
                    "ttft_ms": float,
                    "prompt_tokens": int
                }
            ]
        """
        prompt_tokens = self.estimate_input_tokens(prompt)
        candidates = []
        
        for model in self.models:
            ttft = self.estimate_ttft(prompt_tokens, model.name)
            
            if ttft <= max_ttft_ms:
                candidates.append({
                    "model_name": model.name,
                    "client_model_name": model.client_model_name,
                    "ttft_ms": round(ttft, 2),
                    "prompt_tokens": prompt_tokens,
                    "model_size_gb": model.size_gb,
                    "model_shards": model.shards
                })
        
        # Sort by TTFT (fastest first)
        return sorted(candidates, key=lambda x: x["ttft_ms"])
    
    def get_fastest_model(self, prompt: str) -> Dict[str, any]:
        """
        Get the fastest model for a given prompt based on TTFT.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Dictionary with fastest model information including predictions
        """
        predictions = self.predict_latencies(prompt)
        fastest_model = min(predictions.items(), key=lambda x: x[1]["ttft_ms"])
        
        model_obj = self._get_model(fastest_model[0])
        
        return {
            "model_name": fastest_model[0],
            "client_model_name": model_obj.client_model_name,
            **fastest_model[1]
        }
    
    def _get_model(self, model_name: str) -> ModelSpec:
        """
        Get model specification by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelSpec object
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in self._model_lookup:
            available_models = list(self._model_lookup.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        return self._model_lookup[model_name]
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names
        """
        return [model.name for model in self.models]


# Convenience function for quick predictions
def predict_latency(
    prompt: str, 
    model_name: str, 
    expected_output_tokens: int = None
) -> Dict[str, float]:
    """
    Quick latency prediction for a single model using trained token predictor.
    
    Args:
        prompt: Input prompt text
        model_name: Name of the model
        expected_output_tokens: Expected output tokens (uses predictor if None)
        
    Returns:
        Dictionary with latency predictions
    """
    predictor = LatencyPredictor()
    predictions = predictor.predict_latencies(prompt, expected_output_tokens)
    
    if model_name not in predictions:
        available_models = list(predictions.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return predictions[model_name]


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing LatencyPredictor with Token Predictor Integration...")
    
    try:
        # Initialize predictor
        predictor = LatencyPredictor()
        
        # Test queries with different complexity levels
        test_queries = [
            "What is Python?",
            "How do I debug Python code that's throwing an error?",
            "Write a function to sort a list of numbers in ascending order and explain the algorithm.",
            "Explain the concept of machine learning, deep learning, and neural networks in detail.",
            "Compare the performance characteristics of different sorting algorithms."
        ]
        
        print(f"\n🔮 Testing latency predictions:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: '{query[:60]}...' ---")
            
            # Get predictions for all models
            predictions = predictor.predict_latencies(query)
            
            for model_name, pred in predictions.items():
                print(f"  {model_name:15} | "
                      f"TTFT: {pred['ttft_ms']:6.1f}ms | "
                      f"Full: {pred['full_latency_ms']:6.1f}ms | "
                      f"Tokens: {pred['prompt_tokens']}→{pred['expected_output_tokens']}")
            
            # Get fastest model recommendation
            fastest = predictor.get_fastest_model(query)
            print(f"  🏆 Fastest: {fastest['model_name']} ({fastest['ttft_ms']:.1f}ms)")
        
        # Test budget-based recommendations
        print(f"\n💰 Models under 100ms TTFT budget:")
        budget_recs = predictor.recommend_models_for_budget(test_queries[0], 100.0)
        for rec in budget_recs:
            print(f"  {rec['model_name']:15} | TTFT: {rec['ttft_ms']:6.1f}ms")
        
        print(f"\n📋 Available models: {predictor.get_available_models()}")
        print("\n✅ LatencyPredictor test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("🔧 Make sure token_predictor.py and model artifacts are available!")
```

```
"""
token_predictor.py - Production Token Prediction Module

This module provides the TokenPredictorNode class for predicting response token counts
using the trained Random Forest model and optimized feature extraction.

Usage:
    predictor = TokenPredictorNode()
    predicted_tokens = predictor.predict("How do I debug Python code?")
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
import logging
from pathlib import Path

# Import our optimized feature extraction
from token_estimator_feature_collection import (
    extract_optimized_features, 
    get_label_encoders,
    set_label_encoders,
    TokenEstimatorFeatureCollection
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenPredictorNode:
    """
    Production token predictor using Random Forest model with optimized features.
    
    This class handles loading the trained model, extracting features from queries,
    and predicting response token counts for LLM routing decisions.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 encoders_path: Optional[str] = None):
        """
        Initialize the TokenPredictorNode.
        
        Args:
            model_path: Path to the trained Random Forest model (.pkl file)
            encoders_path: Path to the label encoders (.pkl file) - optional
        """
        
        # Default paths relative to this file
        default_base_path = Path(__file__).parent / "model_artifacts"
        
        self.model_path = model_path or str(default_base_path / "random_forest_token_predictor.pkl")
        self.encoders_path = encoders_path or str(default_base_path / "label_encoders.pkl")
        
        # Initialize components
        self.model = None
        self.selected_features = None
        self.feature_collector = None
        self.is_loaded = False
        
        # Load model components
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and associated components."""
        try:
            logger.info("🔄 Loading Random Forest token predictor model...")
            
            # Load the trained Random Forest model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Loaded model from: {self.model_path}")
            
            # Load label encoders (optional - we can work without them)
            if os.path.exists(self.encoders_path):
                encoders = joblib.load(self.encoders_path)
                set_label_encoders(encoders)
                logger.info(f"✅ Loaded encoders from: {self.encoders_path}")
            else:
                logger.info(f"⚠️ Encoders file not found: {self.encoders_path} - using built-in encoding")
            
            # Set features based on our extraction methods (don't require external file)
            self._set_default_features()
            logger.info(f"✅ Using {len(self.selected_features)} features based on our extraction methods")
            
            # Initialize feature collector
            self.feature_collector = TokenEstimatorFeatureCollection()
            
            self.is_loaded = True
            logger.info("✅ TokenPredictorNode initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _set_default_features(self) -> None:
        """Set default features based on our feature extraction methods."""
        self.selected_features = [
            # Core numerical features (that we can extract)
            'complexity_score', 'word_count', 'has_questions', 'query_token_length',
            'char_count', 'caps_ratio', 'punctuation_density', 'avg_word_length',
            
            # Category features (that we can extract) 
            'category0', 'category1', 'category2', 'category3', 
            'category4', 'category5', 'category6',
            
            # Encoded categorical features (that we can encode)
            'question_type_encoded', 'other_subcategory_encoded', 'query_context_encoded'
        ]
        logger.info(f"✅ Using default features: {len(self.selected_features)} features")
    
    def _extract_feature_vector(self, query: str) -> np.ndarray:
        """
        Extract feature vector from query text.
        
        Args:
            query: Input query text
            
        Returns:
            numpy array of feature values in the correct order
        """
        try:
            # Extract all features using optimized feature collection
            features_dict = self.feature_collector.extract_optimized_features(query)
            
            # Create feature vector in the correct order
            feature_vector = []
            for feature_name in self.selected_features:
                if feature_name in features_dict:
                    feature_vector.append(features_dict[feature_name])
                else:
                    # Use default value for missing features
                    feature_vector.append(0)
                    logger.warning(f"⚠️ Missing feature '{feature_name}', using default value 0")
            
            return np.array(feature_vector)
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for query: '{query[:50]}...' Error: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}")
    
    def _execute(self, input_data: str) -> int:
        """
        Execute prediction on input data (main method as specified by Christopher).
        
        Args:
            input_data: Query text to predict tokens for
            
        Returns:
            Integer number of predicted tokens
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        if not input_data or not isinstance(input_data, str):
            logger.warning("⚠️ Empty or invalid input_data provided")
            return 50  # Default fallback value
        
        try:
            # Extract feature vector from input_data using utils functions
            feature_vector = self._extract_feature_vector(input_data)
            
            # Ask the model to predict and return integer
            prediction = self.model.predict([feature_vector])[0]
            
            # Ensure prediction is a positive integer
            predicted_tokens = max(1, int(round(prediction)))
            
            logger.debug(f"🔮 Prediction: '{input_data[:50]}...' → {predicted_tokens} tokens")
            
            return predicted_tokens
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for input_data: '{input_data[:50]}...' Error: {str(e)}")
            # Return reasonable fallback based on query length
            fallback = max(10, len(input_data.split()) * 3)
            logger.warning(f"⚠️ Using fallback prediction: {fallback} tokens")
            return fallback
    
    def predict(self, query: str) -> int:
        """
        Predict the number of response tokens for a given query.
        Wrapper around _execute for backward compatibility.
        
        Args:
            query: Input query text
            
        Returns:
            Predicted number of response tokens (integer)
        """
        return self._execute(query)
    
    def predict_batch(self, queries: list) -> list:
        """
        Predict token counts for a batch of queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of predicted token counts
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        predictions = []
        for query in queries:
            try:
                prediction = self.predict(query)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"❌ Batch prediction failed for query: '{query[:50]}...'")
                predictions.append(50)  # Fallback value
        
        logger.info(f"✅ Batch prediction complete: {len(predictions)} queries processed")
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise RuntimeError("Model does not support feature importance.")
        
        importance_dict = {}
        for feature, importance in zip(self.selected_features, self.model.feature_importances_):
            importance_dict[feature] = float(importance)
        
        return importance_dict
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "n_features": len(self.selected_features),
            "features": self.selected_features,
            "model_path": self.model_path,
            "encoders_path": self.encoders_path
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info["max_depth"] = self.model.max_depth
        if hasattr(self.model, 'oob_score_'):
            info["oob_score"] = self.model.oob_score_
            
        return info
    
    def reload_model(self) -> None:
        """Reload the model from disk."""
        logger.info("🔄 Reloading model...")
        self.is_loaded = False
        self._load_model()


# Convenience functions for direct usage
_global_predictor = None

def get_predictor() -> TokenPredictorNode:
    """Get or create the global token predictor instance."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = TokenPredictorNode()
    return _global_predictor

def predict_tokens(query: str) -> int:
    """
    Convenience function to predict tokens for a single query.
    
    Args:
        query: Input query text
        
    Returns:
        Predicted number of response tokens
    """
    predictor = get_predictor()
    return predictor.predict(query)

def predict_tokens_batch(queries: list) -> list:
    """
    Convenience function to predict tokens for multiple queries.
    
    Args:
        queries: List of query strings
        
    Returns:
        List of predicted token counts
    """
    predictor = get_predictor()
    return predictor.predict_batch(queries)


# Example usage and testing
if __name__ == "__main__":
    # Test the token predictor
    print("🧪 Testing TokenPredictorNode...")
    
    try:
        # Initialize predictor
        predictor = TokenPredictorNode()
        
        # Test queries
        test_queries = [
            "What is Python?",
            "How do I debug Python code that's throwing an error?",
            "Write a function to sort a list of numbers in ascending order.",
            "Explain the concept of machine learning in simple terms.",
            "What are the main differences between SQL and NoSQL databases?"
        ]
        
        print(f"\n🔮 Testing predictions:")
        for query in test_queries:
            tokens = predictor.predict(query)
            print(f"   '{query[:50]}...' → {tokens} tokens")
        
        # Test batch prediction
        batch_predictions = predictor.predict_batch(test_queries)
        print(f"\n📊 Batch prediction: {batch_predictions}")
        
        # Show model info
        model_info = predictor.get_model_info()
        print(f"\n📋 Model Info:")
        print(f"   Status: {model_info['status']}")
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Features: {model_info['n_features']}")
        
        print("\n✅ TokenPredictorNode test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("🔧 Make sure you've trained the model using the retraining notebook first!")

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
