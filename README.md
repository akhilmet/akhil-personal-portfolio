```
"""
token_predictor.py - Production Token Prediction Node

This module provides the TokenPredictorNode class for predicting response token counts
using the trained Random Forest model and optimized feature extraction.
Integrates with the router ensemble graph architecture.

Usage:
    # As ensemble node:
    predictor_node = TokenPredictorNode()
    result = predictor_node._execute(input_data)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
import logging
from pathlib import Path

# Import ensemble graph node base class
from ..ensemble_graph.ensemble_graph_node import EnsembleGraphNode

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

class TokenPredictorNode(EnsembleGraphNode):
    """
    Production token predictor node using Random Forest model with optimized features.
    
    This class handles loading the trained model, extracting features from queries,
    and predicting response token counts for LLM routing decisions.
    
    Inherits from EnsembleGraphNode to integrate with the router ensemble.
    """
    
    def __init__(self):
        """Initialize the TokenPredictorNode for ensemble graph integration."""
        name: str = "TokenPredictor"
        description = "Predict the expected output token size."
        super().__init__(name, description)
        
        # Default paths relative to this file
        default_base_path = Path(__file__).parent / "model_artifacts"
        
        self.model_path = str(default_base_path / "random_forest_token_predictor.pkl")
        self.encoders_path = str(default_base_path / "label_encoders.pkl")
        
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
    
    def _execute(self, input_data):
        """
        Execute prediction on input data (main ensemble graph method).
        
        Args:
            input_data: Dictionary containing query and other context from ensemble graph
            
        Returns:
            Integer predicted token count (for ensemble graph)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Extract query from input_data dictionary (as per Dwayne's guidance)
        query = input_data.get('query', '')
        
        if not query or not isinstance(query, str):
            logger.warning("⚠️ Empty or invalid query in input_data")
            return 50  # Default fallback value
        
        try:
            # Extract feature vector from query using utils functions
            feature_vector = self._extract_feature_vector(query)
            
            # Ask the model to predict and return integer
            prediction = self.model.predict([feature_vector])[0]
            
            # Ensure prediction is a positive integer
            predicted_tokens = max(1, int(round(prediction)))
            
            logger.debug(f"🔮 Prediction: '{query[:50]}...' → {predicted_tokens} tokens")
            
            # Return the actual integer result (not a formatted string)
            return predicted_tokens
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for query: '{query[:50]}...' Error: {str(e)}")
            # Return reasonable fallback based on query length
            fallback = max(10, len(query.split()) * 3)
            logger.warning(f"⚠️ Using fallback prediction: {fallback} tokens")
            return fallback
    
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
    
    def predict_tokens(self, query: str) -> int:
        """
        Predict the number of response tokens for a given query.
        Helper method for direct token prediction (standalone usage).
        
        Args:
            query: Input query text
            
        Returns:
            Predicted number of response tokens (integer)
        """
        # Simulate input_data format for _execute method
        input_data = {'query': query}
        result = self._execute(input_data)
        return result
    
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
                prediction = self.predict_tokens(query)
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


# Convenience functions for direct usage (when not using ensemble graph)
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
    return predictor.predict_tokens(query)

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
    # Test the token predictor node
    print("🧪 Testing TokenPredictorNode...")
    
    try:
        # Initialize predictor node
        predictor = TokenPredictorNode()
        
        # Test queries
        test_queries = [
            "What is Python?",
            "How do I debug Python code that's throwing an error?",
            "Write a function to sort a list of numbers in ascending order.",
            "Explain the concept of machine learning in simple terms.",
            "What are the main differences between SQL and NoSQL databases?"
        ]
        
        print(f"\n🔮 Testing ensemble graph integration:")
        for query in test_queries:
            # Test _execute method (ensemble graph interface)
            input_data = {'query': query}
            result = predictor._execute(input_data)
            print(f"   Query: '{query[:50]}...'")
            print(f"   Result: {result} tokens")
        
        print(f"\n🔮 Testing direct prediction:")
        for query in test_queries:
            tokens = predictor.predict_tokens(query)
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
