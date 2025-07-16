```
"""
Latency Predictor for Query Router

Hardware-based latency prediction for intelligent LLM routing using the
trained token predictor model and Mistral tokenizer for accurate estimates.

This module provides latency estimation based on A10 GPU memory bandwidth,
model specifications, and predicted response token counts to enable optimal
model selection in the query router.

Hardware Configuration:
- GPU Type: A10 (600 GB/s memory bandwidth per GPU)
- Models: All models use 4 A10 GPUs for consistent performance
"""

import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import ensemble graph node base class
try:
    from ..ensemble_graph.ensemble_graph_node import EnsembleGraphNode
except ImportError:
    # Fallback for when running standalone or testing
    class EnsembleGraphNode:
        def __init__(self, name, description):
            self.name = name
            self.description = description

# Import Mistral tokenizer for accurate token counting
try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
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


class LatencyPredictorNode(EnsembleGraphNode):
    """
    Hardware-based latency predictor node for LLM routing decisions.
    
    Predicts Time-to-First-Token (TTFT) and full completion latency
    based on model size, GPU memory bandwidth, and accurate token counts
    using the trained token predictor model.
    
    Inherits from EnsembleGraphNode to integrate with the router ensemble.
    """
    
    # Default model configurations for Capital One AI Sandbox (from production data)
    DEFAULT_MODELS = [
        ModelSpec("llama-3.1-8b", size_gb=16, shards=4, client_model_name="genai-llama3-8b-sandbox-1"),
        ModelSpec("llama-3.3-70b", size_gb=140, shards=4, client_model_name="genai-llama3-70b-sandbox-1"), 
        ModelSpec("mixtral-8x7b", size_gb=87, shards=4, client_model_name="genai-mixtral-8x7b-sandbox-1"),
    ]
    
    def __init__(self):
        """Initialize the LatencyPredictorNode for ensemble graph integration."""
        name: str = "LatencyPredictor"
        description = "Predict the latency of processing based on token prediction."
        super().__init__(name, description)
        
        # Initialize latency predictor components
        self.models = self.DEFAULT_MODELS
        self.bandwidth_gbps = 600.0  # A10 GPU bandwidth
        self._model_lookup = {model.name: model for model in self.models}
    
    def _execute(self, input_data):
        """
        Execute latency prediction on input data.
        
        Args:
            input_data: Dictionary containing query and other context from ensemble graph
            
        Returns:
            Dictionary with latency predictions for all models
        """
        # Extract inputs from the ensemble graph (as per Dwayne's guidance)
        query = input_data.get('query', '')
        
        # Get token prediction from TokenPredictor node output (if available)
        token_prediction = input_data.get('TokenPredictor', None)
        
        if not query:
            return {"error": "No query provided"}
        
        try:
            # Predict latencies for all models
            predictions = self.predict_latencies(query, token_prediction)
            
            # Return the predictions dictionary (graph takes care of the rest)
            return predictions
            
        except Exception as e:
            return {"error": f"Latency prediction failed: {str(e)}"}
    
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
                # Use proper Mistral chat completion format
                messages = [UserMessage(content=text)]
                request = ChatCompletionRequest(messages=messages)
                tokens = tokenizer.encode_chat_completion(request)
                return len(tokens.tokens)
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
    
    def estimate_output_tokens(self, text: str, token_prediction: int = None) -> int:
        """
        Estimate expected output token count.
        
        Uses the token prediction from TokenPredictor node if available,
        otherwise falls back to heuristic-based estimation.
        
        Args:
            text: Input prompt text
            token_prediction: Token prediction from TokenPredictor node
            
        Returns:
            Predicted output token count
        """
        if not text or not text.strip():
            return 50  # Default fallback
        
        # Use token prediction from TokenPredictor node if available
        if token_prediction is not None and isinstance(token_prediction, (int, float)):
            return max(1, int(token_prediction))
        
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
        
        # Effective bandwidth scales with A10 GPU shards
        effective_bandwidth = self.bandwidth_gbps * model.shards
        
        # Base per-token latency (ms) - accounts for A10 memory access patterns
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
        
        # Effective bandwidth scales with A10 GPU shards
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
            expected_output_tokens: Expected number of output tokens from TokenPredictor node.
                                  If None, uses heuristic estimation.
                                  
        Returns:
            Dictionary mapping model names to latency predictions
        """
        # Get accurate input token count
        prompt_tokens = self.estimate_input_tokens(prompt)
        
        # Get expected output token count (from TokenPredictor or fallback)
        if expected_output_tokens is None:
            expected_output_tokens = self.estimate_output_tokens(prompt)
        else:
            expected_output_tokens = self.estimate_output_tokens(prompt, expected_output_tokens)
        
        results = {}
        
        for model in self.models:
            ttft = self.estimate_ttft(prompt_tokens, model.name)
            full_latency = self.estimate_full_latency(
                prompt_tokens, expected_output_tokens, model.name
            )
            
            results[model.name] = {
                "ttft_ms": round(ttft, 2),
                "full_latency_ms": round(full_latency, 2),
                "prompt_tokens": prompt_tokens,           # INPUT tokens
                "expected_output_tokens": expected_output_tokens,  # OUTPUT tokens (different!)
                "model_size_gb": model.size_gb,
                "model_shards": model.shards
            }
        
        return results
    
    def recommend_models_for_budget(
        self, 
        prompt: str, 
        max_ttft_ms: float,
        expected_output_tokens: int = None
    ) -> List[Dict[str, any]]:
        """
        Recommend models that meet a TTFT budget requirement.
        
        Args:
            prompt: Input prompt text
            max_ttft_ms: Maximum acceptable TTFT in milliseconds
            expected_output_tokens: Expected output tokens from TokenPredictor node
            
        Returns:
            List of model recommendations sorted by TTFT (fastest first)
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
    
    def get_fastest_model(self, prompt: str, expected_output_tokens: int = None) -> Dict[str, any]:
        """
        Get the fastest model for a given prompt based on TTFT.
        
        Args:
            prompt: Input prompt text
            expected_output_tokens: Expected output tokens from TokenPredictor node
            
        Returns:
            Dictionary with fastest model information including predictions
        """
        predictions = self.predict_latencies(prompt, expected_output_tokens)
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


# Convenience function for quick predictions (when not using ensemble graph)
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
    predictor = LatencyPredictorNode()
    predictions = predictor.predict_latencies(prompt, expected_output_tokens)
    
    if model_name not in predictions:
        available_models = list(predictions.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return predictions[model_name]


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing LatencyPredictorNode with A10 GPU Integration...")
    
    try:
        # Initialize predictor node
        predictor = LatencyPredictorNode()
        
        # Test queries with different complexity levels
        test_queries = [
            "What is Python?",
            "How do I debug Python code that's throwing an error?",
            "Write a function to sort a list of numbers in ascending order and explain the algorithm.",
            "Explain the concept of machine learning, deep learning, and neural networks in detail.",
            "Compare the performance characteristics of different sorting algorithms."
        ]
        
        print(f"\n🔮 Testing latency predictions with A10 GPUs (600 GB/s each):")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: '{query[:60]}...' ---")
            
            # Test ensemble graph integration
            input_data = {'query': query, 'TokenPredictor': 150}  # Mock token prediction
            result = predictor._execute(input_data)
            
            if "error" not in result:
                for model_name, pred in result.items():
                    # Show shard info for context
                    model_obj = predictor._get_model(model_name)
                    print(f"  {model_name:15} ({model_obj.shards} A10s) | "
                          f"TTFT: {pred['ttft_ms']:6.1f}ms | "
                          f"Full: {pred['full_latency_ms']:6.1f}ms | "
                          f"Input: {pred['prompt_tokens']} → Output: {pred['expected_output_tokens']} tokens")
            else:
                print(f"  Error: {result['error']}")
        
        # Test budget-based recommendations
        print(f"\n💰 Models under 500ms TTFT budget (A10 adjusted):")
        budget_recs = predictor.recommend_models_for_budget(test_queries[0], 500.0, 120)
        for rec in budget_recs:
            model_obj = predictor._get_model(rec['model_name'])
            print(f"  {rec['model_name']:15} ({model_obj.shards} A10s) | TTFT: {rec['ttft_ms']:6.1f}ms")
        
        print(f"\n📋 Available models: {predictor.get_available_models()}")
        print("\n✅ LatencyPredictorNode test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("🔧 Make sure token_predictor.py and model artifacts are available!")
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
