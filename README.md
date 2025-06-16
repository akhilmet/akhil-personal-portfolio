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

## License

This project is licensed under the MIT License.
---
# Capital One AI Sandbox - Enterprise Latency Predictor

Copy and paste each cell below into separate Jupyter notebook cells in your AI Sandbox environment:

## Cell 1: Capital One AI Sandbox Setup

```python
# Capital One AI Sandbox - Enterprise Latency Predictor
# Hardware-based latency prediction for LLM routing

from c1.aiml.inference_client import Client
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import numpy as np
import time
import re

# Initialize the Capital One AI client
client = Client()

print("🚀 Capital One AI Sandbox - Latency Predictor Demo")
print("=" * 60)
print("✅ AI Client initialized")
```

## Cell 2: Model Specifications for Capital One Environment

```python
# =============================================================================
# CAPITAL ONE MODEL SPECIFICATIONS
# =============================================================================
class ModelSpec:
    def __init__(self, name: str, size_gb: float, shards: int = 1, client_model_name: str = None):
        self.name = name
        self.size_gb = size_gb
        self.shards = shards
        self.client_model_name = client_model_name or name
    
    def __repr__(self):
        return f"ModelSpec({self.name}, {self.size_gb}GB, {self.shards} shards)"

# Your Capital One AI Sandbox model configurations
# Update client_model_name based on available models in your environment
AVAILABLE_MODELS: List[ModelSpec] = [
    ModelSpec("llama-3.1-8B", size_gb=16, shards=1, client_model_name="llama-3.1-8b"),
    ModelSpec("llama-3.3-70B", size_gb=140, shards=8, client_model_name="llama-3.3-70b"), 
    ModelSpec("mixtral-8x7B", size_gb=87, shards=4, client_model_name="mixtral-8x7b"),
]

print("📊 Available Models in Capital One AI Sandbox:")
for model in AVAILABLE_MODELS:
    print(f"  • {model.name} -> {model.client_model_name}")
```

## Cell 3: Token Counting with Capital One Client

```python
# =============================================================================
# TOKEN COUNTING FOR CAPITAL ONE ENVIRONMENT
# =============================================================================
def get_token_count_from_client(text: str, model_name: str = "llama-3.1-8b") -> int:
    """
    Get accurate token count using Capital One AI client.
    This uses the actual tokenizer from the model.
    """
    try:
        # Use the client to get token count (if available)
        # This is a placeholder - adjust based on actual Capital One client API
        response = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": text}],
            max_tokens=1,  # Minimal tokens to get count
            return_token_count=True  # Adjust based on actual API
        )
        
        # Extract token count from response metadata
        if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens'):
            return response.usage.prompt_tokens
        else:
            # Fallback to approximation if token count not available
            return estimate_token_count(text)
            
    except Exception as e:
        print(f"Warning: Could not get exact token count, using approximation. Error: {e}")
        return estimate_token_count(text)

def estimate_token_count(text: str) -> int:
    """
    Fallback token estimation for Capital One environment.
    Based on Llama tokenization patterns.
    """
    # More accurate approximation for Llama models
    # Accounts for punctuation, spaces, and typical token boundaries
    words = text.split()
    
    # Base token count (roughly 1.3 tokens per word for English)
    base_tokens = len(words) * 1.3
    
    # Add tokens for punctuation and special characters
    punct_tokens = len(re.findall(r'[.,!?;:(){}[\]"\'`]', text)) * 0.5
    
    # Add tokens for numbers (numbers often become multiple tokens)
    number_tokens = len(re.findall(r'\d+', text)) * 0.7
    
    return max(1, int(base_tokens + punct_tokens + number_tokens))

def llama_token_count(text: str) -> int:
    """Main token counting function for Capital One environment"""
    return get_token_count_from_client(text)

print("✅ Token counting functions loaded for Capital One AI Sandbox")
```

## Cell 4: Latency Estimation Engine

```python
# =============================================================================
# LATENCY ESTIMATION ENGINE
# =============================================================================
def estimate_ttft(
    prompt_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
) -> float:
    """Estimate Time-to-First-Token (TTFT) in milliseconds"""
    # Effective bandwidth scales with GPU shards
    effective_bandwidth = bandwidth_gbps * model.shards
    
    # Base per-token latency
    base_latency_ms = (model.size_gb / effective_bandwidth) * 1000.0
    
    # Prefill cost (process entire prompt)
    prefill_cost = prompt_tokens * base_latency_ms
    
    # First token decode cost
    decode_cost = base_latency_ms
    
    return prefill_cost + decode_cost

def estimate_full_latency(
    prompt_tokens: int,
    expected_output_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
) -> float:
    """Estimate end-to-end latency in milliseconds"""
    effective_bandwidth = bandwidth_gbps * model.shards
    base_latency_ms = (model.size_gb / effective_bandwidth) * 1000.0
    
    # Prefill phase
    prefill_cost = prompt_tokens * base_latency_ms
    
    # Decode phase
    decode_per_token = base_latency_ms
    total_decode_cost = expected_output_tokens * decode_per_token
    
    return prefill_cost + total_decode_cost

def choose_model_for_ttft_budget(
    prompt_text: str,
    budget_ms: float,
) -> List[Dict]:
    """Return models that meet TTFT budget, sorted by speed"""
    prompt_tokens = llama_token_count(prompt_text)
    
    candidates = []
    for model in AVAILABLE_MODELS:
        ttft = estimate_ttft(
            prompt_tokens=prompt_tokens,
            model=model,
        )
        if ttft <= budget_ms:
            candidates.append({
                "model": model.name,
                "client_model": model.client_model_name,
                "ttft_ms": round(ttft, 2),
                "tokens": prompt_tokens,
            })
    
    return sorted(candidates, key=lambda x: x["ttft_ms"])

print("✅ Latency estimation engine loaded")
```

## Cell 5: Real Latency Testing with Capital One Client

```python
# =============================================================================
# REAL LATENCY TESTING WITH CAPITAL ONE CLIENT
# =============================================================================
def measure_actual_latency(prompt: str, model_name: str, max_tokens: int = 50) -> Dict:
    """
    Measure actual latency using Capital One AI client.
    Returns both TTFT and total completion time.
    """
    try:
        start_time = time.time()
        
        # Make request to Capital One AI client
        response = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,  # Low temperature for consistent timing
        )
        
        end_time = time.time()
        total_latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Extract response details
        response_text = response.choices[0].message.content if response.choices else ""
        
        return {
            "success": True,
            "total_latency_ms": round(total_latency, 2),
            "response_length": len(response_text),
            "response_tokens": estimate_token_count(response_text),
            "model": model_name,
            "prompt_tokens": estimate_token_count(prompt)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": model_name
        }

def compare_predicted_vs_actual(prompt: str, expected_output: int = 50) -> pd.DataFrame:
    """
    Compare predicted latencies with actual measured latencies.
    """
    results = []
    prompt_tokens = llama_token_count(prompt)
    
    print(f"🧪 Testing prompt: {prompt[:60]}...")
    print(f"📊 Prompt tokens: {prompt_tokens}")
    print("=" * 50)
    
    for model in AVAILABLE_MODELS:
        # Predicted latency
        predicted_ttft = estimate_ttft(prompt_tokens, model)
        predicted_full = estimate_full_latency(prompt_tokens, expected_output, model)
        
        # Actual latency
        print(f"Testing {model.name}...")
        actual_result = measure_actual_latency(prompt, model.client_model_name, expected_output)
        
        if actual_result["success"]:
            results.append({
                "model": model.name,
                "predicted_ttft_ms": round(predicted_ttft, 2),
                "predicted_full_ms": round(predicted_full, 2),
                "actual_total_ms": actual_result["total_latency_ms"],
                "prediction_accuracy": round(predicted_full / actual_result["total_latency_ms"], 2),
                "prompt_tokens": prompt_tokens,
                "response_tokens": actual_result["response_tokens"],
                "shards": model.shards,
                "size_gb": model.size_gb
            })
            print(f"  ✅ Predicted: {predicted_full:.1f}ms | Actual: {actual_result['total_latency_ms']:.1f}ms")
        else:
            print(f"  ❌ Error: {actual_result['error']}")
    
    return pd.DataFrame(results)

print("✅ Real latency testing functions loaded")
```

## Cell 6: Capital One Test Scenarios

```python
# =============================================================================
# CAPITAL ONE ENTERPRISE TEST SCENARIOS
# =============================================================================
capital_one_scenarios = [
    {
        "name": "Customer Service Query",
        "prompt": "A customer is asking about their credit card rewards balance. How should I respond professionally?",
        "expected_output": 80,
        "category": "customer_service"
    },
    {
        "name": "Risk Analysis",
        "prompt": "Analyze the key risk factors that should be considered when evaluating a commercial loan application.",
        "expected_output": 150,
        "category": "risk_analysis"
    },
    {
        "name": "Regulatory Compliance",
        "prompt": "Explain the key requirements of the Fair Credit Reporting Act (FCRA) for financial institutions.",
        "expected_output": 200,
        "category": "compliance"
    },
    {
        "name": "Data Analysis Query",
        "prompt": "Given transaction data, what SQL query would help identify potential fraudulent patterns?",
        "expected_output": 120,
        "category": "data_analysis"
    },
    {
        "name": "Simple FAQ",
        "prompt": "What are the business hours for Capital One customer service?",
        "expected_output": 30,
        "category": "simple_faq"
    }
]

print(f"🏦 Capital One Enterprise Test Scenarios: {len(capital_one_scenarios)} scenarios")
for scenario in capital_one_scenarios:
    print(f"  • {scenario['name']} ({scenario['category']})")
```

## Cell 7: Run Predictions on Capital One Scenarios

```python
# =============================================================================
# RUN PREDICTIONS ON CAPITAL ONE SCENARIOS
# =============================================================================
results = []

print("🚀 Running Latency Predictions for Capital One Scenarios")
print("=" * 60)

for scenario in capital_one_scenarios:
    prompt = scenario["prompt"]
    expected_output = scenario["expected_output"]
    
    # Get token count
    tokens = llama_token_count(prompt)
    
    print(f"\n📝 {scenario['name']}")
    print(f"   Category: {scenario['category']}")
    print(f"   Tokens: {tokens}")
    
    # Calculate predictions for each model
    for model in AVAILABLE_MODELS:
        ttft = estimate_ttft(tokens, model)
        full_latency = estimate_full_latency(tokens, expected_output, model)
        
        results.append({
            "scenario": scenario["name"],
            "category": scenario["category"],
            "model": model.name,
            "client_model": model.client_model_name,
            "prompt_tokens": tokens,
            "output_tokens": expected_output,
            "ttft_ms": round(ttft, 1),
            "full_latency_ms": round(full_latency, 1),
            "shards": model.shards,
            "size_gb": model.size_gb
        })
        
        print(f"   {model.name:15} | TTFT: {ttft:6.1f}ms | Full: {full_latency:7.1f}ms")

# Create results dataframe
df = pd.DataFrame(results)
print(f"\n📊 Capital One Results Summary")
print("=" * 50)
print(df.groupby(['model', 'category'])[['ttft_ms', 'full_latency_ms']].mean().round(1))
```

## Cell 8: Capital One Visualization Dashboard

```python
# =============================================================================
# CAPITAL ONE VISUALIZATION DASHBOARD
# =============================================================================
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Capital One AI Sandbox - Latency Predictor Analysis', fontsize=16, fontweight='bold')

# 1. TTFT by Model and Business Category
ax1 = axes[0, 0]
ttft_pivot = df.pivot_table(values='ttft_ms', index='category', columns='model', aggfunc='mean')
sns.heatmap(ttft_pivot, annot=True, fmt='.1f', cmap='Blues', ax=ax1)
ax1.set_title('Time to First Token by Business Category')
ax1.set_xlabel('Model')
ax1.set_ylabel('Business Use Case')

# 2. Model Performance Distribution
ax2 = axes[0, 1]
df_plot = df.copy()
df_plot['model_short'] = df_plot['model'].str.replace('llama-', '').str.replace('mixtral-', 'mix-')
sns.boxplot(data=df_plot, x='model_short', y='full_latency_ms', ax=ax2)
ax2.set_title('Latency Distribution Across Models')
ax2.set_xlabel('Model')
ax2.set_ylabel('Full Latency (ms)')
ax2.tick_params(axis='x', rotation=45)

# 3. Business Use Case Complexity
ax3 = axes[1, 0]
category_stats = df.groupby('category').agg({
    'prompt_tokens': 'mean',
    'full_latency_ms': 'mean'
}).reset_index()
sns.scatterplot(data=category_stats, x='prompt_tokens', y='full_latency_ms', ax=ax3, s=100)
for i, row in category_stats.iterrows():
    ax3.annotate(row['category'], (row['prompt_tokens'], row['full_latency_ms']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax3.set_title('Use Case Complexity vs Latency')
ax3.set_xlabel('Average Prompt Tokens')
ax3.set_ylabel('Average Latency (ms)')

# 4. GPU Scaling Efficiency
ax4 = axes[1, 1]
scaling_data = df.groupby('model')[['shards', 'ttft_ms', 'size_gb']].first().reset_index()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
scatter = ax4.scatter(scaling_data['shards'], scaling_data['ttft_ms'], 
                     s=scaling_data['size_gb']*2, alpha=0.7, c=colors)
for i, row in scaling_data.iterrows():
    ax4.annotate(row['model'].replace('llama-', '').replace('mixtral-', 'mix-'), 
                (row['shards'], row['ttft_ms']), xytext=(5, 5), textcoords='offset points')
ax4.set_title('GPU Scaling vs TTFT (bubble size = model size)')
ax4.set_xlabel('Number of GPU Shards')
ax4.set_ylabel('Average TTFT (ms)')

plt.tight_layout()
plt.show()
```

## Cell 9: Real vs Predicted Latency Test

```python
# =============================================================================
# REAL VS PREDICTED LATENCY VALIDATION
# =============================================================================
print("🎯 Real vs Predicted Latency Validation")
print("=" * 50)

# Test with a representative prompt
test_prompt = "Analyze the potential risks and benefits of implementing a new digital banking feature for mobile customers."

# Compare predictions with actual measurements
validation_df = compare_predicted_vs_actual(test_prompt, expected_output=100)

if not validation_df.empty:
    print("\n📊 Prediction Accuracy Results:")
    print(validation_df[['model', 'predicted_full_ms', 'actual_total_ms', 'prediction_accuracy']])
    
    # Visualize accuracy
    plt.figure(figsize=(10, 6))
    x = range(len(validation_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], validation_df['predicted_full_ms'], 
            width, label='Predicted', alpha=0.7)
    plt.bar([i + width/2 for i in x], validation_df['actual_total_ms'], 
            width, label='Actual', alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Latency (ms)')
    plt.title('Predicted vs Actual Latency Comparison')
    plt.xticks(x, validation_df['model'])
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Calculate overall accuracy
    avg_accuracy = validation_df['prediction_accuracy'].mean()
    print(f"\n🎯 Average Prediction Accuracy: {avg_accuracy:.2f}x")
    print("   (1.0 = perfect prediction, <1.0 = under-prediction, >1.0 = over-prediction)")
else:
    print("❌ Could not validate with real measurements. Check your Capital One AI client setup.")
```

## Cell 10: Capital One Routing Recommendations

```python
# =============================================================================
# CAPITAL ONE ROUTING RECOMMENDATIONS
# =============================================================================
print("🏦 Capital One AI Routing Recommendations")
print("=" * 60)

# Define business scenarios with different requirements
business_scenarios = [
    {"name": "Real-time Customer Chat", "max_ttft_ms": 100, "priority": "speed"},
    {"name": "Risk Analysis Report", "max_ttft_ms": 500, "priority": "accuracy"},
    {"name": "Quick FAQ Response", "max_ttft_ms": 50, "priority": "speed"},
    {"name": "Regulatory Document Review", "max_ttft_ms": 1000, "priority": "accuracy"},
]

test_prompts = [
    "What's my account balance?",
    "Analyze credit risk factors for this loan application",
    "When is your customer service open?",
    "Review this compliance document for GDPR requirements"
]

print("📋 Routing Recommendations by Business Scenario:")
for i, scenario in enumerate(business_scenarios):
    print(f"\n💼 {scenario['name']} (Max TTFT: {scenario['max_ttft_ms']}ms)")
    
    candidates = choose_model_for_ttft_budget(test_prompts[i], scenario['max_ttft_ms'])
    
    if candidates:
        print(f"   ✅ Recommended models:")
        for candidate in candidates[:3]:  # Top 3 recommendations
            print(f"      • {candidate['model']:15} - {candidate['ttft_ms']:6.1f}ms")
    else:
        print(f"   ⚠️  No models meet {scenario['max_ttft_ms']}ms budget")
        # Show fastest available option
        all_candidates = choose_model_for_ttft_budget(test_prompts[i], 10000)  # High budget
        if all_candidates:
            fastest = all_candidates[0]
            print(f"      Fastest available: {fastest['model']} - {fastest['ttft_ms']:.1f}ms")

print(f"\n🚀 Integration Ready!")
print("   Next: Integrate these predictions with your Capital One AI routing logic")
```

---

## 🏦 **Capital One AI Sandbox Instructions:**

1. **Run Cell 1** first to initialize the Capital One AI client
2. **Update model names** in Cell 2 based on your available models
3. **Test token counting** in Cell 3 (may need API adjustments)
4. **Run prediction engine** (Cells 4-7) for theoretical analysis
5. **Validate with real tests** (Cell 9) using actual Capital One AI calls
6. **Get routing recommendations** (Cell 10) for your business scenarios

## ⚙️ **Setup Notes:**

- **Model Names:** Update `client_model_name` in Cell 2 to match your sandbox
- **Token Counting:** Adjust Cell 3 based on Capital One client API capabilities  
- **Error Handling:** The code includes fallbacks for API limitations
- **Business Context:** Examples tailored for financial services use cases

This integrates your hardware-based latency predictor with the Capital One AI Sandbox environment!


Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
