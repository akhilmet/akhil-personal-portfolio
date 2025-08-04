```
Contextual Multi-Armed Bandits - Code Logic
Core Data Structures
pythonclass ContextualBandit:
    def __init__(self):
        self.models = ['llama-8b', 'mistral-8x7b', 'llama-70b']
        
        # For each model, track:
        self.A = {}  # Context covariance matrix (5x5)
        self.b = {}  # Context-reward correlation vector (5x1)
        
        for model in self.models:
            self.A[model] = np.eye(5)      # Identity matrix
            self.b[model] = np.zeros(5)    # Zero vector
        
        self.alpha = 0.1  # Exploration parameter
Context Vector Creation
pythondef create_context(self, tokens, has_pii, quality_score, latency, cost):
    """Convert query info into numerical vector"""
    return np.array([
        tokens / 1000,                    # Normalized tokens
        1.0 if has_pii else 0.0,         # Binary PII flag
        quality_score,                    # Already 0-1
        latency / 10000,                  # Normalized latency (ms)
        cost * 1000                       # Normalized cost
    ])
Model Selection (Upper Confidence Bound)
pythondef select_model(self, context):
    """Choose model using confidence + exploration"""
    
    # Hard constraint first
    if context[1] == 1.0:  # PII = True
        return 'llama-8b'
    
    best_model = None
    best_score = -float('inf')
    
    for model in self.models:
        # Calculate expected reward (what we think will happen)
        A_inv = np.linalg.inv(self.A[model])
        theta = A_inv.dot(self.b[model])
        expected_reward = context.dot(theta)
        
        # Calculate confidence bonus (exploration)
        confidence_bonus = self.alpha * np.sqrt(
            context.dot(A_inv).dot(context)
        )
        
        # Upper Confidence Bound = Expected + Bonus
        ucb_score = expected_reward + confidence_bonus
        
        if ucb_score > best_score:
            best_score = ucb_score
            best_model = model
    
    return best_model
Learning from Outcomes
pythondef update_model(self, model_used, context, actual_outcome):
    """Learn from what actually happened"""
    
    # Calculate reward based on actual performance
    reward = self.calculate_reward(actual_outcome)
    
    # Update model parameters
    self.A[model_used] += np.outer(context, context)  # Update covariance
    self.b[model_used] += reward * context            # Update correlation
    
    # Gradually reduce exploration over time
    self.alpha = max(0.01, self.alpha * 0.999)

def calculate_reward(self, outcome):
    """Convert actual performance to reward (-1 to +1)"""
    cost_efficiency = min(outcome['expected_cost'] / outcome['actual_cost'], 2.0)
    speed_efficiency = min(outcome['expected_latency'] / outcome['actual_latency'], 2.0)
    quality_achievement = outcome['actual_quality']
    
    # Weighted combination
    reward = (
        0.4 * (cost_efficiency - 1.0) +      # Cost bonus/penalty
        0.3 * (speed_efficiency - 1.0) +     # Speed bonus/penalty  
        0.3 * (quality_achievement - 0.5)    # Quality bonus/penalty
    )
    
    return np.clip(reward, -1, 1)  # Keep in [-1, 1] range
Complete Usage Flow
pythondef route_query(self, tokens, has_pii, quality_score, latency, cost):
    """Main routing function"""
    
    # 1. Create context vector
    context = self.create_context(tokens, has_pii, quality_score, latency, cost)
    
    # 2. Select model using UCB
    chosen_model = self.select_model(context)
    
    # 3. Execute query and get results
    actual_outcome = self.execute_query(chosen_model, query)
    
    # 4. Learn from the outcome
    self.update_model(chosen_model, context, actual_outcome)
    
    return chosen_model

Learning Evolution Example
# Week 1: High exploration (alpha = 0.1)
# Confidence bonuses are large → tries different models frequently

# Month 3: Reduced exploration (alpha ≈ 0.05)  
# More confident in decisions → mostly exploits learned patterns

# Month 6: Minimal exploration (alpha ≈ 0.01)
# Very confident → only explores when genuinely uncertain
```


```
Dynamic Weight Adjustment Logic
Base Weights
pythonweights = {
    'cost_importance': 0.3,
    'speed_importance': 0.3, 
    'quality_importance': 0.3,
    'privacy_importance': 0.1
}
Token-Based Adjustments
python# Simple queries - prioritize cost and speed
if tokens < 350:
    weights['cost_importance'] += 0.1
    weights['speed_importance'] += 0.1
    weights['quality_importance'] -= 0.2

# Complex queries - prioritize quality  
elif tokens > 1000:
    weights['quality_importance'] += 0.15
    weights['cost_importance'] -= 0.075
    weights['speed_importance'] -= 0.075
Context-Based Adjustments
python# Large quality differences between models
if max(quality_scores) - min(quality_scores) > 0.3:
    weights['quality_importance'] += 0.15
    weights['cost_importance'] -= 0.075
    weights['speed_importance'] -= 0.075

# Large speed differences between models  
if max(latency_scores) - min(latency_scores) > 2000:  # milliseconds
    weights['speed_importance'] += 0.15
    weights['cost_importance'] -= 0.075
    weights['quality_importance'] -= 0.075

# Expensive queries
if estimated_cost > 0.10:
    weights['cost_importance'] += 0.15
    weights['quality_importance'] -= 0.075
    weights['speed_importance'] -= 0.075

Optimization Formula
Objective Function (Minimize)
pythonfor each model:
    cost = tokens * model_cost_per_token
    speed = latency_ms / 1000  # Convert to seconds
    quality = quality_score   # 0-1 scale
    
    objective_score = (
        weights['cost_importance'] * cost +
        weights['speed_importance'] * speed -
        weights['quality_importance'] * quality  # Subtract because higher is better
    )

return model_with_lowest_score
Model Costs
pythonmodel_costs = {
    'llama-8b': 0.0001,      # $0.0001 per token (local)
    'mistral-8x7b': 0.0003,  # $0.0003 per token (remote)  
    'llama-70b': 0.0008      # $0.0008 per token (remote)
}

Decision Flow Summary
Query Input → Check PII → Check Token Count → Apply Rules → Adjust Weights → Optimize → Return Model

1. PII? → llama-8b (done)
2. < 350 tokens? → llama-8b (unless big quality gap)
3. > 1000 tokens? → Remote models only (unless they're bad)  
4. Medium tokens? → Optimize all models
5. Apply dynamic weights based on context
6. Calculate objective scores and pick minimum



Example 1: Simple Math Question (200 tokens)
python# Input
tokens = 200
has_pii = False
quality_scores = {'llama-8b': 0.6, 'mistral-8x7b': 0.7, 'llama-70b': 0.8}
latency_scores = {'llama-8b': 800, 'mistral-8x7b': 2200, 'llama-70b': 3500}
cost_per_token = 0.0002

# Decision Logic:
# 1. No PII ✓
# 2. tokens (200) < 350 → Simple query
# 3. Quality gap: 0.8 - 0.6 = 0.2 < 0.4 → Not significant
# 4. Route to "llama-8b"

result = "llama-8b"  # Fast and cheap for simple queries
Example 2: Creative Writing (300 tokens, big quality gap)
python# Input  
tokens = 300
has_pii = False
quality_scores = {'llama-8b': 0.3, 'mistral-8x7b': 0.6, 'llama-70b': 0.9}
latency_scores = {'llama-8b': 600, 'mistral-8x7b': 1800, 'llama-70b': 2800}
cost_per_token = 0.0003

# Decision Logic:
# 1. No PII ✓  
# 2. tokens (300) < 350 → Simple query
# 3. Quality gap: 0.9 - 0.3 = 0.6 > 0.4 → Significant gap!
# 4. Use full optimization
# 5. Adjusted weights favor quality due to large gap
# 6. Calculate scores for all models

result = "mistral-8x7b" or "llama-70b"  # Depends on optimization

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
