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

import re
from typing import List, Dict
from transformers import AutoTokenizer

# --------------------------------------------------------------------------
# Latency Estimator with GPU Scaling
#
# - Uses Llama 3.1 tokenizer for accurate token counts.
# - Estimates TTFT and full latency based purely on GPU memory bandwidth,
#   scaled by the number of GPUs (shards).
# - Applies simple task penalties (math ×1.5, code ×2.0).
# - For now, uses a placeholder “expected_output_tokens”:
#     • Typically taken from API max_tokens
#     • Otherwise defaults to a fixed average (e.g., 50)
# --------------------------------------------------------------------------

# 1. Load Llama 3.1 tokenizer
#    Adjust the model ID or path as needed.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1")

def llama_token_count(text: str) -> int:
    """
    Return the number of tokens in `text` according to the Llama 3.1 tokenizer.
    We disable special tokens so we count exactly the prompt tokens.
    """
    enc = tokenizer(text, return_length=True, add_special_tokens=False)
    if isinstance(enc, dict) and "length" in enc:
        return int(enc["length"])
    return len(enc["input_ids"])

def detect_math(prompt: str) -> bool:
    """
    Return True if the prompt contains digits or basic arithmetic operators.
    Used to apply a compute penalty for math reasoning tasks.
    """
    return bool(re.search(r"[0-9]+|[\+\-\*/=]", prompt))

def detect_code(prompt: str) -> bool:
    """
    Return True if the prompt contains common code keywords or syntax.
    Used to apply a larger penalty for code-generation tasks.
    """
    keywords = ["def", "return", "import", "class", "{", "}", ";", "=>"]
    return any(kw in prompt for kw in keywords)

class ModelSpec:
    def __init__(self, name: str, size_gb: float, shards: int = 1):
        """
        name: model identifier, e.g. "llama-3.1-8B"
        size_gb: model size in GB (e.g. 16 for 8B)
        shards: number of GPUs the model is sharded across
        """
        self.name = name
        self.size_gb = size_gb
        self.shards = shards

# Define your models and how many GPUs they use
AVAILABLE_MODELS: List[ModelSpec] = [
    ModelSpec("llama-3.1-8B", size_gb=16, shards=1),
    ModelSpec("llama-3.3-70B", size_gb=140, shards=8),
    ModelSpec("mixtral-8x7B", size_gb=87, shards=4),
]

def estimate_ttft(
    prompt_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
    is_math: bool = False,
    is_code: bool = False,
) -> float:
    """
    Estimate Time-to-First-Token (TTFT) in ms:
      1. Compute per-token latency as (model_size_GB / (bandwidth_GBps * shards)) * 1000 ms.
      2. Prefill = prompt_tokens * per-token latency.
      3. Decode first token = per-token latency, with task penalties.
    """
    # 1. Effective bandwidth scales linearly with number of GPUs
    effective_bandwidth = bandwidth_gbps * model.shards

    # 2. Base per-token latency (ms)
    base_latency_ms = (model.size_gb / effective_bandwidth) * 1000.0

    # 3. Prefill cost (process entire prompt in parallel, but we use same formula for simplicity)
    prefill_cost = prompt_tokens * base_latency_ms

    # 4. Decode first token cost
    decode_cost = base_latency_ms

    # 5. Apply penalties for task type
    if is_math:
        decode_cost *= 1.5
    if is_code:
        decode_cost *= 2.0

    # 6. Total TTFT
    return prefill_cost + decode_cost

def estimate_full_latency(
    prompt_tokens: int,
    expected_output_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
    is_math: bool = False,
    is_code: bool = False,
) -> float:
    """
    Estimate end-to-end latency (ms) until the last token:
      - Prefill: prompt_tokens × per-token latency
      - Decode: expected_output_tokens × per-token latency with penalties
    expected_output_tokens is a placeholder (e.g., request.max_tokens or average = 50).
    """
    effective_bandwidth = bandwidth_gbps * model.shards
    base_latency_ms = (model.size_gb / effective_bandwidth) * 1000.0

    # Prefill phase
    prefill_cost = prompt_tokens * base_latency_ms

    # Decode phase
    decode_per_token = base_latency_ms
    if is_math:
        decode_per_token *= 1.5
    if is_code:
        decode_per_token *= 2.0
    total_decode_cost = expected_output_tokens * decode_per_token

    return prefill_cost + total_decode_cost

def choose_model_for_ttft_budget(
    prompt_text: str,
    budget_ms: float,
) -> List[Dict]:
    """
    Return models whose estimated TTFT ≤ budget_ms.
    - Tokenizes using llama_token_count.
    - Detects math/code flags.
    - Sorts candidates by ascending TTFT.
    """
    # Token count for the prompt
    prompt_tokens = llama_token_count(prompt_text)

    # Task flags
    math_flag = detect_math(prompt_text)
    code_flag = detect_code(prompt_text)

    # Evaluate each model
    candidates: List[Dict] = []
    for m in AVAILABLE_MODELS:
        ttft = estimate_ttft(
            prompt_tokens=prompt_tokens,
            model=m,
            is_math=math_flag,
            is_code=code_flag,
        )
        if ttft <= budget_ms:
            candidates.append({"model": m.name, "ttft_ms": ttft})

    # Return sorted by fastest TTFT
    return sorted(candidates, key=lambda x: x["ttft_ms"])


Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
