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

from transformers import AutoTokenizer
import re
from typing import Optional, List, Dict

# ------------------------------------------------------------------------------
# Latency Estimator Using Llama 3.1 Tokenizer and Naive Hardware-Based Latency
#
# This module provides functions to:
#  - Count tokens in prompts using the Llama 3.1 tokenizer.
#  - Detect simple math/code flags in prompts.
#  - Estimate Time-to-First-Token (TTFT) and full latency based purely on hardware bandwidth:
#      per-token latency = model_size_GB / bandwidth_GBps * 1000 ms
#  - Route a prompt to models based on a TTFT budget.
#
# Naive approach:
#  - We assume both prefill and decode phases share the same per-token latency computed
#    from GPU memory bandwidth (e.g., 600 GB/s for A10). In reality, prefill is compute-bound
#    and can be faster; here we use the same memory-bound formula for simplicity.
#  - No overhead multiplier is applied; this gives a theoretical lower bound based on bandwidth.
#  - Task penalties (math ×1.5, code ×2.0) are applied to decode cost to reflect extra compute patterns.
#
# Requirements:
#  - transformers library with Llama 3.1 tokenizer available.
#  - Adjust model identifier if needed.
# ------------------------------------------------------------------------------

# 1. Load Llama 3.1 tokenizer for accurate token counts.
#    Replace "meta-llama/Llama-3.1" with the correct model ID or local path as needed.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1")

def llama_token_count(text: str) -> int:
    """
    Count tokens in `text` using the Llama 3.1 tokenizer.
    Returns the number of subword tokens the model will see for the prompt.
    """
    # Disable special tokens so we count only the prompt
    encoded = tokenizer(text, return_length=True, add_special_tokens=False)
    if isinstance(encoded, dict) and "length" in encoded:
        return int(encoded["length"])
    # Fallback if return_length not supported:
    return len(encoded["input_ids"])

def detect_math(prompt: str) -> bool:
    """
    Simple math detection: checks for digits or basic operators.
    If prompt contains arithmetic patterns, return True.
    """
    return bool(re.search(r"[0-9]+|[\+\-\*/=]", prompt))

def detect_code(prompt: str) -> bool:
    """
    Simple code detection: checks for common code keywords or syntax.
    If prompt looks like code, return True.
    """
    keywords = ["def", "function", "return", "{", "}", ";", "import", "class", "=>", "printf", "#include"]
    return any(kw in prompt for kw in keywords)

# 2. Model specification (single GPU only)
class ModelSpec:
    def __init__(self, name: str, size_gb: float, fits_on_single_gpu: bool):
        """
        name: model identifier (e.g., "llama-3.1-8B")
        size_gb: model size in GB (approximate, e.g., 16 for 8B)
        fits_on_single_gpu: True if it can run on a single A10 without sharding
        """
        self.name = name
        self.size_gb = size_gb
        self.fits_on_single_gpu = fits_on_single_gpu

# Define available models that fit on a single A10
AVAILABLE_MODELS: List[ModelSpec] = [
    ModelSpec("llama-3.1-8B", size_gb=16, fits_on_single_gpu=True),
    # Add more single-GPU-compatible models here if available
]

def estimate_ttft(
    prompt_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
    is_math: bool = False,
    is_code: bool = False,
) -> Optional[float]:
    """
    Estimate Time-to-First-Token (TTFT) in milliseconds for a single A10 run.
    Naive hardware-based latency: per-token latency = model_size_GB / bandwidth_GBps * 1000 ms.
    - prompt_tokens: number of input tokens.
    - model: ModelSpec instance.
    - bandwidth_gbps: GPU memory bandwidth in GB/s (default 600 for A10).
    - is_math/is_code: apply penalty on decode cost.
    Returns:
      - TTFT in ms if model fits on single GPU, else None.
    """
    if not model.fits_on_single_gpu:
        return None

    # Compute per-token latency from hardware bandwidth (naive lower bound):
    # model_size_GB / bandwidth_GBps gives seconds per token; multiply by 1000 to get ms/token.
    base_latency_ms = (model.size_gb / bandwidth_gbps) * 1000.0

    # Prefill cost: treat like memory-bound as well (naive).
    prefill_cost = prompt_tokens * base_latency_ms  # ms for prompt processing

    # Decode cost for first token: same base latency
    decode_cost = base_latency_ms  # ms for first generated token

    # Apply task penalties on decode phase:
    penalty = 1.0
    if is_math:
        penalty *= 1.5
    if is_code:
        penalty *= 2.0
    decode_cost *= penalty

    # Total TTFT: prefill + first-token decode
    return prefill_cost + decode_cost

def estimate_full_latency(
    prompt_tokens: int,
    expected_output_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
    is_math: bool = False,
    is_code: bool = False,
) -> Optional[float]:
    """
    Estimate total latency (ms) until last token is generated for a single A10 run.
    Uses the same per-token latency for prefill and decode (naive).
    - prompt_tokens: number of input tokens.
    - expected_output_tokens: heuristic estimate (e.g., API max_tokens or average).
    - model: ModelSpec instance.
    Returns total latency in ms, or None if model doesn't fit.
    """
    if not model.fits_on_single_gpu:
        return None

    base_latency_ms = (model.size_gb / bandwidth_gbps) * 1000.0

    # Prefill phase: process all prompt tokens
    prefill_cost = prompt_tokens * base_latency_ms

    # Decode phase: process expected output tokens
    decode_per_token = base_latency_ms
    penalty = 1.0
    if is_math:
        penalty *= 1.5
    if is_code:
        penalty *= 2.0
    decode_per_token *= penalty

    total_decode_cost = expected_output_tokens * decode_per_token

    return prefill_cost + total_decode_cost

def choose_model_for_ttft_budget(
    prompt_text: str,
    budget_ms: float,
    models: List[ModelSpec],
) -> List[Dict]:
    """
    Select models whose estimated TTFT <= budget_ms.
    - Tokenize prompt_text using llama_token_count for accurate prompt_tokens.
    - For each single-GPU-compatible model, estimate TTFT with naive hardware-based latency.
    - Return list of {'model': name, 'ttft_ms': value}, sorted by ttft_ms ascending.
    """
    prompt_tokens = llama_token_count(prompt_text)
    is_math = detect_math(prompt_text)
    is_code = detect_code(prompt_text)

    candidates = []
    for m in models:
        ttft = estimate_ttft(
            prompt_tokens=prompt_tokens,
            model=m,
            is_math=is_math,
            is_code=is_code
        )
        if ttft is not None and ttft <= budget_ms:
            candidates.append({"model": m.name, "ttft_ms": ttft})
    candidates.sort(key=lambda x: x["ttft_ms"])
    return candidates

# Example usage
if __name__ == "__main__":
    prompt = "def add(a, b): return a + b"
    prompt_tokens = llama_token_count(prompt)
    is_math_flag = detect_math(prompt)
    is_code_flag = detect_code(prompt)

    # Heuristic for expected output tokens (e.g., API max_tokens or average)
    expected_output_tokens = 50

    # TTFT budget in ms
    ttft_budget = 200.0

    # Find models that meet TTFT budget
    candidates = choose_model_for_ttft_budget(
        prompt_text=prompt,
        budget_ms=ttft_budget,
        models=AVAILABLE_MODELS
    )

    if candidates:
        print("Models meeting TTFT budget:")
        for c in candidates:
            print(f"  {c['model']}: estimated TTFT {c['ttft_ms']:.1f} ms")
        chosen_name = candidates[0]["model"]
    else:
        print("No model meets the TTFT budget; defaulting to fastest available model.")
        chosen_name = AVAILABLE_MODELS[0].name

    # Retrieve ModelSpec for chosen model
    chosen_spec = next(m for m in AVAILABLE_MODELS if m.name == chosen_name)

    # Estimate full latency for chosen model
    full_latency = estimate_full_latency(
        prompt_tokens=prompt_tokens,
        expected_output_tokens=expected_output_tokens,
        model=chosen_spec,
        is_math=is_math_flag,
        is_code=is_code_flag
    )
    print(f"Estimated full latency for {chosen_spec.name} with {expected_output_tokens} output tokens: {full_latency:.1f} ms")


Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
