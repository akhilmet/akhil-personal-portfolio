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
from typing import Optional, List, Dict

# --------------------------------------------------------------------------
# Fallback Latency Estimator Without transformers
#
# Uses approximate tokenization until Llama tokenizer is available.
# Hardware-based naive latency: per-token = model_size_GB / bandwidth_GBps *1000 ms.
# --------------------------------------------------------------------------

def approx_tokenize(text: str) -> list:
    """
    Approximate tokenization: split on words and punctuation.
    Under- or over-estimates real token count.
    """
    return re.findall(r"\w+|[^\w\s]", text)

def approx_token_count(text: str) -> int:
    return len(approx_tokenize(text))

def detect_math(prompt: str) -> bool:
    return bool(re.search(r"[0-9]+|[\+\-\*/=]", prompt))

def detect_code(prompt: str) -> bool:
    keywords = ["def", "function", "return", "{", "}", ";", "import", "class", "=>", "printf", "#include"]
    return any(kw in prompt for kw in keywords)

class ModelSpec:
    def __init__(self, name: str, size_gb: float, fits_on_single_gpu: bool):
        self.name = name
        self.size_gb = size_gb
        self.fits_on_single_gpu = fits_on_single_gpu

AVAILABLE_MODELS: List[ModelSpec] = [
    ModelSpec("llama-3.1-8B", size_gb=16, fits_on_single_gpu=True),
]

def estimate_ttft(
    prompt_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
    is_math: bool = False,
    is_code: bool = False,
) -> Optional[float]:
    if not model.fits_on_single_gpu:
        return None
    base_latency_ms = (model.size_gb / bandwidth_gbps) * 1000.0
    prefill_cost = prompt_tokens * base_latency_ms
    decode_cost = base_latency_ms
    penalty = 1.0
    if is_math:
        penalty *= 1.5
    if is_code:
        penalty *= 2.0
    decode_cost *= penalty
    return prefill_cost + decode_cost

def estimate_full_latency(
    prompt_tokens: int,
    expected_output_tokens: int,
    model: ModelSpec,
    bandwidth_gbps: float = 600.0,
    is_math: bool = False,
    is_code: bool = False,
) -> Optional[float]:
    if not model.fits_on_single_gpu:
        return None
    base_latency_ms = (model.size_gb / bandwidth_gbps) * 1000.0
    prefill_cost = prompt_tokens * base_latency_ms
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
    prompt_tokens = approx_token_count(prompt_text)
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

if __name__ == "__main__":
    prompt = "def add(a, b): return a + b"
    prompt_tokens = approx_token_count(prompt)
    is_math_flag = detect_math(prompt)
    is_code_flag = detect_code(prompt)

    expected_output_tokens = 50
    ttft_budget = 200.0

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

    chosen_spec = next(m for m in AVAILABLE_MODELS if m.name == chosen_name)
    full_latency = estimate_full_latency(
        prompt_tokens=prompt_tokens,
        expected_output_tokens=expected_output_tokens,
        model=chosen_spec,
        is_math=is_math_flag,
        is_code=is_code_flag
    )
    print(f"Estimated full latency for {chosen_spec.name}: {full_latency:.1f} ms")

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
