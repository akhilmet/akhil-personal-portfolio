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

Below is the **end-to-end design** and then a full Jupyter-notebook–style code listing. First, **here’s exactly what happens at each step**, then you can copy & paste the cells one by one.

---

## 🔍 1. End-to-end Workflow

1. **Data Ingestion**

   * Load the Parquet splits for MMLU (`auxiliary_train`, `validation`, `test`).
   * Load GSM8K and HumanEval (via Hugging Face Datasets or local JSONL) into pandas.

2. **Data Cleaning**

   * Drop rows missing prompts or answers.
   * (If you have real latency logs) filter out extreme outliers.

3. **Prompt Serialization**

   * **MMLU**: join `question` + all multiple-choice `choices` into one string.
   * **GSM8K/HumanEval**: use `question` or `prompt` directly.

4. **Tokenizer Setup**

   * Instantiate each model’s enterprise tokenizer (LLaMA 3.1 8B, LLaMA 3.3 70B, Mistral 8×7B) with `trust_remote_code=True, use_fast=True`.

5. **Feature Engineering** (per sample, per model)

   * **Raw counts**

     * `input_length`  = number of tokens in prompt
     * `output_length` = number of tokens in expected answer (or your capped max)
   * **Task flags**

     * `is_math` = 1 for GSM8K, else 0
     * `is_code` = 1 for HumanEval, else 0
     * **penalty\_factor** = 1.5 if `is_math`, 2.0 if `is_code`, else 1.0
     * `weighted_input_length` = `input_length * penalty_factor`
   * **Complexity counters**

     * `math_op_count`  = count of `+ - * / = ^` in prompt
     * `code_kw_count`  = count of Python keywords (`def`, `class`, `import`, `for`, `while`, `if`, `else`, `return`) in prompt
   * **Model specifics**

     * `param_count`        = 8e9, 70e9, or 8e9
     * `tokens_per_param`   = `input_length / param_count`

6. **Assemble & Persist**

   * Concatenate all features + measured `latency_ms` into one big DataFrame.
   * Save to disk (`processed_train.parquet`, etc.) for reproducibility.

7. **Model Training**

   * Split into train/validation/test.
   * Build a sklearn `Pipeline`:

     * `StandardScaler` on numeric features
     * `XGBRegressor`
   * Fit on training data, tune on validation, evaluate on test (MAE/R²).

8. **Deployment**

   * `joblib.dump(pipeline, "latency_predictor.joblib")`
   * In your query router:

     1. Extract the same features from each incoming prompt + metadata
     2. `pipeline.predict(…)` → estimated latency
     3. Route to the model/hardware combo that minimizes latency / cost

---

## 📓 2. Full Jupyter Notebook Code

```python
# Cell 1: Install Dependencies
!pip install transformers datasets scikit-learn xgboost pandas tqdm joblib --quiet
```

```python
# Cell 2: Imports
import os, json, re
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
from tqdm import tqdm
```

```python
# Cell 3: Load Parquet (MMLU) + HF Datasets (GSM8K, HumanEval)
df_mmlu      = pd.read_parquet("auxiliary_train-00000-of-00001.parquet")
ds_gsm8k     = load_dataset("gsm8k", "main", split="train")
df_gsm8k     = pd.DataFrame(ds_gsm8k)
ds_humaneval = load_dataset("openai_humaneval", split="test")
df_humaneval = pd.DataFrame(ds_humaneval)
```

```python
# Cell 4: Clean
for df in (df_mmlu, df_gsm8k, df_humaneval):
    df.dropna(subset=["question","answer"] if "answer" in df.columns else ["prompt"], inplace=True)
```

```python
# Cell 5: Prompt Serialization
# MMLU: question + choices
def mmlu_prompt(row):
    opts = row["choices"]  # list like ["A. …", "B. …", …]
    return row["question"] + "\n" + "\n".join(opts)

df_mmlu["prompt"] = df_mmlu.apply(mmlu_prompt, axis=1)
# GSM8K & HumanEval already have 'question' or 'prompt'
df_gsm8k["prompt"]     = df_gsm8k["question"]
df_humaneval["prompt"] = df_humaneval["prompt"]
```

```python
# Cell 6: Tokenizer & Model Metadata
models = {
  "LLaMA-3.1-8B": { "tokenizer":"meta-llama/Llama-3-8b-hf",  "params": 8e9 },
  "LLaMA-3.3-70B":{ "tokenizer":"meta-llama/Llama-3-70b-hf", "params":70e9 },
  "Mistral-8x7B":{ "tokenizer":"mistralai/mistral-8x7b",   "params": 8e9 },
}
tokenizers = {
  name: AutoTokenizer.from_pretrained(cfg["tokenizer"], trust_remote_code=True, use_fast=True)
  for name,cfg in models.items()
}
```

```python
# Cell 7: Feature Engineering Functions
# regexes for counting
MATH_OPS   = re.compile(r"[+\-*/^=]")
CODE_KWDS  = re.compile(r"\b(def|class|import|for|while|if|else|return)\b")

def featurize(df, task_name, model_name, latency_fn):
    tok       = tokenizers[model_name]
    p         = models[model_name]["params"]
    records   = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{task_name}@{model_name}"):
        prmpt = row["prompt"]
        out   = row.get("answer","")
        inp_len  = len(tok.encode(prmpt))
        out_len  = len(tok.encode(out))
        is_math  = 1 if task_name=="GSM8K"     else 0
        is_code  = 1 if task_name=="HumanEval" else 0
        penalty  = 1.5 if is_math else (2.0 if is_code else 1.0)
        win_len  = inp_len * penalty
        mop_cnt  = len(MATH_OPS.findall(prmpt))
        ckw_cnt  = len(CODE_KWDS.findall(prmpt))
        tpp      = inp_len / p

        records.append({
          "input_length":         inp_len,
          "output_length":        out_len,
          "is_math":              is_math,
          "is_code":              is_code,
          "penalty_factor":       penalty,
          "weighted_input_len":   win_len,
          "math_op_count":        mop_cnt,
          "code_kw_count":        ckw_cnt,
          "param_count":          p,
          "tokens_per_param":     tpp,
          # measured or simulated latency (ms)
          "latency_ms":           latency_fn(inp_len, out_len, model_name)
        })
    return pd.DataFrame.from_records(records)
```

```python
# Cell 8: Simulate (or replace with real timing) & Build Full FeatureSet
def simulate_latency(i,o,m): 
    base = 5*i + 10*o
    mult = 1.5 if m=="GSM8K" else (2.0 if m=="HumanEval" else 1.0)
    return base * mult + np.random.normal(0,20)

parts = []
for model_name in models:
    parts.append( featurize(df_mmlu,      "MMLU",      model_name, simulate_latency) )
    parts.append( featurize(df_gsm8k,     "GSM8K",     model_name, simulate_latency) )
    parts.append( featurize(df_humaneval, "HumanEval", model_name, simulate_latency) )

df_all = pd.concat(parts, ignore_index=True)
df_all.to_parquet("processed_latency_data.parquet", index=False)
```

```python
# Cell 9: Train/Test Split
X = df_all.drop(columns=["latency_ms"])
y = df_all["latency_ms"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

```python
# Cell 10: Build & Fit Pipeline
numeric_feats = list(X.columns)
pipe = Pipeline([
  ("scaler", StandardScaler()), 
  ("xgb",    XGBRegressor(tree_method="hist", enable_categorical=False))
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, "latency_predictor.joblib")
```

```python
# Cell 11: Evaluation
preds = pipe.predict(X_test)
from sklearn.metrics import mean_absolute_error, r2_score
print("MAE:", mean_absolute_error(y_test, preds))
print("R² :", r2_score(y_test, preds))
```

```python
# Cell 12: Router Helper Function
import joblib
router = joblib.load("latency_predictor.joblib")

def predict_latency(prompt, task, model_name, max_out=50):
    tok = tokenizers[model_name]
    i   = len(tok.encode(prompt))
    o   = max_out
    im  = 1 if task=="GSM8K" else 0
    ic  = 1 if task=="HumanEval" else 0
    pen = 1.5 if im else (2.0 if ic else 1.0)
    win = i * pen
    mop = len(MATH_OPS.findall(prompt))
    ckw = len(CODE_KWDS.findall(prompt))
    p   = models[model_name]["params"]
    tpp = i / p

    feat = pd.DataFrame([{
      "input_length":i, "output_length":o,
      "is_math":im, "is_code":ic,
      "penalty_factor":pen,
      "weighted_input_len":win,
      "math_op_count":mop,
      "code_kw_count":ckw,
      "param_count":p,
      "tokens_per_param":tpp
    }])
    return router.predict(feat)[0]
```

---

🎉 That’s the **complete** pipeline—from raw Parquet/JSONL through feature engineering (token counts, task-type flags with penalties, complexity counters, model-param ratios), all the way to XGBoost training and a live inference helper you can drop into your query router.

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
