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

# **EDA Analysis for Latency Predictor Feature Engineering**

## **Dataset Overview**

So basically I'm using three datasets to train this latency predictor: MMLU (academic stuff), GSM8K (math word problems), and HumanEval (coding challenges). These datasets are pretty solid representations of different types of workloads that hit LLMs - you've got your basic knowledge retrieval, mathematical reasoning, and code generation.

**MMLU Schema Reference:**
```json
{
  "question": "string",
  "subject": "string", 
  "choices": ["string", "string", "string", "string"],
  "answer": "A|B|C|D"
}
```

**GSM8K Schema Reference:**
```json
{
  "question": "string",
  "answer": "string"
}
```

**HumanEval Schema Reference:**
```json
{
  "task_id": "string",
  "prompt": "string",
  "canonical_solution": "string",
  "test": "string",
  "entry_point": "string"
}
```

## **Target Models & Theoretical Latency**

We're working with three models that have pretty different performance characteristics:

**Llama 3.1 8B**: Our baseline speed demon, averaging 20-35ms/token. This thing's fast but obviously not gonna nail the super complex stuff.

**Llama 3.3 70B**: The heavy hitter, clocking in at 55-95ms/token. Way slower but handles complex reasoning much better.

**Mixtral 8x7B**: The wildcard at around 40-65ms/token. MoE architecture makes it weird to predict since it's not just about parameter count.

Using the 0.24ms/token baseline from transformer research [(Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198), the theoretical latency breakdown looks like:
- **Llama 3.1 8B**: 1.0x factor (baseline)
- **Llama 3.3 70B**: ~3.5x factor (way more parameters = way more compute)
- **Mixtral 8x7B**: ~2.1x factor (MoE efficiency but still bigger than 8B)

## **Key Findings**

### **Text Patterns Are Pretty Predictable**
MMLU queries are usually short and sweet (60-90 chars), GSM8K gets wordier with all the setup (120-180 chars), and HumanEval is verbose AF with all the function specs (180-250 chars). This tracks with research showing input length is basically the main driver of inference time [(Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361).

### **Task Complexity Actually Matters**
Different datasets hit the models differently:
- **MMLU**: Mostly just memory recall, pretty lightweight
- **GSM8K**: Multi-step reasoning that makes the model work harder
- **HumanEval**: Creative generation that really taxes the attention mechanisms

Research backs this up - reasoning tasks light up way more attention heads [(Clark et al., 2019)](https://arxiv.org/abs/1906.04341).

## **Feature Engineering Game Plan**

### **Core Features (The Heavy Hitters)**
**Token Count**: This is gonna be our MVP feature. The 0.24ms/token baseline is solid and token count correlates like crazy with latency.

**Task Type Flags**: Binary flags for math and code tasks. Math gets a 1.5x penalty based on [(Hendrycks et al., 2021)](https://arxiv.org/abs/2103.03874) showing math reasoning needs extra compute cycles. Code gets hit with 2.0x because syntax validation and logic checking is expensive [(Chen et al., 2021)](https://arxiv.org/abs/2107.03374).

**Dataset Indicators**: One-hot encoding for MMLU/GSM8K/HumanEval since each has distinct processing patterns.

### **Supporting Features**
**Complexity Counters**: Count of math operators, code keywords, question words - basically quantifying how gnarly the query is beyond just "has math" or "has code".

**Model-Specific Stuff**: Tokens per parameter ratios since different model sizes handle the same input totally differently.

### **The Math Behind the Madness**

The complexity multipliers aren't just random numbers:
- **1.5x for math**: Research shows mathematical reasoning requires additional forward passes in transformers
- **2.0x for code**: Programming tasks need syntax validation + logical consistency checks
- **Model scaling factors**: The 3.5x for 70B aligns with scaling laws [(Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)

For Mixtral, the 2.1x factor accounts for MoE efficiency - it's not as bad as a dense 56B model but definitely slower than the 8B.

## **Expected Performance**

Based on this analysis, token count + complexity flags + dataset type should get us like 90% of the way there for accurate predictions. The remaining edge cases can probably be handled with interaction features.

Targeting MAPE < 15% which is pretty standard for production ML systems. Cross-dataset validation should ensure we're not just memorizing patterns from one specific task type.

This setup should give us a solid foundation for training Random Forest models that can actually predict latency accurately across all three models while keeping inference time fast enough for real-time routing.

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
