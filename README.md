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

Core Features (The Heavy Hitters)

Token Count: This is gonna be our MVP feature.

Core Latency Calculation
Theoretical Foundation:
Latency per token = Model Size (GB) / GPU Memory Bandwidth (GB/s)
For our A10 GPU setup (600 GB/s bandwidth):

Llama 3.1 8B (16GB): ~27ms/token theoretical minimum
Llama 3.3 70B (140GB): Cannot run on single A10 (exceeds 24GB VRAM)
Mixtral 8x7B (87GB): Cannot run on single A10

Key Insight: Modern LLM inference is memory-bandwidth bound, not compute-bound. The A10's 125 TFLOPS processing power is bottlenecked by 600 GB/s memory throughput.
Token Processing Breakdown:

Input tokens: 0.17ms/token (prefill - parallel processing, compute-bound)
Output tokens: 27-35ms/token (decode - sequential generation, memory-bound)
Why this matters: A 100-token input + 50-token output = 17ms + 1,750ms = ~1.77 seconds total
Model Type: Hardware compatibility flags since only 8B models fit on single A10
Task Type Flags: Binary flags for math and code tasks. Math gets a 1.5x penalty based on (Hendrycks et al., 2021) showing math reasoning needs extra compute cycles. Code gets hit with 2.0x because syntax validation and logic checking is expensive (Chen et al., 2021).

Supporting Features
Complexity Counters: Count of math operators, code keywords, question words - basically quantifying how complex the query is beyond just "has math" or "has code".

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
