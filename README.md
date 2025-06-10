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

Model Selection Decisions
Primary Focus Models:

Llama 3.1 8B (70% of training data) - Primary production model, fastest inference
Llama 3.3 70B (20% of training data) - Premium quality model for complex tasks
Llama 3.2 11B (10% of training data) - Balanced middle-tier option

Models Not Selected (Rationale):

Llama 3 70B - Superseded by Llama 3.1 70B which offers better performance and efficiency for same resource cost
Llama 3 8B - Outperformed by Llama 3.1 8B in all benchmarks while requiring identical infrastructure
Mixtral 8x7B - Complex MoE architecture makes latency prediction inconsistent due to dynamic expert routing
Llama 4 Scout - Experimental status creates production risk with unpredictable performance patterns
Llama 3.1 70B - Redundant with Llama 3.3 70B which provides superior capabilities for same resource requirements

Decision Rationale: Focused on the three most strategically important models rather than all 8 to ensure high-quality predictions for production traffic patterns. Llama 3.1 8B handles majority of queries, so prediction accuracy here is critical for system performance.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
