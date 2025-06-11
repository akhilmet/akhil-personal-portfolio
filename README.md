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

Load & Clean

Read the auxiliary_train, validation and test Parquet files.

Drop any rows with missing questions or answers and filter out extreme latency outliers.

Prompt Serialization

For MMLU: concatenate the question and multiple-choice options into a single prompt string.

For GSM8K & HumanEval: use the raw question/prompt fields directly.

Tokenization & Length Features

Use each model’s enterprise tokenizer (LLaMA 3.1 8B, LLaMA 3.3 70B, Mistral 8×7B) to encode prompts and answers.

Compute input_length and output_length as the respective token counts.

Add System Metadata

Insert fixed hardware specs (e.g. GPU memory, batch size) and any decoding hyperparams.

Categorical Encoding

One-hot encode task (MMLU/GSM8K/HumanEval) and model_name.

Drop one level per category to avoid multicollinearity.

Assemble Final DataFrame

Combine all numeric and dummy columns with the measured latency_ms target.

Export the result as X (features) and y (latencies) ready for training.

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
