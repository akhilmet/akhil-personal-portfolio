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
## **Dataset Schemas**

### **MMLU (Academic Questions)**
**Schema:**
```json
{
  "question": "string",
  "subject": "string", 
  "choices": ["string", "string", "string", "string"],
  "answer": "A|B|C|D"
}
```

**Sample:**
```json
{
  "question": "What is the primary function of mitochondria?",
  "subject": "high_school_biology",
  "choices": ["A: Protein synthesis", "B: Energy production", "C: DNA replication", "D: Waste removal"],
  "answer": "B"
}
```

### **GSM8K (Math Problems)**
**Schema:**
```json
{
  "question": "string",
  "answer": "string"
}
```

**Sample:**
```json
{
  "question": "Sarah has 24 apples. She gives 8 away and buys 15 more. How many apples does she have total?",
  "answer": "Sarah gives away 8: 24-8=16. Then buys 15 more: 16+15=31. #### 31"
}
```

### **HumanEval (Code Generation)**
**Schema:**
```json
{
  "task_id": "string",
  "prompt": "string",
  "canonical_solution": "string",
  "test": "string",
  "entry_point": "string"
}
```

**Sample:**
```json
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if any two numbers are closer than threshold \"\"\"\n",
  "canonical_solution": "    for i, x in enumerate(numbers):\n        for j, y in enumerate(numbers):\n            if i != j and abs(x-y) < threshold:\n                return True\n    return False",
  "test": "def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) == False",
  "entry_point": "has_close_elements"
}
```
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
