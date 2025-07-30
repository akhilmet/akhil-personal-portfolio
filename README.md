# Token Predictor Development - Two Slide Presentation

## **Slide 1: Initial Development & Overfitting Challenge**

### Token Estimator Development

**Objective:** Estimate the total tokens a model is likely to consume (input + output) before routing the prompt

**Initial Approach:**
- Used to calculate total cost for querying each LLM
- Essential component for Latency Prediction

**Model Training Results:**
- Initially trained Linear Regression, Random Forest, and XGBoost models to determine the best fit
- Dataset comprised of 4000 NLP queries and 1000 coding question queries
- **Challenge Identified:** Model was overfitting to the limited dataset
- High variance in predictions across different query types
- Poor generalization to unseen query patterns

**Key Learning:** Need for larger, more diverse dataset and different modeling approach to achieve production-ready accuracy

---

## **Slide 2: Enhanced Approach with Magpie Dataset & MLP**

### Token Estimator Enhancement - Precision Over Raw Accuracy

**New Dataset & Architecture:**
- Switched to **Magpie-Llama-3.1-Pro-DPO-100k dataset** (100,000 diverse instruction-response pairs)
- Implemented **PyTorch MLP (Multi-Layer Perceptron)** with 4-layer neural network
- **2 Core Features:** 384-dimensional sentence embeddings + sophisticated complexity score

**Precision vs Accuracy Focus:**
- **Previous Goal:** Lower MAE numbers at any cost → Led to overfitting
- **New Goal:** Consistent, reliable predictions across diverse query types
- **Result:** More precise predictions that generalize well to production scenarios

**Architecture Benefits:**
- **Semantic Understanding:** 384D embeddings capture deep linguistic patterns
- **Production Ready:** No metadata dependencies, query-only approach  
- **Robust Predictions:** Neural network handles varying query complexity effectively
- **Consistent Performance:** Eliminates wild prediction swings (like 724 tokens for "What is Python?")

**Key Achievement:** Transformed from overfitted narrow accuracy to robust precision across diverse real-world queries
---

## Project Milestones

### **Milestone 1: Custom Routing System Development**
We built our own query routing system from scratch rather than using existing solutions like RouteLLM, establishing development infrastructure including GitHub repository setup and access to pre-trained models from Hugging Face. This foundational work enabled rapid prototyping and experimentation with different routing approaches tailored specifically to our requirements.

### **Milestone 2: Token Predictor Development**
We replaced basic word-count heuristics with a machine learning model trained on the Magpie-Llama-3.1-Pro-DPO-100k dataset, achieving ~165 MAE improvement from ~230 MAE baseline. The model was distilled down to just two key features: input tokens and semantic analysis through vector embeddings, providing a clean and efficient prediction approach.

### **Milestone 3: Hardware-Based Latency Estimator**
We implemented a physics-based latency prediction system that utilizes A10 GPU specifications and token estimations to calculate both Time-To-First-Token (TTFT) and full response latency. The system provides accurate timing predictions essential for SLA compliance and user experience optimization.

### **Milestone 4: Privacy Classification & Intent Detection**
We integrated Capital One's DataProfiler tool for automated PII detection and developed intent classification capabilities to categorize queries by type and complexity. These components ensure regulatory compliance while enabling more intelligent routing decisions based on query characteristics.

### **Milestone 5: Multi-Objective Decision Engine**
We designed and implemented a sophisticated decision framework using the formula: min Objective = λ₁Cost + λ₂Latency - λ₃Quality + λ₄Privacy, with tunable weights based on business preferences. The system balances competing objectives while maintaining transparency in routing decisions.

### **Milestone 6: Integration Testing & Documentation**
We completed comprehensive system integration testing, performance validation, and created detailed documentation including deployment procedures and operational monitoring guidelines. The system is production-ready with robust error handling and fallback mechanisms.

---

## Results and Takeaways

### Key Results
- **Token Prediction Accuracy**: Achieved ~165 MAE improvement (from ~230 to ~165 MAE) using ML-based approach with streamlined features: input tokens and semantic analysis through vector embeddings
- **System Integration**: Successfully integrated all components into a cohesive ensemble graph architecture that processes queries through multiple prediction stages
- **Production Readiness**: Delivered a fully functional MVP with comprehensive documentation, error handling, and monitoring capabilities

### Technical Learnings
- **Feature Engineering Impact**: Distilling from 15+ features to just two key features (input tokens and semantic vector embeddings) maintained prediction accuracy while improving model simplicity and efficiency
- **Architecture Benefits**: Ensemble graph design provided modularity and scalability while maintaining clean separation of concerns between prediction components
- **Real-world Complexity**: Balancing multiple objectives (cost, latency, quality, privacy) requires sophisticated decision frameworks and careful weight tuning

### Next Steps for Continued Development
- **Token Prediction Model Enhancement**: Continue improving the token prediction accuracy through additional training data, refined semantic embeddings, and advanced ML techniques
- **Advanced Decision Engines**: Implement contextual bandits and TOPSIS methods for adaptive learning and more sophisticated multi-criteria optimization
- **Performance Optimization**: Add caching mechanisms and similarity search to reduce redundant computations and improve response times
- **Expanded Model Support**: Extend routing capabilities to support additional LLM providers and model variants for increased flexibility and cost optimization opportunities

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
