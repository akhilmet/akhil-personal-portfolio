```

# Example usage and testing
if __name__ == "__main__":
    # Test the token predictor
    print("🧪 Testing TokenPredictorNode...")
    
    try:
        # Initialize predictor
        predictor = TokenPredictorNode()
        
        # Test queries
        test_queries = [
            "What is Python?",
            "How do I debug Python code that's throwing an error?",
            "Write a function to sort a list of numbers in ascending order.",
            "Explain the concept of machine learning in simple terms.",
            "What are the main differences between SQL and NoSQL databases?"
        ]
        
        print(f"\n🔮 Testing predictions:")
        for query in test_queries:
            tokens = predictor.predict(query)
            print(f"   '{query[:50]}...' → {tokens} tokens")
        
        # Test batch prediction
        batch_predictions = predictor.predict_batch(test_queries)
        print(f"\n📊 Batch prediction: {batch_predictions}")
        
        # Show model info
        model_info = predictor.get_model_info()
        print(f"\n📋 Model Info:")
        print(f"   Status: {model_info['status']}")
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Features: {model_info['n_features']}")
        
        print("\n✅ TokenPredictorNode test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("🔧 Make sure you've trained the model using the retraining notebook first!")
```
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

### **GitHub README Ready:**
This format is optimized for copying into GitHub README.md and then easily transferring to any Jupyter environment. Each cell is clearly separated and ready for enterprise use.

## License

This project is licensed under the MIT License.
---

Thank you for visiting my portfolio! I hope you find my work and experiences interesting. If you have any questions or just want to say hi, don't hesitate to contact me!
