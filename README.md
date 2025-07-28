```
# test_basic_functions.py
from token_estimator_feature_collection import TokenEstimatorFeatureCollection

def test_basic_functions():
    # Initialize the class
    feature_collector = TokenEstimatorFeatureCollection()
    
    # Test query
    test_query = "What is machine learning and how does it work?"
    
    print("Testing Basic Token Estimator Functions")
    print("=" * 50)
    print(f"Query: '{test_query}'")
    print("-" * 50)
    
    try:
        # Test 1: Input tokens
        print("Testing get_token_length()...")
        input_tokens = feature_collector.get_token_length(test_query)
        print(f"✅ Input tokens: {input_tokens}")
        
        # Test 2: Sentence embeddings  
        print("\nTesting get_query_embedding()...")
        embedding = feature_collector.get_query_embedding(test_query)
        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Sample values: {embedding[:5]}")
        
        print(f"\n🎉 Both functions work! Ready for Chris's MLP integration.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functions()


```
Title: feat|test|chore|bug|doc: Add sentence transformer embeddings to token predictor
Description
Context
Updated token predictor to use MiniLM sentence embeddings for better semantic understanding of queries, replacing basic feature engineering approach.
Updates

Added get_query_embedding() method to extract 384-dimensional sentence embeddings
Updated get_token_length() to use Mistral tokenizer for consistency
Configured model path to use Kubeflow volume storage (query-routing-systems-datasets/model_artifacts/)
```

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
