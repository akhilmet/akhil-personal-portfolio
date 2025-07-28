```
# Cell 1: Save the sentence transformer model
import os
from sentence_transformers import SentenceTransformer

model_save_path = "./model_artifacts/sentence_transformer"
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)

# Cell 2: Save the trained MLP models
torch.save(reg_model.state_dict(), "./model_artifacts/regression_model.pth")
torch.save(cls_model.state_dict(), "./model_artifacts/classification_model.pth")

# Cell 3: Download model to local machine
import shutil
import os

shutil.make_archive('sentence_transformer_model', 'zip', './model_artifacts/')
print("Model zipped as 'sentence_transformer_model.zip'")
print("Download it from the Jupyter file browser")

# Cell 4: Complete TokenEstimatorFeatureCollection class
import re
import string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from sentence_transformers import SentenceTransformer

tokenizer = MistralTokenizer.v3()

class TokenEstimatorFeatureCollection:
    """
    Feature collection class for token predictor.
    Can be instantiated per request or used as singleton.
    """
    
    def get_token_length(self, text):
        """
        Calculates the token length of the input text using Mistral tokenizer.
        
        Args:
            text (str): The input text.
            
        Returns:
            int: The number of tokens.
        """
        try:
            req = ChatCompletionRequest(messages=[UserMessage(content=text)])
            enc = tokenizer.encode_chat_completion(req)
            return len(enc.tokens)
        except Exception as e:
            return max(1, len(text) // 4)
    
    def get_query_embedding(self, query_text):
        """
        Returns the embeddings for a given query text as a numpy array.
        
        Args:
            query_text (str): The input query text.
            
        Returns:
            numpy.ndarray: The embedding vector as an array.
        """
        import os
        
        # Use Kubeflow volume path like Chris showed for Magpie dataset
        base_dir = os.path.join(
            os.getcwd(),
            "query-routing-systems-datasets", 
            "model_artifacts"
        )
        model_path = os.path.join(base_dir, "sentence_transformer")
        
        model = SentenceTransformer(model_path)
        embedding = model.encode([query_text])
        return embedding[0]
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
