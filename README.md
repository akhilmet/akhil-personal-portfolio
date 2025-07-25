```
# Cell 1: Import basic libraries
import pandas as pd
import pyarrow
import os
import numpy as np

# Cell 2: Load and combine Magpie dataset
df0 = pd.read_parquet("/home/jovyan/query-routing-systems-datasets/Magpie-Llama-3.1-Pro-DPO-100k-v0.1/data/" + "test-00000-of-00001.parquet")
df1 = pd.read_parquet("/home/jovyan/query-routing-systems-datasets/Magpie-Llama-3.1-Pro-DPO-100k-v0.1/data/" + "train-00000-of-00004.parquet")
df2 = pd.read_parquet("/home/jovyan/query-routing-systems-datasets/Magpie-Llama-3.1-Pro-DPO-100k-v0.1/data/" + "train-00001-of-00004.parquet")
df3 = pd.read_parquet("/home/jovyan/query-routing-systems-datasets/Magpie-Llama-3.1-Pro-DPO-100k-v0.1/data/" + "train-00002-of-00004.parquet")
df4 = pd.read_parquet("/home/jovyan/query-routing-systems-datasets/Magpie-Llama-3.1-Pro-DPO-100k-v0.1/data/" + "train-00003-of-00004.parquet")

# Cell 3: Combine datasets
list_of_dataframes = [df1, df2, df3, df4, df0]
combined_df = pd.concat(list_of_dataframes, ignore_index=True)

# Cell 4: Clean and preview data
combined_df = combined_df.dropna()
combined_df.head()

# Cell 5: Setup tokenizers
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
tokenizer = MistralTokenizer.v3()
def count_tokens(text):
    req = ChatCompletionRequest(messages=[UserMessage(content=text)])
    enc = tokenizer.encode_chat_completion(req)
    return len(enc.tokens)

# Cell 6: Alternative tokenizer
import tiktoken
enc = tiktoken.get_encoding("gpt2")
def count_tokens_gpt(text):
    tokens = enc.encode(text)
    return len(tokens)

# Cell 7: Create input_output_df
input_output_df = combined_df[['instruction', 'responses']].copy()

# Cell 8: Calculate input tokens
input_output_df['input_tokens'] = input_output_df['instruction'].apply(count_tokens_gpt)

# Cell 9: Calculate output tokens
input_output_df['output_tokens'] = input_output_df['responses'].apply(count_tokens_gpt)

# Cell 10: Create working copy
temp = input_output_df.copy()

# Cell 11: Analyze distribution and create balanced token length categories
print("Token distribution analysis:")
print(temp['output_tokens'].describe(percentiles=[.33, .66, .75, .90, .95]))

def categorize_tokens(token_count):
    if token_count < 150:
        return 0  # short
    elif token_count < 450:
        return 1  # medium
    else:
        return 2  # long

temp['token_category'] = temp['output_tokens'].apply(categorize_tokens)
print("\nCategory distribution:")
print(temp['token_category'].value_counts().sort_index())

# Cell 12: Import sentence transformer and ML libraries
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import numpy as np

# Cell 13: Generate sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings...")
embeddings = model.encode(temp['instruction'].tolist(), show_progress_bar=True)
print(f"Generated embeddings shape: {embeddings.shape}")

# Cell 14: Add embeddings to dataframe
num_new_columns = len(embeddings[0])
for i in range(num_new_columns):
    temp[f'embed_{i}'] = embeddings[:, i]

# Cell 15: Neural Network Regression Model
class EmbeddingRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)

# Cell 16: Neural Network Classification Model
class EmbeddingClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.logsoftmax(self.fc4(x))
        return x

# Cell 17: Dataset class for embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, features, targets, task_type='regression'):
        self.features = torch.FloatTensor(features)
        if task_type == 'regression':
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Cell 18: Prepare features and data splits
feature_columns = ['input_tokens'] + [f'embed_{i}' for i in range(384)]
X = temp[feature_columns].values
y_reg = temp['output_tokens'].values
y_cls = temp['token_category'].values

X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

# Cell 19: Create datasets and loaders
train_reg_dataset = EmbeddingDataset(X_train, y_train_reg, 'regression')
test_reg_dataset = EmbeddingDataset(X_test, y_test_reg, 'regression')
train_cls_dataset = EmbeddingDataset(X_train, y_train_cls, 'classification')
test_cls_dataset = EmbeddingDataset(X_test, y_test_cls, 'classification')

train_reg_loader = DataLoader(train_reg_dataset, batch_size=32, shuffle=True)
test_reg_loader = DataLoader(test_reg_dataset, batch_size=32, shuffle=False)
train_cls_loader = DataLoader(train_cls_dataset, batch_size=32, shuffle=True)
test_cls_loader = DataLoader(test_cls_dataset, batch_size=32, shuffle=False)

# Cell 20: Setup device and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = len(feature_columns)

reg_model = EmbeddingRegressionModel(input_dim, 256).to(device)
cls_model = EmbeddingClassificationModel(input_dim, 256, 3).to(device)

# Cell 21: Train regression model
reg_criterion = nn.MSELoss()
reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=0.001)

num_epochs = 10
reg_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for features, targets in train_reg_loader:
        features, targets = features.to(device), targets.to(device)
        
        reg_optimizer.zero_grad()
        outputs = reg_model(features)
        loss = reg_criterion(outputs, targets)
        loss.backward()
        reg_optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_reg_loader)
    print(f"Regression Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Cell 22: Evaluate regression model
reg_model.eval()
reg_predictions = []
reg_actuals = []

with torch.no_grad():
    for features, targets in test_reg_loader:
        features, targets = features.to(device), targets.to(device)
        outputs = reg_model(features)
        
        reg_predictions.extend(outputs.cpu().numpy())
        reg_actuals.extend(targets.cpu().numpy())

reg_predictions = np.array(reg_predictions)
reg_actuals = np.array(reg_actuals)

reg_mse = mean_squared_error(reg_actuals, reg_predictions)
reg_mae = mean_absolute_error(reg_actuals, reg_predictions)

print(f"Embedding Regression Model Performance:")
print(f"Mean Squared Error (MSE): {reg_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(reg_mse):.4f}")
print(f"Mean Absolute Error (MAE): {reg_mae:.4f}")

# Cell 23: Train classification model
cls_criterion = nn.NLLLoss()
cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001)

cls_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for features, targets in train_cls_loader:
        features, targets = features.to(device), targets.to(device)
        
        cls_optimizer.zero_grad()
        outputs = cls_model(features)
        loss = cls_criterion(outputs, targets)
        loss.backward()
        cls_optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_cls_loader)
    print(f"Classification Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Cell 24: Evaluate classification model
cls_model.eval()
cls_predictions = []
cls_actuals = []

with torch.no_grad():
    for features, targets in test_cls_loader:
        features, targets = features.to(device), targets.to(device)
        outputs = cls_model(features)
        predictions = torch.argmax(outputs, dim=-1)
        
        cls_predictions.extend(predictions.cpu().numpy())
        cls_actuals.extend(targets.cpu().numpy())

cls_predictions = np.array(cls_predictions)
cls_actuals = np.array(cls_actuals)

cls_accuracy = accuracy_score(cls_actuals, cls_predictions)
print(f"Embedding Classification Model Performance:")
print(f"Accuracy: {cls_accuracy:.4f}")
print(f"Classification Report:")
print(classification_report(cls_actuals, cls_predictions, target_names=['Short', 'Medium', 'Long']))

# Cell 25: Test both models on sample queries
test_queries = [
    "What is machine learning?",
    "Implement a neural network from scratch using Python and explain each component in detail",
    "How do I sort a list?",
    "Explain the differences between supervised and unsupervised learning, provide examples, and discuss when to use each approach"
]

def predict_tokens_embedding(query, reg_model, cls_model, sentence_model, device):
    query_embedding = sentence_model.encode([query])
    input_tokens = len(enc.encode(query))
    features = np.concatenate([[input_tokens], query_embedding[0]])
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reg_model.eval()
        cls_model.eval()
        
        reg_output = reg_model(features_tensor)
        cls_output = cls_model(features_tensor)
        cls_pred = torch.argmax(cls_output, dim=-1)
        
        categories = ['Short', 'Medium', 'Long']
        
    return reg_output.cpu().item(), categories[cls_pred.cpu().item()]

print("Test Predictions:")
for query in test_queries:
    predicted_tokens, predicted_category = predict_tokens_embedding(query, reg_model, cls_model, model, device)
    print(f"Query: {query[:50]}...")
    print(f"Predicted Tokens: {predicted_tokens:.1f}")
    print(f"Predicted Category: {predicted_category}")
    print("-" * 60)
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
