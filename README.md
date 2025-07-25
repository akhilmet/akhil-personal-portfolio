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
input_output_df = combined_df[['instruction', 'response']].copy()

# Cell 8: Calculate input tokens
input_output_df['input_tokens'] = input_output_df['instruction'].apply(count_tokens_gpt)

# Cell 9: Calculate output tokens
input_output_df['output_tokens'] = input_output_df['response'].apply(count_tokens_gpt)

# Cell 10: Create token length categories
def categorize_tokens(token_count):
    if token_count < 50:
        return 0  # short
    elif token_count < 300:
        return 1  # medium
    else:
        return 2  # long

input_output_df['token_category'] = input_output_df['output_tokens'].apply(categorize_tokens)

# Cell 11: Import BERT and ML libraries
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, AutoConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import numpy as np

# Cell 12: BERT Regression Model
class BertRegressionModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output

# Cell 13: BERT Classification Model
class BertClassificationModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim, num_classes):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.logsoftmax(self.fc2(output))
        return output

# Cell 14: Dataset class for regression
class TokenRegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=512):
        self.texts = texts.reset_index(drop=True)
        self.targets = targets.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        target = self.targets.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.float)
        }

# Cell 15: Dataset class for classification
class TokenClassificationDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=512):
        self.texts = texts.reset_index(drop=True)
        self.targets = targets.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        target = self.targets.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.long)
        }

# Cell 16: Setup BERT components
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
config = AutoConfig.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cell 17: Prepare data splits
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    input_output_df['instruction'], 
    input_output_df['output_tokens'], 
    input_output_df['token_category'], 
    test_size=0.2, 
    random_state=42
)

# Cell 18: Create regression datasets and loaders
train_reg_dataset = TokenRegressionDataset(X_train, y_train_reg, bert_tokenizer)
test_reg_dataset = TokenRegressionDataset(X_test, y_test_reg, bert_tokenizer)

train_reg_loader = DataLoader(train_reg_dataset, batch_size=16, shuffle=True)
test_reg_loader = DataLoader(test_reg_dataset, batch_size=16, shuffle=False)

# Cell 19: Train regression model
reg_model = BertRegressionModel(config, 'bert-base-uncased', hidden_dim=128).to(device)
reg_criterion = nn.MSELoss()
reg_optimizer = torch.optim.AdamW(params=reg_model.parameters(), lr=1e-4)

num_epochs = 3
reg_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_reg_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        reg_optimizer.zero_grad()
        outputs = reg_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = reg_criterion(outputs, targets)
        loss.backward()
        reg_optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_reg_loader)
    print(f"Regression Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Cell 20: Evaluate regression model
reg_model.eval()
reg_predictions = []
reg_actuals = []

with torch.no_grad():
    for batch in test_reg_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        outputs = reg_model(input_ids=input_ids, attention_mask=attention_mask)
        
        reg_predictions.extend(outputs.cpu().numpy())
        reg_actuals.extend(targets.cpu().numpy())

reg_predictions = np.array(reg_predictions)
reg_actuals = np.array(reg_actuals)

reg_mse = mean_squared_error(reg_actuals, reg_predictions)
reg_mae = mean_absolute_error(reg_actuals, reg_predictions)

print(f"BERT Regression Model Performance:")
print(f"Mean Squared Error (MSE): {reg_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(reg_mse):.4f}")
print(f"Mean Absolute Error (MAE): {reg_mae:.4f}")

# Cell 21: Create classification datasets and loaders
train_cls_dataset = TokenClassificationDataset(X_train, y_train_cls, bert_tokenizer)
test_cls_dataset = TokenClassificationDataset(X_test, y_test_cls, bert_tokenizer)

train_cls_loader = DataLoader(train_cls_dataset, batch_size=16, shuffle=True)
test_cls_loader = DataLoader(test_cls_dataset, batch_size=16, shuffle=False)

# Cell 22: Train classification model
cls_model = BertClassificationModel(config, 'bert-base-uncased', hidden_dim=128, num_classes=3).to(device)
cls_criterion = nn.NLLLoss()
cls_optimizer = torch.optim.AdamW(params=cls_model.parameters(), lr=1e-4)

cls_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_cls_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        cls_optimizer.zero_grad()
        outputs = cls_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = cls_criterion(outputs, targets)
        loss.backward()
        cls_optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_cls_loader)
    print(f"Classification Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Cell 23: Evaluate classification model
cls_model.eval()
cls_predictions = []
cls_actuals = []

with torch.no_grad():
    for batch in test_cls_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        outputs = cls_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs, dim=-1)
        
        cls_predictions.extend(predictions.cpu().numpy())
        cls_actuals.extend(targets.cpu().numpy())

cls_predictions = np.array(cls_predictions)
cls_actuals = np.array(cls_actuals)

cls_accuracy = accuracy_score(cls_actuals, cls_predictions)
print(f"BERT Classification Model Performance:")
print(f"Accuracy: {cls_accuracy:.4f}")
print(f"Classification Report:")
print(classification_report(cls_actuals, cls_predictions, target_names=['Short', 'Medium', 'Long']))

# Cell 24: Test both models on sample queries
test_queries = [
    "What is machine learning?",
    "Implement a neural network from scratch using Python and explain each component in detail",
    "How do I sort a list?",
    "Explain the differences between supervised and unsupervised learning, provide examples, and discuss when to use each approach"
]

def predict_tokens(query, reg_model, cls_model, tokenizer, device):
    encoding = tokenizer(
        query,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        reg_model.eval()
        cls_model.eval()
        
        reg_output = reg_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = cls_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_pred = torch.argmax(cls_output, dim=-1)
        
        categories = ['Short', 'Medium', 'Long']
        
    return reg_output.cpu().item(), categories[cls_pred.cpu().item()]

print("Test Predictions:")
for query in test_queries:
    predicted_tokens, predicted_category = predict_tokens(query, reg_model, cls_model, bert_tokenizer, device)
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
