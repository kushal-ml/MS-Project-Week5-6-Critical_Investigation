import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import faiss
import os

# Load PubMedQA dataset
def load_pubmedqa(sample_size=1000):  # Added sample_size parameter
    print("Current working directory:", os.getcwd())
    with open('pubmedqa/data/ori_pqaa.json', 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for key, item in data.items():
        processed_data.append({
            'id': key,
            'QUESTION': item['QUESTION'],
            'CONTEXTS': item['CONTEXTS'],
            'LONG_ANSWER': item['LONG_ANSWER'],
            'final_decision': item['final_decision']
        })
    
    # Convert to DataFrame and sample
    df = pd.DataFrame(processed_data)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Using {len(df)} samples from the dataset")
    return df

class EMFT(torch.nn.Module):
    def __init__(self, base_model):
        super(EMFT, self).__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)
        self.linear1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        
        x = self.layer_norm(embeddings)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        
        return x
    
class PubMedQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256, max_contexts=5):  # Reduced max_length
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_contexts = max_contexts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        question = item['QUESTION']
        contexts = item['CONTEXTS'][:self.max_contexts]  # Limit number of contexts
        answer = item['LONG_ANSWER']
        
        # Tokenize inputs
        question_tokens = self.tokenizer(question, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        context_tokens = [self.tokenizer(context, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt') for context in contexts]
        
        # Pad contexts to max_contexts
        while len(context_tokens) < self.max_contexts:
            context_tokens.append(self.tokenizer('', max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'))
            
        context_ids = torch.stack([ct['input_ids'].squeeze(0) for ct in context_tokens])
        context_mask = torch.stack([ct['attention_mask'].squeeze(0) for ct in context_tokens])
        
        answer_tokens = self.tokenizer(answer, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'question_ids': question_tokens['input_ids'].squeeze(0),
            'question_mask': question_tokens['attention_mask'].squeeze(0),
            'context_ids': context_ids,
            'context_mask': context_mask,
            'answer_ids': answer_tokens['input_ids'].squeeze(0),
            'answer_mask': answer_tokens['attention_mask'].squeeze(0),
        }

def train(model, train_loader, epochs=3, lr=2e-5, accumulation_steps=4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            try:
                question_ids = batch['question_ids'].to(device)
                question_mask = batch['question_mask'].to(device)
                context_ids = batch['context_ids'].to(device)
                context_mask = batch['context_mask'].to(device)
                answer_ids = batch['answer_ids'].to(device)
                answer_mask = batch['answer_mask'].to(device)
                
                batch_size, num_contexts, seq_len = context_ids.shape
                
                with torch.cuda.amp.autocast():  # Use mixed precision
                    q_emb = model(question_ids, question_mask)
                    c_emb = model(context_ids.view(-1, seq_len), context_mask.view(-1, seq_len)).view(batch_size, num_contexts, -1)
                    a_emb = model(answer_ids, answer_mask)
                    
                    # Compute similarity scores
                    pos_scores = torch.sum(q_emb * a_emb, dim=1)
                    neg_scores = torch.max(torch.sum(q_emb.unsqueeze(1) * c_emb, dim=2), dim=1)[0]
                    
                    # Compute loss (contrastive learning)
                    loss = torch.mean(torch.clamp(neg_scores - pos_scores + 0.5, min=0))
                
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps
                
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                
                # Print batch loss
                if i % 10 == 0:  # Print every 10 batches
                    print(f"Batch {i}, Loss: {loss.item()}")
                
            except RuntimeError as e:
                print(f"Error in batch {i}: {str(e)}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")
        
        # Save model after each epoch
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')

def create_index(model, data_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Creating index"):
            context_ids = batch['context_ids'].to(device)
            context_mask = batch['context_mask'].to(device)
            batch_size, num_contexts, seq_len = context_ids.shape
            embeddings = model(context_ids.view(-1, seq_len), context_mask.view(-1, seq_len)).view(batch_size, num_contexts, -1)
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.vstack([emb for batch_emb in all_embeddings for emb in batch_emb])
    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)
    return index

def evaluate(model, test_data, index, k=10):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    correct = 0
    total = 0
    
    for _, item in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
        question = item['QUESTION']
        contexts = item['CONTEXTS']
        
        question_tokens = tokenizer(question, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        question_ids = question_tokens['input_ids'].to(device)
        question_mask = question_tokens['attention_mask'].to(device)
        
        with torch.no_grad():
            q_emb = model(question_ids, question_mask).cpu().numpy()
        
        _, I = index.search(q_emb, k)
        retrieved_contexts = [ctx for idx in I[0] for ctx in test_data.iloc[idx // len(contexts)]['CONTEXTS']]
        
        if any(ctx in contexts for ctx in retrieved_contexts[:k]):
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"Top-{k} Accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
     # Load a sample of the data (e.g., 1000 samples)
    pubmedqa_data = load_pubmedqa(sample_size=1000)
    
    # Split into train and test
    train_data, test_data = train_test_split(pubmedqa_data, test_size=0.2, random_state=42)

    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    base_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = EMFT(base_model)

    # Prepare datasets and dataloaders with smaller batch size
    train_dataset = PubMedQADataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    test_dataset = PubMedQADataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Train the model
    try:
        train(model, train_loader)
        print("Training completed successfully!")
        
        # Save the final model
        torch.save(model.state_dict(), 'model_final.pt')
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")