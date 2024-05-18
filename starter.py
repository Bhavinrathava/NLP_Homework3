from transformers import AutoTokenizer, BertModel
import torch.optim as optim
import torch.nn as nn
import torch
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class MCQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        encoded_inputs = self.tokenizer(instance[0], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        label = torch.tensor(instance[1])
        return encoded_inputs['input_ids'].squeeze(0), encoded_inputs['attention_mask'].squeeze(0), label

class BertForMCQA(nn.Module):
    def __init__(self, bert_model):
        super(BertForMCQA, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
        logits = self.linear(cls_output)
        return logits

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), desc=f"Training") as pbar:
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f"Evaluating") as pbar:
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def main():
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    test = []
    valid = []
    
    file_name = 'train_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            label = 1 if j == ans else 0
            train.append([text, label])
                
    file_name = 'dev_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            label = 1 if j == ans else 0
            valid.append([text, label])
        
    file_name = 'test_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            label = 1 if j == ans else 0
            test.append([text, label])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    model = BertForMCQA(bert_model)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    # Load data into DataLoader
    max_length = 128
    batch_size = 16

    train_dataset = MCQADataset(train, tokenizer, max_length)
    valid_dataset = MCQADataset(valid, tokenizer, max_length)
    test_dataset = MCQADataset(test, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 3
    for epoch in range(epochs):
        train_loss = train_model(model, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_accuracy = evaluate_model(model, valid_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

    # Test the model
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
