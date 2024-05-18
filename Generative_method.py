import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

class MCADataSet(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]

        question = instance['question']['stem']
        choices = instance['question']['choices']
        fact = instance['fact1']
        answer_key = instance['answerKey']

        prompt = f'[START] {fact} [SEP] {question} [SEP] [A] {choices[0]["text"]} [SEP] [B] {choices[1]["text"]} [SEP] [C] {choices[2]["text"]} [SEP] [D] {choices[3]["text"]} [ANSWER]'

        encoded_inputs = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        label_tensor = torch.tensor(self.tokenizer.encode(answer_key, add_special_tokens=False, return_tensors='pt').squeeze(), dtype=torch.long)

        return encoded_inputs['input_ids'].squeeze(0), encoded_inputs['attention_mask'].squeeze(0), label_tensor
    
class BertForMCQA(nn.Module):
    def __init__(self, bert_model, take_last_token=True):
        super(BertForMCQA, self).__init__()
        self.bert = bert_model
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

def read_data(file_path):
    '''
    Read Data from JSONL file and return as list
    '''

    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def train_model(model, dataloader, optimizer, device, epochs, criterion):
    
    model.train()
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        with tqdm(total=len(dataloader), desc=f"Training Epoch {epoch+1}/{epochs}") as pbar:
            for batch in dataloader:

                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                #print(outputs.logits.shape, labels.shape)
                # Take the first token as the prediction
                logits = outputs.logits[:, -1, :]
                loss = criterion(logits, labels)
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
                train_loss += loss.item()

        train_losses.append(train_loss / len(dataloader))
        print(f"Epoch {epoch+1}/{epochs} Loss: {train_loss / len(dataloader)}")

    return train_losses
    
def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    val_losses = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        loss = criterion(logits, labels)
        val_losses += loss.item()
    print(f"Validation Loss: {val_losses / len(dataloader)}")
    return val_losses / len(dataloader)

def generate_predictions(model, data_loader, tokenizer, device):
    model.eval()
    predictions = []
    references = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits[:,-1,:].argmax(dim=-1)
            pred_labels = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            true_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(pred_labels)
            references.extend(true_labels)
    return predictions, references


def calculate_accuracy(predictions, references):
    correct = 0
    total = len(predictions)
    for pred, ref in zip(predictions, references):
        if pred.strip() == ref.strip():
            correct += 1
    return correct / total


def main():

    #Define Hyper Parameters 
    model_name = 'gpt-2'
    epochs = 10 
    learning_rate = 3e-5
    take_last_token = True
    max_length = 128
    batch_size = 8
    # Load Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = BertForMCQA(model, take_last_token=take_last_token)

    # Load Data
    data = read_data('train_complete.jsonl')
    dataset = MCADataSet(data, tokenizer, max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    data = read_data('dev_complete.jsonl')
    dataset = MCADataSet(data, tokenizer, max_length)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    data = read_data('test_complete.jsonl')
    dataset = MCADataSet(data, tokenizer, max_length)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move Model to Device
    model.to(device)

    #Define Loss Function
    criterion = nn.CrossEntropyLoss()

    # Train Model
    train_losses = train_model(model, train_loader, optimizer, device, epochs, criterion)
    val_loss = evaluate_model(model, val_loader, device, criterion)

    # Generate Predictions
    predictions, references = generate_predictions(model, test_loader, tokenizer, device)
    accuracy = calculate_accuracy(predictions, references)
    print(f"Test Accuracy: {accuracy}")

if __name__ == '__main__':
    main()


    
    
