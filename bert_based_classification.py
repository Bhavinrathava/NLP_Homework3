import torch.nn as nn 
import torch.nn.functional as F
from transformers import BertModel
import torch 
from tqdm import tqdm
import json
from transformers import BertTokenizer

from Dataset import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, model_name, num_classes, batch_size=8):
        super(Model, self).__init__()
        self.pre_trained = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.pre_trained.config.hidden_size * 4, num_classes)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.view(-1, 128)
        attention_mask = attention_mask.view(-1, 128)
        output = self.pre_trained(input_ids = input_ids, attention_mask = attention_mask)
        output = output.pooler_output.view(batch_size,  -1)
        output = self.fc(output)
        return output
    
def convert_to_train_data(encoded_choices):
    input_ids, attention_mask = [], []
    for choice in encoded_choices:
        input_ids.append(choice['input_ids'])
        attention_mask.append(choice['attention_mask'])

    input_ids = torch.stack(input_ids).transpose(0, 1).reshape(-1, 128)
    attention_mask = torch.stack(attention_mask).transpose(0, 1).reshape(-1, 128)

    return input_ids, attention_mask

def train(model, optimizer, criterion, epochs, train_loader, val_loader, batch_size=8):

    train_losses, val_losses, train_acc, val_acc = [], [], 0, 0 

    for epoch in tqdm(range(epochs)):
        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            encoded_choices, labels = batch['encoded_choices'], batch['label']
            input_ids, attention_mask = convert_to_train_data(encoded_choices)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            output = model(input_ids, attention_mask)
            #print(output.shape, labels.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader)):
                encoded_choices, labels = batch['encoded_choices'], batch['label']
                input_ids, attention_mask = convert_to_train_data(encoded_choices)
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                output = model(input_ids, attention_mask)
                
                loss = criterion(output, labels)
                val_losses.append(loss.item())



    return train_losses, val_losses, train_acc, val_acc

def evaluate():
    test_loss, test_acc = 0, 0
    
    return test_loss, test_acc

def predict():
    pass

if __name__ == '__main__':
    
    # Hyperparameters
    model_name = 'bert-base-uncased'
    num_classes = 4
    epochs = 10 
    learning_rate = 0.001
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128
    batch_size = 8

    # Define Model 
    model = Model(model_name, num_classes)
    model.to(device)
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define Loss Function
    criterion = nn.CrossEntropyLoss()

    # Load Data
    data = []
    with open('train_complete.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    train_loader = get_loader(data, tokenizer, max_length, batch_size)

    data = []
    with open('dev_complete.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    val_loader = get_loader(data, tokenizer, max_length, batch_size)

    data = []
    with open('test_complete.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    test_loader = get_loader(data, tokenizer, max_length, batch_size)
    

    # Print Model Summary
    #print(model)

    # Print Dataloader Summary
    #print(f'Length of Train Dataloader : {len(train_loader)}')
    #print(f'Length of Validation Dataloader : {len(val_loader)}')
    #print(f'Length of Test Dataloader : {len(test_loader)}')


    # Train Model
    train_losses, val_losses, train_acc, val_acc = train(model, optimizer, criterion, epochs, train_loader, val_loader)

    # Evaluate Model
    test_loss, test_acc = evaluate(model, criterion, test_loader)


