import torch.nn as nn 
import torch.nn.functional as F
from transformers import BertModel
import torch 
from tqdm import tqdm
import json
from transformers import BertTokenizer

from Dataset import get_loader

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, model_name, num_classes, batch_size=8, freeze_bert=False):
        super(Model, self).__init__()
        self.pre_trained = BertModel.from_pretrained(model_name)

        if(freeze_bert):
            for param in self.pre_trained.parameters():
                param.requires_grad = False
        

        self.fc = nn.Linear(self.pre_trained.config.hidden_size , 4)

    def forward(self, input_ids, attention_mask):
        #print(input_ids.shape, attention_mask.shape)
        output = self.pre_trained(input_ids = input_ids, attention_mask = attention_mask)
        output = output.pooler_output
        output = self.fc(output)
        return output
    
def convert_to_train_data(encoded_choices, batch_size=8):
    input_ids, attention_mask = [], []
    #print(f'Encoded Choices SHape: {len(encoded_choices)}')
    for choice in encoded_choices:
        input_ids.append(choice['input_ids'])
        attention_mask.append(choice['attention_mask'])

    input_ids = torch.stack(input_ids).transpose(0, 1).reshape(-1, 128)
    attention_mask = torch.stack(attention_mask).transpose(0, 1).reshape(-1, 128)

    return input_ids, attention_mask

def train(model, optimizer, criterion, epochs, train_loader, val_loader, batch_size=8):

    train_losses, val_losses, train_acc, val_acc = [], [], 0, 0 

    for epoch in (range(epochs)):
        model.train()
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}/{epochs}") as pbar:
            for i, batch in (enumerate(train_loader)):
                optimizer.zero_grad()
                encoded_choices, labels = batch['encoded_choices'], batch['label']

                input_ids, attention_mask = encoded_choices['input_ids'], encoded_choices['attention_mask']

                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                labels = labels.to(device)
                output = model(input_ids, attention_mask)

                # measure accuracy
                pred = output.argmax(dim=1, keepdim=True)
                train_acc += pred.eq(labels.view_as(pred)).sum().item()

                loss = criterion(output, labels)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
                
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        train_acc /= len(train_loader)
            
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Validating ") as pbar:
                for i, batch in (enumerate(val_loader)):
                    encoded_choices, labels = batch['encoded_choices'], batch['label']
                    input_ids, attention_mask = encoded_choices['input_ids'], encoded_choices['attention_mask']
                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                    labels = labels.to(device)
                    output = model(input_ids, attention_mask)
                    
                    # measure accuracy
                    pred = output.argmax(dim=1, keepdim=True)
                    val_acc += pred.eq(labels.view_as(pred)).sum().item()
                    loss = criterion(output, labels)
                    
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update(1)
                    
                    val_losses.append(loss.item())
        val_acc /= len(val_loader)

        print(f'Epoch {epoch+1}/{epochs} Train Loss: {sum(train_losses)/len(train_loader)} Train Accuracy: {train_acc * 100}')
        print(f'Epoch {epoch+1}/{epochs} Validation Loss: {sum(val_losses)/len(val_loader)} Validation Accuracy: {val_acc * 100}')


    return train_losses, val_losses, train_acc, val_acc

def evaluate(model, criterion, test_loader, batch_size=8, freeze_bert=True):
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f"Evaluating") as pbar:
            for i, batch in (enumerate(test_loader)):
                encoded_choices, labels = batch['encoded_choices'], batch['label']
                input_ids, attention_mask = encoded_choices['input_ids'], encoded_choices['attention_mask']
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                labels = labels.to(device)
                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)
                test_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

                # measure accuracy
                pred = output.argmax(dim=1, keepdim=True)
                test_acc += pred.eq(labels.view_as(pred)).sum().item()

    print(f'Test Loss: {test_loss/len(test_loader)}')
    print(f'Test Accuracy: {test_acc/len(test_loader) * 100}')
    return test_loss/len(test_loader), test_acc / len(test_loader)

def predict():
    pass

if __name__ == '__main__':
    
    # Hyperparameters
    model_name = 'bert-base-uncased'
    num_classes = 4
    epochs = 3
    learning_rate = 0.0001
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    batch_size = 2
    freeze_bert = True
    # Define Model 
    model = Model(model_name, num_classes, batch_size, freeze_bert)
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
    
    # Train Model
    train_losses, val_losses, train_acc, val_acc = train(model, optimizer, criterion, epochs, train_loader, val_loader)

    # Evaluate Model
    test_loss, test_acc = evaluate(model, criterion, test_loader)


