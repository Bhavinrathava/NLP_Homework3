import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']['stem']
        choices = item['question']['choices']
        fact = item['fact1']
        answer_key = item['answerKey']
    
        input_text = ""
        for choice, choice_val in zip(choices, ["A ","B ","C ", "D "]) : 
            formatted_text = f"[CLS] {fact} [SEP] {question} [SEP] {choice} [SEP] {choice['text']} [END]"
            input_text += formatted_text
        
        inputs = self.tokenizer.encode_plus(
                input_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        encoded_choices = {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze()}

        # One Hot Encoding
        label = ord(answer_key) - ord('A')

        label = torch.tensor(label, dtype=torch.long)

        return {
            'encoded_choices': encoded_choices,
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_loader(data, tokenizer, max_length, batch_size):
    dataset = QADataset(data, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data = [
        {
            'question': {
                'stem': 'What is the capital of France?',
                'choices': [
                    {'label': 'A', 'text': 'London'},
                    {'label': 'B', 'text': 'Paris'},
                    {'label': 'C', 'text': 'Berlin'},
                    {'label': 'D', 'text': 'Madrid'}
                ]
            },
            'fact1': 'France is a country in Europe.',
            'answerKey': 'B'
        },
        {
            'question': {
                'stem': 'What is the capital of Germany?',
                'choices': [
                    {'label': 'A', 'text': 'London'},
                    {'label': 'B', 'text': 'Paris'},
                    {'label': 'C', 'text': 'Berlin'},
                    {'label': 'D', 'text': 'Madrid'}
                ]
            },
            'fact1': 'Germany is a country in Europe.',
            'answerKey': 'C'
        }
    ]

    loader = get_loader(data, tokenizer, 128, 2)
    for batch in loader:
        print(batch['encoded_choices'])
        print(batch['label'])
        break