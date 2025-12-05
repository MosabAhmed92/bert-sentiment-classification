import torch 
from torch.utils.data import Dataset

import pandas as pd 
from transformers import AutoTokenizer

class BERTDataset(Dataset):
    def __init__(self, datapath):

        self.path = datapath
        self.max_length = 256
        self.data = pd.read_csv(datapath)
        self.reviews = self.data['review'].to_list()
        self.label = self.data['sentiment'].map({'negative' : 0.0, 'positive' : 1.0}).to_list()

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.label[idx]

        encodings = self.tokenizer(review, padding = 'max_length',truncation = True,
                                        max_length = self.max_length, return_tensors = 'pt')
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        token_type_ids = encodings['token_type_ids']
        
        return {'input_ids' : input_ids.flatten(),
                'attention_mask' : attention_mask.flatten(),
                'label' : torch.tensor(label, dtype = torch.float32)}

if __name__ == '__main__':
    data = BERTDataset('./Data/IMDB Dataset.csv')

    print(f'The length of our data is : {len(data)}')

    d = data[0]

    print('The Sample Sentence : ')
    print('------------------------------------------------')
    print(data.reviews[1])

    print(f"the inputs ids of the sample  is : {d['input_ids'].shape}")
    print('------------------------------------------------')

    print(f"the attention mask shape of the sample is {d['attention_mask'].shape}")
    print('------------------------------------------------')

    print(f"the label of the sample is {d['label']}")