import torch 
import torch.nn as nn 
from transformers import BertModel, AutoTokenizer


class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        print("Loading the Pretrained Model")

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Linear(in_features = 768, out_features=1)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)

        last_hidden_state = outputs.last_hidden_state

        cls_vector = last_hidden_state[:, 0, :]

        logits = self.classifier(cls_vector)

        return logits
    


if __name__ == '__main__':

    # Setup
    model = SentimentModel()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Dummy Data 
    text_batch = ['I love this movie', 'This is terrible']
    inputs = tokenizer(text_batch, padding = True, return_tensors = 'pt')


    # Forward pass 

    print('Feeding Data ....')

    output = model(inputs['input_ids'], inputs['attention_mask'])


    # Verification 

    print('Output Shape {output.shape}')

    if output.shape == (2, 1):
        print('Success ...')
    else:
        print('Failed')









