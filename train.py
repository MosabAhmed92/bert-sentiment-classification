import torch 
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from data import BERTDataset
from sentiment_model import SentimentModel


def train():
    # device Setup 
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Hyperparameters 
    batch_size = 16
    learning_rate = 0.00002
    epochs = 2

    # loading the dataset 
    data = BERTDataset('./Data/IMDB Dataset.csv')

    # splitting parameters
    train_size = int(0.8*len(data))
    test_size = int(len(data) - train_size)

    # The Split
    train_set, test_set = random_split(dataset = data, lengths = [train_size, test_size])

    # Data Loaders
    train_dl = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    # instantiate the Model 
    model = SentimentModel().to(device)
    optimizer = Adam(params = model.parameters(), lr = learning_rate)
    criterion = BCEWithLogitsLoss()

    # looping 
    for epoch in (range(epochs)):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_dl):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 1
            optimizer.zero_grad()
            # 2
            predictions = model(input_ids, attention_mask)
            # 3
            # we use label.view(-1, 1) to make it as a column vector not a row vector .. not [[256]] but [[256, 1]]
            loss = criterion(predictions, labels.view(-1, 1))
            # 4
            loss.backward()
            # 5
            optimizer.step()

            epoch_loss += loss.item()


            if batch_idx%10 == 0:
                print(f"Epoch : {epoch} | Batch : {batch_idx / len(train_dl)} | Loss : {loss.item():.4f}")
        
        print(f"Epoch {epoch + 1} Finishe. Avg Loss : {epoch_loss/ len(train_dl):.4f}")


if __name__ == '__main__':
    train()