import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bagOfWords, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

#print(intents)

allWords = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w, tag))

#print(allWords)

ignore_symbols = ['?', '!', ',', '.']
allWords = [stem(w) for w in allWords if w not in ignore_symbols]
allWords = sorted(set(allWords))    #making it as set to remove the duplicates

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bagOfWords(pattern_sentence, allWords)
    #print(bag)
    x_train.append(bag)

    label = tags.index(tag)     #extracting the index
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__ (self, index):
        return self.x_data[index], self.y_data[index]

    def __len__ (self):
        return self.n_samples

#hyperparameter
batch_size = 8
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = chatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        #forward
        output = model(words.float())
        loss = criterion(output, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}/{num_epochs}\tloss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data ={
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "allWords": allWords,
    "tags": tags
}

FILE = "data.pth"

torch.save(data, FILE)

print('training completed!')