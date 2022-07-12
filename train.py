import json
from socketserver import DatagramRequestHandler
from nltk_utils import bagOfWords, tokenize, stem
import numpy as np

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

class chatDataset(dataset):
    def _init__(self):
        n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __get_item__ (self, index):
        return self.x_data(indx), self.y_data(indx)

    def __get_len__(self):
        return self.n_samples

#hyperparameters
batch_size = 8

