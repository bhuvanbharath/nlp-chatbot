import random
import json
from statistics import mode
import torch
from model import NeuralNet
from nltk_utils import bagOfWords, tokenize, stem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

with open ('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"

data = torch.load(FILE)

input_size = data['input_size']
model_state = data['model_state']
hidden_size = data['hidden_size']
output_size = data['output_size']
allWords = data['allWords']
tags = data['tags']

model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()    #evaluation mode

bot_name = "Mantech"
print("Let's chat! Type 'quit' to exit")

while(True):
    user_input = input("You: ")

    if (user_input == 'quit'):
        break

    user_input = tokenize(user_input)
    x = bagOfWords(user_input, allWords)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x.float())
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if (prob > 0.75):
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I don't understand this!")
