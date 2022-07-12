from sre_parse import Tokenizer
import nltk
#nltk.download('punkt')     #call only once to download the punkt package
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word).lower()

def bagOfWords(tokenized_sentence, allWords):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(allWords), dtype=np.int32)

    for indx, w in enumerate(allWords):
        if w in tokenized_sentence:
            bag[indx] = 1
    
    return bag

# sen = ["hello", "where", "is", "it"]
# wrds = ["hey", "hello", "come", "is", "when", "it"]
# bow = bagOfWords(sen, wrds)
# print(bow)




# a = "how can i get the products?"
# a = tokenize(a)
# print(a)

# words = ["enjoyable", "beginning", "beautiful"]
# stemmer_words = [stem(w) for w in words]
# print(stemmer_words)
