import nltk
#nltk.download('punkt')     #call only once to download the punkt package
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word).lower()

a = "how can i get the products?"
a = tokenize(a)
print(a)

words = ["enjoyable", "beginning", "beautiful"]
stemmer_words = [stem(w) for w in words]
print(stemmer_words)
