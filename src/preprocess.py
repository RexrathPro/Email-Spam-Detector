import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in stop_words and w not in punct]
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)
