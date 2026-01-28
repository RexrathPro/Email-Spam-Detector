import pickle
from preprocess import transform_text

model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

def predict(text):
    transformed = transform_text(text)
    vect = vectorizer.transform([transformed])
    result = model.predict(vect)[0]
    return result
