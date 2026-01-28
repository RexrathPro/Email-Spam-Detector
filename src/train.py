import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from src.preprocess import transform_text

# Load CSV
df = pd.read_csv("data/hamber.csv")

# Rename columns
df.rename(columns={"Message": "text", "Category": "label"}, inplace=True)

# Dataset already uses 0/1 so no mapping needed

# Apply preprocessing
df["transformed_text"] = df["text"].apply(transform_text)

# Feature + target
X = df["transformed_text"]
y = df["label"]

# TF-IDF
tfidf = TfidfVectorizer()
X_vec = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Save artifacts
pickle.dump(tfidf, open("models/vectorizer.pkl", "wb"))
pickle.dump(model, open("models/model.pkl", "wb"))

print("\nTraining completed successfully!")
