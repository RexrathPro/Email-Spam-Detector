# Email Spam Detector using Machine Learning

This project builds a machine learning system capable of classifying emails as **Spam** or **Ham (Not Spam)** using classical NLP methods. Multiple algorithms were benchmarked, and the best performing model was selected for potential deployment.

---

##  Overview

Spam filtering is a real-world NLP problem used everywhere in production email systems (Gmail, Outlook, Yahoo, corporate mail servers, etc).  
The goal of this project was to:

- Process raw email text
- Extract meaningful features (TF-IDF)
- Train and compare multiple ML models
- Select the best performer
- Prepare model for deployment (serialization)

---

##  Dataset

The dataset contains **5728 labeled email messages** with:

- `text` — Email content
- `spam` — Binary label (1 = spam, 0 = ham)

Dataset size: ~8.5MB  
Labels are relatively balanced for training.

---

## Text Preprocessing & Feature Engineering

Preprocessing steps applied:

- Lowercasing
- Tokenization
- Removal of punctuation
- Removal of stopwords
- Stemming (Porter Stemmer)
- TF-IDF vectorization (`TfidfVectorizer`)

These techniques convert raw text into machine-learnable numeric features.

---

##  Models Evaluated

The following machine learning models were trained and compared:

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Support Vector Machine (SVM) | Linear / Kernel |
| Bernoulli Naive Bayes | Probabilistic |
| Multinomial Naive Bayes | Probabilistic |
| K-Nearest Neighbors (KNN) | Distance-based |
| Decision Tree | Tree-based |
| Random Forest | Ensemble (Bagging) |
| Extra Trees | Ensemble (Bagging) |
| Gradient Boosting | Ensemble (Boosting) |
| AdaBoost | Ensemble (Boosting) |
| Stacking Classifier | Meta-ensemble |

---

##  Results

Final evaluation metrics on test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Bernoulli Naive Bayes** | **0.9826** | **0.9823** | **0.8740** | **0.9250** |
| Extra Trees | 0.9787 | 0.9906 | 0.8346 | 0.9059 |
| SVM | 0.9738 | 1.0000 | 0.7874 | 0.8810 |
| Random Forest | 0.9729 | 1.0000 | 0.7795 | 0.8761 |
| Multinomial NB | 0.9719 | 1.0000 | 0.7717 | 0.8711 |
| Logistic Regression | 0.9593 | 0.9670 | 0.6929 | 0.8073 |
| Gradient Boosting | 0.9593 | 0.9775 | 0.6850 | 0.8056 |
| AdaBoost | 0.9302 | 0.8667 | 0.5118 | 0.6436 |
| KNN | 0.9128 | 1.0000 | 0.2913 | 0.4512 |
| Decision Tree | 0.9467 | 0.8158 | 0.7323 | 0.7718 |

> **Best performer:** `Bernoulli Naive Bayes` due to highest F1-score, strong precision and recall, and small model size.

---

##  Key Insight

Despite the rise of transformer models, classical ML methods like Naive Bayes remain extremely strong for spam filtering when combined with TF-IDF.

---

##  Model Deployment Preparation

The following artifacts were serialized for future deployment:

```
pickle.dump(tfidf, open('vectorizer.pkl','wb'))
pickle.dump(bnb, open('model.pkl','wb'))
```

Alternative deployment (recommended):

```
Pipeline([
  ('tfidf', TfidfVectorizer()),
  ('bnb', BernoulliNB())
])
```

Saved as a single artifact for API serving.

---

##  Tech Stack

**Languages**
- Python

**Libraries**
- scikit-learn
- pandas
- numpy
- nltk
- matplotlib / seaborn

---

##  Project Structure

```
Email-Spam-Detector/
│
├── data/                 # dataset (optional)
├── notebooks/            # Jupyter experiments
├── src/                  # code modules
├── models/               # serialized models (optional)
├── requirements.txt
└── README.md
```

---

##  How to Run

```bash
pip install -r requirements.txt
```

Train:

```bash
python src/train.py
```

Predict (future API):

```python
model.predict(["You have won a FREE prize!"])
```

---

##  Future Improvements

Planned enhancements:

- [ ] Deploy via FastAPI / Flask
- [ ] Add Streamlit Web UI
- [ ] Threshold tuning for spam control
- [ ] Track experiments with MLflow
- [ ] Integrate transformer models (BERT / DistilBERT)
- [ ] Add SHAP explainability for feature importance
- [ ] Add monitoring for production metrics

---

##  Why This Project Matters

This project represents a real ML engineering workflow:

✔ Data Cleaning  
✔ NLP Feature Engineering  
✔ Model Benchmarking  
✔ Evaluation Metrics  
✔ Serialization  
✔ Deployment-readiness  

This aligns very closely with actual **ML Engineer internship** expectations.

---

## 📜 License

MIT License

---

## 👤 Author

**Pratham (RexrathPro)**  
ML & Software Engineering Enthusiast

