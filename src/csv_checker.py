import pandas as pd
df = pd.read_csv('data/hamber.csv')
df.rename(columns={'Message': 'text', 'Category': 'label'}, inplace=True)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})
print(df['label'].value_counts())
