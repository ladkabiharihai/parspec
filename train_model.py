import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import pickle
from text_extraction import extract_text_from_pdf
from text_embedding import get_text_embedding

data = pd.read_excel("DataSet.xlsx", sheet_name="train_data")

data['text'] = data['datasheet_link'].apply(extract_text_from_pdf)
data.dropna(subset=['text'], inplace=True)
data['embedding'] = data['text'].apply(get_text_embedding)

X = np.vstack(data['embedding'].values)
y = data['target_col'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

lgbm_classifier = LGBMClassifier(n_estimators=200, random_state=42)
lgbm_classifier.fit(X_train_balanced, y_train_balanced)

y_pred = lgbm_classifier.predict(X_val)
print("Validation Report:")
print(classification_report(y_val, y_pred))

with open("lgbm_model.pkl", "wb") as f:
    pickle.dump(lgbm_classifier, f)
