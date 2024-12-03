import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from text_extraction import extract_text_from_pdf
from text_embedding import get_text_embedding

data = pd.read_excel("DataSet.xlsx", sheet_name="train_data")
data['text'] = data['datasheet_link'].apply(extract_text_from_pdf)
data.dropna(subset=['text'], inplace=True)

data['embedding'] = data['text'].apply(get_text_embedding)

X_train, X_val, y_train, y_val = train_test_split(
    list(data['embedding']), data['target_col'], test_size=0.2, random_state=42
)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)
print(classification_report(y_val, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(classifier, f)
