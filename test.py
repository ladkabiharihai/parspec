import pandas as pd
from predict_pipeline import predict_class

test_data = pd.read_excel("DataSet.xlsx", sheet_name="test_data")

predictions = []
for url in test_data['datasheet_link']:
    predicted_class, _ = predict_class(url)
    predictions.append(predicted_class)

test_data['predicted_class'] = predictions

from sklearn.metrics import classification_report
print(classification_report(test_data['target_col'], test_data['predicted_class']))