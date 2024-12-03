import pandas as pd
from sklearn.metrics import classification_report
from predict_pipeline import predict_class

def evaluate_test_data(test_file_path, sheet_name="test_data"):
    print("Loading test data...")
    test_data = pd.read_excel(test_file_path, sheet_name=sheet_name)

    predictions = []
    true_labels = test_data['target_col'].tolist()

    print("Processing test dataset...")
    for idx, row in test_data.iterrows():
        pdf_url = row['datasheet_link']
        true_label = row['target_col']

        try:
            predicted_class, _ = predict_class(pdf_url)
            predictions.append(predicted_class)
            print(f"Processed {idx + 1}/{len(test_data)}: {predicted_class} (True: {true_label})")
        except Exception as e:
            print(f"Error processing {pdf_url}: {e}")
            predictions.append("Error extracting text")

    test_data['predicted_class'] = predictions

    output_path = "test_results.csv"
    test_data.to_csv(output_path, index=False)
    print(f"Test results saved to {output_path}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, labels=["cable", "fuses", "lighting", "others"]))

if __name__ == "__main__":
    test_file_path = "DataSet.xlsx"
    
    evaluate_test_data(test_file_path)
