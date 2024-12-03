import pickle
from text_extraction import extract_text_from_pdf
from text_embedding import get_text_embedding

with open("model.pkl", "rb") as f:
    classifier = pickle.load(f)

def predict_class(pdf_url):
    text = extract_text_from_pdf(pdf_url)
    if not text:
        return "Error extracting text", None
    embedding = get_text_embedding(text)
    predicted_class = classifier.predict([embedding])[0]
    probabilities = classifier.predict_proba([embedding])[0]
    return predicted_class, probabilities

