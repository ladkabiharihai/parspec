from fastapi import FastAPI, HTTPException
from predict_pipeline import predict_class

app = FastAPI()

@app.post("/predict/")
def predict(pdf_url: str):
    try:
        predicted_class, probabilities = predict_class(pdf_url)
        if probabilities is None:
            raise HTTPException(status_code=400, detail="Failed to process the PDF.")
        return {
            "predicted_class": predicted_class,
            "class_probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

