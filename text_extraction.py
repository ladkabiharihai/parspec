import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from retry import retry

LOCAL_DIR = "pdf_files"
os.makedirs(LOCAL_DIR, exist_ok=True)

@retry(tries=3, delay=2)
def download_pdf(pdf_url, save_path):
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return save_path

    try:
        print(f"Downloading: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded: {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
        return None

def extract_text_from_pdf(pdf_url):
    try:
        pdf_filename = os.path.basename(pdf_url.split('?')[0])
        save_path = os.path.join(LOCAL_DIR, pdf_filename)

        local_pdf_path = download_pdf(pdf_url, save_path)
        if not local_pdf_path:
            return None

        with open(local_pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        print(f"Extracted text from {local_pdf_path}")
        return text.strip()
    except Exception as e:
        print(f"Failed to extract text from {pdf_url}: {e}")
        return None
