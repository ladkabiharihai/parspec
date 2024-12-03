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
        return save_path
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
        return None

def extract_text_from_pdf(pdf_url):
    pdf_filename = os.path.basename(pdf_url.split('?')[0])
    save_path = os.path.join(LOCAL_DIR, pdf_filename)
    local_pdf_path = download_pdf(pdf_url, save_path)
    if not local_pdf_path:
        return None
    try:
        with open(local_pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        print(f"Failed to extract text from {pdf_url}: {e}")
        return None
