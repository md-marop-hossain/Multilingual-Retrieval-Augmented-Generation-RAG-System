import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io, os, re, unicodedata

# ---------- Windows Tesseract Configuration ----------
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set language
lang = "ben"  # Bengali language OCR
pdf_path = "data/HSC26_Bangla.pdf"
output_path = "data/cleaned_text.txt"
os.makedirs("data", exist_ok=True)

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text

def extract_text_with_ocr(pdf_path, lang="ben"):
    doc = fitz.open(pdf_path)
    all_text = ""

    for i, page in enumerate(doc):
        print(f"Processing page {i + 1}/{len(doc)}")
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            text = pytesseract.image_to_string(img, lang=lang)
        except pytesseract.TesseractError:
            text = pytesseract.image_to_string(img, lang="eng")
        all_text += text + "\n"
    
    return clean_text(all_text)

if __name__ == "__main__":
    result = extract_text_with_ocr(pdf_path, lang=lang)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    print("✅ OCR completed and saved to", output_path)
