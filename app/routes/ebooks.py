from fastapi import APIRouter, UploadFile, File
from ..services import ebook_processor
import shutil
import os

router = APIRouter()

UPLOAD_DIR = "uploaded_ebooks"

@router.post("/upload/")
async def upload_ebook(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if file.filename.endswith(".pdf"):
        text = ebook_processor.extract_text_from_pdf(file_path)
    elif file.filename.endswith(".epub"):
        text = ebook_processor.extract_text_from_epub(file_path)
    elif file.filename.endswith(".docx"):
        text = ebook_processor.extract_text_from_docx(file_path)
    else:
        return {"error": "Unsupported file format"}

    return {"text": text}
