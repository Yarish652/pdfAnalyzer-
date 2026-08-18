from fastapi import FastAPI, UploadFile, File
from pdf_reader import extract_text
from chunker import sentence_chunker



app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print("Received file:", file.filename)

    text = extract_text(file.file)
    print("extracted text")
    print(text)

    chunks = sentence_chunker(text, 3)

    for i,  chunk in enumerate(chunks):
        print(f"\n--chunk {i+1}--")
        print(chunk)


    return {
        "success": True,
        "message": "PDF received successfully",
        "filename": file.filename
    }