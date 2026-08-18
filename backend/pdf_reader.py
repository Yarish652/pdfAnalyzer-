from pypdf import PdfReader


def extract_text(file):
    reader = PdfReader(file)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: 
            text += page_text + "\n"

    return text