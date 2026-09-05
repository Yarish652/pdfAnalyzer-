from pypdf import PdfReader


def extract_document(file):
    reader = PdfReader(file)

    document = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if not page_text:
            continue

        blocks = [
            block.strip()
            for block in page_text.split("\n")
            if block.strip()
        ]

        document.append({
            "page": page_number,
            "blocks": blocks
        })

    return document