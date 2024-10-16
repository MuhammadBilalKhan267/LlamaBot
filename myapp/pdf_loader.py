from PyPDF2 import PdfReader
from langchain.docstore.document import Document

def load_pdf_from_file(file_object):
    pdf_reader = PdfReader(file_object)

    text = ""

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    document =  Document(page_content=text, metadata={})

    return document

