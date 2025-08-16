# For loading PDF files
from langchain_community.document_loaders import PyPDFLoader
# For Breaking docs to smaller, semantically meaningful chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdfs(uploaded_files, chunk_size=5000, chunk_overlap=500):
    documents = []
    for uploaded_file in uploaded_files:
        # Construct a temporary filename on disk by prefixing 
        # temp_ to the original filename (uploaded_file.name).
        temppdf=f"./temp_{uploaded_file.name}"
        # Use 'wb' mode to write non-text data such as PDFs, images, etc.
        with open(temppdf, "wb") as f:
            # Write raw PDF bytes to temp_df   
            f.write(uploaded_file.getvalue())
            # store original file name for later use
            file_name = uploaded_file.name

        # Load the PDF file using PyPDFLoader.
        loader = PyPDFLoader(temppdf)
        # Defining docs to hold a list of Document objects
        docs = loader.load()
        # Appending as list items to the 'documents' list that was initialised earlier.
        documents.extend(docs)
        # Split the documents into smaller chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(documents)
    return splits