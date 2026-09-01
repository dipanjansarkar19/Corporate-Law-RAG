import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "vectorstore1000"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Copy the key from your OpenAI account and add it to a .env file in the root of this project."
        )
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}/. Add one or more PDFs there first."
        )

    print(f"Step 1/4: Loading {len(pdf_paths)} PDFs...")

    documents = []
    for path in pdf_paths:
        doc_name = os.path.splitext(os.path.basename(path))[0]  # e.g. "ibc_2016"
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_doc"] = doc_name
        documents.extend(pages)
        print(f'Loaded {len(pages)} pages from {os.path.basename(path)}')
    print(f'Total pages across all documents: {len(documents)}')


    print('Step 2/4: Splitting into chunks...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\nSection", 
                      "\n\n",
                        "\n",
                          " "],
    )
    chunks = splitter.split_documents(documents)
    print(f'Created {len(chunks)} chunks.')


    print('Step 3/4: Creating embeddings and building the vector store...')
    print("This call the OpenAI API once per chunk- may take a few minutes")

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    Chroma.from_documents(chunks, embeddings, persist_directory= PERSIST_DIR)

    print(F"Step 4/4 done. Saved to '{PERSIST_DIR}/'.")

    print (f'You can now run: streamlit run app.py')

if __name__ == "__main__":
    main()