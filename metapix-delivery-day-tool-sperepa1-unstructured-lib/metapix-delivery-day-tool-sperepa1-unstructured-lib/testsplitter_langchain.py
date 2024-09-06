import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Hard-code the OpenAI API key
OPENAI_API_KEY = ''

# Define the file path for the Mustang model
#MUSTANG_FILE_PATH = "C:/Users/SPEREPA1/Downloads/2024_Ford_Mustang_Owners_Manual_version_1_om_EN-US.pdf"
#MUSTANG_FILE_PATH = "C:/Users/SPEREPA1/Downloads/24MYMustangQRG.pdf"
#MUSTANG_FILE_PATH = "C:/Users/SPEREPA1/Downloads/24MY_F150_QRG_ENG_V1.pdf"
MUSTANG_FILE_PATH = "C:/Users/SPEREPA1/Downloads/2024_Ford_F-150_Owners_Manual_version_1_om_EN-US.pdf"

# Step 1: Extract Text from PDF
def extract_text_from_pdf(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    text = ""
    try:
        reader = PdfReader(filename)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        if not text:
            raise ValueError("No text extracted from the PDF.")
        
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from the PDF: {e}")
        return ""

# Step 2: Split Text into Chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        raise ValueError("No text chunks were created. Please check the input text and splitting logic.")
    
    return chunks

# Step 3: Create Embeddings and Vector Store
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    try:
        embedded_texts = embeddings.embed_documents(chunks)
    except Exception as e:
        print(f"An error occurred while generating embeddings: {e}")
        embedded_texts = []

    if not embedded_texts:
        raise ValueError("No embeddings were generated. Please check the input texts and API key.")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Step 4: Retrieve Answer to Question
def answer_question(question, vectorstore):
    retriever = vectorstore.as_retriever()
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff",
        llm=llm,
        return_source_documents=True
    )
    results = qa_chain.invoke({"query": question})
    answer = results['result']
    source_docs = results['source_documents'][:3]  # Limit to top 3 source documents for brevity
    
    return answer, source_docs

# Step 5: Format Answer with Source Text
def format_answer_with_source(question, vectorstore):
    answer, source_docs = answer_question(question, vectorstore)
    formatted_answer = f"Answer: {answer}\n\nSources:\n"
    
    for i, doc in enumerate(source_docs):
        formatted_answer += f"Source {i+1}: {doc.page_content[:500]}...\n"  # Show more characters for longer output
    
    return formatted_answer

# Main function to orchestrate the process
def main():
    filename = MUSTANG_FILE_PATH
    text = extract_text_from_pdf(filename)

    if not text:
        print("No text extracted from the PDF.")
        return

    chunks = split_text_into_chunks(text)
    vectorstore = create_vector_store(chunks)

    # Example usage
    question = "Please explain how to use the FordPass Integration feature in f150 "
    formatted_answer = format_answer_with_source(question, vectorstore)
    print(formatted_answer)

if __name__ == "__main__":
    main()
