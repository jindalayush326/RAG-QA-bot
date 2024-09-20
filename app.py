# Import Libraries
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Set up OpenAI and Pinecone
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone

pc = Pinecone(api_key=pinecone_api_key)

# Read PDF document
def read_doc(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Divide documents into chunks
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

# Setup Pinecone index
def setup_pinecone_index(embeddings, chunks, index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]),  
            metric="cosine",  
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    ids = [str(i) for i in range(len(embeddings))]
    vectors_to_upsert = [
        {
            "id": ids[i],
            "values": embeddings[i].tolist(),
            "metadata": {"text": chunks[i]},
        }
        for i in range(len(embeddings))
    ]
    index.upsert(vectors=vectors_to_upsert)

    return index

# Retrieve query from Pinecone
def retrieve_query(query, index, k=2):
    matching_results = index.query(query, top_k=k)
    return matching_results['matches']

# Retrieve answers using QA chain
def retrieve_answers(query, index, chain):
    doc_search = retrieve_query(query, index)
    docs = [match['metadata']['text'] for match in doc_search]
    response = chain.run(input_documents=docs, question=query)
    return response, docs

# Streamlit interface
def main():
    st.title("Retrieval-Augmented Generation (RAG) QA Bot")

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded PDF
        file_path = "task.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process PDF
        documents = read_doc(file_path)
        chunks = chunk_data(documents)
        
        # Generate embeddings
        # Generate embeddings
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
        embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])

        
        # Setup Pinecone index
        index_name = "langchain"
        index = setup_pinecone_index(embeddings, chunks, index_name)

        # Load QA model
        llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        # Query input
        query = st.text_input("Ask a question about the document:")
        
        if query:
            answer, retrieved_docs = retrieve_answers(query, index, qa_chain)
            st.write("Answer:", answer)
            st.write("Retrieved Document Segments:")
            for doc in retrieved_docs:
                st.write("-", doc)

if __name__ == "__main__":
    main()
