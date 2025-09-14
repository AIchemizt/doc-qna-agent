import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.tools import ArxivQueryRun

def main():
    """Main function to run the Q&A agent."""
    # --- SETUP ---
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("🔴 GOOGLE_API_KEY not found in .env file.")

    # Configuration
    FAISS_INDEX_PATH = "faiss_index"
    EMBEDDING_MODEL = "hkunlp/instructor-large"

    print("Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Loading FAISS vector store...")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"🔴 FAISS index not found at {FAISS_INDEX_PATH}. Please run ingest.py first.")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    print("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

    # --- INITIALIZE CORE COMPONENTS ---
    
    # 1. Document Q&A Chain (The primary tool)
    doc_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    # 2. ArXiv Tool (The bonus tool)
    arxiv_tool = ArxivQueryRun()

    print("\n✅ Agent is ready.")
    print("--- Document & ArXiv Q&A Agent ---")
    print("Default mode is searching your PDFs. To search ArXiv, start your query with 'arxiv'.")
    print("Type 'exit' to quit.")

    # --- MAIN LOOP ---
    while True:
        try:
            user_query = input("\n[You]: ")
            if user_query.lower() == 'exit':
                print("Exiting...")
                break
            
            # Simple Router Logic
            if user_query.lower().startswith("arxiv"):
                print("[Agent is searching ArXiv...]")
                # Extract the actual search term after "arxiv "
                search_term = user_query[5:].strip()
                if not search_term:
                    response = "Please provide a search term after 'arxiv'."
                else:
                    response = arxiv_tool.run(search_term)
            else:
                print("[Agent is searching local documents...]")
                # Use the main document Q&A chain
                response_dict = doc_qa_chain.invoke({"query": user_query})
                response = response_dict['result']

            print(f"\n[Agent]: {response}")

        except Exception as e:
            print(f"🔴 An error occurred: {e}")

if __name__ == "__main__":
    main()