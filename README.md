# Document Q&A Agent with ArXiv Integration

This project is an AI agent that answers questions based on a collection of PDF documents. It uses a local vector database for document retrieval and the Gemini API for language understanding. It also includes a tool to search ArXiv for external papers.

## Features

-   **PDF Ingestion:** Processes multiple PDF documents from a local directory.
-   **Vector Search:** Uses FAISS with local sentence-transformer embeddings to create a searchable knowledge base.
-   **LLM Integration:** Leverages the Google Gemini API (`gemini-1.5-flash`) for question answering and reasoning.
-   **Tool Usage (Bonus):** Integrates the ArXiv API as a callable tool for searching external academic papers.
-   **Simple Router:** A keyword-based router directs queries to either the local document store or the ArXiv tool.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AIchemizt/doc-qna-agent.git
    cd doc-qna-agent
    ```

2.  **Create and activate a virtual environment:**
    -   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your API Key:**
    -   Create a `.env` file in the root directory.
    -   Add your Google Gemini API key to it:
      ```
      GOOGLE_API_KEY="your_api_key_here"
      ```

## Usage

1.  **Add PDFs:** Place your PDF files inside the `docs/` directory.

2.  **Build the Knowledge Base:** Run the ingestion script once. This processes the PDFs into a local FAISS vector store. This step can take a few minutes.
    ```bash
    python ingest.py
    ```

3.  **Run the Agent:** Start the main application to begin asking questions.
    ```bash
    python main.py
    ```
    -   **To query your documents:** Just type your question.
    -   **To search ArXiv:** Start your query with the keyword `arxiv`.
    -   Type `exit` to quit.