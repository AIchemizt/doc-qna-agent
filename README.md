# Document Q&A Agent with ArXiv Integration

This project is an AI agent that answers questions based on a collection of PDF documents. It uses a local vector database for document retrieval and the Gemini API for language understanding. It also includes a tool to search ArXiv for external papers.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd doc-qna-agent
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure your API Key:**
    - Create a `.env` file in the root directory.
    - Add your Google Gemini API key to it:
      ```
      GOOGLE_API_KEY="your_api_key_here"
      ```

## Usage

1.  **Add PDFs:** Place your PDF files inside the `docs/` directory.

2.  **Ingest Documents:** Run the ingestion script once to build the vector database. This will take a few minutes.
    ```bash
    python ingest.py
    ```

3.  **Run the Agent:** Start the main application to begin asking questions.
    ```bash
    python main.py
    ```
    - To ask about your documents, just type your question.
    - To search ArXiv, start your query with the keyword `arxiv`.
    - Type `exit` to quit.