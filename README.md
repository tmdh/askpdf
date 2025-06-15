# AskPDF

A document question-answering application built with Streamlit that allows users to upload PDFs and ask questions about their content.

## Features

- Upload multiple PDF documents
- Ask questions about the content of uploaded PDFs
- Powered by LangChain and OpenRouter API
- Uses local embeddings with HuggingFace models
- Vector storage with ChromaDB

## Setup

1. Clone the repository
```bash
git clone https://github.com/tmdh/askpdf.git
cd askpdf
```

2. Install dependencies using uv
```bash
pip install -r pyproject.toml
```

3. Create a `.env` file with your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Upload PDF documents using the sidebar
3. Enter your questions in the text area
4. Click Submit to get answers based on the document content

## Technical Details

- Uses Sentence Transformers for text embeddings
  - Embedding model: `all-MiniLM-L6-v2`
  - Text splitting model: `all-mpnet-base-v2`
- Implements RAG (Retrieval Augmented Generation) for accurate answers
- Uses `deepseek/deepseek-chat-v3-0324` model via OpenRouter API
- ChromaDB for vector storage and similarity search
- Document chunking with semantic text splitting (100 tokens per chunk with 50% overlap)

## License

MIT

## Author

Built by [tmdh](https://github.com/tmdh)
