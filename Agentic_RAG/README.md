# Agentic RAG

Lightweight Retrieval-Augmented Generation (RAG) demo using local documents, FAISS embeddings, and a Groq-backed LLM with a Gradio UI.

Quick Start
- Create a Python virtual environment and install dependencies:

	```bash
	python -m venv .venv
	.venv\Scripts\activate
	pip install -r requirements.txt
	```

- Set the required environment variable:

	```powershell
	setx GROQ_API_KEY "<your_api_key>"
	```

- Add docs to the `data/` folder (PDF or TXT), then run:

	```bash
	python app.py
	```

Files
- **Requirements:** [Agentic_RAG/requirements.txt](Agentic_RAG/requirements.txt)
- **App entrypoint:** [Agentic_RAG/app.py](Agentic_RAG/app.py)
- **Documents:** [Agentic_RAG/data/](Agentic_RAG/data/)

Notes
- The app launches a Gradio UI (default: http://127.0.0.1:7860).
- Ensure `GROQ_API_KEY` is set; the app raises an error if it's missing.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` and FAISS for retrieval.

License
- MIT (choose a license that fits your project)
