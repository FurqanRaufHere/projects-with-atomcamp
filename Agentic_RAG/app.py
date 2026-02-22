import os
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ------------------------
# Load API Key
# ------------------------

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not set in Space secrets.")

# ------------------------
# Initialize LLM
# ------------------------

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=512
)

# ------------------------
# Load Documents
# ------------------------

DATA_DIR = "./data"

def load_documents(data_dir):
    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = filename
            documents.append(doc)

    return documents

documents = load_documents(DATA_DIR)

# ------------------------
# Chunking
# ------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunks = text_splitter.split_documents(documents)

# ------------------------
# Embeddings + FAISS
# ------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------------
# Prompt
# ------------------------

prompt_template = """
You are an AI assistant that answers ONLY using provided context.

If answer not found, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ------------------------
# RAG Pipeline
# ------------------------

def rag_pipeline(query):
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return "I don't know based on the provided documents.", ""

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    final_prompt = prompt.format(context=context, question=query)

    response = llm.invoke(final_prompt)
    answer = response.content

    sources = list(set(doc.metadata["source"] for doc in retrieved_docs))
    source_text = "\n".join(sources)

    return answer, source_text

# ------------------------
# Gradio UI
# ------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Agentic AI RAG Assistant")

    question_input = gr.Textbox(label="Your Question", lines=2)
    answer_output = gr.Textbox(label="Answer", lines=8)
    source_output = gr.Textbox(label="Sources", lines=4)

    submit = gr.Button("Submit")

    submit.click(
        fn=rag_pipeline,
        inputs=question_input,
        outputs=[answer_output, source_output]
    )

demo.launch()
