
import os
import tempfile
import gradio as gr

from google import genai
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END
from typing import TypedDict


# -----------------------------
# Gemini Setup
# -----------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")

client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

vector_db = None
uploaded_file_name = None


# -----------------------------
# Gemini Embeddings
# -----------------------------

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )

            embeddings.append(result.embeddings[0].values)

        return embeddings

    def embed_query(self, text):
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        return result.embeddings[0].values


embedding_model = GeminiEmbeddings()


# -----------------------------
# Upload + Process PDF
# -----------------------------

def process_pdf(file):

    global vector_db
    global uploaded_file_name

    if file is None:
        return "Please upload a PDF."

    try:

        reader = PdfReader(file)

        document_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                document_text += text + "\n"

        if not document_text.strip():
            return "No readable text found in this PDF."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = splitter.split_text(document_text)

        documents = [
            Document(page_content=chunk)
            for chunk in chunks
        ]

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model
        )

        uploaded_file_name = os.path.basename(file)

        return (
            f"✅ Document processed successfully!\n\n"
            f"File: {uploaded_file_name}\n"
            f"Pages: {len(reader.pages)}\n"
            f"Chunks: {len(chunks)}"
        )

    except Exception as e:
        return f"Error processing document: {str(e)}"


# -----------------------------
# Tool
# -----------------------------

@tool
def search_document(query: str) -> str:
    """Search the uploaded document for relevant information."""

    if vector_db is None:
        return "No document has been uploaded."

    docs = vector_db.similarity_search(query, k=3)

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# -----------------------------
# LangGraph
# -----------------------------

class AgentState(TypedDict):
    user_query: str
    retrieved_context: str
    final_answer: str


def retrieve_node(state):

    context = search_document.invoke(
        state["user_query"]
    )

    return {
        "retrieved_context": context
    }


def answer_node(state):

    prompt = f"""
You are ActionDoc AI.

Answer the user's question using ONLY the uploaded document.

DOCUMENT CONTEXT:
{state["retrieved_context"]}

USER QUESTION:
{state["user_query"]}

Give a clear response.

When relevant include:
- Important information
- Tasks
- Deadlines
- Required items
- Action plan

If information is unavailable, say "Not specified".
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return {
        "final_answer": response.text
    }


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", END)

actiondoc_graph = workflow.compile()


# -----------------------------
# Final Agent
# -----------------------------

def ask_actiondoc(question):

    if vector_db is None:
        return "Please upload and process a PDF first."

    if not question.strip():
        return "Please enter a question."

    try:

        result = actiondoc_graph.invoke({
            "user_query": question
        })

        return result["final_answer"]

    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------
# Gradio UI
# -----------------------------

with gr.Blocks(title="ActionDoc AI") as demo:

    gr.Markdown(
        """
        # 📄 ActionDoc AI
        ### Document-to-Action AI Agent

        Upload a PDF and ask questions about tasks,
        deadlines, instructions and important information.
        """
    )

    pdf_file = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    process_button = gr.Button(
        "Process Document"
    )

    document_status = gr.Textbox(
        label="Document Status"
    )

    process_button.click(
        process_pdf,
        inputs=pdf_file,
        outputs=document_status
    )

    question = gr.Textbox(
        label="Ask ActionDoc AI",
        placeholder="Example: What are the important deadlines?"
    )

    ask_button = gr.Button(
        "Ask Agent"
    )

    answer = gr.Markdown()

    ask_button.click(
        ask_actiondoc,
        inputs=question,
        outputs=answer
    )


# -----------------------------
# Start Server
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
