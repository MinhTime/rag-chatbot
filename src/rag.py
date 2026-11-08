"""The RAG chatbot: retriever + LLM served by Ollama, wired with LangChain."""
import re

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

SYSTEM = ("Bạn là trợ lý hỏi đáp tiếng Việt. Trả lời thật ngắn gọn, chỉ một cụm từ "
          "hoặc một câu ngắn, không giải thích.")

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + " Chỉ dùng thông tin trong NGỮ CẢNH. "
               "Nếu ngữ cảnh không có câu trả lời, hãy nói: Không biết."),
    ("human", "NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}\nTRẢ LỜI:"),
])

NO_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + " Nếu không chắc, hãy nói: Không biết."),
    ("human", "CÂU HỎI: {question}\nTRẢ LỜI:"),
])


def get_embeddings(model_name, device="cuda"):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device},
                                 encode_kwargs={"normalize_embeddings": True, "batch_size": 32})


def build_index(chunks, embeddings):
    # vectors are L2-normalised, so L2 ranking == cosine ranking
    return FAISS.from_documents(chunks, embeddings)


def get_llm(cfg, model=None, json_mode=False):
    model = model or cfg.llm_model
    extra = {"reasoning": cfg.think} if model.startswith("qwen3") else {}  # thinking models only
    return ChatOllama(model=model, base_url=cfg.ollama_url,
                      temperature=cfg.temperature, num_ctx=cfg.num_ctx,
                      num_predict=cfg.num_predict, format="json" if json_mode else None, **extra)


def strip_think(text):
    """Drop a <think>...</think> block if a reasoning model emits one."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def format_context(docs):
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


class RAGChatbot:
    def __init__(self, retriever, llm, k=4):
        self.retriever, self.k = retriever, k
        self.rag_chain = RAG_PROMPT | llm | StrOutputParser() | RunnableLambda(strip_think)
        self.plain_chain = NO_RAG_PROMPT | llm | StrOutputParser() | RunnableLambda(strip_think)

    def retrieve(self, question, k=None):
        return self.retriever.search(question, k or self.k)

    def answer_with(self, question, docs):
        return self.rag_chain.invoke({"context": format_context(docs), "question": question})

    def ask(self, question):
        docs = self.retrieve(question)
        return {"answer": self.answer_with(question, docs), "docs": docs}

    def ask_no_rag(self, question):
        return {"answer": self.plain_chain.invoke({"question": question}).strip(), "docs": []}

