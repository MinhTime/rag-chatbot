"""The RAG chatbot: bge-m3 embeddings + FAISS + Qwen2.5 (Ollama), wired with LangChain."""
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    return ChatOllama(model=model or cfg.llm_model, base_url=cfg.ollama_url,
                      temperature=cfg.temperature, num_ctx=cfg.num_ctx,
                      num_predict=cfg.num_predict, format="json" if json_mode else None)


def format_context(docs):
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


class RAGChatbot:
    def __init__(self, index, llm, k=4):
        self.index, self.k = index, k
        self.rag_chain = RAG_PROMPT | llm | StrOutputParser()
        self.plain_chain = NO_RAG_PROMPT | llm | StrOutputParser()

    def retrieve(self, question, k=None):
        return self.index.similarity_search(question, k=k or self.k)

    def ask(self, question):
        docs = self.retrieve(question)
        answer = self.rag_chain.invoke({"context": format_context(docs), "question": question})
        return {"answer": answer.strip(), "docs": docs}

    def ask_no_rag(self, question):
        return {"answer": self.plain_chain.invoke({"question": question}).strip(), "docs": []}


def batch_retrieve_ctx_ids(index, embeddings, questions, k):
    """Fast retrieval for many questions at once: returns ctx_id lists (top-k chunks)."""
    q = np.asarray(embeddings.embed_documents(questions), dtype="float32")
    _, ids = index.index.search(q, k)
    store, id_map = index.docstore, index.index_to_docstore_id
    return [[store.search(id_map[i]).metadata["ctx_id"] for i in row if i >= 0] for row in ids]
