"""LLM-as-judge with the local model: correctness (vs gold) and faithfulness (vs context)."""
import json
import re

from langchain_core.prompts import ChatPromptTemplate

CORRECTNESS = ChatPromptTemplate.from_messages([
    ("system", "You grade answers to Vietnamese questions. Compare the MODEL ANSWER with the "
               "GOLD ANSWER. Score 1-5: 5 = same meaning as gold, 4 = correct but with extra or "
               "missing minor detail, 3 = partly correct, 2 = mostly wrong, 1 = wrong or "
               "'Không biết'. Reply only JSON: {{\"score\": <1-5>, \"reason\": \"<short>\"}}"),
    ("human", "QUESTION: {question}\nGOLD ANSWER: {gold}\nMODEL ANSWER: {answer}"),
])

FAITHFULNESS = ChatPromptTemplate.from_messages([
    ("system", "You check if an answer is supported by the given CONTEXT. Score 1-5: "
               "5 = fully supported, 3 = partly supported, 1 = not supported or invented. "
               "If the answer is 'Không biết', score 5 only if the context really lacks the "
               "answer, else 1. Reply only JSON: {{\"score\": <1-5>, \"reason\": \"<short>\"}}"),
    ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER: {answer}"),
])


def parse_score(text):
    try:
        s = int(json.loads(text)["score"])
    except Exception:
        m = re.search(r"[1-5]", text or "")
        s = int(m.group()) if m else 1
    return min(max(s, 1), 5)


class Judge:
    def __init__(self, llm_json):
        self.correct_chain = CORRECTNESS | llm_json
        self.faith_chain = FAITHFULNESS | llm_json

    def correctness(self, question, golds, answer):
        out = self.correct_chain.invoke({"question": question, "gold": " | ".join(golds),
                                         "answer": answer})
        return parse_score(out.content)

    def faithfulness(self, question, context, answer):
        out = self.faith_chain.invoke({"question": question, "context": context, "answer": answer})
        return parse_score(out.content)
