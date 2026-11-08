"""LLM-as-judge with the local model: correctness (vs gold) and faithfulness (vs context)."""
import json
import re

from langchain_core.prompts import ChatPromptTemplate

# Single user message (no system role): works the same for every judge model family.
CORRECTNESS = ChatPromptTemplate.from_messages([
    ("human", "You grade answers to Vietnamese questions. Compare the MODEL ANSWER with the "
               "GOLD ANSWER. Score 1-5: 5 = same meaning as gold, 4 = correct but with extra or "
               "missing minor detail, 3 = partly correct, 2 = mostly wrong, 1 = wrong or "
               "'Không biết'. Reply only JSON: {{\"score\": <1-5>, \"reason\": \"<short>\"}}\n\n"
               "QUESTION: {question}\nGOLD ANSWER: {gold}\nMODEL ANSWER: {answer}"),
])

FAITHFULNESS = ChatPromptTemplate.from_messages([
    ("human", "You check if an answer is supported by the given CONTEXT. Score 1-5: "
               "5 = fully supported, 3 = partly supported, 1 = not supported or invented. "
               "If the answer is 'Không biết', score 5 only if the context really lacks the "
               "answer, else 1. Reply only JSON: {{\"score\": <1-5>, \"reason\": \"<short>\"}}\n\n"
               "CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER: {answer}"),
])


def parse_score(text):
    try:
        s = int(json.loads(text)["score"])
    except Exception:
        m = re.search(r"[1-5]", text or "")
        s = int(m.group()) if m else 1
    return min(max(s, 1), 5)


class Judge:
    """Scores 1-5. Tries JSON mode, then plain text; returns None if the judge model fails
    (e.g. Ollama aborts a degenerate, repeating generation) so one bad call cannot kill a run."""

    def __init__(self, llm_json, llm_plain=None):
        self.llms = [llm for llm in (llm_json, llm_plain) if llm is not None]
        self.failures = 0

    def _score(self, prompt, inputs):
        for llm in self.llms:
            try:
                return parse_score((prompt | llm).invoke(inputs).content)
            except Exception as e:  # noqa: BLE001 - any backend error -> next attempt
                last = e
        self.failures += 1
        print(f"judge failed: {type(last).__name__}: {str(last)[:120]}")
        return None

    def correctness(self, question, golds, answer):
        return self._score(CORRECTNESS, {"question": question, "gold": " | ".join(golds),
                                         "answer": answer})

    def faithfulness(self, question, context, answer):
        return self._score(FAITHFULNESS, {"question": question, "context": context,
                                          "answer": answer})
