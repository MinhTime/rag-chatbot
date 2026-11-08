from src.metrics import contains, exact_match, f1, hit_at_k, normalize, reciprocal_rank


def test_normalize_vietnamese():
    assert normalize("  Thế kỷ XII! ") == "thế kỷ xii"
    assert normalize("“Hà Nội”.") == "hà nội"


def test_em_f1_contains():
    gold = ["thế kỷ XII"]
    assert exact_match("Thế kỷ XII.", gold) == 1.0
    assert exact_match("vào thế kỷ XII", gold) == 0.0
    assert contains("vào thế kỷ XII", gold) == 1.0
    assert 0.8 < f1("vào thế kỷ XII", gold) < 1.0
    assert f1("Không biết", gold) == 0.0


def test_retrieval_metrics():
    assert hit_at_k([5, 3, 9], 3, 2) == 1.0
    assert hit_at_k([5, 3, 9], 9, 2) == 0.0
    assert reciprocal_rank([5, 3, 9], 3, 4) == 0.5
    assert reciprocal_rank([5, 3, 9], 7, 4) == 0.0


def test_growing_test_set_keeps_original_50():
    from src.data import sample_questions
    qs = [{"id": f"q{i:04d}"} for i in range(1000)]
    q50, q200 = sample_questions(qs, 50, 42), sample_questions(qs, 200, 42)
    assert q200[:50] == q50
    assert len({q["id"] for q in q200}) == 200


def test_notebook_ships_every_src_module_it_imports():
    """Regression: notebooks write src/ via %%writefile; a missing module breaks Kaggle runs."""
    import re
    from pathlib import Path

    import scripts.build_notebook as bn
    root = Path(bn.ROOT)
    shipped = set(bn.SRC)
    cells = (root / "scripts" / "build_notebook.py").read_text()
    imported = {m + ".py" for m in re.findall(r"from src\.(\w+) import", cells)}
    for f in shipped:
        imported |= {m + ".py" for m in re.findall(r"from \.(\w+) import",
                                                   (root / "src" / f).read_text())}
    assert imported <= shipped, imported - shipped


def test_judge_falls_back_and_never_crashes():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from src.judge import Judge

    class Broken(FakeListChatModel):
        def invoke(self, *a, **k):
            raise RuntimeError("prediction aborted, token repeat limit reached")

    ok = FakeListChatModel(responses=["Score: 4"])
    assert Judge(Broken(responses=["x"]), ok).correctness("q", ["g"], "a") == 4
    j = Judge(Broken(responses=["x"]), Broken(responses=["x"]))
    assert j.correctness("q", ["g"], "a") is None and j.failures == 1
