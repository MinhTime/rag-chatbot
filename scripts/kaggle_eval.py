"""Run an experiment version on Kaggle and gate it against a baseline.

    python scripts/kaggle_eval.py run v1_hybrid --baseline v0_dense   # push, wait, collect, gate
    python scripts/kaggle_eval.py push v2_rerank
    python scripts/kaggle_eval.py status v2_rerank
    python scripts/kaggle_eval.py collect v2_rerank
    python scripts/kaggle_eval.py gate v2_rerank --baseline v1_hybrid

Needs KAGGLE_USERNAME / KAGGLE_KEY in the environment. Used by CI (.github/workflows/eval.yml).
Exit code 1 from `gate`/`run` = the new version is worse than the baseline.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_notebook import build  # noqa: E402

DATASET = "gamind/uit-viquad-2-0"

# Gate: the new version may not be worse than the baseline by more than this (absolute points).
# Answer metrics use the largest test set both versions have (200, else the original 50).
GATE = {
    ("retrieval_all_dev", "Hit@4"): 0.005,   # 3,852 questions: tight
    ("retrieval_all_dev", "MRR@10"): 0.005,
    ("TEST", "rag", "F1"): 0.02,
    ("TEST", "rag", "Judge>=4"): 0.04,
}
N_BOOT = 2000


def kaggle(*args, check=True):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout + r.stderr


def slug(version):
    return "rag-" + re.sub(r"[^a-z0-9]+", "-", version.lower()).strip("-")


def ref(version):
    return f"{os.environ['KAGGLE_USERNAME']}/{slug(version)}"


def push(version):
    build(version)
    with tempfile.TemporaryDirectory() as d:
        nb = ROOT / "notebooks" / f"{version}_kaggle.ipynb"
        shutil.copy(nb, Path(d) / f"{slug(version)}.ipynb")
        meta = {"id": ref(version), "title": slug(version).replace("-", " "),
                "code_file": f"{slug(version)}.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "enable_tpu": False, "enable_internet": True, "dataset_sources": [DATASET],
                "kernel_sources": [], "competition_sources": [], "model_sources": [],
                "machine_shape": "NvidiaTeslaT4"}
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print(kaggle("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4").strip())


def status(version):
    out = kaggle("kernels", "status", ref(version), check=False)
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    return m.group(1).lower() if m else "unknown"


def collect(version):
    dest = ROOT / "results" / version
    with tempfile.TemporaryDirectory() as d:
        kaggle("kernels", "output", ref(version), "-p", d, "-o", "-q", check=False)
        src = Path(d) / "results" / version
        if not src.exists():
            raise FileNotFoundError(f"no results/{version} in kernel output")
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.name not in ("ollama.log", "index.tar.gz"):
                shutil.copy(f, dest / f.name)
        if (src / "index.tar.gz").exists():  # serving artifact: keep outside the git tree
            (ROOT / "artifacts").mkdir(exist_ok=True)
            shutil.copy(src / "index.tar.gz", ROOT / "artifacts" / f"index-{version}.tar.gz")
        for log in Path(d).glob("*.log"):
            shutil.copy(log, dest / "kaggle_run.log")
    print(f"collected -> {dest}")
    return dest


def get(d, path):
    for k in path:
        d = d[k]
    return d


def paired_bootstrap(version, baseline, n):
    """95% CI of mean F1(new) - F1(old) over the same questions (paired bootstrap)."""
    import numpy as np
    import pandas as pd
    a = pd.read_csv(ROOT / "results" / baseline / "predictions.csv").head(n)
    b = pd.read_csv(ROOT / "results" / version / "predictions.csv").head(n)
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    d = (m["rag_f1_b"] - m["rag_f1_a"]).to_numpy()
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (N_BOOT, len(d))).mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(d)


def gate(version, baseline):
    new = json.loads((ROOT / "results" / version / "metrics.json").read_text())
    old = json.loads((ROOT / "results" / baseline / "metrics.json").read_text())
    sizes = [int(k[4:]) for k in new if k.startswith("test") and k in old]
    test = f"test{max(sizes)}"
    mean, lo, hi, n = paired_bootstrap(version, baseline, max(sizes))
    rows, ok = [], True
    for path, tol in GATE.items():
        path = tuple(test if p == "TEST" else p for p in path)
        a, b = get(old, path), get(new, path)
        passed = b >= a - tol
        if path[-1] == "F1" and not passed and hi >= 0:
            passed = True  # drop is within noise: the 95% CI still includes 0
        ok &= passed
        name = "/".join(path) if path[0].startswith("test") else path[-1]
        rows.append(f"| {name} | {a:.3f} | {b:.3f} | {b - a:+.3f} | "
                    f"{'✅' if passed else '❌'} |")
    table = "\n".join([f"### Eval gate: `{version}` vs `{baseline}`", "",
                       "| metric | baseline | new | Δ | pass |",
                       "|---|---:|---:|---:|:---:|", *rows,
                       "", f"Paired bootstrap on {n} questions: ΔF1 = {mean:+.3f} "
                       f"(95% CI {lo:+.3f} .. {hi:+.3f})", "", f"**{'PASS' if ok else 'FAIL'}**"])
    print(table)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(table + "\n")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "collect", "gate", "run"])
    ap.add_argument("version")
    ap.add_argument("--baseline", default="v0_dense")
    ap.add_argument("--timeout_min", type=int, default=90)
    a = ap.parse_args()
    if a.cmd == "push":
        push(a.version)
    elif a.cmd == "status":
        print(status(a.version))
    elif a.cmd == "collect":
        collect(a.version)
    elif a.cmd == "gate":
        sys.exit(0 if gate(a.version, a.baseline) else 1)
    else:
        push(a.version)
        deadline = time.time() + a.timeout_min * 60
        time.sleep(60)
        while (st := status(a.version)) in ("queued", "running", "unknown"):
            if time.time() > deadline:
                sys.exit(f"timeout, last status {st}")
            time.sleep(60)
        if st != "complete":
            sys.exit(f"kernel finished with status {st}")
        collect(a.version)
        sys.exit(0 if gate(a.version, a.baseline) else 1)


if __name__ == "__main__":
    main()
