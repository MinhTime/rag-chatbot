"""Start a local Ollama server and pull a model (used by the Kaggle notebook)."""
import json
import subprocess
import time
import urllib.request


def server_up(url="http://localhost:11434"):
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def start_server(log_path="ollama.log", url="http://localhost:11434", timeout=60):
    if server_up(url):
        return None
    proc = subprocess.Popen(["ollama", "serve"], stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(timeout):
        if server_up(url):
            return proc
        time.sleep(1)
    raise RuntimeError(f"Ollama did not start, see {log_path}")


def pull(model):
    subprocess.run(["ollama", "pull", model], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def model_info(model, url="http://localhost:11434"):
    req = urllib.request.Request(f"{url}/api/show", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req))
    return d.get("details", {})
