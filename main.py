import sys, os
sys.path.append(os.path.dirname(__file__))

from orchestrator.orchestrator import Orchestrator


if __name__ == "__main__":
    Orchestrator().run_once()  # change to run_loop() for continuous mode
from core.rag_qa import NomadQA
from core.utils import read_yaml, ROOT

settings = read_yaml(ROOT / "config" / "settings.yaml")
qa = NomadQA(settings)

while True:
    q = input("\nAsk Nomad a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    result = qa.ask(q)
    print(f"Nomad: {result['reply']}")
