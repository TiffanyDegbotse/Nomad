import sys, os
sys.path.append(os.path.dirname(__file__))

from orchestrator.orchestrator import Orchestrator


if __name__ == "__main__":
    Orchestrator().run_once()  # change to run_loop() for continuous mode
