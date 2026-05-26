import os
import sys

# Add backend to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arq import run_worker
from app.core.arq import WorkerSettings

# Import to register ARQ task functions
import app.services.knowledge_tasks  # noqa: F401

if __name__ == "__main__":
    run_worker(WorkerSettings)
