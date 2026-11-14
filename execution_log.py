"""Save a separate JSON execution log for each question."""

from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

LOG_DIRECTORY = Path(__file__).resolve().parent / "logs"


def save_execution_log(execution_log: dict) -> Path:
    """Write one UTF-8 JSON file per question without overwriting earlier turns."""
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    question_time = datetime.fromisoformat(execution_log["asked_at"]).strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_path = LOG_DIRECTORY / f"question_{question_time}.json"

    if log_path.exists():
        log_path = LOG_DIRECTORY / f"question_{question_time}_{uuid4().hex[:5]}.json"

    with log_path.open("x", encoding="utf-8") as log_file:
        json.dump(execution_log, log_file, ensure_ascii=False, indent=2)
    
    return log_path
