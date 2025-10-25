"""
Utility functions used across all modules
Shared by ALL team members
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

def log_event(message: str, level: str = "INFO"):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {"INFO": "ℹ️ ", "WARN": "⚠️ ", "ERROR": "❌", "DEBUG": "🔍"}.get(level, "")
    print(f"[{timestamp}] [{level}] {prefix} {message}")

def print_header(title: str, width: int = 60):
    """Print formatted section header"""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")

def save_json(data: Any, filepath: str):
    """Save data to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log_event(f"Saved to {filepath}")

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    log_event(f"Loaded from {filepath}")
    return data

def clean_text(text: str) -> str:
    """Clean text: remove HTML tags, extra whitespace"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:\(\)\-\'\"—]', '', text)
    return text.strip()

def batch_list(items: List, batch_size: int) -> List[List]:
    """Split list into batches"""
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

if __name__ == "__main__":
    print_header("Testing Utils")
    log_event("Utils module is working!")
