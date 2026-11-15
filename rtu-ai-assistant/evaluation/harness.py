"""
Evaluation Harness for RTU AI Academic Assistant.
Serves as the common test runner, session lifecycle manager, and reporter.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Ensure backend path resolution
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.database.session import SessionLocal
from app.retrieval.pipeline import RetrievalPipeline
from app.services.chat_service import ChatService
from app.services.importance_service import ImportanceService
from app.services.planner_service import PlannerService
from app.services.quiz_service import QuizService

logger = logging.getLogger("rtu.eval.harness")

class EvaluationHarness:
    """
    Manages end-to-end evaluation runs, datasets, and structured output formatting.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or str(CURRENT_DIR / "eval_dataset_v2_100.json")
        self.db = None
        self.pipeline = None
        self.chat_service = None
        self.dataset = []
        self._initialized = False

    def setup(self):
        """Initializes database connection and eager retrieval indexes once."""
        if self._initialized:
            return
        
        logger.info("[Harness] Bootstrapping RTU Evaluation Harness...")
        self.db = SessionLocal()
        self.pipeline = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
        self.pipeline.load_indexes(self.db)
        self.chat_service = ChatService(pipeline=self.pipeline)
        self.load_dataset()
        self._initialized = True
        logger.info("[Harness] RTU Evaluation Harness ready.")

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads evaluation queries from JSON dataset file."""
        if not os.path.exists(self.dataset_path):
            logger.warning(f"[Harness] Dataset file not found: {self.dataset_path}")
            self.dataset = []
            return self.dataset

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        logger.info(f"[Harness] Loaded {len(self.dataset)} evaluation queries from {self.dataset_path}")
        return self.dataset

    def run_benchmark(
        self,
        name: str,
        eval_fn: Callable[[Dict[str, Any], Any], Dict[str, Any]],
        limit: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> Dict[str, Any]:
        """
        Executes eval_fn across the dataset, measuring per-item latency and aggregating metrics.
        """
        self.setup()
        records = self.dataset
        if filter_fn:
            records = [r for r in records if filter_fn(r)]
        if limit:
            records = records[:limit]

        logger.info(f"[Harness] Starting benchmark '{name}' across {len(records)} test cases...")
        start_time = time.perf_counter()
        results = []
        durations = []

        for idx, item in enumerate(records):
            item_start = time.perf_counter()
            try:
                res = eval_fn(item, self)
                item_dur = (time.perf_counter() - item_start) * 1000.0
                res["latency_ms"] = round(item_dur, 2)
                res["query_id"] = item.get("query_id", f"q_{idx}")
                res["subject"] = item.get("subject", "General")
                results.append(res)
                durations.append(item_dur)
            except Exception as e:
                logger.error(f"[Harness] Error on query {item.get('query_id')}: {e}", exc_info=True)
                results.append({
                    "query_id": item.get("query_id", f"q_{idx}"),
                    "error": str(e),
                    "passed": False,
                    "latency_ms": round((time.perf_counter() - item_start) * 1000.0, 2)
                })

        total_elapsed = time.perf_counter() - start_time
        return {
            "benchmark": name,
            "total_items": len(records),
            "total_elapsed_sec": round(total_elapsed, 2),
            "avg_latency_ms": round(sum(durations) / max(len(durations), 1), 2),
            "results": results
        }

    def save_report(self, data: Dict[str, Any], output_path: str):
        """Saves evaluation report to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[Harness] Report written to {out}")

    def teardown(self):
        """Closes open database sessions."""
        if self.db:
            self.db.close()
            self._initialized = False
