"""
Crash-Course Planner Service for RTU Exam Helper Chatbot
Implements Section 9.2 of Implementation Plan Version 2.
Time-boxed knapsack algorithm:
Scores topics as `marks_value / study_time`, fills available hours from highest score down,
reserves a 10% buffer, and schedules a final rapid formula/diagram revision block.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.services.importance_service import ImportanceService

class PlannerService:
    def __init__(self):
        self.importance_service = ImportanceService()

    def generate_plan(
        self,
        db: Session,
        subject_id: str,
        hours_available: float = 5.0,
        student_level: str = "year2"
    ) -> Dict[str, Any]:
        """
        Creates an hour-by-hour crash-course preparation schedule:
        1. Fetch ranked topics for subject
        2. Compute score = marks_yield / study_minutes
        3. Fill available time, reserve 10% buffer, append formula/diagram revision block
        """
        ranking_data = self.importance_service.get_ranked_topics(db, subject_id)
        if "error" in ranking_data:
            return ranking_data

        ranked_topics = ranking_data["ranked_topics"]
        subject_name = ranking_data["subject"]
        tier = ranking_data["tier"]

        # Total available minutes
        total_available_minutes = int(hours_available * 60)

        # Scale buffer and revision proportionally so short budgets (0.5h, 1h) still have study time
        if total_available_minutes <= 60:
            buffer_minutes = max(3, int(total_available_minutes * 0.10))
            revision_block_minutes = max(5, int(total_available_minutes * 0.15))
            min_topic_slot = 10
        elif total_available_minutes <= 120:
            buffer_minutes = max(8, int(total_available_minutes * 0.10))
            revision_block_minutes = max(15, int(total_available_minutes * 0.15))
            min_topic_slot = 15
        else:
            buffer_minutes = max(15, int(total_available_minutes * 0.10))
            revision_block_minutes = max(30, int(total_available_minutes * 0.15))
            min_topic_slot = 20

        study_budget_minutes = max(10, total_available_minutes - buffer_minutes - revision_block_minutes)

        # Score topics by marks efficiency
        scored_topics = []
        for t in ranked_topics:
            if total_available_minutes <= 60:
                t_mins = min_topic_slot
            elif total_available_minutes <= 120:
                t_mins = max(min_topic_slot, min(25, t.get("est_study_minutes", 30)))
            else:
                t_mins = max(min_topic_slot, t.get("est_study_minutes", 45))

            if tier in ["Tier A", "Tier B"] and t["total_marks"] > 0:
                # Weighted marks score: total_marks * 1.5 + years_appeared * 5
                marks_yield = (t["total_marks"] * 1.2) + (t["years_appeared"] * 4.0)
            else:
                # Syllabus default heuristic: Unit 1-3 weighted higher
                unit_weight = 10 if t["unit_no"] in [1, 2, 4] else 7
                marks_yield = unit_weight

            efficiency_score = marks_yield / t_mins
            scored_topics.append({
                **t,
                "study_minutes": t_mins,
                "efficiency_score": round(efficiency_score, 3),
                "marks_yield": round(marks_yield, 1)
            })

        # Sort by efficiency score descending
        scored_topics.sort(key=lambda x: x["efficiency_score"], reverse=True)

        # Greedy knapsack allocation
        allocated_topics = []
        spent_minutes = 0
        current_time_offset = 0

        for t in scored_topics:
            if spent_minutes + t["study_minutes"] <= study_budget_minutes:
                start_m = current_time_offset
                end_m = start_m + t["study_minutes"]
                
                allocated_topics.append({
                    "slot_number": len(allocated_topics) + 1,
                    "time_window": f"{start_m // 60}h {start_m % 60}m - {end_m // 60}h {end_m % 60}m",
                    "topic_id": t["topic_id"],
                    "topic_name": t["topic_name"],
                    "unit_no": t["unit_no"],
                    "duration_minutes": t["study_minutes"],
                    "marks_priority": t["importance_level"],
                    "historical_frequency": f"Appeared in {t['years_appeared']} past papers ({t['total_marks']} marks)",
                    "action": "Read definition, memorize 1 core diagram, review 10-mark steps"
                })
                spent_minutes += t["study_minutes"]
                current_time_offset = end_m
            else:
                continue

        # Add Final Revision Session
        rev_start = current_time_offset
        rev_end = rev_start + revision_block_minutes
        allocated_topics.append({
            "slot_number": len(allocated_topics) + 1,
            "time_window": f"{rev_start // 60}h {rev_start % 60}m - {rev_end // 60}h {rev_end % 60}m",
            "topic_id": "REVISION_SHEET",
            "topic_name": f"Comprehensive {subject_name} Rapid Revision Block",
            "unit_no": "All Units",
            "duration_minutes": revision_block_minutes,
            "marks_priority": "CRUCIAL (Final Recall)",
            "historical_frequency": "Covers all key formula sheets and state diagrams",
            "action": "Revise 2-mark concise definitions, formulas, and box-framed algorithms"
        })

        return {
            "subject": subject_name,
            "short_name": ranking_data["short_name"],
            "tier": tier,
            "total_hours": hours_available,
            "hours_available": hours_available,
            "total_budget_minutes": total_available_minutes,
            "allocated_study_minutes": spent_minutes,
            "buffer_minutes": buffer_minutes,
            "revision_minutes": revision_block_minutes,
            "total_topics_covered": len(allocated_topics) - 1,
            "schedule": allocated_topics,
            "final_revision_block": allocated_topics[-1] if allocated_topics else None,
            "senior_advice": (
                f"You have {hours_available} hours left! We prioritized the highest marks-per-minute topics first. "
                f"Stick to the timer: 10 minutes on definitions, 20 minutes on the main diagram and algorithm steps. "
                f"A 10% rest buffer ({buffer_minutes} mins) and final {revision_block_minutes}-min formula review are already locked in."
            )
        }
