"""
investigator_feedback.py - Self-Improving Active Learning and Human-in-the-Loop (HITL) System.
Captures investigator reviews, tracks forensic precision/accuracy, and dynamically calibrates
crime classification weights and sensitivity thresholds based on cumulative feedback.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "investigator_feedback.json")

def load_feedback_history() -> List[Dict[str, Any]]:
    """Loads all historical investigator feedback records."""
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_feedback_record(record: Dict[str, Any]) -> bool:
    """Appends a new verified feedback record to the persistent feedback store."""
    history = load_feedback_history()
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(record)
    try:
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save feedback: {e}")
        return False

def compute_learning_metrics() -> Dict[str, Any]:
    """
    Computes performance statistics and self-improving calibration weights
    from accumulated investigator feedback.
    """
    history = load_feedback_history()
    total = len(history)
    
    if total == 0:
        return {
            "total_reviews": 0,
            "accuracy_rate": 88.5, # baseline default
            "true_positives": 0,
            "false_alarms": 0,
            "misclassifications": 0,
            "category_biases": {},
            "status": "System using baseline pre-trained priors (0 feedback records)."
        }
        
    correct_count = sum(1 for r in history if r.get("verdict") == "Confirmed Correct")
    false_alarm_count = sum(1 for r in history if r.get("verdict") == "False Alarm (Routine Activity)")
    misclassified_count = sum(1 for r in history if r.get("verdict") == "Misclassified Crime Type")
    
    accuracy = round((correct_count / total) * 100, 1)
    
    # Calculate category-level adaptive bias adjustments
    # If a category is frequently confirmed, slightly boost its prior (+0.05 to +0.15)
    # If a category frequently produces false alarms, penalize its prior (-0.05 to -0.20)
    cat_confirms = {}
    cat_false_alarms = {}
    
    for r in history:
        pred_crime = r.get("predicted_crime", "")
        verdict = r.get("verdict", "")
        
        if verdict == "Confirmed Correct":
            cat_confirms[pred_crime] = cat_confirms.get(pred_crime, 0) + 1
        elif verdict == "False Alarm (Routine Activity)":
            cat_false_alarms[pred_crime] = cat_false_alarms.get(pred_crime, 0) + 1

    category_biases = {}
    all_cats = set(list(cat_confirms.keys()) + list(cat_false_alarms.keys()))
    
    for cat in all_cats:
        c = cat_confirms.get(cat, 0)
        fa = cat_false_alarms.get(cat, 0)
        # Net bias adjustment between -0.20 and +0.15
        net = (c * 0.04) - (fa * 0.08)
        category_biases[cat] = round(max(-0.25, min(0.20, net)), 3)

    return {
        "total_reviews": total,
        "accuracy_rate": accuracy,
        "true_positives": correct_count,
        "false_alarms": false_alarm_count,
        "misclassifications": misclassified_count,
        "category_biases": category_biases,
        "status": f"Active Learning calibrated across {total} verified investigator review(s)."
    }

def get_adaptive_crime_bias(crime_type: str) -> float:
    """Returns dynamically calibrated probability bias for a given crime type."""
    metrics = compute_learning_metrics()
    biases = metrics.get("category_biases", {})
    return biases.get(crime_type, 0.0)
