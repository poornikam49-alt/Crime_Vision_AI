"""
crime_classifier.py - Multi-modal Crime Classification and Alert Generation Engine.
Fuses Computer Vision (DIP/motion/edge dynamics), ML/NN anomalies, and NLP textual cues
to identify crime type, severity, confidence, and alert notifications.
"""

import re
import numpy as np
from typing import Dict, Any, List, Tuple
import investigator_feedback

CRIME_TAXONOMY = {
    "Physical Assault / Violent Encounter": {
        "severity": "CRITICAL",
        "keywords": [
            "fight", "fighting", "assault", "assaulted", "punch", "punched", "punching",
            "struggle", "struggling", "attack", "attacked", "violence", "violent", "hit", "hitting",
            "brawl", "kicking", "kicked", "beating", "shoved", "wrestling", "battery", "scuffle",
            "altercation", "physical"
        ],
        "visual_profile": {"motion_spike": 0.05, "jerk_threshold": 0.025, "edge_change": 0.02},
        "description": "Rapid physical altercation, erratic high-velocity struggle, or physical confrontation."
    },
    "Armed Robbery / Weapon Threat": {
        "severity": "CRITICAL",
        "keywords": [
            "weapon", "gun", "knife", "knives", "pistol", "firearm", "blade", "armed", "threat",
            "threatened", "hostage", "pointing", "demanding", "robbery", "robbed", "machete", "holdup"
        ],
        "visual_profile": {"motion_spike": 0.05, "jerk_threshold": 0.025, "edge_change": 0.03},
        "description": "Brandishing a visible weapon, armed intimidation, or coercive robbery."
    },
    "Burglary / Perimeter Intrusion": {
        "severity": "HIGH",
        "keywords": [
            "broken window", "forced entry", "break-in", "broke in", "unauthorized access", "trespass",
            "trespassing", "perimeter", "door jimmied", "night", "alarm", "after-hours", "climbing",
            "breach", "breached", "intruder", "intrusion", "burglar", "burglary"
        ],
        "visual_profile": {"motion_spike": 0.04, "jerk_threshold": 0.02, "edge_change": 0.03},
        "description": "Unauthorized perimeter entry, forced access point breach, or trespassing."
    },
    "Theft / Larceny / Quick Grab": {
        "severity": "HIGH",
        "keywords": [
            "theft", "stolen", "steal", "stealing", "snatch", "snatched", "snatching", "grab", "grabbed",
            "missing property", "bag", "laptop", "shoplifting", "fleeing", "ran away", "purse",
            "backpack", "picked", "thief"
        ],
        "visual_profile": {"motion_spike": 0.05, "jerk_threshold": 0.025, "edge_change": 0.02},
        "description": "Sudden property grab, opportunistic theft, or rapid escape trajectory."
    },
    "Vandalism / Property Destruction": {
        "severity": "MEDIUM",
        "keywords": [
            "vandalism", "vandal", "damage", "damaging", "graffiti", "spray", "smashed", "shattered",
            "destroyed", "property damage", "broken glass", "kicking door", "tagging", "defaced"
        ],
        "visual_profile": {"motion_spike": 0.04, "jerk_threshold": 0.02, "edge_change": 0.04},
        "description": "Localized intentional property destruction, defacement, or breakage."
    },
    "Suspicious Loitering / Reconnaissance": {
        "severity": "MEDIUM",
        "keywords": [
            "loitering", "loiter", "prowling", "casing", "waiting around",
            "lurking", "reconnaissance", "peering", "hovering", "masked", "casing joint"
        ],
        "visual_profile": {"motion_spike": 0.02, "jerk_threshold": 0.01, "edge_change": 0.01},
        "description": "Prolonged unauthorized dwelling, casing entryways, or reconnaissance behavior."
    }
}

def analyze_incident_text(text: str) -> Dict[str, Any]:
    """Analyzes user description / notes for crime indicators."""
    if not text:
        return {"clean": "", "matched_keywords": [], "scores": {k: 0.0 for k in CRIME_TAXONOMY}}
    
    clean = re.sub(r"\s+", " ", text).strip().lower()
    matched_all = set()
    category_scores = {}
    
    for crime, details in CRIME_TAXONOMY.items():
        score = 0.0
        for kw in details["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", clean):
                score += 1.0
                matched_all.add(kw)
            elif kw in clean:
                score += 0.5
                matched_all.add(kw)
        category_scores[crime] = score
        
    return {
        "clean": clean,
        "matched_keywords": sorted(list(matched_all)),
        "scores": category_scores
    }

def extract_visual_dynamics(df) -> Dict[str, float]:
    """Computes advanced statistical and dynamic indicators from frame features."""
    if df.empty:
        return {
            "peak_motion": 0.0, "mean_motion": 0.0, "motion_std": 0.0,
            "motion_jerk": 0.0, "edge_variance": 0.0, "ml_outlier_ratio": 0.0,
            "nn_high_activity_ratio": 0.0, "peak_nn_prob": 0.0,
            "flagged_duration_sec": 0.0, "peak_timestamp": 0.0
        }
        
    motion = df["motion_ratio"].to_numpy()
    edges = df["edge_ratio"].to_numpy()
    
    peak_motion = float(np.max(motion)) if len(motion) else 0.0
    mean_motion = float(np.mean(motion)) if len(motion) else 0.0
    motion_std = float(np.std(motion)) if len(motion) else 0.0
    
    # Motion jerk (derivative of motion)
    motion_diff = np.abs(np.diff(motion)) if len(motion) > 1 else np.array([0.0])
    motion_jerk = float(np.max(motion_diff)) if len(motion_diff) else 0.0
    
    edge_variance = float(np.std(edges)) if len(edges) else 0.0
    
    ml_flags = df.get("ml_anomaly", np.zeros(len(df), dtype=bool))
    ml_outlier_ratio = float(np.mean(ml_flags)) if len(df) else 0.0
    
    nn_flags = df.get("nn_high_activity", np.zeros(len(df), dtype=bool))
    nn_high_activity_ratio = float(np.mean(nn_flags)) if len(df) else 0.0
    
    nn_probs = df.get("nn_high_activity_probability", np.zeros(len(df)))
    peak_nn_prob = float(np.max(nn_probs)) if len(nn_probs) else 0.0
    
    # Peak timestamp
    peak_idx = int(np.argmax(motion)) if len(motion) else 0
    peak_timestamp = float(df.iloc[peak_idx]["timestamp_seconds"]) if len(df) else 0.0
    
    # Combined flagged frames duration
    flagged = df[(df.get("ml_anomaly", False)) | (df.get("nn_high_activity", False))]
    flagged_duration = len(flagged) * (df["timestamp_seconds"].iloc[1] - df["timestamp_seconds"].iloc[0]) if len(df) > 1 else 0.0
    
    return {
        "peak_motion": peak_motion,
        "mean_motion": mean_motion,
        "motion_std": motion_std,
        "motion_jerk": motion_jerk,
        "edge_variance": edge_variance,
        "ml_outlier_ratio": ml_outlier_ratio,
        "nn_high_activity_ratio": nn_high_activity_ratio,
        "peak_nn_prob": peak_nn_prob,
        "flagged_duration_sec": max(0.0, flagged_duration),
        "peak_timestamp": peak_timestamp
    }

def classify_crime(df, text_description: str = "") -> Dict[str, Any]:
    """
    Multi-modal Crime Classification Engine.
    Combines visual dynamics, NLP incident cues, and statistical outliers.
    """
    text_info = analyze_incident_text(text_description)
    vd = extract_visual_dynamics(df)
    
    # Raw scoring per crime category
    crime_scores = {}
    
    for crime, data in CRIME_TAXONOMY.items():
        v_prof = data["visual_profile"]
        
        # 1. Visual dynamics score
        v_score = 0.0
        if "Loitering" in crime:
            if vd["peak_motion"] > 0.05 or vd["motion_jerk"] > 0.025:
                v_score = 0.05  # High kinetic jerk disqualifies loitering
            else:
                v_score = 0.50 if (0.01 <= vd["mean_motion"] <= 0.04) else 0.10
        else:
            if vd["peak_motion"] >= v_prof["motion_spike"]:
                v_score += 0.45 * min(2.0, vd["peak_motion"] / v_prof["motion_spike"])
            if vd["motion_jerk"] >= v_prof["jerk_threshold"]:
                v_score += 0.40 * min(2.0, vd["motion_jerk"] / v_prof["jerk_threshold"])
            if vd["edge_variance"] >= v_prof["edge_change"]:
                v_score += 0.25 * min(2.0, vd["edge_variance"] / v_prof["edge_change"])
            
        # Specific heuristic boosts based on distinct signatures:
        if "Assault" in crime and (vd["motion_jerk"] >= 0.03 or vd["peak_motion"] >= 0.05):
            v_score += 0.45
        elif "Burglary" in crime and vd["edge_variance"] >= 0.025 and vd["peak_motion"] >= 0.04:
            v_score += 0.35
        elif "Theft" in crime and vd["motion_jerk"] >= 0.025:
            v_score += 0.35
        elif "Vandalism" in crime and vd["edge_variance"] >= 0.03 and vd["peak_motion"] >= 0.035:
            v_score += 0.35
            
        # 2. Textual NLP score (0.0 to 1.5)
        t_raw = text_info["scores"].get(crime, 0.0)
        t_score = min(1.5, t_raw * 0.75) if t_raw > 0 else 0.0
        
        # 3. ML/NN Anomaly boost (0.0 to 0.4)
        anomaly_score = (vd["ml_outlier_ratio"] * 0.5) + (vd["peak_nn_prob"] * 0.5)
        
        # Multi-modal fusion weights
        if text_description.strip() and t_raw > 0:
            combined_score = (v_score * 0.35) + (t_score * 0.55) + (anomaly_score * 0.10)
        elif text_description.strip():
            combined_score = (v_score * 0.65) + (anomaly_score * 0.35)
        else:
            combined_score = (v_score * 0.70) + (anomaly_score * 0.30)
            
        # Incorporate self-improving active learning bias
        learned_bias = investigator_feedback.get_adaptive_crime_bias(crime)
        combined_score = max(0.0, combined_score + learned_bias)
            
        crime_scores[crime] = round(float(combined_score), 4)

    # Sort candidates
    sorted_crimes = sorted(crime_scores.items(), key=lambda x: x[1], reverse=True)
    top_crime, top_score = sorted_crimes[0]
    second_crime, second_score = sorted_crimes[1] if len(sorted_crimes) > 1 else ("None", 0.0)
    
    # Balanced review gate:
    # These features detect unusual activity, NOT a crime itself. Because the
    # previous gate required every visual threshold to be exceeded at once,
    # it could miss staged fight/assault videos. We now flag a clip for human
    # review when multiple independent activity signals agree.
    visual_signals = [
        vd["peak_motion"] >= 0.05,
        vd["motion_jerk"] >= 0.025,
        vd["edge_variance"] >= 0.025,
        vd["ml_outlier_ratio"] >= 0.10,
        vd["peak_nn_prob"] >= 0.65,
    ]
    # Binary prototype decision rule. It is deliberately based on combinations
    # of signals rather than motion alone. This is a heuristic, not a trained
    # crime-recognition model.
    visual_review_required = (
        (vd["motion_jerk"] >= 0.025 and vd["peak_motion"] >= 0.04)
        or (vd["ml_outlier_ratio"] >= 0.10 and vd["peak_nn_prob"] >= 0.65)
        or (
            vd["peak_motion"] >= 0.07
            and vd["motion_jerk"] >= 0.02
            and vd["edge_variance"] >= 0.025
        )
    )

    # Specific incident words in investigator notes can support a candidate
    # classification, but ordinary words such as "night" or "bag" cannot.
    specific_terms = {
        "fight", "fighting", "assault", "assaulted", "punch", "punched", "punching",
        "struggle", "struggling", "attack", "attacked", "violence", "violent", "brawl",
        "kicking", "kicked", "beating", "robbery", "robbed", "burglary", "burglar",
        "forced entry", "break-in", "broke in", "intruder", "intrusion", "theft", "stolen",
        "steal", "stealing", "snatch", "snatched", "shoplifting", "vandalism", "vandal",
        "smashed", "shattered", "property damage", "weapon", "gun", "knife", "firearm",
        "hostage", "threatened", "murder", "stabbed", "stabbing", "shooting"
    }
    has_specific_text = any(
        re.search(r"\\b" + re.escape(term) + r"\\b", text_info["clean"])
        if " " not in term else term in text_info["clean"]
        for term in specific_terms
    )
    has_text_confirmation = bool(has_specific_text and text_info["matched_keywords"])

    # Report a possible incident for review; do not claim that a crime is proven.
    is_crime = bool(has_text_confirmation or visual_review_required)

    if not is_crime:
        primary_crime = "No Strong Incident Indicators (Routine / Inconclusive)"
        severity = "LOW"
        confidence = max(0.50, 1.0 - top_score)
        alert_level = "NORMAL"
    else:
        if has_text_confirmation:
            primary_crime = top_crime
            base_severity = CRIME_TAXONOMY[top_crime]["severity"]
            if vd["peak_motion"] > 0.12 or vd["motion_jerk"] > 0.07 or base_severity == "CRITICAL":
                severity = "CRITICAL"
            elif vd["peak_motion"] > 0.06 or base_severity == "HIGH":
                severity = "HIGH"
            else:
                severity = base_severity
        else:
            # Visual dynamics alone cannot reliably identify a specific crime.
            primary_crime = "Possible suspicious / violent activity (type unconfirmed)"
            severity = "MEDIUM"
        alert_level = severity
        confidence = min(0.90, max(0.50, 0.45 + (top_score * 0.35)))

    # Compile clear, actionable alert message
    if is_crime:
        alert_message = (
            f"⚠️ POSSIBLE INCIDENT — HUMAN REVIEW REQUIRED: {primary_crime.upper()} | Severity: [{severity}]\n"
            f"Potential incident indicators observed around timestamp {vd['peak_timestamp']:.2f}s "
            f"(Peak motion: {vd['peak_motion']*100:.1f}%, Jerk: {vd['motion_jerk']*100:.1f}%). "
            f"Model score: {confidence*100:.1f}%. This is not proof that a crime occurred; verify the footage manually."
        )
    else:
        alert_message = (
            f"✅ NO STRONG CRIME INDICATORS DETECTED | Routine / inconclusive activity\n"
            f"The available motion/anomaly signals do not meet the conservative incident-review threshold "
            f"(Peak motion: {vd['peak_motion']*100:.1f}%, NN activity score: {vd['peak_nn_prob']*100:.1f}%). "
            f"This does not prove that no crime occurred."
        )
        
    # Generate itemized evidence points detected in the video
    evidence_points = []
    if vd["peak_motion"] >= 0.05:
        evidence_points.append({
            "type": "Motion Dynamics Spike",
            "detail": f"Sudden motion surge of {vd['peak_motion']*100:.1f}% observed at timestamp {vd['peak_timestamp']:.2f}s."
        })
    if vd["motion_jerk"] >= 0.03:
        evidence_points.append({
            "type": "Violent Jerk / Rapid Acceleration",
            "detail": f"Sudden velocity discontinuity ({vd['motion_jerk']*100:.1f}%) indicative of rapid physical struggle or sprint."
        })
    if vd["edge_variance"] >= 0.025:
        evidence_points.append({
            "type": "Structural / Edge Disturbance",
            "detail": f"High edge profile variation ({vd['edge_variance']*100:.2f}%), indicating object displacement or localized intrusion."
        })
    if vd["ml_outlier_ratio"] > 0:
        evidence_points.append({
            "type": "Statistical Isolation Anomaly",
            "detail": f"{vd['ml_outlier_ratio']*100:.1f}% of sampled frames statistically diverge from background baseline."
        })
    if vd["peak_nn_prob"] >= 0.50:
        evidence_points.append({
            "type": "Neural Network High-Activity Flag",
            "detail": f"Neural classifier signaled high-threat probability peaking at {vd['peak_nn_prob']*100:.1f}%."
        })
    if text_info["matched_keywords"]:
        evidence_points.append({
            "type": "Witness / Log Keyword Correlation",
            "detail": f"Incident notes corroborated with high-risk terminology: {', '.join(text_info['matched_keywords'])}."
        })
        
    if not evidence_points:
        evidence_points.append({
            "type": "Baseline Consistency",
            "detail": "No significant visual, edge, or motion anomalies exceeded investigation thresholds."
        })

    # Normalized probability distribution across taxonomy
    sum_scores = sum(crime_scores.values()) or 1.0
    probabilities = {k: round(v / sum_scores, 3) for k, v in crime_scores.items()}

    return {
        "is_crime_detected": is_crime,
        "primary_crime": primary_crime,
        "secondary_crime": second_crime,
        "severity": severity,
        "confidence": round(confidence, 3),
        "alert_level": alert_level,
        "alert_message": alert_message,
        "evidence_points": evidence_points,
        "probabilities": probabilities,
        "matched_keywords": text_info["matched_keywords"],
        "visual_dynamics": vd
    }
