"""
scene_change_detector.py - Missing Object and Permanent Scene Change Detection Engine.
Identifies removed valuable assets, missing property, and permanent structural scene damages
by comparing pre-incident baseline frames against post-incident surveillance states.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

def extract_baseline_and_post_frames(video_path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Extracts pre-incident baseline reference frame and post-incident terminal frame from video.
    Returns: (baseline_bgr, post_bgr, baseline_timestamp, post_timestamp)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video for scene change analysis.")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Baseline frame: taken at ~0.5s or frame 5
    baseline_idx = min(max(2, int(fps * 0.4)), max(1, total_frames // 6))
    cap.set(cv2.CAP_PROP_POS_FRAMES, baseline_idx)
    ok_base, base_frame = cap.read()
    
    # Post-incident frame: taken near end (~85% of duration)
    post_idx = max(baseline_idx + 1, int(total_frames * 0.88))
    cap.set(cv2.CAP_PROP_POS_FRAMES, post_idx)
    ok_post, post_frame = cap.read()
    
    cap.release()
    
    if not ok_base or base_frame is None:
        raise ValueError("Failed to extract baseline frame.")
    if not ok_post or post_frame is None:
        post_frame = base_frame.copy()
        
    base_ts = round(baseline_idx / fps, 2)
    post_ts = round(post_idx / fps, 2)
    return base_frame, post_frame, base_ts, post_ts

def detect_scene_changes(
    baseline_frame: np.ndarray,
    post_frame: np.ndarray,
    min_area_ratio: float = 0.003
) -> Dict[str, Any]:
    """
    Detects persistent static scene changes and missing objects between two frames.
    """
    h, w = baseline_frame.shape[:2]
    total_area = h * w
    min_contour_area = total_area * min_area_ratio

    # Convert to grayscale
    gray_base = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(post_frame, cv2.COLOR_BGR2GRAY)

    # Blur to remove camera sensor noise
    blur_base = cv2.GaussianBlur(gray_base, (7, 7), 0)
    blur_post = cv2.GaussianBlur(gray_post, (7, 7), 0)

    # Persistent difference mask
    diff = cv2.absdiff(blur_post, blur_base)
    _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    
    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Edge analysis to distinguish missing object vs structural damage
    edges_base = cv2.Canny(blur_base, 50, 150)
    edges_post = cv2.Canny(blur_post, 50, 150)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_base = baseline_frame.copy()
    annotated_post = post_frame.copy()
    
    detected_changes = []
    change_idx = 1
    
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_contour_area:
            x, y, bw, bh = cv2.boundingRect(c)
            roi_base_edges = edges_base[y:y+bh, x:x+bw]
            roi_post_edges = edges_post[y:y+bh, x:x+bw]
            
            base_edge_density = float(np.mean(roi_base_edges > 0))
            post_edge_density = float(np.mean(roi_post_edges > 0))
            
            # Classification: Missing object vs structural scene change
            # If baseline had edges/object that disappeared in post -> Missing Object
            if base_edge_density > post_edge_density * 1.3:
                change_type = "Missing / Removed Asset"
                sev = "HIGH"
                box_color = (0, 0, 255) # Red
                desc = "Object was present during baseline monitoring but has been removed or displaced."
            elif post_edge_density > base_edge_density * 1.3:
                change_type = "Structural Damage / Surface Disruption"
                sev = "HIGH"
                box_color = (0, 140, 255) # Orange
                desc = "New structural edge disruption, shattered element, or forced perimeter alteration."
            else:
                change_type = "Persistent Spatial Scene Alteration"
                sev = "MEDIUM"
                box_color = (0, 220, 255) # Yellow
                desc = "Significant permanent surface difference detected between pre and post incident states."

            # Draw on both frames
            label = f"SCN-{change_idx:02d}: {change_type.split('/')[0].strip()}"
            cv2.rectangle(annotated_base, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(annotated_base, f"ORIGINAL: {label}", (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.rectangle(annotated_post, (x, y), (x + bw, y + bh), box_color, 2)
            cv2.putText(annotated_post, f"ALTERED: {label}", (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

            detected_changes.append({
                "id": f"SCN-{change_idx:02d}",
                "type": change_type,
                "severity": sev,
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "area_percent": round((area / total_area) * 100, 2),
                "description": desc,
                "base_edges": round(base_edge_density, 3),
                "post_edges": round(post_edge_density, 3)
            })
            change_idx += 1

    # Visual Difference Mask Overlay (Jet Colormap)
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_MAGMA)
    diff_blended = cv2.addWeighted(post_frame, 0.55, diff_colored, 0.45, 0)

    has_scene_change = len(detected_changes) > 0
    missing_assets_count = sum(1 for it in detected_changes if "Missing" in it["type"])
    damage_count = sum(1 for it in detected_changes if "Structural" in it["type"])

    summary = (
        f"🚨 SCENE ALTERATION DETECTED: {len(detected_changes)} persistent modification(s) identified. "
        f"({missing_assets_count} missing property flags, {damage_count} structural damage zones)."
        if has_scene_change else
        "✅ SCENE INTEGRITY PRESERVED: No permanent missing objects or structural changes detected between baseline and post-incident states."
    )

    return {
        "has_scene_change": has_scene_change,
        "summary": summary,
        "changes_count": len(detected_changes),
        "missing_assets_count": missing_assets_count,
        "structural_damage_count": damage_count,
        "changes": detected_changes,
        "annotated_baseline_rgb": cv2.cvtColor(annotated_base, cv2.COLOR_BGR2RGB),
        "annotated_post_rgb": cv2.cvtColor(annotated_post, cv2.COLOR_BGR2RGB),
        "diff_blended_rgb": cv2.cvtColor(diff_blended, cv2.COLOR_BGR2RGB),
        "raw_diff_mask": thresh
    }

def analyze_video_scene_changes(video_path: str) -> Dict[str, Any]:
    """Convenience end-to-end wrapper for analyzing video scene changes."""
    base_bgr, post_bgr, base_ts, post_ts = extract_baseline_and_post_frames(video_path)
    res = detect_scene_changes(base_bgr, post_bgr)
    res["baseline_timestamp"] = base_ts
    res["post_timestamp"] = post_ts
    return res
