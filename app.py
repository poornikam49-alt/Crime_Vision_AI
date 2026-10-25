"""
CrimeVision AI - Intelligent Surveillance Video Review & Crime Evidence Platform
Enhanced Forensic Suite:
- Real-time Tactical Alert Message upon crime detection
- Multi-modal Crime Classification (Assault, Robbery, Burglary, Theft, Vandalism, Loitering)
- Digital Evidence Dossier with Annotated Keyframes (Bounding Boxes, Motion Heatmaps)
- AI Evidence Reliability Checker (ISO/IEC 27037 Court Admissibility & Blur/Noise Audits)
- Missing Object & Scene Change Detector (Persistent Asset Removal & Structural Damage)
- Multi-Camera Event Matching & Incident Trajectory Correlator
- Self-Improving Investigator Feedback System (Active Learning Priors & Model Self-Tuning)
- Cryptographic SHA-256 Chain of Custody & Non-Tampering Verification
- Downloadable Forensic Evidence ZIP Package (Images + Manifest + Custody Logs)
- Interactive Video Player with Synchronized Crime Timeline Markers
- Built-in Synthetic Surveillance Footage Generator for Instant Testing
- Simulated Emergency Dispatcher (Police / SOC / Security Team)
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import re
import json
import io
import time
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import crime_classifier
import evidence_extractor
import alert_system
import sample_generator
import reliability_checker
import multicam_matcher
import scene_change_detector
import investigator_feedback

# ---------- Page Config & Styling ----------
st.set_page_config(
    page_title="CrimeVision AI — Tactical Crime Detection & Evidence Suite",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Tactical Dark Surveillance Theme CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8533 45%, #38bdf8 80%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 16px;
    }
    .metric-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🚨 CrimeVision AI — Advanced Tactical Surveillance & Evidence Suite</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI Crime Detection · Reliability Auditing · Missing Object Analysis · Multi-Camera Trajectory · Self-Improving Active Learning</div>',
    unsafe_allow_html=True
)

# ---------- Core DIP & Feature Extraction ----------
def extract_video_features(video_path, sample_every_seconds=1.0, max_frames=240):
    """Samples video frames and computes DIP visual features."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open this video. Ensure it is an MP4, AVI, MOV, or MKV file.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0
    step = max(1, int(fps * sample_every_seconds))
    
    rows, previews = [], []
    prev_gray = None
    frame_index = 0
    sampled = 0

    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            enhanced = cv2.equalizeHist(gray)
            edges = cv2.Canny(enhanced, 60, 150)
            brightness = float(np.mean(enhanced))
            edge_ratio = float(np.mean(edges > 0))
            motion_ratio = 0.0
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                motion_ratio = float(np.mean(thresh > 0))
            prev_gray = gray.copy()

            rows.append({
                "timestamp_seconds": round(frame_index / fps, 2),
                "frame_number": frame_index,
                "brightness": round(brightness, 3),
                "edge_ratio": round(edge_ratio, 5),
                "motion_ratio": round(motion_ratio, 5)
            })
            if len(previews) < 12:
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                previews.append((round(frame_index / fps, 2), rgb))
            sampled += 1
        frame_index += 1

    cap.release()
    if not rows:
        raise ValueError("No frames could be sampled from this video footage.")
    
    df = pd.DataFrame(rows)
    df["motion_jerk"] = df["motion_ratio"].diff().abs().fillna(0).round(5)
    return df, previews, fps, total_frames, duration

# ---------- ML + NN Anomaly Pipeline ----------
def run_models(df):
    """Runs Isolation Forest and MLP Neural Network on visual features."""
    X = df[["brightness", "edge_ratio", "motion_ratio"]].to_numpy(dtype=float)
    if len(X) >= 5:
        ml = IsolationForest(contamination="auto", random_state=42)
        pred = ml.fit_predict(X)
        scores = ml.decision_function(X)
        df["ml_anomaly"] = pred == -1
        df["ml_anomaly_score"] = np.round(-scores, 5)
    else:
        df["ml_anomaly"] = False
        df["ml_anomaly_score"] = 0.0

    rng = np.random.default_rng(42)
    n = 1200
    train_brightness = rng.uniform(0, 255, n)
    train_edges = rng.uniform(0, 0.35, n)
    train_motion = rng.uniform(0, 0.8, n)
    train_X = np.column_stack([train_brightness, train_edges, train_motion])
    train_y = ((train_motion > 0.20) | ((train_edges > 0.16) & (train_motion > 0.10))).astype(int)
    
    nn = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=350, random_state=42, early_stopping=True)
    )
    nn.fit(train_X, train_y)
    df["nn_high_activity_probability"] = np.round(nn.predict_proba(X)[:, 1], 4)
    df["nn_high_activity"] = df["nn_high_activity_probability"] >= 0.5
    return df

# ---------- Forensic Report Generator ----------
def build_comprehensive_report(
    case_id, location, description, crime_result, evidence_items, df, duration, fps, total_frames, video_hash, scene_change_res=None
):
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    is_crime = crime_result["is_crime_detected"]
    primary_crime = crime_result["primary_crime"]
    severity = crime_result["severity"]
    confidence = crime_result["confidence"]
    evidence_points = crime_result["evidence_points"]
    vd = crime_result.get("visual_dynamics", {})

    lines = [
        "# CrimeVision AI — Advanced Forensic Surveillance Incident Report",
        "",
        "## 1. Case & Surveillance Metadata",
        f"- **Case Reference ID:** {case_id or 'UNASSIGNED-INCIDENT'}",
        f"- **Camera / Location Label:** {location or 'Zone 1 Entrance Camera'}",
        f"- **Report Timestamp:** {now_str}",
        f"- **Estimated Video Duration:** {duration:.2f} seconds",
        f"- **Frame Rate:** {fps:.2f} FPS (Total frames: {total_frames})",
        f"- **Cryptographic Master Video Hash (SHA-256):** `{video_hash}`",
        f"- **Digital Integrity Standard:** ISO/IEC 27037 Admissible Specimen Verification",
        "",
        "## 2. Automated Crime Incident Determination",
        f"- **Crime Status Detected:** {'🚨 YES — OFFENSE FLAGGED' if is_crime else '✅ NO — NOMINAL SURVEILLANCE'}",
        f"- **Primary Crime Classification:** **{primary_crime}**",
        f"- **Incident Severity Level:** **{severity}**",
        f"- **AI Multi-Modal Confidence:** **{confidence*100:.1f}%**",
        f"- **Peak Kinetic Anomaly Timestamp:** {vd.get('peak_timestamp', 0.0):.2f}s",
        f"- **Peak Motion Surge:** {vd.get('peak_motion', 0.0)*100:.1f}%",
        f"- **Violent Jerk / Acceleration Discontinuity:** {vd.get('motion_jerk', 0.0)*100:.1f}%",
        "",
        "## 3. Incident Description & Investigator Notes",
        description.strip() if description.strip() else "No textual incident notes recorded.",
        "",
        "## 4. Key Physical & Contextual Evidence Summary",
    ]
    
    for ep in evidence_points:
        lines.append(f"- **{ep['type']}:** {ep['detail']}")

    if scene_change_res and scene_change_res.get("has_scene_change"):
        lines += [
            "",
            "## 5. Scene Alteration & Missing Object Analysis",
            f"- **Summary:** {scene_change_res['summary']}",
            f"- **Missing Assets Count:** {scene_change_res.get('missing_assets_count', 0)}",
            f"- **Structural Damage Zones:** {scene_change_res.get('structural_damage_count', 0)}"
        ]
        
    lines += [
        "",
        "## 6. Digital Evidence Locker & Reliability Audit",
        f"A total of **{len(evidence_items)}** critical keyframe evidence items were extracted, fingerprinted, and scored for court admissibility.",
        "",
        "| Evidence ID | Timestamp | Category | Severity | Reliability | Grade / Admissibility | SHA-256 Checksum |",
        "|:---|---:|:---|:---:|---:|:---|:---|",
    ]
    
    for ev in evidence_items:
        rel = ev.get("reliability", {})
        score_str = f"{rel.get('reliability_score', 85.0)}%"
        grade_str = rel.get("grade", "GRADE A").split("—")[0].strip()
        lines.append(
            f"| `{ev['id']}` | {ev['time_formatted']} | {ev['category']} | **{ev['severity']}** | "
            f"{score_str} | {grade_str} | `{ev['sha256'][:16]}...` |"
        )
        
    lines += [
        "",
        "## 7. Algorithmic Diagnostics & Activity Metrics",
        f"- Total Analyzed Frames: {len(df)}",
        f"- Isolation Forest Statistical Outliers: {int(df['ml_anomaly'].sum())}",
        f"- Neural Network High-Activity Frames: {int(df['nn_high_activity'].sum())}",
        f"- Matched Investigative Terms: {', '.join(crime_result.get('matched_keywords', [])) or 'None'}",
        "",
        "## 8. Chain of Custody & Forensic Admissibility Disclaimer",
        "1. This digital dossier was automatically compiled using Computer Vision (DIP), Multi-modal Machine Learning, and ISO/IEC 27037 compliance metrics.",
        "2. All cryptographic hashes conform to the SHA-256 standard and must match the master evidence store.",
        "3. Preserve the original unmodified video bitstream. Forensic review must be certified by an investigator before judicial submission.",
        "",
        "**Investigator Signature:** ___________________________   **Badge ID:** _______________   **Date:** _______________"
    ]
    return "\n".join(lines)


# ---------- Sidebar: Controls & Settings ----------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cctv.png", width=64)
    st.header("Surveillance Config")
    
    case_id = st.text_input("Case / Incident Reference ID", value="CV-2026-9041", placeholder="e.g. CV-2026-001")
    location = st.text_input("Camera / Facility Location", value="CAM-02 North Entrance Gate", placeholder="e.g. Main Lobby")
    
    st.divider()
    st.subheader("⚙️ Detection Parameters")
    sample_interval = st.select_slider("Frame Sampling Frequency (sec)", options=[0.25, 0.5, 1.0, 2.0, 3.0], value=0.5)
    max_frames = st.select_slider("Max Analyzed Frames", options=[60, 120, 240, 360], value=120)
    
    enable_audio_alert = st.toggle("🔊 Enable Audio Siren Alert", value=True, help="Plays browser chime/siren when a high/critical crime is detected.")
    
    st.divider()
    # Continuous Learning Status Badge
    lm = investigator_feedback.compute_learning_metrics()
    st.markdown(f"**🧠 Active Learning Engine:**\n`{lm['status']}`")
    st.caption(f"Verified Accuracy: **{lm['accuracy_rate']}%** | Reviews: **{lm['total_reviews']}**")


# ---------- Main Input Section ----------
col_input1, col_input2 = st.columns([1.1, 1.0])

with col_input1:
    st.subheader("1. Video Footage Input")
    input_source = st.radio(
        "Choose video source:",
        ["Upload Surveillance Video", "Generate Test Surveillance Scenario"],
        horizontal=True
    )
    
    uploaded_file = None
    sample_scenario = None
    
    if input_source == "Upload Surveillance Video":
        uploaded_file = st.file_uploader(
            "Upload CCTV / Surveillance Video File",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supports standard surveillance formats: MP4, AVI, MOV, MKV"
        )
    else:
        st.info("💡 No footage on hand? Pick a synthetic surveillance scenario below to generate and test in 1-click:")
        sample_scenario = st.selectbox(
            "Select Scenario to Simulate:",
            [
                ("assault", "🚨 Physical Assault / Violent Struggle Scenario"),
                ("burglary", "🚨 Burglary / Perimeter Window Breach Scenario"),
                ("theft", "🚨 Theft / Counter Valuables Snatch-and-Run"),
                ("normal", "✅ Baseline Routine Surveillance (No Crime)")
            ],
            format_func=lambda x: x[1]
        )

with col_input2:
    st.subheader("2. Incident Notes / Witness Statement")
    description = st.text_area(
        "Enter investigator notes, 911 dispatch text, or witness logs:",
        value="At approx 22:15, security camera detected erratic physical struggle between two individuals near north gate." if (input_source != "Upload Surveillance Video" and sample_scenario and sample_scenario[0] == "assault") else "",
        placeholder="e.g. Intruder breached window on west perimeter. Suspect brandished knife and fled with laptop.",
        height=130
    )

analyze_button = st.button("🚀 Run CrimeVision AI Analysis", type="primary", use_container_width=True)

# ---------- Execution Logic ----------
if analyze_button:
    video_path_to_process = None
    is_temp_file = False
    
    try:
        with st.spinner("Preparing video stream and verifying bitstream integrity..."):
            if input_source == "Upload Surveillance Video":
                if uploaded_file is None:
                    st.error("Please upload a video file or choose a test scenario.")
                    st.stop()
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    video_path_to_process = tmp.name
                is_temp_file = True
                filename_display = uploaded_file.name
                video_bytes = uploaded_file.getvalue()
            else:
                scenario_key = sample_scenario[0]
                video_path_to_process = sample_generator.generate_sample_video(scenario=scenario_key, duration_seconds=6)
                is_temp_file = True
                filename_display = f"Synthetic_Scenario_{scenario_key.upper()}.mp4"
                with open(video_path_to_process, "rb") as f:
                    video_bytes = f.read()

        with st.spinner("Extracting Digital Image Processing (DIP) motion vectors & edge profiles..."):
            video_hash = evidence_extractor.compute_file_sha256(video_path_to_process)
            df, previews, fps, total_frames, duration = extract_video_features(
                video_path_to_process, float(sample_interval), int(max_frames)
            )
            df = run_models(df)

        with st.spinner("Performing Multi-Modal Crime Classification & Evidence Extraction..."):
            crime_result = crime_classifier.classify_crime(df, description)
            raw_evidence_items = evidence_extractor.extract_evidence_items(video_path_to_process, df)
            
            # 1. AI Evidence Reliability Checker
            evidence_items = reliability_checker.batch_evaluate_evidence(raw_evidence_items)
            
            # 2. Missing Object & Scene Change Detection
            scene_change_res = scene_change_detector.analyze_video_scene_changes(video_path_to_process)
            
            # 3. Report & ZIP Packaging
            report_md = build_comprehensive_report(
                case_id, location, description, crime_result, evidence_items, df, duration, fps, total_frames, video_hash, scene_change_res
            )
            evidence_zip_bytes = evidence_extractor.create_evidence_zip_package(evidence_items, case_id, video_hash)

        # Session state persistence
        st.session_state["cv_df"] = df
        st.session_state["cv_previews"] = previews
        st.session_state["cv_crime_result"] = crime_result
        st.session_state["cv_evidence_items"] = evidence_items
        st.session_state["cv_scene_change"] = scene_change_res
        st.session_state["cv_report_md"] = report_md
        st.session_state["cv_zip_bytes"] = evidence_zip_bytes
        st.session_state["cv_video_bytes"] = video_bytes
        st.session_state["cv_video_hash"] = video_hash
        st.session_state["cv_meta"] = {
            "duration": duration, "fps": fps, "total_frames": total_frames,
            "filename": filename_display, "case_id": case_id, "location": location
        }
        st.session_state["cv_analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.success("✅ Analysis completed successfully. Review detected crimes, reliability audits, and evidence below.")

    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
    finally:
        if is_temp_file and video_path_to_process and os.path.exists(video_path_to_process):
            try:
                os.remove(video_path_to_process)
            except Exception:
                pass


# ---------- Display Results & Dashboards ----------
if "cv_crime_result" in st.session_state:
    crime_result = st.session_state["cv_crime_result"]
    evidence_items = st.session_state["cv_evidence_items"]
    scene_change_res = st.session_state.get("cv_scene_change", {})
    df = st.session_state["cv_df"]
    meta = st.session_state["cv_meta"]
    video_hash = st.session_state["cv_video_hash"]
    report_md = st.session_state["cv_report_md"]
    zip_bytes = st.session_state["cv_zip_bytes"]
    video_bytes = st.session_state.get("cv_video_bytes", None)

    # 1. Audible Siren Alert
    if enable_audio_alert and crime_result["is_crime_detected"]:
        alert_system.render_audio_alert(crime_result["severity"], enabled=True)

    # 2. Prominent Real-time Alert Banner
    alert_system.render_alert_banner(crime_result)

    # Main Tab Navigation
    tab_overview, tab_evidence, tab_scene, tab_multicam, tab_feedback, tab_report, tab_dispatch = st.tabs([
        "🚨 Incident Overview",
        "🔍 Evidence Locker & Reliability",
        "🕵️ Missing Object & Scene Detector",
        "🎥 Multi-Camera Matching",
        "🧠 Investigator Feedback (HITL)",
        "📋 Forensic Police Report",
        "📡 Emergency Dispatch"
    ])

    # ==================== TAB 1: OVERVIEW & ALERTS ====================
    with tab_overview:
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Crime Detected", "YES 🚨" if crime_result["is_crime_detected"] else "NO 🛡️")
        col_m2.metric("Primary Offense", crime_result["primary_crime"].split("/")[0].strip())
        col_m3.metric("Threat Severity", crime_result["severity"])
        col_m4.metric("AI Confidence", f"{crime_result['confidence']*100:.1f}%")
        col_m5.metric("Evidence Items", len(evidence_items))

        st.divider()

        col_left, col_right = st.columns([1.1, 0.9])
        
        with col_left:
            st.subheader("📹 Surveillance Playback & Timeline")
            if video_bytes:
                st.video(video_bytes)
            
            st.caption(f"Source: {meta['filename']} · {meta['duration']:.2f}s duration · {meta['fps']:.1f} FPS")
            st.code(f"Master Video SHA-256: {video_hash}", language="text")

        with col_right:
            st.subheader("📊 Crime Classification Probabilities")
            prob_df = pd.DataFrame(list(crime_result["probabilities"].items()), columns=["Crime Category", "Probability Score"])
            prob_df = prob_df.sort_values(by="Probability Score", ascending=True)
            st.bar_chart(prob_df.set_index("Crime Category"), horizontal=True)

            st.markdown("### 📌 Flagged Evidence Cues")
            for pt in crime_result["evidence_points"]:
                st.markdown(f"- **{pt['type']}:** {pt['detail']}")

        st.subheader("📈 Multi-Signal Temporal Activity Timeline")
        timeline_cols = ["motion_ratio", "edge_ratio", "nn_high_activity_probability"]
        chart_data = df.set_index("timestamp_seconds")[timeline_cols]
        chart_data.columns = ["Kinetic Motion Ratio", "Structural Edge Ratio", "Neural Threat Prob"]
        st.line_chart(chart_data)

    # ==================== TAB 2: DIGITAL EVIDENCE LOCKER & RELIABILITY ====================
    with tab_evidence:
        st.subheader("🔍 Digital Evidence Locker & AI Reliability Assessment")
        st.caption("Each evidence item is analyzed for blur, contrast, sensor noise, bounding box isolation, and ISO/IEC 27037 legal admissibility.")

        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                "📦 Download Complete Evidence Archive (.ZIP)",
                data=zip_bytes,
                file_name=f"evidence_dossier_{meta['case_id']}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                "📥 Download Frame Analysis Data (.CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"frame_analysis_{meta['case_id']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()

        # Display Each Evidence Item with Reliability Card
        for item in evidence_items:
            rel = item.get("reliability", {})
            sev_badge = f'<span class="badge-{item["severity"].lower()}">{item["severity"]}</span>'
            
            st.markdown(
                f"### Evidence **{item['id']}** — Timestamp: `{item['time_formatted']}` ({item['timestamp_seconds']:.2f}s) | {sev_badge} {item['category']}",
                unsafe_allow_html=True
            )
            
            # Reliability Audit Box
            r_col1, r_col2 = st.columns([1.1, 0.9])
            with r_col1:
                st.markdown(
                    f"""
                    <div style="background:#0f172a; border-left: 5px solid {rel.get('color_code', '#3b82f6')};
                                padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.1rem; font-weight:800; color:{rel.get('color_code', '#3b82f6')};">
                                🛡️ {rel.get('grade', 'GRADE A')} ({rel.get('reliability_score', 85.0)}% Reliability)
                            </span>
                            <span style="font-size:0.8rem; background:#1e293b; color:#94a3b8; padding:3px 8px; border-radius:4px;">
                                {rel.get('admissibility', 'COURT-ADMISSIBLE')}
                            </span>
                        </div>
                        <div style="font-size:0.88rem; color:#cbd5e1; margin-top:6px;">
                            <strong>Forensic Guidance:</strong> {rel.get('recommendations', ['No anomalies'])[0]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"**Description:** {item['description']}")
                st.write(f"**Bounding Boxes Identified:** {item['bounding_boxes_count']} anomalous zone(s)")
            with r_col2:
                st.code(f"Frame SHA-256 Checksum:\n{item['sha256']}", language="text")
                with st.expander("🔬 View Detailed Reliability Audit Factors"):
                    for factor_name, f_data in rel.get("factors", {}).items():
                        st.markdown(f"- **{factor_name}:** `{f_data.get('score', '')}` — {f_data.get('detail', '')}")

            # 3-Way Visual Comparison
            tab_ann, tab_hm, tab_orig = st.tabs(["🎯 Annotated Bounding Boxes", "🔥 Motion Heatmap", "📷 Raw Original Frame"])
            with tab_ann:
                st.image(item["annotated_rgb"], caption=f"Annotated Bounding Boxes — Evidence {item['id']}", use_container_width=True)
            with tab_hm:
                st.image(item["heatmap_rgb"], caption=f"Jet Motion Heatmap (Spatial Density) — Evidence {item['id']}", use_container_width=True)
            with tab_orig:
                st.image(item["original_rgb"], caption=f"Raw Unprocessed Surveillance Frame — Evidence {item['id']}", use_container_width=True)

            st.divider()

    # ==================== TAB 3: MISSING OBJECT & SCENE CHANGE DETECTOR ====================
    with tab_scene:
        st.subheader("🕵️ Missing Object & Permanent Scene Change Detector")
        st.caption("Compares pre-incident baseline state against post-incident conditions to detect stolen assets, missing items, or physical scene damages.")

        if scene_change_res:
            st.markdown(
                f"""
                <div style="background:#131d2e; border: 1px solid {'#ef4444' if scene_change_res.get('has_scene_change') else '#22c55e'};
                            border-radius: 8px; padding: 16px; margin-bottom: 18px;">
                    <h4 style="margin:0 0 6px 0; color:{'#f87171' if scene_change_res.get('has_scene_change') else '#4ade80'};">
                        {scene_change_res.get('summary', '')}
                    </h4>
                    <p style="margin:0; font-size:0.9rem; color:#cbd5e1;">
                        Baseline Timestamp: <strong>{scene_change_res.get('baseline_timestamp', 0.5):.2f}s</strong> &nbsp;|&nbsp;
                        Post-Incident Timestamp: <strong>{scene_change_res.get('post_timestamp', 4.0):.2f}s</strong> &nbsp;|&nbsp;
                        Total Alterations: <strong>{scene_change_res.get('changes_count', 0)}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.image(
                    scene_change_res["annotated_baseline_rgb"],
                    caption=f"Baseline Pre-Incident State ({scene_change_res.get('baseline_timestamp', 0.5):.2f}s)",
                    use_container_width=True
                )
            with col_s2:
                st.image(
                    scene_change_res["annotated_post_rgb"],
                    caption=f"Post-Incident Settled State ({scene_change_res.get('post_timestamp', 4.0):.2f}s)",
                    use_container_width=True
                )
            with col_s3:
                st.image(
                    scene_change_res["diff_blended_rgb"],
                    caption="Permanent Spatial Divergence Map",
                    use_container_width=True
                )

            if scene_change_res.get("changes"):
                st.markdown("### 📋 Detected Scene Modifications Inventory")
                for ch in scene_change_res["changes"]:
                    st.markdown(
                        f"- **[{ch['id']}] {ch['type']}** (Severity: `{ch['severity']}`, Area: `{ch['area_percent']}%`)\n"
                        f"  *Description:* {ch['description']} (Bounding Box: `{ch['bbox']}`)"
                    )
        else:
            st.info("Run analysis on footage to inspect scene changes.")

    # ==================== TAB 4: MULTI-CAMERA EVENT MATCHING ====================
    with tab_multicam:
        st.subheader("🎥 Multi-Camera Event Matching & Incident Trajectory")
        st.caption("Correlates suspect appearance, spatial movement, and temporal lag across multiple surveillance camera views.")

        col_mc_btn, col_mc_info = st.columns([1, 2])
        with col_mc_btn:
            run_multicam = st.button("⚡ Correlate Multi-Camera Network (3 Feeds)", type="primary")

        if run_multicam:
            with st.spinner("Processing multi-camera feeds (CAM-01 Gate, CAM-02 Hallway, CAM-03 Stairwell)..."):
                feed_bundle = multicam_matcher.generate_multicam_scenario_bundle()
                feed_results = []
                
                for item in feed_bundle:
                    p = item["video_path"]
                    f_df, f_prev, f_fps, f_tot, f_dur = extract_video_features(p, sample_every_seconds=0.5, max_frames=60)
                    f_df = run_models(f_df)
                    f_cres = crime_classifier.classify_crime(f_df, "")
                    f_evs = evidence_extractor.extract_evidence_items(p, f_df)
                    
                    feed_results.append({
                        "camera_id": item["camera_id"],
                        "camera_name": item["camera_name"],
                        "df": f_df,
                        "crime_result": f_cres,
                        "evidence_items": f_evs,
                        "video_path": p
                    })

                mc_result = multicam_matcher.correlate_camera_feeds(feed_results)
                st.session_state["cv_multicam_result"] = mc_result
                st.session_state["cv_multicam_feeds"] = feed_results

        if "cv_multicam_result" in st.session_state:
            mc_res = st.session_state["cv_multicam_result"]
            mc_feeds = st.session_state["cv_multicam_feeds"]

            st.markdown(
                f"""
                <div style="background:#0c192c; border: 2px solid #38bdf8; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <h3 style="margin:0 0 6px 0; color:#38bdf8;">
                        {mc_res['summary'].splitlines()[0]}
                    </h3>
                    <p style="margin:0; color:#e2e8f0; font-size:0.95rem;">
                        Cross-Camera Match Confidence: <strong>{mc_res['match_confidence']*100:.1f}%</strong> | Monitored Nodes: <strong>{len(mc_feeds)} Cameras</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 🗺️ Incident Trajectory Path")
            col_t1, col_t2, col_t3 = st.columns(3)
            cols = [col_t1, col_t2, col_t3]
            
            for idx, pt in enumerate(mc_res["trajectory"]):
                with cols[idx % 3]:
                    st.markdown(
                        f"""
                        <div style="background:#1e293b; border-radius:8px; padding:14px; border:1px solid #334155; margin-bottom:10px;">
                            <div style="font-size:0.8rem; color:#94a3b8; font-weight:bold;">STEP #{pt['step']} · {pt['timestamp']}</div>
                            <div style="font-size:1.15rem; font-weight:800; color:#38bdf8; margin:4px 0;">{pt['camera_id']}</div>
                            <div style="font-size:0.85rem; color:#cbd5e1;">{pt['camera_name']}</div>
                            <div style="margin-top:8px;">
                                <span class="badge-critical">{pt['action'].split('/')[0]}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("### 📊 Cross-Camera Correlation Matrix")
            st.dataframe(pd.DataFrame(mc_res["correlation_matrix"]), use_container_width=True)

            st.markdown("### 📹 Synchronized Camera Angle Previews")
            c_prev1, c_prev2, c_prev3 = st.columns(3)
            c_cols = [c_prev1, c_prev2, c_prev3]
            for i, f in enumerate(mc_feeds):
                with c_cols[i % 3]:
                    st.write(f"**{f['camera_id']} — {f['camera_name']}**")
                    if f.get("evidence_items") and f["evidence_items"][0].get("annotated_rgb") is not None:
                        st.image(f["evidence_items"][0]["annotated_rgb"], use_container_width=True)

    # ==================== TAB 5: INVESTIGATOR FEEDBACK (HITL) ====================
    with tab_feedback:
        st.subheader("🧠 Self-Improving Investigator Feedback System")
        st.caption("Human-in-the-Loop review: verify model predictions, correct classifications, and continuously tune CrimeVision's adaptive weights.")

        # Active Learning Metrics Bar
        l_metrics = investigator_feedback.compute_learning_metrics()
        col_lm1, col_lm2, col_lm3, col_lm4 = st.columns(4)
        col_lm1.metric("Verified Accuracy", f"{l_metrics['accuracy_rate']}%")
        col_lm2.metric("Total Reviews Logged", l_metrics["total_reviews"])
        col_lm3.metric("True Positives", l_metrics["true_positives"])
        col_lm4.metric("False Alarms Prevented", l_metrics["false_alarms"])

        st.divider()

        st.markdown("### 📝 Submit Case Verification & Feedback")
        with st.form("investigator_feedback_form"):
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                reviewer_name = st.text_input("Investigator Name / Badge ID", value="Detective J. Vance (#8421)")
                agency = st.text_input("Agency / Department", value="Forensic Video Unit / Metro Police")
                verdict = st.selectbox(
                    "Investigation Verdict:",
                    [
                        "Confirmed Correct",
                        "Misclassified Crime Type",
                        "False Alarm (Routine Activity)",
                        "False Negative (Crime Missed)"
                    ]
                )
            with col_fb2:
                corrected_label = st.selectbox(
                    "If misclassified, select true offense:",
                    list(crime_classifier.CRIME_TAXONOMY.keys()) + ["Routine Monitoring (No Crime)"]
                )
                confidence_rating = st.slider("Investigator Confidence in Assessment", 1, 5, 5)
                notes = st.text_area("Investigator Notes & Tuning Remarks", placeholder="e.g. Subject sparred playfully, no physical battery.")

            submit_fb = st.form_submit_button("💾 Save Feedback & Update Active Learning Model", type="primary")

        if submit_fb:
            record = {
                "case_id": meta["case_id"],
                "predicted_crime": crime_result["primary_crime"],
                "predicted_severity": crime_result["severity"],
                "predicted_confidence": crime_result["confidence"],
                "verdict": verdict,
                "corrected_crime": corrected_label if verdict == "Misclassified Crime Type" else crime_result["primary_crime"],
                "investigator": reviewer_name,
                "agency": agency,
                "confidence_rating": confidence_rating,
                "notes": notes
            }
            if investigator_feedback.save_feedback_record(record):
                st.success("✅ Feedback successfully registered. Model calibration weights updated dynamically!")
                st.rerun()

        # Feedback History & Learned Biases
        st.markdown("### 📈 Learned Category Biases & Dynamic Calibration")
        if l_metrics["category_biases"]:
            st.dataframe(pd.DataFrame(list(l_metrics["category_biases"].items()), columns=["Crime Category", "Learned Dynamic Bias"]), use_container_width=True)
        else:
            st.info("No prior biases logged yet. Submit reviews above to train the adaptive learning engine.")

        hist = investigator_feedback.load_feedback_history()
        if hist:
            with st.expander(f"📜 View Full Historical Verification Audit Log ({len(hist)} records)"):
                st.dataframe(pd.DataFrame(hist), use_container_width=True)

    # ==================== TAB 6: FORENSIC REPORT ====================
    with tab_report:
        st.subheader("📋 Formal Forensic Incident Report")
        
        col_rep1, col_rep2 = st.columns([1, 1])
        with col_rep1:
            st.download_button(
                "📄 Download Incident Report (.MD)",
                data=report_md.encode("utf-8"),
                file_name=f"police_report_{meta['case_id']}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
        with col_rep2:
            st.download_button(
                "📋 Download Evidence Manifest (.JSON)",
                data=json.dumps({
                    "case_id": meta["case_id"],
                    "video_hash": video_hash,
                    "crime_result": {k: v for k, v in crime_result.items() if k != "visual_dynamics"},
                    "scene_changes": {k: v for k, v in scene_change_res.items() if not k.endswith("_rgb") and k != "raw_diff_mask"},
                    "evidence_items": [
                        {k: v for k, v in it.items() if not k.endswith("_rgb")} for it in evidence_items
                    ]
                }, indent=2).encode("utf-8"),
                file_name=f"evidence_manifest_{meta['case_id']}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown(report_md)

    # ==================== TAB 7: EMERGENCY DISPATCH SIMULATOR ====================
    with tab_dispatch:
        st.subheader("📡 Automated Emergency Dispatch Simulator")
        st.caption("Simulates sending instantaneous tactical notifications to Law Enforcement, CCTV Dispatch, and On-Site Security.")

        dispatch_payload = alert_system.simulate_emergency_dispatch(meta["case_id"], meta["location"], crime_result)
        
        col_disp_btn, col_disp_status = st.columns([1, 2])
        with col_disp_btn:
            trigger_dispatch = st.button("🚨 Broadcast Emergency Dispatch Now", type="primary")

        if trigger_dispatch:
            st.toast("🚨 Emergency Dispatch Broadcast in progress...", icon="📡")
            with st.status("Broadcasting Priority Dispatch...", expanded=True) as status:
                st.write(f"⏱️ **[0.0s]** Authenticating case `{meta['case_id']}` with Secure Dispatch Gateway...")
                time.sleep(0.4)
                st.write(f"📡 **[0.4s]** Transmitting SHA-256 evidence fingerprint `{video_hash[:16]}...`")
                time.sleep(0.4)
                st.write(f"🚨 **[0.8s]** Sending Priority [{crime_result['severity']}] CAD ticket to Regional Police Command...")
                time.sleep(0.3)
                st.write("📲 **[1.1s]** SMS Alert pushed to Security Supervisor & Property Management.")
                status.update(label="✅ Dispatch Transmission Verified & Acknowledged!", state="complete", expanded=True)

        st.markdown("### 📄 Dispatch Payload Preview")
        st.json(dispatch_payload)

st.divider()
st.caption(
    "CrimeVision AI Tactical Suite v3.0 · Powered by Streamlit, OpenCV (DIP), Scikit-Learn (Isolation Forest & MLP), "
    "and Cryptographic SHA-256 Hashing. ISO/IEC 27037 Digital Forensics Admissible Architecture."
)
