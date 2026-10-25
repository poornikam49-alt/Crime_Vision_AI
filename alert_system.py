"""
alert_system.py - Real-time Crime Alert UI, Audio Alert Generator, and Emergency Dispatch Simulator.
Provides interactive alerts, browser-based siren/chime triggers, and simulated law enforcement dispatch.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any

def render_audio_alert(severity: str = "CRITICAL", enabled: bool = True):
    """
    Renders an HTML5 Web Audio API synthesizer that plays an audible alert tone
    directly in the user's browser if enabled and severity is HIGH or CRITICAL.
    """
    if not enabled or severity not in ["CRITICAL", "HIGH"]:
        return

    freq1 = 880 if severity == "CRITICAL" else 660
    freq2 = 587 if severity == "CRITICAL" else 440
    duration = 0.35

    audio_html = f"""
    <script>
    (function() {{
        try {{
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            
            function playBeep(freq, startTime, dur) {{
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(freq, startTime);
                gain.gain.setValueAtTime(0.15, startTime);
                gain.gain.exponentialRampToValueAtTime(0.001, startTime + dur);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(startTime);
                osc.stop(startTime + dur);
            }}
            
            const now = ctx.currentTime;
            playBeep({freq1}, now, {duration});
            playBeep({freq2}, now + {duration}, {duration});
            playBeep({freq1}, now + {duration * 2}, {duration});
        }} catch(e) {{
            console.log("Audio alert blocked by browser autoplay policy until interaction:", e);
        }}
    }})();
    </script>
    """
    st.components.v1.html(audio_html, height=0, width=0)

def render_alert_banner(crime_result: Dict[str, Any]):
    """Renders a prominent, high-visibility tactical alert banner for detected crimes."""
    is_crime = crime_result["is_crime_detected"]
    severity = crime_result["severity"]
    primary_crime = crime_result["primary_crime"]
    confidence = crime_result["confidence"]
    alert_msg = crime_result["alert_message"]
    vd = crime_result.get("visual_dynamics", {})
    peak_ts = vd.get("peak_timestamp", 0.0)

    if not is_crime:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #102a1b 0%, #0d1f14 100%);
                        border: 2px solid #28a745; border-radius: 10px; padding: 18px 24px;
                        margin-bottom: 20px; box-shadow: 0 4px 15px rgba(40,167,69,0.2);">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="background-color: #28a745; color: white; padding: 4px 12px;
                                     border-radius: 6px; font-weight: bold; font-size: 0.85rem; letter-spacing: 1px;">
                            STATUS: NOMINAL
                        </span>
                        <h2 style="color: #4ade80; margin: 10px 0 4px 0; font-size: 1.5rem;">
                            🛡️ No Active Crime Detected
                        </h2>
                        <p style="color: #a7f3d0; margin: 0; font-size: 0.95rem;">
                            Surveillance monitoring confirms baseline activity. No violent motion spikes or perimeter breaches identified.
                        </p>
                    </div>
                    <div style="text-align: right; border-left: 1px solid #1e4d2b; padding-left: 20px;">
                        <div style="color: #86efac; font-size: 0.85rem;">SURVEILLANCE CONFIDENCE</div>
                        <div style="color: white; font-size: 1.8rem; font-weight: bold;">{confidence*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Color scheme according to severity
    if severity == "CRITICAL":
        border_col = "#ff1744"
        bg_gradient = "linear-gradient(135deg, #3d050b 0%, #1f0205 100%)"
        badge_bg = "#d50000"
        title_col = "#ff5252"
        shadow = "0 0 25px rgba(255, 23, 68, 0.45)"
        icon = "🚨"
    elif severity == "HIGH":
        border_col = "#ff9100"
        bg_gradient = "linear-gradient(135deg, #361700 0%, #1c0c00 100%)"
        badge_bg = "#e65100"
        title_col = "#ffab40"
        shadow = "0 0 20px rgba(255, 145, 0, 0.35)"
        icon = "⚠️"
    else: # MEDIUM
        border_col = "#ffd600"
        bg_gradient = "linear-gradient(135deg, #332b00 0%, #191500 100%)"
        badge_bg = "#f57f17"
        title_col = "#ffff00"
        shadow = "0 0 15px rgba(255, 214, 0, 0.3)"
        icon = "⚡"

    st.markdown(
        f"""
        <div style="background: {bg_gradient};
                    border: 2px solid {border_col}; border-radius: 12px; padding: 20px 26px;
                    margin-bottom: 22px; box-shadow: {shadow};">
            <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                <div style="flex: 1; min-width: 280px;">
                    <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 8px;">
                        <span style="background-color: {badge_bg}; color: white; padding: 4px 12px;
                                     border-radius: 6px; font-weight: 800; font-size: 0.85rem; letter-spacing: 1px;">
                            PRIORITY {severity} ALERT
                        </span>
                        <span style="color: #cbd5e1; font-size: 0.85rem;">
                            ⏱️ Incident Peak: <strong>{peak_ts:.2f}s</strong>
                        </span>
                    </div>
                    <h2 style="color: {title_col}; margin: 4px 0 8px 0; font-size: 1.65rem; font-weight: 800;">
                        {icon} CRIME DETECTED: {primary_crime.upper()}
                    </h2>
                    <p style="color: #e2e8f0; margin: 0; font-size: 1.0rem; line-height: 1.45;">
                        <strong>Alert Dispatch:</strong> {alert_msg}
                    </p>
                </div>
                <div style="text-align: right; background: rgba(0,0,0,0.35); padding: 12px 20px;
                            border-radius: 8px; border: 1px solid {border_col}44;">
                    <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">INCIDENT CONFIDENCE</div>
                    <div style="color: white; font-size: 2.1rem; font-weight: 900;">{confidence*100:.1f}%</div>
                    <div style="color: {title_col}; font-size: 0.8rem; font-weight: bold;">SEVERITY: {severity}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def simulate_emergency_dispatch(case_id: str, location: str, crime_result: Dict[str, Any]):
    """Simulates sending an urgent dispatch notification to emergency services or private security."""
    primary_crime = crime_result["primary_crime"]
    severity = crime_result["severity"]
    confidence = crime_result["confidence"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dispatch_payload = {
        "event": "CRIMEVISION_EMERGENCY_DISPATCH",
        "timestamp": now_str,
        "case_id": case_id or "AUTO-GEN-" + datetime.now().strftime("%H%M%S"),
        "location": location or "Zone 1 Main Surveillance Camera",
        "crime_type": primary_crime,
        "priority_level": severity,
        "ai_confidence": f"{confidence*100:.1f}%",
        "action_required": "IMMEDIATE LAW ENFORCEMENT & SECURITY DISPATCH",
        "recipient_channels": [
            "Emergency 911 / Police Dispatch CAD Link",
            "CCTV Command Control Center (SMS/Pager)",
            "On-Duty Security Supervisor Alert"
        ]
    }

    return dispatch_payload
