"""
sample_generator.py - Generates realistic synthetic surveillance video clips
for testing CrimeVision AI crime detection and evidence analysis.
"""

import cv2
import numpy as np
import tempfile
import os

def create_surveillance_background(width=480, height=270):
    """Creates a realistic surveillance camera background."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Wall (top 60%)
    img[0:int(height * 0.65), :] = [60, 65, 70]
    # Floor (bottom 35%)
    img[int(height * 0.65):, :] = [45, 50, 52]
    # Floor line / skirting board
    cv2.line(img, (0, int(height * 0.65)), (width, int(height * 0.65)), (90, 95, 100), 2)
    # Door frame on right
    cv2.rectangle(img, (int(width * 0.72), int(height * 0.2)), (int(width * 0.88), int(height * 0.65)), (40, 45, 48), -1)
    cv2.rectangle(img, (int(width * 0.72), int(height * 0.2)), (int(width * 0.88), int(height * 0.65)), (120, 125, 130), 2)
    # Window / restricted zone on left
    cv2.rectangle(img, (int(width * 0.12), int(height * 0.18)), (int(width * 0.32), int(height * 0.45)), (110, 120, 130), 2)
    cv2.line(img, (int(width * 0.22), int(height * 0.18)), (int(width * 0.22), int(height * 0.45)), (110, 120, 130), 1)
    return img

def draw_person(frame, x, y, size=1.0, color=(180, 180, 190)):
    """Draws a stylized surveillance figure."""
    head_r = int(10 * size)
    torso_h = int(35 * size)
    torso_w = int(14 * size)
    # Head
    cv2.circle(frame, (int(x), int(y - torso_h - head_r)), head_r, color, -1)
    # Torso
    cv2.rectangle(frame, (int(x - torso_w // 2), int(y - torso_h)), (int(x + torso_w // 2), int(y - torso_h // 3)), color, -1)
    # Legs
    cv2.line(frame, (int(x - 4 * size), int(y - torso_h // 3)), (int(x - 6 * size), int(y)), color, int(4 * size))
    cv2.line(frame, (int(x + 4 * size), int(y - torso_h // 3)), (int(x + 6 * size), int(y)), color, int(4 * size))

def generate_sample_video(scenario="assault", duration_seconds=6, fps=20):
    """
    Generates a sample video for a specified crime scenario.
    Returns the temporary file path of the generated MP4.
    """
    width, height = 480, 270
    total_frames = int(duration_seconds * fps)
    
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    bg = create_surveillance_background(width, height)
    rng = np.random.default_rng(101)
    
    for f in range(total_frames):
        t = f / fps
        frame = bg.copy()
        
        # Add subtle scanline noise without blow-up
        if f % 4 == 0:
            frame[::8, :] = np.clip(frame[::8, :].astype(np.int16) - 5, 0, 255).astype(np.uint8)
        
        if scenario == "assault":
            # Walking towards each other, then violent struggle at t=2.0 - 4.5s
            if t < 2.0:
                p1_x = 100 + t * 45
                p2_x = 380 - t * 45
                draw_person(frame, p1_x, 210, 1.0, (190, 170, 160))
                draw_person(frame, p2_x, 210, 1.0, (140, 150, 200))
            elif 2.0 <= t < 4.5:
                # Rapid erratic motion (violent struggle)
                jitter1 = rng.uniform(-18, 18)
                jitter2 = rng.uniform(-18, 18)
                p1_x = 220 + jitter1
                p2_x = 235 + jitter2
                draw_person(frame, p1_x, 210 + rng.uniform(-6, 6), 1.0, (210, 140, 140))
                draw_person(frame, p2_x, 210 + rng.uniform(-6, 6), 1.0, (130, 130, 220))
                # Sudden arm flailing / impact lines
                cv2.line(frame, (int(p1_x), 170), (int(p2_x + jitter1), 165), (220, 200, 180), 3)
            else:
                # Fleeing / post-incident
                p1_x = 220 - (t - 4.5) * 80
                draw_person(frame, max(40, p1_x), 210, 1.0, (190, 170, 160))
                # Victim down
                cv2.rectangle(frame, (230, 200), (270, 215), (130, 130, 220), -1)
                
        elif scenario == "burglary":
            # Perimeter intrusion: Empty room until t=2.0s, then window breach
            if t >= 2.0:
                # Breach at window (x=110, y=100)
                intruder_x = 110 + (t - 2.0) * 35
                intruder_y = 120 + min(90, (t - 2.0) * 40)
                # Flashlight beam cone
                pts = np.array([[int(intruder_x), int(intruder_y - 20)],
                                [int(intruder_x + 90), int(intruder_y + 40)],
                                [int(intruder_x + 60), int(intruder_y + 70)]], np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (180, 220, 240))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                draw_person(frame, intruder_x, intruder_y, 0.9, (80, 80, 90))
                
        elif scenario == "theft":
            # Bag/Valuable on counter; thief grabs and runs
            # Counter / table
            cv2.rectangle(frame, (190, 180), (280, 215), (70, 75, 80), -1)
            if t < 2.5:
                # Item present
                cv2.rectangle(frame, (225, 168), (245, 180), (40, 180, 220), -1)
                # Person approaching
                p_x = 120 + t * 40
                draw_person(frame, p_x, 210, 1.0, (160, 170, 180))
            elif 2.5 <= t < 3.2:
                # Grab motion
                cv2.rectangle(frame, (230, 168), (250, 180), (40, 180, 220), -1)
                draw_person(frame, 220, 210, 1.0, (160, 170, 180))
                cv2.line(frame, (220, 175), (235, 175), (200, 190, 180), 3)
            else:
                # Item stolen, rapid sprint away
                p_x = 220 + (t - 3.2) * 110
                draw_person(frame, min(470, p_x), 210, 1.0, (160, 170, 180))
                
        else: # "normal"
            # Steady walking across scene
            p_x = 60 + t * 55
            draw_person(frame, p_x, 210, 1.0, (170, 180, 190))
            
        # Add CCTV overlay stamp
        timecode = f"CAM-03 [LIVE REC] 2026-10-24 22:{14 + int(t//60):02d}:{int(t%60):02d}.{int((t*10)%10)}"
        cv2.putText(frame, timecode, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (width - 25, 20), 5, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 55, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        out.write(frame)
        
    out.release()
    return out_path
