"""
Forest Fire AI Monitoring System — Enterprise Edition
Unified Real-Time Vision, Incident State Machine, Multi-Channel Alert & PDF Analytics Platform.

Channels: Telegram & Email ONLY (SMS / Twilio completely removed).
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

# Load environment configuration
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH, override=True)
    else:
        load_dotenv()
except ImportError:
    pass

# Database integration
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    from mysql.connector import IntegrityError
except ImportError:
    mysql = None
    MySQLError = Exception
    IntegrityError = Exception

from src.database import (
    add_forest_officer,
    get_active_officers,
    get_all_officers,
    get_database_connection,
    get_db_config,
    get_recent_alerts,
    queue_alert,
    test_connection,
    update_alert_status,
    update_recipient_status,
)

from src.arduino_manager import (
    ArduinoManager,
    arduino_engine,
    list_available_ports,
)
from src.incident_manager import (
    IncidentManager,
    generate_all_12_charts,
    generate_incident_analytics_pdf,
    get_alert_cooldown_seconds,
    get_confidence_threshold,
    get_min_fire_frames,
    get_min_no_fire_frames,
    incident_engine,
    is_pc_alarm_active,
    is_pc_alarm_test_active,
    stop_pc_alarm,
    test_pc_alarm,
    trigger_pc_alarm,
)

from src.notifications import (
    dispatch_emergency_alert,
    format_google_maps_url,
    generate_incident_pdf_report,
    get_alert_dispatch_history,
    get_cooldown_status,
    get_latest_alert_status,
    get_system_config_status,
    handle_confirmed_fire_alert,
    handle_final_pdf_on_resolution,
    handleFireAlert,
    reset_cooldown,
    send_email_alert,
    send_telegram_message,
    start_alert_worker,
    validate_e164_phone,
    verify_telegram_bot,
)

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

try:
    from streamlit_js_eval import get_geolocation
except ImportError:
    get_geolocation = None


# ============================================================
# PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="Forest Fire AI Monitoring System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Light Theme Override */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.8) !important;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #b91c1c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: #0f172a;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }
    /* Buttons and Card Styling */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
    }
    .alarm-banner-active {
        background: linear-gradient(90deg, #dc2626, #b91c1c);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
    }
    .alarm-banner-safe {
        background: linear-gradient(90deg, #16a34a, #15803d);
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ASSETS & MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
ALARM_MP3 = ASSETS_DIR / "alarm.mp3"
ALARM_WAV = ASSETS_DIR / "alarm.wav"


# ============================================================
# PERSISTENT SESSION STATE INITIALIZATION
# ============================================================

if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.latitude = None
    st.session_state.longitude = None
    st.session_state.accuracy = None
    st.session_state.camera_running = False
    st.session_state.alarm_test_message = None
    st.session_state.alarm_test_timestamp = 0.0
    st.session_state.telegram_test_message = None
    st.session_state.email_test_message = None
    st.session_state.camera_mode = "Direct USB/Laptop Webcam"
    st.session_state.uploaded_video_stats = None
    st.session_state.buzzer_test_message = None
    st.session_state.arduino_msg = None
    st.session_state.arduino_error = None
    st.session_state.arduino_selected_port = None
    st.session_state.arduino_auto_attempted = False

# Auto-probe and connect to detected Arduino on initial load if available
if not arduino_engine.is_connected and not st.session_state.get("arduino_auto_attempted", False):
    st.session_state.arduino_auto_attempted = True
    avail_ports = list_available_ports()
    arduino_candidates = [p["port"] for p in avail_ports if p.get("is_arduino")]
    if arduino_candidates:
        cand_port = arduino_candidates[0]
        st.session_state.arduino_selected_port = cand_port
        ok, err = arduino_engine.connect(cand_port, baud_rate=9600)
        if ok:
            st.session_state.arduino_msg = f"Connected to {cand_port}"
            st.session_state.arduino_error = None
        else:
            st.session_state.arduino_error = err

# Ensure background alert dispatcher thread is running (Point 4)
start_alert_worker()


# Thread-safe location store
LOCATION_LOCK = threading.Lock()
LOCATION = {
    "latitude": None,
    "longitude": None,
    "accuracy": None,
    "url": "GPS location unavailable",
}


def getCurrentLocation() -> Dict[str, Any]:
    """Retrieve GPS coordinates from browser or session state."""
    lat = st.session_state.get("latitude")
    lon = st.session_state.get("longitude")
    acc = st.session_state.get("accuracy")

    if (lat is None or lon is None) and get_geolocation is not None:
        try:
            browser_loc = get_geolocation()
            if isinstance(browser_loc, dict):
                coords = browser_loc.get("coords", {})
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                acc = coords.get("accuracy")
                if lat is not None and lon is not None:
                    st.session_state.latitude = lat
                    st.session_state.longitude = lon
                    st.session_state.accuracy = acc
        except Exception:
            pass

    if lat is not None and lon is not None:
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            acc_f = float(acc) if acc is not None else None
            url, status = format_google_maps_url(lat_f, lon_f)
            with LOCATION_LOCK:
                LOCATION["latitude"] = lat_f
                LOCATION["longitude"] = lon_f
                LOCATION["accuracy"] = acc_f
                LOCATION["url"] = url
            return {"latitude": lat_f, "longitude": lon_f, "accuracy": acc_f, "url": url, "status": "AVAILABLE"}
        except (ValueError, TypeError):
            pass

    return {"latitude": None, "longitude": None, "accuracy": None, "url": "GPS location unavailable", "status": "UNAVAILABLE"}


# ============================================================
# YOLO MODEL LOADER
# ============================================================

@st.cache_resource
def load_yolo_model():
    """Load and cache YOLO model."""
    if not MODEL_PATH.exists():
        return None
    try:
        model = YOLO(str(MODEL_PATH))
        return model
    except Exception as e:
        print(f"[YOLO] Model load error: {e}")
        return None


MODEL = load_yolo_model()


def is_flame_color_present(box_crop: np.ndarray, min_flame_ratio: float = 0.035) -> Tuple[bool, float]:
    """
    Verify whether a detected bounding box contains genuine flame-colored pixels
    (warm red/orange/yellow spectrum, high saturation, and high value/luminance).
    Filters out false positives on plain walls, furniture, lights, clothes, and skin.
    """
    if box_crop is None or box_crop.size == 0:
        return False, 0.0

    try:
        hsv = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
        # Warm flame range: Hue in [0, 35] (red, orange, yellow) or [160, 180] (deep flame red)
        # Saturation >= 45 (rejects white walls, grayish reflections, monochrome surfaces)
        # Value >= 90 (rejects dark shadows and dark objects)
        mask1 = cv2.inRange(hsv, np.array([0, 45, 90]), np.array([35, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 45, 90]), np.array([180, 255, 255]))
        flame_mask = cv2.bitwise_or(mask1, mask2)

        total_pixels = box_crop.shape[0] * box_crop.shape[1]
        if total_pixels == 0:
            return False, 0.0

        flame_ratio = np.count_nonzero(flame_mask) / float(total_pixels)
        return (flame_ratio >= min_flame_ratio), flame_ratio
    except Exception:
        return True, 1.0


def detect_fire_in_frame(
    frame: np.ndarray,
    conf_thresh: float = 0.55,
    min_box_area_ratio: float = 0.003,
) -> Tuple[np.ndarray, List[Dict[str, Any]], float, str]:
    """
    Run YOLO inference on a single frame.
    Returns (annotated_frame, detections_list, max_fire_confidence, dominant_label).
    """
    if MODEL is None or frame is None:
        return frame, [], 0.0, "normal"

    h, w = frame.shape[:2]
    total_area = float(h * w) if (h > 0 and w > 0) else 1.0

    detections = []
    max_fire_conf = 0.0
    has_smoke = False
    annotated = frame.copy()

    try:
        results = MODEL.predict(frame, conf=conf_thresh, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = MODEL.names.get(cls_id, "unknown").lower()
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                # Filter tiny noise boxes and extreme slivers
                bx_w = max(0, xyxy[2] - xyxy[0])
                bx_h = max(0, xyxy[3] - xyxy[1])
                area_ratio = (bx_w * bx_h) / total_area

                if area_ratio < min_box_area_ratio or area_ratio > 0.98:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 1, min(w, x2))
                y2 = max(y1 + 1, min(h, y2))
                box_crop = frame[y1:y2, x1:x2]

                # Trained model has {0: 'Fire'}
                if "fire" in cls_name or cls_name == "0":
                    # Color verification: reject false positive boxes on plain walls/furniture/lights without flame hues
                    is_flame, flame_ratio = is_flame_color_present(box_crop)
                    if not is_flame and conf < 0.85:
                        continue

                    if conf > max_fire_conf:
                        max_fire_conf = conf
                    label_text = f"FIRE {conf:.1%}"
                    color = (0, 0, 255)  # Red (BGR)
                    detections.append({"class": "FIRE", "confidence": conf, "box": [x1, y1, x2, y2]})
                else:
                    label_text = f"{cls_name.upper()} {conf:.1%}"
                    color = (0, 255, 0)
                    detections.append({"class": cls_name.upper(), "confidence": conf, "box": [x1, y1, x2, y2]})

                # Draw bounding box & label HUD
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label_text,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        dominant_label = "fire" if max_fire_conf >= conf_thresh else "normal"
        return annotated, detections, max_fire_conf, dominant_label

    except Exception as exc:
        print(f"[YOLO] Frame prediction error: {exc}")
        return frame, [], 0.0, "normal"


# ============================================================
# WEBRTC VIDEO FRAME CALLBACK
# ============================================================

def webrtc_video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Thread-safe WebRTC frame callback."""
    img = frame.to_ndarray(format="bgr24")
    if img is None:
        return frame

    thresh = get_confidence_threshold()
    annotated, detections, max_conf, label = detect_fire_in_frame(img, conf_thresh=thresh)

    with LOCATION_LOCK:
        loc_lat = LOCATION["latitude"]
        loc_lon = LOCATION["longitude"]
        loc_acc = LOCATION["accuracy"]

    # Process frame into central incident engine
    incident_engine.process_frame(
        image=img,
        detections=detections,
        fire_confidence=max_conf,
        label=label,
        detection_mode="LIVE CAMERA (WebRTC)",
        latitude=loc_lat,
        longitude=loc_lon,
        accuracy=loc_acc,
        conf_threshold=thresh,
    )

    # Automatic buzzer actuation during WebRTC fire detection
    if max_conf >= thresh and label == "fire":
        try:
            arduino_engine.notify_fire_alert(True)
        except Exception:
            pass

    # Overlay MQ-2 Sensor HUD
    try:
        hw_s = arduino_engine.get_status()
        if hw_s["is_connected"] and hw_s.get("has_reading", False):
            s_val = hw_s['latest_mq2_value']
            s_stat = hw_s.get('sensor_status', 'NORMAL')
            thresh = int(hw_s.get('mq2_threshold', 300))
            b_lbl = "BUZZER:ON" if hw_s.get("buzzer_active") else "BUZZER:OFF"
            hud_text = f"MQ-2: {s_val} / 1023 [{s_stat}] | Thresh: {thresh} | {b_lbl}"
            hud_color = (0, 0, 255) if s_stat in ("WARNING", "CRITICAL") else (0, 255, 0)
            cv2.putText(annotated, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
    except Exception:
        pass

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ============================================================
# HEADER & ALARM STATE HUD
# ============================================================

st.markdown('<div class="main-header">🔥 Forest Fire AI Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Real-Time AI Computer Vision Surveillance & Rapid Incident Response</div>', unsafe_allow_html=True)

# Location status
current_location = getCurrentLocation()

# Alarm State Display (Point 7)
alarm_is_active = is_pc_alarm_active()
if alarm_is_active:
    st.markdown('<div class="alarm-banner-active">🚨 FIRE ALARM ACTIVE &bull; INCIDENT STATE DETECTED</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alarm-banner-safe">🟢 FIRE ALARM STOPPED &bull; NORMAL SURVEILLANCE STATE</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR CONTROLS & DIAGNOSTIC TESTS
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    
    # Independent Channel Diagnostics (Point 17)
    st.markdown("#### 🧪 Channel Diagnostics & Tests")
    st.caption("Perform isolated tests without modifying fire incident statistics.")

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        if st.button("🔔 TELEGRAM", width="stretch", help="Test Telegram bot dispatch"):
            st.session_state.telegram_test_message = "Sending Telegram test message..."
            ok, err = send_telegram_message("🔔 <b>TELEGRAM DIAGNOSTIC TEST</b>\nForest Fire System Telegram connectivity verified successfully.")
            if ok:
                st.session_state.telegram_test_message = "✅ Telegram test delivered!"
            else:
                st.session_state.telegram_test_message = f"❌ Telegram test failed: {err}"
            st.rerun()

    with t_col2:
        if st.button("📧 EMAIL", width="stretch", help="Test Email SMTP dispatch"):
            st.session_state.email_test_message = "Sending Email test message..."
            ok, err = send_email_alert(
                subject="📧 FOREST FIRE SYSTEM - SMTP DIAGNOSTIC TEST",
                body="This is an independent SMTP email diagnostic test from the Forest Fire Monitoring System.",
            )
            if ok:
                st.session_state.email_test_message = "✅ Email test delivered!"
            else:
                st.session_state.email_test_message = f"❌ Email test failed: {err}"
            st.rerun()

    with t_col3:
        if st.button("🚨 BUZZER", width="stretch", help="Test physical Arduino buzzer"):
            if arduino_engine.is_connected:
                ok = arduino_engine.test_buzzer(duration_seconds=2.0)
                if ok:
                    st.session_state.buzzer_test_message = "🚨 Arduino Buzzer triggered!"
                else:
                    st.session_state.buzzer_test_message = "⚠️ Buzzer busy / cooldown"
            else:
                st.session_state.buzzer_test_message = "⚠️ Arduino not connected"
            st.rerun()

    if st.session_state.telegram_test_message:
        if "✅" in st.session_state.telegram_test_message:
            st.success(st.session_state.telegram_test_message)
        else:
            st.warning(st.session_state.telegram_test_message)

    if st.session_state.email_test_message:
        if "✅" in st.session_state.email_test_message:
            st.success(st.session_state.email_test_message)
        else:
            st.warning(st.session_state.email_test_message)

    if st.session_state.get("buzzer_test_message"):
        if "🚨" in st.session_state.buzzer_test_message or "🔔" in st.session_state.buzzer_test_message:
            st.success(st.session_state.buzzer_test_message)
        else:
            st.warning(st.session_state.buzzer_test_message)
        st.session_state.buzzer_test_message = None

    st.markdown("---")
    st.markdown("#### 🔌 Arduino Hardware Hub")
    st.caption("Arduino Uno + MQ-2 Smoke Sensor (A0) + Buzzer (Pin 8)")

    if st.session_state.get("arduino_error"):
        st.error(st.session_state.arduino_error)
    elif st.session_state.get("arduino_msg"):
        st.success(st.session_state.arduino_msg)

    hw_status = arduino_engine.get_status()
    ports = list_available_ports()
    port_options = [p["port"] for p in ports]

    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        if port_options:
            default_idx = 0
            if hw_status["port"] in port_options:
                default_idx = port_options.index(hw_status["port"])
            elif st.session_state.get("arduino_selected_port") in port_options:
                default_idx = port_options.index(st.session_state.arduino_selected_port)
            selected_port = st.selectbox(
                "Serial COM Port",
                options=port_options,
                index=default_idx,
                format_func=lambda x: f"{x} ({next((p['description'] for p in ports if p['port'] == x), '')})",
                label_visibility="collapsed",
            )
            st.session_state.arduino_selected_port = selected_port
        else:
            selected_port = None
            st.selectbox("Serial COM Port", options=["No COM ports detected"], disabled=True, label_visibility="collapsed")
    with c_col2:
        if st.button("🔄 Scan", width="stretch", help="Scan for USB serial ports"):
            st.session_state.arduino_auto_attempted = False
            st.session_state.arduino_error = None
            st.rerun()

    # Connect / Disconnect Action
    if hw_status["is_connected"]:
        st.success(f"🟢 Connected: `{hw_status['port']}`")
        if st.button("🔴 Disconnect Hardware", width="stretch"):
            arduino_engine.disconnect()
            st.session_state.arduino_msg = "Arduino disconnected."
            st.session_state.arduino_error = None
            st.rerun()
    else:
        st.info("⚪ Disconnected (Vision-Only Active)")
        if st.button("🔌 Connect Arduino", width="stretch", disabled=not selected_port, type="primary"):
            if selected_port:
                ok, err = arduino_engine.connect(selected_port, baud_rate=9600)
                if ok:
                    st.session_state.arduino_msg = f"Connected to {selected_port}"
                    st.session_state.arduino_error = None
                else:
                    st.session_state.arduino_msg = None
                    st.session_state.arduino_error = err
                st.rerun()

    # Live MQ-2 Sensor Range Display (0 – 1023 ADC)
    # Uses @st.fragment(run_every=...) so this block auto-reruns every 1.5 s
    # and always shows the latest reading from the background serial thread.
    @st.fragment(run_every=1.5)
    def _mq2_live_telemetry():
        hw_s = arduino_engine.get_status()
        is_conn = hw_s["is_connected"]
        has_reading = hw_s.get("has_reading", False)
        current_thresh_f = min(1023, max(0, int(hw_s["smoke_threshold"])))
        buzz_active_f = hw_s["buzzer_active"]
        b_mode_f = hw_s.get("buzzer_mode", "PASSIVE_TONE")
        buzz_badge = f"🔊 BUZZER ACTIVE ({b_mode_f} on D8)" if buzz_active_f else "🔇 BUZZER OFF (Pin D8 SILENT)"

        st.markdown("##### 📟 Live MQ-2 Sensor Telemetry")
        if is_conn and has_reading:
            val_f = int(hw_s["latest_mq2_value"])
            progress_f = min(1.0, max(0.0, val_f / 1023.0))
            stat_f = hw_s.get("sensor_status", "NORMAL")
            stat_badge = "🚨 CRITICAL" if stat_f == "CRITICAL" else ("⚠️ WARNING" if stat_f == "WARNING" else "🟢 NORMAL")
            st.markdown(f"**Sensor Reading:** `MQ-2: {val_f} / 1023` &bull; {stat_badge}")
            st.markdown(f"**Selected Threshold:** `MQ-2: {current_thresh_f} / 1023`")
            if buzz_active_f:
                st.error(buzz_badge)
            else:
                st.success(buzz_badge)
            st.progress(progress_f, text=f"Raw ADC Value: {val_f} / 1023 ({progress_f:.1%})")
            st.caption(f"🕒 Timestamp: `{hw_s.get('last_reading_str', 'N/A')}`")
        elif is_conn:
            st.info("⏳ Sensor stream initializing… (waiting for first reading)")
            st.markdown(f"**Selected Threshold:** `MQ-2: {current_thresh_f} / 1023`")
            st.caption(buzz_badge)
        else:
            st.warning("⚪ Sensor reading unavailable (Arduino disconnected)")
            st.markdown(f"**Selected Threshold:** `MQ-2: {current_thresh_f} / 1023`")

    _mq2_live_telemetry()

    # Hardware Buzzer Driver Mode (Passive Tone vs Active DC)
    st.markdown("##### 🔊 Hardware Buzzer Driver Mode")
    mode_options = {
        "PASSIVE_TONE": "🔊 Passive Buzzer (tone 2000Hz) [Default / Works for ALL Buzzers]",
        "ACTIVE_HIGH": "⚡ Active Buzzer (digitalWrite HIGH 5V) [Active Buzzer Only]",
        "ACTIVE_LOW": "🔄 Active Buzzer Module (digitalWrite LOW 0V)",
    }
    cur_b_mode = hw_status.get("buzzer_mode", "PASSIVE_TONE")
    mode_keys = list(mode_options.keys())
    cur_mode_idx = mode_keys.index(cur_b_mode) if cur_b_mode in mode_keys else 0
    selected_b_mode = st.selectbox(
        "Buzzer Hardware Type",
        options=mode_keys,
        index=cur_mode_idx,
        format_func=lambda k: mode_options[k],
        help="2-pin piezo buzzers from Arduino starter kits MUST use 'Passive Buzzer (tone 2000Hz)'. Constant DC 5V produces silence on passive buzzers.",
    )
    if selected_b_mode != cur_b_mode:
        arduino_engine.set_buzzer_mode(selected_b_mode)
        st.rerun()

    if selected_b_mode == "ACTIVE_HIGH":
        st.caption("⚠️ *Note:* If your buzzer makes no sound, switch to **Passive Buzzer (tone 2000Hz)**. 2-pin piezo buzzers require an oscillating frequency.")

    # Quick One-Click MQ-2 Test Presets (Thresholds 13, 22, 30, 300)
    st.markdown("##### 🧪 MQ-2 Threshold Test Presets")
    st.caption("Select a test threshold to verify buzzer activation on COM9 Pin D8:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🎯 Test 13", width="stretch", help="Set threshold to 13 (current reading 39 triggers buzzer)"):
            arduino_engine.set_threshold(13, warning_threshold=9)
            st.rerun()
    with p_col2:
        if st.button("🎯 Test 22", width="stretch", help="Set threshold to 22 (reading 39 triggers buzzer)"):
            arduino_engine.set_threshold(22, warning_threshold=15)
            st.rerun()
    with p_col3:
        if st.button("🎯 Test 30", width="stretch", help="Set threshold to 30 (reading 39 triggers buzzer)"):
            arduino_engine.set_threshold(30, warning_threshold=20)
            st.rerun()
    with p_col4:
        if st.button("🔄 Std (300)", width="stretch", help="Reset threshold to standard 300 (silences buzzer)"):
            arduino_engine.set_threshold(300, warning_threshold=200)
            st.rerun()

    # MQ-2 Threshold Controls (Configurable alert thresholds across 0-1023 measurement range)
    current_thresh = min(1023, max(0, int(hw_status["smoke_threshold"])))
    warn_thresh = st.slider(
        "MQ-2 Warning Threshold (A0)",
        min_value=0,
        max_value=1023,
        value=min(1023, max(0, int(hw_status.get("warning_threshold", 200)))),
        step=1,
        help="Analog threshold for MQ-2 warning status on range 0 to 1023 (default: 200)",
    )
    if warn_thresh != int(hw_status.get("warning_threshold", 200)):
        arduino_engine.set_warning_threshold(int(warn_thresh))

    mq2_thresh_val = st.slider(
        "MQ-2 Critical Fire Threshold (A0)",
        min_value=0,
        max_value=1023,
        value=current_thresh,
        step=1,
        help="Analog threshold for MQ-2 critical smoke/fire detection on range 0 to 1023 (Test values: 13, 22, 30; Default: 300)",
    )
    if mq2_thresh_val != current_thresh:
        arduino_engine.set_threshold(int(mq2_thresh_val))

    # Automatic Buzzer & Cooldown Controls
    auto_buzz = st.checkbox(
        "Automatic Arduino Buzzer Alert",
        value=bool(hw_status.get("auto_buzzer_enabled", True)),
        help="Automatically sound buzzer when reading >= threshold or when fire is confirmed",
    )
    if auto_buzz != bool(hw_status.get("auto_buzzer_enabled", True)):
        arduino_engine.set_auto_buzzer(auto_buzz)

    buzz_cooldown = st.slider(
        "Buzzer Alert Cooldown (seconds)",
        min_value=1,
        max_value=60,
        value=int(hw_status.get("buzzer_cooldown", 10)),
        step=1,
        help="Minimum quiet cooldown interval between buzzer alert activations (default: 10s)",
    )
    if buzz_cooldown != int(hw_status.get("buzzer_cooldown", 10)):
        arduino_engine.set_cooldown(float(buzz_cooldown))

    # Multi-Modal Decision Fusion Rule
    st.markdown("##### 🧠 Multi-Modal Decision Fusion")
    current_rule = incident_engine.fusion_rule
    fusion_options = ["Vision OR MQ-2", "Vision AND MQ-2", "Vision Only", "MQ-2 Only"]
    rule_idx = fusion_options.index(current_rule) if current_rule in fusion_options else 0
    selected_rule = st.selectbox(
        "Alert Trigger Rule",
        options=fusion_options,
        index=rule_idx,
        help="Determines when fire alert & buzzer activate based on YOLO vision and MQ-2 sensor.",
    )
    if selected_rule != current_rule:
        incident_engine.set_fusion_rule(selected_rule)

    # Quick Buzzer Manual Controls
    st.markdown("##### 🚨 Hardware Buzzer Controls & Tests")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🔔 TEST BUZZER", width="stretch", disabled=not hw_status["is_connected"], help="Sends BUZZER_TEST to Arduino Uno (2s tone 2000Hz pulse on Pin D8)"):
            ok = arduino_engine.test_buzzer(duration_seconds=2.0)
            if ok:
                st.session_state.buzzer_test_message = "🔔 Arduino Buzzer test triggered (BUZZER_TEST)!"
            else:
                st.session_state.buzzer_test_message = "⚠️ Buzzer busy / serial unavailable"
            st.rerun()
    with b_col2:
        if hw_status["buzzer_active"]:
            if st.button("🔇 Mute Buzzer", width="stretch", type="secondary"):
                arduino_engine.stop_buzzer(force=True)
                st.rerun()
        else:
            if st.button("🔊 Force Buzzer", width="stretch", disabled=not hw_status["is_connected"], help="Manually activate buzzer in current driver mode"):
                arduino_engine.set_buzzer(True)
                st.rerun()

    # Multi-Mode Diagnostic Sweep Button
    if st.button("🔍 Multi-Mode Diagnostic Sweep (4s)", width="stretch", disabled=not hw_status["is_connected"], help="Sequentially tests: 1s Tone 2000Hz (Passive), 1s DC HIGH (Active), 1s DC LOW (Active-LOW). Listen to hear which mode makes sound on your hardware!"):
        ok = arduino_engine.run_buzzer_diagnostic()
        if ok:
            st.session_state.buzzer_test_message = "🔍 Diagnostic sweep running on Arduino Uno! Listen to identify which phase sounds."
        st.rerun()

    # Arduino Serial Feedback Monitor
    st.markdown("##### 📡 Arduino Serial Debug Monitor")
    st.code(
        f"Arduino:    {'Connected' if hw_status['is_connected'] else 'Disconnected'}\n"
        f"Last RX:    {hw_status.get('last_rx', 'Waiting for data...')}\n"
        f"Last TX:    {hw_status.get('last_tx', 'None')}\n"
        f"Buzzer ACK: {hw_status.get('buzzer_ack', 'None')}",
        language="text"
    )
    last_ack = hw_status.get("last_hardware_ack")
    if last_ack:
        st.caption(f"**Latest Hardware Reply:** `{last_ack}`")

    serial_history = hw_status.get("serial_logs", [])
    if serial_history:
        with st.expander("📜 Arduino Serial Log (Last 10 events)"):
            for ts, msg in reversed(serial_history[-10:]):
                st.text(f"[{ts}] {msg}")

    st.markdown("---")
    st.markdown("#### 🔄 Monitoring Session Reset")
    if st.button("🔄 NEW MONITORING SESSION", width="stretch", type="primary"):
        incident_engine.reset_session()
        st.session_state.camera_running = False
        st.success("New monitoring session started. All frame counters reset.")
        st.rerun()

    st.markdown("#### ⚙️ Detection Thresholds")
    conf_thresh_slider = st.slider("Fire Confidence Threshold", 0.40, 0.95, float(get_confidence_threshold()), 0.05)
    confirm_frames_slider = st.number_input("Fire Confirmation Frames", 1, 20, int(get_min_fire_frames()))
    clear_frames_slider = st.number_input("Resolution Clean Frames", 1, 50, int(get_min_no_fire_frames()))

    # Real-time synchronization of detection sliders with incident engine
    incident_engine.set_detection_thresholds(
        confidence=conf_thresh_slider,
        min_fire_frames=int(confirm_frames_slider),
        min_clean_frames=int(clear_frames_slider),
    )

    st.markdown("---")
    st.markdown("#### 📍 GPS Location")
    if current_location["status"] == "AVAILABLE":
        st.success(f"📍 GPS: {current_location['latitude']:.4f}, {current_location['longitude']:.4f}")
    else:
        st.warning("GPS: Unavailable (Manual override available)")

    with st.expander("✏️ Manual GPS Override"):
        man_lat = st.text_input("Latitude", value=str(st.session_state.latitude or "12.9716"))
        man_lon = st.text_input("Longitude", value=str(st.session_state.longitude or "77.5946"))
        if st.button("Save GPS Coordinates", width="stretch"):
            try:
                st.session_state.latitude = float(man_lat)
                st.session_state.longitude = float(man_lon)
                st.session_state.accuracy = 10.0
                st.success("GPS Coordinates updated.")
                st.rerun()
            except ValueError:
                st.error("Enter valid numeric coordinates.")


# ============================================================
# MAIN TABS (LIVE CAMERA, VIDEO ANALYSIS, OFFICERS & ALERTS)
# ============================================================

tab_live, tab_video, tab_officers, tab_reports = st.tabs([
    "📹 Live Camera Surveillance",
    "📼 Video File Analysis",
    "🧑‍🚒 Forest Officers & Dispatches",
    "📑 Incident Analytics & PDF Reports",
])


# ============================================================
# TAB 1: LIVE CAMERA SURVEILLANCE
# ============================================================

with tab_live:
    col_cam, col_stats = st.columns([1.6, 1.0], gap="large")

    with col_cam:
        st.markdown("### 📹 Live Camera Stream")
        cam_source = st.radio(
            "Select Camera Input Mode:",
            ["Direct USB/Laptop Webcam", "📱 Iriun 4K Mobile Webcam", "WebRTC Browser Stream"],
            horizontal=True,
            key="camera_mode_selector",
        )

        if cam_source == "Direct USB/Laptop Webcam":
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.1, 1.1, 0.8, 1.5])
            with ctrl_col1:
                start_cam = st.button("▶️ Start Live Camera", type="primary", width="stretch", disabled=st.session_state.camera_running)
            with ctrl_col2:
                stop_cam = st.button("⏹️ Stop Live Camera", type="secondary", width="stretch", disabled=not st.session_state.camera_running)
            with ctrl_col3:
                cam_idx = st.number_input("Camera Index", 0, 5, 0, help="Camera hardware index (0 for built-in, 1+ for USB)", key="usb_cam_idx")
            with ctrl_col4:
                usb_res = st.selectbox(
                    "Stream Resolution",
                    [
                        "720p HD (1280x720) [Balanced]",
                        "1080p Full HD (1920x1080) [High Quality]",
                        "480p SD (640x480) [Standard]",
                        "360p Low (640x360) [Fastest FPS]",
                        "4K UHD (3840x2160) [Ultra HD]",
                        "Default / Camera Native"
                    ],
                    index=0,
                    key="usb_cam_res"
                )

            if start_cam:
                st.session_state.camera_running = True
                incident_engine.start_session(source="LIVE CAMERA")
                st.rerun()

            if stop_cam:
                st.session_state.camera_running = False
                incident_engine.stop_session()
                st.rerun()

            live_frame_placeholder = st.empty()

            if st.session_state.camera_running:
                cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(cam_idx))
                if not cap.isOpened():
                    st.error(f"Unable to open camera at index {cam_idx}. Please verify camera permissions, switch to Iriun 4K Webcam, or select WebRTC mode.")
                    st.session_state.camera_running = False
                else:
                    # Apply selected resolution to hardware capture
                    target_w, target_h = None, None
                    if "3840x2160" in usb_res:
                        target_w, target_h = 3840, 2160
                    elif "1920x1080" in usb_res:
                        target_w, target_h = 1920, 1080
                    elif "1280x720" in usb_res:
                        target_w, target_h = 1280, 720
                    elif "640x480" in usb_res:
                        target_w, target_h = 640, 480
                    elif "640x360" in usb_res:
                        target_w, target_h = 640, 360

                    if target_w and target_h:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

                    # Continuous live streaming loop
                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Camera stream interrupted.")
                            break

                        # Enforce selected resolution if driver did not adapt
                        if target_w and target_h and (frame.shape[1] != target_w or frame.shape[0] != target_h):
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                        annotated_frame, detections, max_conf, label = detect_fire_in_frame(
                            frame, conf_thresh=conf_thresh_slider
                        )

                        # Update incident manager session stats
                        with LOCATION_LOCK:
                            loc_lat = LOCATION["latitude"]
                            loc_lon = LOCATION["longitude"]
                            loc_acc = LOCATION["accuracy"]

                        incident_engine.process_frame(
                            image=frame,
                            detections=detections,
                            fire_confidence=max_conf,
                            label=label,
                            detection_mode="LIVE CAMERA",
                            latitude=loc_lat,
                            longitude=loc_lon,
                            accuracy=loc_acc,
                            conf_threshold=conf_thresh_slider,
                            min_fire_frames=int(confirm_frames_slider),
                            min_clean_frames=int(clear_frames_slider),
                        )

                        # Automatic buzzer actuation during confirmed fire incident
                        if max_conf >= conf_thresh_slider and label == "fire" and incident_engine.fire_active:
                            try:
                                arduino_engine.notify_fire_alert(True)
                            except Exception:
                                pass

                        # Overlay MQ-2 Sensor HUD & Resolution Badge on annotated frame
                        try:
                            # Resolution tag (top right)
                            res_badge = f"{frame.shape[1]}x{frame.shape[0]}"
                            cv2.putText(annotated_frame, res_badge, (annotated_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        except Exception:
                            pass

                        try:
                            hw_s = arduino_engine.get_status()
                            if hw_s["is_connected"] and hw_s.get("has_reading", False):
                                s_val = hw_s['latest_mq2_value']
                                s_stat = hw_s.get('sensor_status', 'NORMAL')
                                thresh = int(hw_s.get('mq2_threshold', 300))
                                b_lbl = "BUZZER:ON" if hw_s.get("buzzer_active") else "BUZZER:OFF"
                                hud_text = f"MQ-2: {s_val} / 1023 [{s_stat}] | Thresh: {thresh} | {b_lbl}"
                                hud_color = (0, 0, 255) if s_stat in ("WARNING", "CRITICAL") else (0, 255, 0)
                                cv2.putText(annotated_frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
                        except Exception:
                            pass

                        live_frame_placeholder.image(annotated_frame, channels="BGR", width="stretch")
                        time.sleep(0.03)

                    cap.release()
            else:
                live_frame_placeholder.info("Camera is idle. Click '▶️ Start Live Camera' to start continuous live AI monitoring.")

        elif cam_source == "📱 Iriun 4K Mobile Webcam":
            iriun_conn_type = st.radio(
                "Iriun Connection Method:",
                ["DirectShow Virtual Webcam (USB / Wi-Fi)", "Network Stream (RTSP / HTTP)"],
                horizontal=True,
                key="iriun_conn_type_radio"
            )

            if iriun_conn_type == "DirectShow Virtual Webcam (USB / Wi-Fi)":
                ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.1, 1.1, 0.8, 1.5])
                with ctrl_col1:
                    start_cam = st.button("▶️ Start Iriun 4K Camera", type="primary", width="stretch", disabled=st.session_state.camera_running, key="start_iriun_btn")
                with ctrl_col2:
                    stop_cam = st.button("⏹️ Stop Iriun 4K Camera", type="secondary", width="stretch", disabled=not st.session_state.camera_running, key="stop_iriun_btn")
                with ctrl_col3:
                    iriun_idx = st.number_input("Camera Index", 0, 10, 1, help="Iriun Webcam index on Windows (usually 1 or 2 when laptop camera is 0)", key="iriun_idx_input")
                with ctrl_col4:
                    iriun_res = st.selectbox(
                        "Stream Resolution",
                        [
                            "Full HD 1080p (1920x1080) [Recommended]",
                            "HD 720p (1280x720) [Balanced]",
                            "480p SD (640x480) [Standard]",
                            "360p Low (640x360) [Fastest FPS]",
                            "4K UHD (3840x2160) [Ultra HD]",
                            "Default / App Auto"
                        ],
                        index=0,
                        key="iriun_res_select"
                    )
                iriun_stream_target = int(iriun_idx)
            else:
                ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.1, 1.1, 1.8, 1.3])
                with ctrl_col1:
                    start_cam = st.button("▶️ Start Iriun IP Stream", type="primary", width="stretch", disabled=st.session_state.camera_running, key="start_iriun_net_btn")
                with ctrl_col2:
                    stop_cam = st.button("⏹️ Stop Iriun IP Stream", type="secondary", width="stretch", disabled=not st.session_state.camera_running, key="stop_iriun_net_btn")
                with ctrl_col3:
                    iriun_url = st.text_input("Iriun Stream URL", value="http://192.168.1.100:8080/video", placeholder="http://<phone_ip>:8080/video or rtsp://...", key="iriun_url_input")
                with ctrl_col4:
                    iriun_res = st.selectbox(
                        "Stream Resolution",
                        [
                            "1080p Full HD (1920x1080)",
                            "720p HD (1280x720)",
                            "480p SD (640x480)",
                            "360p Low (640x360)",
                            "Default / Stream Native"
                        ],
                        index=0,
                        key="iriun_net_res_select"
                    )
                iriun_stream_target = iriun_url.strip()

            with st.expander("ℹ️ How to Connect Android Phone via Iriun 4K Webcam & Camera Scanner", expanded=False):
                scan_col, guide_col = st.columns([1.2, 1.8])
                with scan_col:
                    st.markdown("**Detect Connected Cameras**")
                    if st.button("🔍 Scan Camera Indices (0-5)", key="scan_iriun_cams_btn"):
                        detected_cams = []
                        for probe_idx in range(5):
                            probe_cap = cv2.VideoCapture(probe_idx, cv2.CAP_DSHOW)
                            if not probe_cap.isOpened():
                                probe_cap = cv2.VideoCapture(probe_idx)
                            if probe_cap.isOpened():
                                pw = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                ph = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                detected_cams.append(f"• **Index {probe_idx}**: Active ({pw}x{ph})")
                                probe_cap.release()
                        if detected_cams:
                            st.success("**Found active cameras:**\n" + "\n".join(detected_cams))
                        else:
                            st.warning("No camera found on indices 0-4. Please verify phone Iriun app is active.")
                with guide_col:
                    st.markdown("""
                    **Quick Connection Steps:**
                    1. Install **Iriun 4K Webcam** on your Android phone from Google Play Store.
                    2. Install **Iriun Webcam for Windows** from [iriun.com](https://iriun.com).
                    3. Connect phone & laptop using **USB Cable** *(fastest, low latency)* or connect both to the **same Wi-Fi**.
                    4. Open Iriun on phone. It will automatically pair with your PC.
                    5. Set Camera Index to **1** (or 2) above and click **▶️ Start Iriun 4K Camera**.
                    """)

            if start_cam:
                st.session_state.camera_running = True
                incident_engine.start_session(source="LIVE CAMERA (Iriun 4K)")
                st.rerun()

            if stop_cam:
                st.session_state.camera_running = False
                incident_engine.stop_session()
                st.rerun()

            live_frame_placeholder = st.empty()

            if st.session_state.camera_running:
                if iriun_conn_type == "DirectShow Virtual Webcam (USB / Wi-Fi)":
                    cap = cv2.VideoCapture(int(iriun_stream_target), cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(int(iriun_stream_target))
                else:
                    cap = cv2.VideoCapture(str(iriun_stream_target))

                if not cap.isOpened():
                    st.error(f"Unable to open Iriun 4K Webcam ({iriun_stream_target}). Please ensure the Iriun app is open on your Android phone and connected via USB or Wi-Fi.")
                    st.session_state.camera_running = False
                else:
                    # Apply resolution presets for DirectShow
                    target_w, target_h = None, None
                    if "3840x2160" in iriun_res:
                        target_w, target_h = 3840, 2160
                    elif "1920x1080" in iriun_res:
                        target_w, target_h = 1920, 1080
                    elif "1280x720" in iriun_res:
                        target_w, target_h = 1280, 720
                    elif "640x480" in iriun_res:
                        target_w, target_h = 640, 480
                    elif "640x360" in iriun_res:
                        target_w, target_h = 640, 360

                    if target_w and target_h and iriun_conn_type == "DirectShow Virtual Webcam (USB / Wi-Fi)":
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

                    # Continuous live streaming loop
                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Iriun camera stream interrupted.")
                            break

                        # Enforce selected resolution if driver/network stream did not adapt
                        if target_w and target_h and (frame.shape[1] != target_w or frame.shape[0] != target_h):
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                        annotated_frame, detections, max_conf, label = detect_fire_in_frame(
                            frame, conf_thresh=conf_thresh_slider
                        )

                        # Update incident manager session stats
                        with LOCATION_LOCK:
                            loc_lat = LOCATION["latitude"]
                            loc_lon = LOCATION["longitude"]
                            loc_acc = LOCATION["accuracy"]

                        incident_engine.process_frame(
                            image=frame,
                            detections=detections,
                            fire_confidence=max_conf,
                            label=label,
                            detection_mode="LIVE CAMERA (Iriun 4K)",
                            latitude=loc_lat,
                            longitude=loc_lon,
                            accuracy=loc_acc,
                            conf_threshold=conf_thresh_slider,
                            min_fire_frames=int(confirm_frames_slider),
                            min_clean_frames=int(clear_frames_slider),
                        )

                        # Automatic buzzer actuation during confirmed fire incident
                        if max_conf >= conf_thresh_slider and label == "fire" and incident_engine.fire_active:
                            try:
                                arduino_engine.notify_fire_alert(True)
                            except Exception:
                                pass

                        # Overlay MQ-2 Sensor HUD & Resolution Badge on annotated frame
                        try:
                            # Resolution tag (top right)
                            res_badge = f"{frame.shape[1]}x{frame.shape[0]}"
                            cv2.putText(annotated_frame, res_badge, (annotated_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        except Exception:
                            pass

                        try:
                            hw_s = arduino_engine.get_status()
                            if hw_s["is_connected"] and hw_s.get("has_reading", False):
                                s_val = hw_s['latest_mq2_value']
                                s_stat = hw_s.get('sensor_status', 'NORMAL')
                                thresh = int(hw_s.get('mq2_threshold', 300))
                                b_lbl = "BUZZER:ON" if hw_s.get("buzzer_active") else "BUZZER:OFF"
                                hud_text = f"MQ-2: {s_val} / 1023 [{s_stat}] | Thresh: {thresh} | {b_lbl}"
                                hud_color = (0, 0, 255) if s_stat in ("WARNING", "CRITICAL") else (0, 255, 0)
                                cv2.putText(annotated_frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
                        except Exception:
                            pass

                        live_frame_placeholder.image(annotated_frame, channels="BGR", width="stretch")
                        time.sleep(0.03)

                    cap.release()
            else:
                live_frame_placeholder.info("Iriun 4K camera is idle. Open the Iriun app on your Android phone and click '▶️ Start Iriun 4K Camera'.")

        else:
            # WebRTC Streamer
            st.info("Browser WebRTC stream with Google STUN relay. Click 'START' below:")
            webrtc_streamer(
                key="forest-fire-camera-streamer",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False},
                video_frame_callback=webrtc_video_frame_callback,
                async_processing=True,
            )

    # Real-Time Dashboard Statistics (Point 11, 14, 26)
    with col_stats:
        st.markdown("### 📊 Live Monitoring Telemetry")
        stats = incident_engine.get_statistics()
        hw_stats = arduino_engine.get_status()

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("Frames Processed", f"{stats['total_frames']}")
            st.metric("Fire Frames", f"{stats['fire_frames']}")
            st.metric("Current Fire Conf.", f"{incident_engine.current_frame_confidence:.1%}")
            st.metric("Confirmation Conf.", f"{stats.get('confirmation_confidence', 0.0):.1%}")
        with mcol2:
            st.metric("Normal Frames", f"{stats['normal_frames']}")
            st.metric("Smoke Frames", f"{stats['smoke_frames']}")
            st.metric("Max Fire Conf.", f"{stats['max_confidence']:.1%}")
            st.metric("Avg Fire Conf.", f"{stats['average_confidence']:.1%}")

        @st.fragment(run_every=1)
        def render_hardware_telemetry():
            hw_s = arduino_engine.get_status()
            st.markdown("---")
            st.markdown("#### 🔌 Hardware Multi-Modal Telemetry (Arduino Uno + MQ-2)")
            
            # Row 1: Connection & Sensor Status pills
            hc1, hc2 = st.columns(2)
            is_valid = hw_s["is_connected"] and hw_s.get("has_reading", False)
            with hc1:
                if hw_s["is_connected"]:
                    st.success(f"🟢 **Arduino:** Connected (`{hw_s['port']}` @ {hw_s.get('baud_rate', 9600)} baud)")
                else:
                    st.info("⚪ **Arduino:** Disconnected (Vision-Only)")

                thresh = int(hw_s.get('mq2_threshold', 300))
                warn_thresh_val = int(hw_s.get('warning_threshold', 200))
                if is_valid:
                    val = int(hw_s['latest_mq2_value'])
                    st.metric(
                        "MQ-2 Sensor Value",
                        f"{val} / 1023",
                        delta=f"Active Thresh: {thresh} | Warn: {warn_thresh_val}",
                        delta_color="inverse" if hw_s["smoke_detected"] else "normal",
                    )
                    progress_val = min(1.0, max(0.0, val / 1023.0))
                    st.progress(progress_val, text=f"Range: {val} / 1023 ({progress_val:.1%})")
                elif hw_s["is_connected"]:
                    st.metric("MQ-2 Sensor Value", "Initializing...", delta=f"Active Thresh: {thresh}")
                    st.info("⏳ Waiting for incoming serial packets from COM9...")
                else:
                    st.metric("MQ-2 Sensor Value", "Unavailable", delta="Hardware Disconnected")
                    st.warning("⚪ Sensor disconnected (Vision-only active)")
            with hc2:
                stat = hw_s.get("sensor_status", "NORMAL")
                thresh = int(hw_s.get('mq2_threshold', 300))
                if not hw_s["is_connected"]:
                    st.info("⚪ **Sensor Status:** DISCONNECTED")
                elif not hw_s.get("has_reading", False):
                    st.info("⏳ **Sensor Status:** INITIALIZING")
                elif stat == "CRITICAL":
                    st.error(f"🚨 **Sensor Status:** {stat} (Exceeds {thresh})")
                elif stat == "WARNING":
                    st.warning(f"⚠️ **Sensor Status:** {stat}")
                else:
                    st.success(f"🟢 **Sensor Status:** {stat}")

                st.markdown(f"**Selected Threshold:** `{thresh} / 1023`")

                if hw_s["buzzer_active"]:
                    st.error("🔊 **Buzzer:** ACTIVE (Pin D8 tone 2000Hz)")
                else:
                    st.success("🔇 **Buzzer:** OFF (Pin D8 silent)")

                st.caption(f"🕒 **Latest Reading:** `{hw_s.get('last_reading_str', 'No readings yet')}`")

            # Row 2: Live Serial Packet Monitor & Buzzer Test
            st.markdown("##### 📡 Serial Telemetry & Commands")
            s_c1, s_c2 = st.columns([1.5, 1.0])
            with s_c1:
                last_rx = hw_s.get("last_rx", "Waiting for data...")
                last_tx = hw_s.get("last_tx", "None")
                buzzer_ack = hw_s.get("buzzer_ack", "None")
                st.code(
                    f"Arduino:    {'Connected' if hw_s['is_connected'] else 'Disconnected'}\n"
                    f"Last RX:    {last_rx}\n"
                    f"Last TX:    {last_tx}\n"
                    f"Buzzer ACK: {buzzer_ack}",
                    language="text"
                )
            with s_c2:
                if st.button("🔔 TEST BUZZER", key="main_test_buzzer_btn", width="stretch", disabled=not hw_s["is_connected"], help="Sends BUZZER_TEST to Arduino Uno (2s tone 2000Hz pulse on Pin D8)"):
                    arduino_engine.test_buzzer(duration_seconds=2.0)
                    st.rerun()
                if hw_s["buzzer_active"]:
                    if st.button("🔇 MUTE", key="main_mute_buzzer_btn", width="stretch"):
                        arduino_engine.stop_buzzer(force=True)
                        st.rerun()

        render_hardware_telemetry()

        st.markdown("---")
        st.markdown(f"**Incident ID:** `{stats.get('incident_id', 'N/A')}`")
        st.markdown(f"**Incident Status:** `{stats['status']}`")
        st.markdown(f"**PC Alarm:** `{'🔊 ACTIVE' if stats['alarm_active'] else '🔇 OFF'}`")
        st.markdown(f"**Alerts Dispatched for Incident:** `{1 if stats['fire_frames'] > 0 else 0}`")
        st.markdown(f"**Fire Detection Rate:** `{stats['fire_percentage']:.1f}%`")

        if st.button("🔄 Refresh Live Metrics", width="stretch"):
            st.rerun()

    # Dedicated Maximum-Confidence Evidence Section (Point 4 & 19)
    st.markdown("---")
    st.markdown("### 🔥 Maximum-Confidence Fire Evidence Capture")
    ev_col1, ev_col2 = st.columns([1.2, 1.0], gap="large")

    with ev_col1:
        if stats.get("max_fire_frame") is not None:
            st.image(
                stats["max_fire_frame"],
                caption=f"Evidence Frame #{stats.get('max_fire_frame_number', 0)} (Confidence: {stats['max_confidence']:.2%})",
                channels="BGR",
                width="stretch",
            )
        else:
            st.info("No fire evidence frame captured yet in this session.")

    with ev_col2:
        if stats.get("max_fire_frame") is not None:
            st.markdown(f"**Evidence Frame Number:** `Frame #{stats.get('max_fire_frame_number')}`")
            ev_ts = stats.get("max_fire_frame_timestamp")
            st.markdown(f"**Capture Timestamp:** `{ev_ts.strftime('%Y-%m-%d %H:%M:%S UTC') if ev_ts else 'N/A'}`")
            st.markdown(f"**Peak Confidence Score:** `{stats['max_confidence']:.2%}`")
            st.markdown(f"**Class Label:** `FIRE`")
            st.markdown(f"**GPS Coordinates:** `{current_location['url']}`")
        else:
            st.write("Evidence data will appear as soon as a fire signature is detected by YOLO.")


# ============================================================
# TAB 2: VIDEO FILE ANALYSIS
# ============================================================

with tab_video:
    st.markdown("### 📼 Upload Video for Comprehensive Frame-by-Frame AI Analysis")
    uploaded_file = st.file_uploader("Upload Video File (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tfile.write(uploaded_file.read())
        tfile.close()

        vid_col1, vid_col2 = st.columns([1.4, 1.0], gap="large")

        with vid_col1:
            st.video(tfile.name)
            run_analysis = st.button("🚀 Run Frame-by-Frame AI Video Analysis", type="primary", width="stretch")

        with vid_col2:
            st.markdown("#### 🔌 Real-Time MQ-2 Hardware Telemetry")
            telemetry_box = st.empty()

            def render_vid_telemetry_box(hw_s, frame_no=None, total_f=None, fire_count=None, cur_conf=0.0):
                with telemetry_box.container():
                    st.markdown(
                        f"**Arduino Port:** `{hw_s['port'] if hw_s['is_connected'] else 'Disconnected'}` &bull; "
                        f"{'🟢 Connected' if hw_s['is_connected'] else '⚪ Disconnected'}"
                    )
                    is_valid = hw_s["is_connected"] and hw_s.get("has_reading", False)
                    stat = hw_s.get("sensor_status", "NORMAL")
                    thresh = int(hw_s.get("smoke_threshold", 300))

                    if not hw_s["is_connected"]:
                        st.info("⚪ **Sensor Status:** DISCONNECTED")
                    elif not is_valid:
                        st.info("⏳ **Sensor Status:** INITIALIZING (Waiting for COM9 stream)")
                    elif stat == "CRITICAL":
                        st.error(f"🚨 **Sensor Status:** CRITICAL (Threshold {thresh} Exceeded)")
                    elif stat == "WARNING":
                        st.warning("⚠️ **Sensor Status:** WARNING (Elevated)")
                    else:
                        st.success("🟢 **Sensor Status:** NORMAL")

                    m1, m2 = st.columns(2)
                    with m1:
                        if is_valid:
                            val = int(hw_s['latest_mq2_value'])
                            st.metric(
                                "MQ-2 Value",
                                f"{val} / 1023",
                                delta=f"Active Thresh: {thresh} | Warn: {int(hw_s.get('warning_threshold', 200))}",
                                delta_color="inverse" if hw_s["smoke_detected"] else "normal",
                            )
                        elif hw_s["is_connected"]:
                            st.metric("MQ-2 Value", "Initializing...", delta=f"Active Thresh: {thresh}")
                        else:
                            st.metric("MQ-2 Value", "Unavailable", delta="Disconnected")

                    with m2:
                        buzz_active = hw_s["buzzer_active"]
                        st.metric(
                            "Buzzer State",
                            "🔊 ACTIVE (tone 2000Hz)" if buzz_active else "🔇 OFF (silent)",
                            delta="Auto-Alert Active" if hw_s.get("auto_buzzer_enabled") else "Manual Only",
                        )

                    st.markdown(f"**Selected Threshold:** `{thresh} / 1023` &bull; **Buzzer:** `{'🔊 ACTIVE (Pin D8 tone 2000Hz)' if hw_s['buzzer_active'] else '🔇 OFF (Pin D8 silent)'}`")

                    if is_valid:
                        val = int(hw_s['latest_mq2_value'])
                        progress_val = min(1.0, max(0.0, val / 1023.0))
                        st.progress(progress_val, text=f"Raw MQ-2 ADC: {val} / 1023 ({progress_val:.1%})")

                    st.caption(f"🕒 **Latest Reading:** `{hw_s.get('last_reading_str', 'No readings yet')}`")

                    if frame_no is not None:
                        st.markdown("---")
                        st.markdown(f"**Analysis Progress:** Frame `{frame_no} / {total_f}`")
                        st.markdown(f"**Current Frame Confidence:** `{cur_conf:.1%}`")
                        st.markdown(f"**Confirmed Fire Frames:** `{fire_count if fire_count is not None else 0}`")

            # Initial render before running
            render_vid_telemetry_box(arduino_engine.get_status())

        if run_analysis:
            incident_engine.start_session(source="UPLOADED VIDEO")
            cap = cv2.VideoCapture(tfile.name)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

            with vid_col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_preview = st.empty()

            processed_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                processed_count += 1
                annotated_frame, detections, max_conf, label = detect_fire_in_frame(
                    frame, conf_thresh=conf_thresh_slider
                )

                incident_engine.process_frame(
                    image=frame,
                    detections=detections,
                    fire_confidence=max_conf,
                    label=label,
                    detection_mode="UPLOADED VIDEO",
                    latitude=st.session_state.latitude,
                    longitude=st.session_state.longitude,
                    accuracy=st.session_state.accuracy,
                    frame_idx=processed_count,
                    conf_threshold=conf_thresh_slider,
                    min_fire_frames=int(confirm_frames_slider),
                    min_clean_frames=int(clear_frames_slider),
                )

                # Automatic buzzer actuation during confirmed fire incident
                if max_conf >= conf_thresh_slider and label == "fire" and incident_engine.fire_active:
                    try:
                        arduino_engine.notify_fire_alert(True)
                    except Exception:
                        pass

                # Overlay live MQ-2 sensor HUD on preview frame
                try:
                    hw_s = arduino_engine.get_status()
                    if hw_s["is_connected"] and hw_s.get("has_reading", False):
                        s_val = hw_s['latest_mq2_value']
                        s_stat = hw_s.get('sensor_status', 'NORMAL')
                        thresh = int(hw_s.get('mq2_threshold', 300))
                        b_lbl = "BUZZER:ON" if hw_s.get("buzzer_active") else "BUZZER:OFF"
                        hud_text = f"MQ-2: {s_val} / 1023 [{s_stat}] | Thresh: {thresh} | {b_lbl}"
                        hud_color = (0, 0, 255) if s_stat in ("WARNING", "CRITICAL") else (0, 255, 0)
                        cv2.putText(annotated_frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
                except Exception:
                    pass

                if processed_count % 5 == 0 or processed_count == total_vid_frames:
                    progress_bar.progress(min(1.0, processed_count / total_vid_frames))
                    status_text.text(f"Analyzing frame {processed_count}/{total_vid_frames}...")
                    frame_preview.image(annotated_frame, channels="BGR", width="stretch")
                    render_vid_telemetry_box(
                        hw_s,
                        frame_no=processed_count,
                        total_f=total_vid_frames,
                        fire_count=incident_engine.fire_frames_count,
                        cur_conf=max_conf,
                    )

            cap.release()
            had_fire = (incident_engine.fire_frames_count > 0)
            if had_fire and arduino_engine.auto_buzzer_enabled and arduino_engine.is_connected:
                arduino_engine.trigger_fire_alarm(duration_seconds=5.0)
            incident_engine.stop_session()
            progress_bar.progress(1.0)
            status_text.success("✅ Video Analysis Complete!")
            st.session_state.uploaded_video_stats = incident_engine.get_statistics()
            st.rerun()

        if st.session_state.uploaded_video_stats:
            vstats = st.session_state.uploaded_video_stats
            st.markdown("---")
            st.markdown("#### 📊 Video Analysis Results & Multi-Modal Telemetry")
            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            vcol1.metric("Analyzed Frames", vstats["total_frames"])
            vcol2.metric("Fire Frames", vstats["fire_frames"])
            vcol3.metric("Max Fire Conf.", f"{vstats['max_confidence']:.2%}")
            vcol4.metric("Avg Fire Conf.", f"{vstats['average_confidence']:.2%}")

            # Hardware correlation summary
            h_stat = arduino_engine.get_status()
            hw_col1, hw_col2, hw_col3 = st.columns(3)
            with hw_col1:
                if h_stat["is_connected"] and h_stat.get("has_reading", False):
                    st.metric("Latest MQ-2 Sensor Value", f"{h_stat['latest_mq2_value']} / 1023")
                elif h_stat["is_connected"]:
                    st.metric("Latest MQ-2 Sensor Value", "Initializing...")
                else:
                    st.metric("Latest MQ-2 Sensor Value", "Unavailable")
            with hw_col2:
                st.metric("Sensor Status", f"{h_stat.get('sensor_status', 'NORMAL')}")
            with hw_col3:
                st.metric("Fusion Rule Applied", f"{incident_engine.fusion_rule}")


# ============================================================
# TAB 3: FOREST OFFICERS & EMERGENCY BROADCAST DISPATCH
# ============================================================

with tab_officers:
    st.markdown("### 🧑‍🚒 Forest Officer Directory & Emergency Dispatch")
    db_ok, db_msg, db_info = test_connection()
    if db_ok:
        st.success(f"🟢 Database Connected (MySQL - `{db_info.get('database')}` on {db_info.get('host')}:{db_info.get('port')})")
    else:
        st.warning(f"🟡 MySQL Connection Note: {db_msg}")

    off_col1, off_col2 = st.columns([1.2, 1.4], gap="large")

    with off_col1:
        with st.container(border=True):
            st.markdown("#### 📋 Register Forest Officer")
            with st.form("add_officer_form", clear_on_submit=True):
                name_in = st.text_input("Officer Name", placeholder="Ranger John Doe")
                phone_in = st.text_input("Mobile / Telegram Phone", placeholder="+91 98765 43210")
                email_in = st.text_input("Official Email", placeholder="officer@forest.gov")
                region_in = st.text_input("Forest Range / Sector", placeholder="Sector 4 - Pine Ridge")
                submitted = st.form_submit_button("Save Officer", type="primary", width="stretch")

            if submitted:
                if not name_in.strip():
                    st.error("Officer name is required.")
                elif not email_in.strip() or "@" not in email_in:
                    st.error("Valid email address is required.")
                else:
                    try:
                        add_forest_officer(name_in.strip(), phone_in.strip(), email_in.strip(), region_in.strip())
                        st.success(f"Officer '{name_in}' registered successfully.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Database error: {exc}")

        officers = get_active_officers() if get_active_officers else []
        if officers:
            st.markdown("#### 📋 Active Registered Officers")
            st.dataframe(pd.DataFrame(officers), hide_index=True, width="stretch")
        else:
            st.info("No active forest officers currently registered in MySQL.")

    with off_col2:
        with st.container(border=True):
            st.markdown("#### 🚨 Manual Emergency Broadcast Dispatch")
            st.caption("Dispatches verified alert to Telegram AND every registered officer's email with attached PDF analytics report.")
            
            stats = incident_engine.get_statistics()
            default_alert_msg = (
                f"Forest fire emergency detected by AI monitoring system.\n"
                f"Peak Confidence: {stats['max_confidence']:.1%}.\n"
                f"Location: {current_location['url']}.\n"
                f"Immediate response required."
            )
            broadcast_msg = st.text_area("Alert Message Text", value=default_alert_msg, height=110)
            
            if st.button("🚨 DISPATCH EMERGENCY BROADCAST (TELEGRAM + EMAIL)", type="primary", width="stretch"):
                if not broadcast_msg.strip():
                    st.error("Message cannot be empty.")
                else:
                    try:
                        res = handleFireAlert(
                            confidence=max(0.75, stats["max_confidence"]),
                            detectionMode="Manual Dashboard Broadcast",
                            severity="CRITICAL",
                            latitude=current_location["latitude"],
                            longitude=current_location["longitude"],
                            accuracy=current_location["accuracy"],
                            custom_message=broadcast_msg,
                            force=True,
                            evidence_frame=stats.get("max_fire_frame"),
                        )
                        if res.get("success"):
                            st.success(f"✅ Emergency alert dispatched! (Telegram: {res.get('telegram_status')}, Email: {res.get('email_status')})")
                        else:
                            st.warning(f"Alert completed with status: Telegram ({res.get('telegram_status')}), Email ({res.get('email_status')})")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Dispatch error: {err}")

        st.markdown("#### 📋 Session Dispatch History")
        history = get_alert_dispatch_history()
        if history:
            h_rows = []
            for h in history:
                h_rows.append({
                    "Timestamp": h.get("timestamp"),
                    "Mode": h.get("detection_mode"),
                    "Confidence": f"{h.get('confidence', 0):.1%}",
                    "Telegram": h.get("telegram_status"),
                    "Email": h.get("email_status"),
                    "Incident ID": h.get("incident_id"),
                })
            st.dataframe(pd.DataFrame(h_rows), hide_index=True, width="stretch")
        else:
            st.info("No alert dispatches in this session.")


# ============================================================
# TAB 4: INCIDENT ANALYTICS & PDF REPORT GENERATION
# ============================================================

with tab_reports:
    st.markdown("### 📑 Publication-Grade Incident Analytics & 12 Visual Charts")
    stats = incident_engine.get_statistics()

    rep_col1, rep_col2 = st.columns([1.2, 1.0])
    with rep_col1:
        st.markdown(f"**Session ID:** `{stats.get('session_id')}` | **Total Frames:** `{stats['total_frames']}`")
        st.markdown(f"**Fire Detection Rate:** `{stats['fire_percentage']:.2f}%` | **Peak Conf:** `{stats['max_confidence']:.2%}` | **Avg Conf:** `{stats['average_confidence']:.2%}`")
    with rep_col2:
        pdf_bytes = generate_incident_pdf_report(stats)
        if pdf_bytes:
            pdf_filename = f"Forest_Fire_Incident_Report_{stats.get('incident_id', 'session')}.pdf"
            st.download_button(
                label="📄 Download Comprehensive PDF Report (12 Charts + Evidence)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                width="stretch",
            )

    st.markdown("---")
    st.markdown("#### 📈 Visual Analytical Charts")

    charts = generate_all_12_charts(
        telemetry=incident_engine.telemetry_history,
        alert_records=incident_engine.alert_event_history,
        milestones=incident_engine.timeline,
        stats=stats,
        completed_incidents=incident_engine.completed_incidents,
    )

    c_row1_col1, c_row1_col2 = st.columns(2)
    with c_row1_col1:
        if charts.get("chart_1"):
            st.image(charts["chart_1"], caption="Figure 1: AI Fire Confidence Trajectory Over Time", width="stretch")
    with c_row1_col2:
        if charts.get("chart_2"):
            st.image(charts["chart_2"], caption="Figure 2: Binary Fire Detection State Over Time", width="stretch")

    c_row2_col1, c_row2_col2 = st.columns(2)
    with c_row2_col1:
        if charts.get("chart_4"):
            st.image(charts["chart_4"], caption="Figure 3: Detection Frame Classification Distribution", width="stretch")
    with c_row2_col2:
        if charts.get("chart_3"):
            st.image(charts["chart_3"], caption="Figure 4: Fire Detection Frequency by Frame Interval", width="stretch")

    c_row3_col1, c_row3_col2 = st.columns(2)
    with c_row3_col1:
        if charts.get("chart_6"):
            st.image(charts["chart_6"], caption="Figure 5: Fire Confidence Score Distribution", width="stretch")
    with c_row3_col2:
        if charts.get("chart_8"):
            st.image(charts["chart_8"], caption="Figure 6: Cumulative Fire Positive Frame Progression", width="stretch")

    c_row4_col1, c_row4_col2 = st.columns(2)
    with c_row4_col1:
        if charts.get("chart_9"):
            st.image(charts["chart_9"], caption="Figure 7: Alerts Dispatched vs. Cooldown Suppressed", width="stretch")
    with c_row4_col2:
        if charts.get("chart_10"):
            st.image(charts["chart_10"], caption="Figure 8: Dispatched Notifications across Email & Telegram", width="stretch")

    c_row5_col1, c_row5_col2 = st.columns(2)
    with c_row5_col1:
        if charts.get("chart_11"):
            st.image(charts["chart_11"], caption="Figure 9: Incident Lifecycle & Verification Progression", width="stretch")
    with c_row5_col2:
        if charts.get("chart_12"):
            st.image(charts["chart_12"], caption="Figure 10: Incident Phase Verification Breakdown", width="stretch")
