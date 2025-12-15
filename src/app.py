
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from fpdf import FPDF

# Page Config
st.set_page_config(page_title="EQUINOX | Advanced Vision", page_icon="🌑", layout="wide")

# Custom CSS - Dark Neomorphism (Sleek & Premium)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;700;900&display=swap');

    /* Global Settings */
    .stApp {
        background-color: #1e1e24; /* Dark Gunmetal */
        color: #f0f0f0;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Neomorphic Card Container */
    .neo-card {
        background: #1e1e24;
        border-radius: 20px;
        box-shadow:  9px 9px 18px #151519, 
                     -9px -9px 18px #27272f;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid #23232a;
    }
    
    /* Header Styling */
    .equinox-header {
        text-align: center;
        padding: 40px 0;
        margin-bottom: 30px;
        background: #1e1e24;
        border-radius: 30px;
        box-shadow:  15px 15px 30px #151519, 
                     -15px -15px 30px #27272f;
    }
    
    .equinox-title {
        font-size: 4.5rem;
        font-weight: 900;
        letter-spacing: 8px;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 10px 20px rgba(0, 201, 255, 0.2);
    }
    
    .equinox-subtitle {
        font-size: 1.2rem;
        color: #8892b0;
        letter-spacing: 3px;
        margin-top: 10px;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e24;
        border-right: none;
        box-shadow: 5px 0 15px rgba(0,0,0,0.2);
    }
    
    /* Input Fields & Sliders */
    .stTextInput>div>div>input {
        background-color: #1e1e24;
        color: #fff;
        border-radius: 10px;
        border: none;
        box-shadow: inset 4px 4px 8px #151519, 
                    inset -4px -4px 8px #27272f;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(145deg, #202026, #1b1b20);
        color: #00C9FF;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow:  6px 6px 12px #151519, 
                     -6px -6px 12px #27272f;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        color: #92FE9D;
        box-shadow: inset 6px 6px 12px #151519, 
                    inset -6px -6px 12px #27272f;
        transform: translateY(2px);
    }
    
    /* Metric/Result Items */
    .result-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px;
        margin: 10px 0;
        background: #1e1e24;
        border-radius: 12px;
        box-shadow: inset 3px 3px 6px #151519, 
                    inset -3px -3px 6px #27272f;
        border-left: 4px solid #00C9FF;
    }
    .result-label { font-weight: 700; color: #e0e0e0; }
    .result-conf { color: #00C9FF; font-family: monospace; }

    /* --- NEW MILITARY ELEMENTS --- */
    
    /* Blinking Status Light */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    .status-light {
        width: 12px;
        height: 12px;
        background-color: #00ff41;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ff41;
        animation: blink 2s infinite;
        display: inline-block;
        margin-right: 8px;
    }
    
    /* Telemetry Panel */
    .telemetry-box {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #00C9FF;
        border: 1px solid #00C9FF;
        padding: 10px;
        margin-top: 20px;
        background: rgba(0, 201, 255, 0.05);
    }
    
    /* Scanline Animation */
    .scan-container {
        position: relative;
        overflow: hidden;
    }
    .scan-container::after {
        content: " ";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background: linear-gradient(to bottom, transparent 50%, rgba(0, 255, 65, 0.1) 51%, transparent 51%);
        background-size: 100% 4px;
        pointer-events: none;
        z-index: 10;
    }
    
</style>
""", unsafe_allow_html=True)

# Helper: Robust Path Resolver
def resolve_path(filename, local_fallback=None):
    # 1. Check Root (Cloud/Deployment standard)
    if os.path.exists(filename):
        return filename
    
    # 2. Check web_deployment folder (Local test)
    if os.path.exists(os.path.join("web_deployment", filename)):
        return os.path.join("web_deployment", filename)

    # 3. Check final_submission folder (Local test)
    if os.path.exists(os.path.join("final_submission", filename)):
        return os.path.join("final_submission", filename)
        
    # 4. Check specific local paths (Dev environment)
    if local_fallback and os.path.exists(local_fallback):
        return local_fallback
        
    return filename # Return original to fail gracefully later

# Header Section
st.markdown("""
<div class="equinox-header">
    <div class="equinox-title">EQUINOX</div>
    <div class="equinox-subtitle">NEXT-GEN MILITARY INTELLIGENCE</div>
    <div style="margin-top:10px;">
        <span class="status-light"></span> SYSTEM ONLINE // ENCRYPTED UPLINK ESTABLISHED
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🎛️ CONTROL PANEL")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
conf_thresh = st.sidebar.slider("SENSITIVITY", 0.1, 1.0, 0.25, 0.05)

# Model Selector (Streamlit Cloud Ready)
model_choice = st.sidebar.selectbox(
    "NEURAL CORE",
    ("Gold Model (Precision)", "Nuclear Model (Power)", "Dual-Core Ensemble (Max Accuracy)")
)

if model_choice == "Gold Model (Precision)":
    model_path = resolve_path("best.pt", r"runs/train/yolov8l_gold/weights/best.pt")
elif model_choice == "Nuclear Model (Power)":
    model_path = resolve_path("nuclear.pt", r"runs/train/yolov8x_nuclear/weights/best.pt")
else:
    # Ensemble mode doesn't need single model_path, handled in logic
    model_path = None

# Load Model
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    if model_choice == "Dual-Core Ensemble (Max Accuracy)":
        # Check both models for existing
        p1 = resolve_path("best.pt", r"runs/train/yolov8l_gold/weights/best.pt")
        p2 = resolve_path("nuclear.pt", r"runs/train/yolov8x_nuclear/weights/best.pt")
        
        if os.path.exists(p1) and os.path.exists(p2):
             model = load_model(p1) # Load purely for metadata/readiness
             st.sidebar.markdown("""
            <div style="padding:10px; border-radius:10px; background:rgba(0,255,65,0.1); color:#00ff41; text-align:center; margin-top:20px; border: 1px solid #00ff41;">
                ● DUAL-CORE ONLINE
            </div>
            """, unsafe_allow_html=True)
        else:
             st.sidebar.error("⚠️ CORE FILES MISSING")
             model = None
             
    elif model_path and os.path.exists(model_path):
        model = load_model(model_path)
        
        # Dynamic Status Message
        if "Gold" in model_choice:
            status_msg = "● GOLD-CORE ONLINE"
            color = "#00ff41" # Green
            bg = "rgba(0,255,65,0.1)"
        elif "Nuclear" in model_choice:
            status_msg = "● NUCLEAR-CORE ONLINE"
            color = "#00ff41" # Green
            bg = "rgba(0,255,65,0.1)"
        else:
            status_msg = "● NEURAL CORE ACTIVE"
            color = "#00C9FF" # Blue
            bg = "rgba(0,201,255,0.1)"

        st.sidebar.markdown(f"""
        <div style="padding:10px; border-radius:10px; background:{bg}; color:{color}; text-align:center; margin-top:20px; border: 1px solid {color};">
            {status_msg}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("⚠️ CORE OFFLINE")
        model = None
except Exception as e:
    st.sidebar.error(f"SYSTEM FAILURE: {e}")
    model = None

# Real Telemetry (Moved AFTER model loading)
import time
import torch
from torchvision.ops import nms  # Added for Ensemble

# --- CUSTOM WEIGHTED BOX FUSION (Synced with Submission) ---
def custom_wbf(boxes_list, scores_list, labels_list, iou_thr=0.6, skip_box_thr=0.0001):
    if len(boxes_list) == 0:
        return np.array([]), np.array([]), np.array([])
        
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for i in range(len(boxes_list)):
        if len(boxes_list[i]) == 0: continue
        all_boxes.extend(boxes_list[i])
        all_scores.extend(scores_list[i])
        all_labels.extend(labels_list[i])

    if not all_boxes:
        return np.array([]), np.array([]), np.array([])
        
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    order = all_scores.argsort()[::-1]
    all_boxes = all_boxes[order]
    all_scores = all_scores[order]
    all_labels = all_labels[order]

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    used = np.zeros(len(all_boxes), dtype=bool)
    
    for i in range(len(all_boxes)):
        if used[i]: continue
        
        cluster_boxes = [all_boxes[i]]
        cluster_scores = [all_scores[i]]
        label = all_labels[i]
        used[i] = True
        
        for j in range(i + 1, len(all_boxes)):
            if used[j] or all_labels[j] != label: continue
            
            b1 = all_boxes[i]
            b2 = all_boxes[j]
            xx1 = max(b1[0], b2[0]); yy1 = max(b1[1], b2[1])
            xx2 = min(b1[2], b2[2]); yy2 = min(b1[3], b2[3])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            inter = w * h
            area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
            area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
            iou = inter / (area1 + area2 - inter + 1e-6)
            
            if iou > iou_thr:
                cluster_boxes.append(all_boxes[j])
                cluster_scores.append(all_scores[j])
                used[j] = True

        cluster_boxes = np.array(cluster_boxes)
        cluster_scores = np.array(cluster_scores)
        
        total_score = np.sum(cluster_scores)
        weighted_box = np.sum(cluster_boxes * cluster_scores[:, None], axis=0) / total_score
        
        n_models = 2
        boost_factor =  min(len(cluster_scores) / n_models, 1.2) 
        final_score = (total_score / len(cluster_scores)) * boost_factor
        final_score = min(final_score, 1.0)

        keep_boxes.append(weighted_box)
        keep_scores.append(final_score)
        keep_labels.append(label)

    return np.array(keep_boxes), np.array(keep_scores), np.array(keep_labels)


# Initialize session state for latency tracking
if 'inference_time' not in st.session_state:
    st.session_state.inference_time = 0

# System Stats
device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU (INTEL)"
active_classes = len(model.names) if model else 0
latency_display = f"{st.session_state.inference_time:.1f} ms" if st.session_state.inference_time > 0 else "STANDBY"

st.sidebar.markdown(f"""
<div class="telemetry-box">
    <b>📡 SYSTEM TELEMETRY</b><br>
    MODE: {model_choice}<br>
    UNIT: {device_name}<br>
    CLASSES: {active_classes}<br>
    LATENCY: {latency_display}<br>
    STATUS: OPERATIONAL
</div>
""", unsafe_allow_html=True)

# Main Interface
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="neo-card scan-container"><h3>📡 UPLINK FEED</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop Satellite Imagery", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_container_width=True, channels="RGB")
    st.markdown('</div>', unsafe_allow_html=True)
    
# Bottom Container for Full-Width Results
results_container = st.container()

with col2:
    st.markdown('<div class="neo-card"><h3>🎯 TARGET ACQUISITION</h3>', unsafe_allow_html=True)
    if uploaded_file is not None:
        if st.button("ENGAGE ANALYSIS"):
            # HUD Display (Active) with Extra Margin
            st.markdown("""
            <div style="font-family:'Courier New'; font-size:12px; margin:5px 0 40px 0; border:1px solid #00C9FF; padding:8px; background:rgba(0,201,255,0.05);">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>TARGETING ARRAY: <span style="color:#00ff41; text-shadow:0 0 5px #00ff41;">ONLINE</span></span>
                    <span>OPTICS: <span style="color:#00ff41">CALIBRATED</span></span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>LOCK STATUS: <span style="color:#ff003c; text-shadow:0 0 5px #ff003c; animation: blink 0.5s infinite;">ENGAGED</span></span>
                    <span>ZOOM: <span style="color:#00C9FF">1.0x</span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner('PROCESSING NEURAL LAYERS...'):
                start_time = time.time()
                
                # Common variable for results: List of (Name, Score)
                processed_boxes = [] 
                



                # --- ENSEMBLE LOGIC (TTA + WBF) ---
                if model_choice == "Dual-Core Ensemble (Max Accuracy)":
                    p1 = resolve_path("best.pt", r"runs/train/yolov8l_gold/weights/best.pt")
                    p2 = resolve_path("nuclear.pt", r"runs/train/yolov8x_nuclear/weights/best.pt")
                    
                    if not os.path.exists(p1) or not os.path.exists(p2):
                        st.error(f"MISSING MODEL FILES. Gold: {os.path.exists(p1)}, Nuclear: {os.path.exists(p2)}")
                    else:
                        m1 = YOLO(p1)
                        m2 = YOLO(p2)
                        
                        # TTA Pipeline: Original + Flipped
                        candidates_boxes = []
                        candidates_scores = []
                        candidates_labels = []
                        
                        img_h, img_w = image.shape[:2]
                        img_flipped = cv2.flip(image, 1)
                        
                        models = [m1, m2]
                        
                        for model in models:
                            # Pass 1: Original
                            r = model.predict(image, conf=0.10, verbose=False)[0] # Low conf for WBF
                            if len(r.boxes) > 0:
                                candidates_boxes.append(r.boxes.xyxy.cpu().numpy())
                                candidates_scores.append(r.boxes.conf.cpu().numpy())
                                candidates_labels.append(r.boxes.cls.cpu().numpy())
                                
                            # Pass 2: Flipped (TTA)
                            r_flip = model.predict(img_flipped, conf=0.10, verbose=False)[0]
                            if len(r_flip.boxes) > 0:
                                boxes_f = r_flip.boxes.xyxy.cpu().numpy()
                                scores_f = r_flip.boxes.conf.cpu().numpy()
                                labels_f = r_flip.boxes.cls.cpu().numpy()
                                
                                # Invert Flip
                                fixed_boxes = []
                                for b in boxes_f:
                                    x1, y1, x2, y2 = b
                                    fx1 = img_w - x2
                                    fx2 = img_w - x1
                                    fixed_boxes.append([fx1, y1, fx2, y2])
                                    
                                candidates_boxes.append(np.array(fixed_boxes))
                                candidates_scores.append(scores_f)
                                candidates_labels.append(labels_f)
                    
                        # Apply WBF
                        final_boxes, final_scores, final_cls = custom_wbf(
                            candidates_boxes, candidates_scores, candidates_labels, iou_thr=0.60
                        )
                        
                        # Visualization
                        res_plotted = image.copy()
                        names = m1.names
                        
                        # Filter by user threshold
                        for box, score, cls_id in zip(final_boxes, final_scores, final_cls):
                            if score < conf_thresh: continue
                            
                            x1, y1, x2, y2 = map(int, box)
                            label = f"{names[int(cls_id)]} {score:.2f}"
                            cv2.rectangle(res_plotted, (x1, y1), (x2, y2), (0, 255, 65), 2)
                            (w, h), _ = cv2.getTextSize(label, 0, 0.5, 1)
                            cv2.rectangle(res_plotted, (x1, y1 - 20), (x1 + w, y1), (0, 255, 65), -1)
                            cv2.putText(res_plotted, label, (x1, y1 - 5), 0, 0.5, (0, 0, 0), 1)
                            processed_boxes.append((names[int(cls_id)], score))
                            
                        # Display
                        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        st.image(res_plotted_rgb, use_container_width=True)

                else:
                    # --- SINGLE MODEL LOGIC ---
                    if model is not None:
                        results = model.predict(image, conf=conf_thresh)
                        res_plotted = results[0].plot()
                        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        st.image(res_plotted_rgb, use_container_width=True)
                        
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            for box in boxes:
                                cls_id = int(box.cls[0])
                                cls_name = model.names[cls_id].upper()
                                conf = float(box.conf[0])
                                processed_boxes.append((cls_name, conf))

                end_time = time.time()
                st.session_state.inference_time = (end_time - start_time) * 1000
                
                # --- RENDER MISSION INTEL (Bottom Section) ---
                with results_container:
                     st.markdown("""
                     <div style="margin-top: 30px; padding: 20px; background: #1e1e24; border-radius: 15px; border: 1px solid #23232a; box-shadow: 10px 10px 20px #151519, -10px -10px 20px #27272f;">
                        <h2 style="text-align:center; color:#00C9FF; letter-spacing:4px; margin-bottom:20px;">📊 MISSION INTEL</h2>
                     </div>
                     """, unsafe_allow_html=True)
                     
                     c1, c2 = st.columns([1, 1])
                     with c1:
                         st.markdown("<h4>DETECTED SIGNATURES</h4>", unsafe_allow_html=True)
                         if len(processed_boxes) > 0:
                            for name, score in processed_boxes:
                                st.markdown(f"""
                                <div class="result-item">
                                    <span class="result-label">{name.upper()}</span>
                                    <span class="result-conf">{score:.1%}</span>
                                </div>
                                """, unsafe_allow_html=True)
                         else:
                             st.info("NO THREATS IDENTIFIED")
                             
                     with c2:
                         st.markdown("<h4>📝 MISSION LOG</h4>", unsafe_allow_html=True)
                         st.markdown(f"""
                         <div style="font-family:'Courier New'; font-size:12px; color:#8892b0; line-height:1.6;">
                            > [SYSTEM] INITIALIZING NEURAL CORE... OK<br>
                            > [SYSTEM] UPLINK ESTABLISHED... OK<br>
                            > [SCAN] TARGET ACQUIRED: {len(processed_boxes)} SIGNATURES<br>
                            > [ANALYSIS] CONFIDENCE THRESHOLD: {conf_thresh}<br>
                            > [STATUS] MISSION ACTIVE<br>
                         </div>
                         """, unsafe_allow_html=True)
                         
                         # Generate PDF Report
                         pdf = FPDF()
                         pdf.add_page()
                         
                         # Header
                         pdf.set_font("Arial", 'B', 16)
                         pdf.cell(0, 10, "EQUINOX DEFENSE SYSTEM", 0, 1, 'C')
                         pdf.set_font("Arial", 'I', 10)
                         pdf.cell(0, 10, "MISSION INTELLIGENCE REPORT", 0, 1, 'C')
                         pdf.line(10, 30, 200, 30)
                         pdf.ln(10)
                         
                         # Mission Details
                         pdf.set_font("Arial", 'B', 12)
                         pdf.cell(0, 10, "MISSION STATUS: SUCCESS", 0, 1)
                         pdf.set_font("Arial", '', 10)
                         pdf.cell(0, 8, f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
                         pdf.cell(0, 8, f"Unit ID: {device_name}", 0, 1)
                         pdf.cell(0, 8, f"Neural Core: {model_choice}", 0, 1)
                         pdf.cell(0, 8, f"Conf. Threshold: {conf_thresh}", 0, 1)
                         pdf.ln(5)
                         
                         # Signatures Table
                         pdf.set_font("Arial", 'B', 12)
                         pdf.cell(0, 10, "DETECTED SIGNATURES", 0, 1)
                         pdf.set_fill_color(200, 220, 255)
                         pdf.set_font("Arial", 'B', 10)
                         pdf.cell(100, 8, "Signature Class", 1, 0, 'L', 1)
                         pdf.cell(50, 8, "Confidence", 1, 1, 'L', 1)
                         
                         pdf.set_font("Arial", '', 10)
                         if len(processed_boxes) > 0:
                             for name, score in processed_boxes:
                                 pdf.cell(100, 8, name.upper(), 1, 0)
                                 pdf.cell(50, 8, f"{score:.2%}", 1, 1)
                         else:
                             pdf.cell(150, 8, "No active threats detected.", 1, 1)
                             
                         # Footer
                         pdf.ln(10)
                         pdf.set_font("Arial", 'I', 8)
                         pdf.cell(0, 10, f"Generated by Equinox AI | Latency: {latency_display}", 0, 1)
                         
                         # Output to bytes
                         pdf_bytes = pdf.output(dest='S').encode('latin-1')

                         st.markdown("<br>", unsafe_allow_html=True)
                         st.download_button(
                             label="📄 DOWNLOAD MISSION REPORT (PDF)",
                             data=pdf_bytes,
                             file_name="mission_report.pdf",
                             mime="application/pdf",
                             use_container_width=True
                         )

    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#444;">
            AWAITING DATA STREAM...
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br><center style='color:#444; letter-spacing:2px;'>EQUINOX DEFENSE SYSTEMS © 2025</center>", unsafe_allow_html=True)
