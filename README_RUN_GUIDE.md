# 🚀 JUDGE'S QUICK START GUIDE

**Welcome to the Equinox Defense System.**
This guide will help you run our detection system in less than 2 minutes.

---

## ✅ STEP 1: Installation
Open your terminal (Command Prompt or PowerShell) in this folder and run:

```bash
pip install -r requirements.txt
```
*Note: This installs YOLOv8, Streamlit, and SAHI (Slicing Inference).*

---

## 🖥️ STEP 2: Run the Web Dashboard (The "Wow" Factor)
We have built a professional "Mission Control" interface to visualize the detections.

**Run this command:**
```bash
streamlit run src/app.py
```

**How to use the Dashboard:**
1.  A website will open in your browser.
2.  **Look at the Sidebar**:
    *   **"NEURAL CORE"**: This dropdown lets you choose the model.
    *   Select **"Dual-Core Ensemble (Max Accuracy)"** to see our system at full power.
3.  **"Drop Satellite Imagery"**: Drag and drop any image from the `test/images` folder here.
4.  Click **"ENGAGE ANALYSIS"**.
5.  **Result**: The system will scan the image, locate targets, and draw neon bounding boxes around them.

5.  **Result**: The system will scan the image, locate targets, and draw neon bounding boxes around them.

> **⚠️ IMPORTANT**: If the website says **"The app has gone to sleep"**, do not worry! This is normal for cloud deployments to save energy.
> *   Click the button **"Yes, get this app back up"**.
> *   Wait ~15 seconds, and it will be online.

---

## ⚡ STEP 3: Reproduce Our Results (Predictions)
If you want to generate the `.txt` prediction files yourself (the ones provided in `predictions.zip`):

**Run this command:**
```bash
python src/ensemble_predict.py
```
*   This script runs BOTH the **Gold Model** (Precision) and **Nuclear Model** (Power).
*   It merges their results for maximum accuracy.
*   Results will be saved in a new folder: `predictions_ensemble/`.

---

## 📊 STEP 4: Verification (Speed Test)
To verify our claim of **62 FPS** (Real-time efficiency), run:

```bash
python src/benchmark.py
```
*   This will stress-test the model on your hardware and print the FPS (Frames Per Second).

---

## ❓ Troubleshooting
*   **"SAHI not installed"**: Run `pip install sahi`.
*   **"CUDA OOM"**: If you have a small GPU, the website handles errors gracefully. You can also switch to "CPU" mode automatically by just running the code (it auto-detects).
*   **"File not found"**: Ensure you are in the root directory where `README.md` is located.

---
**Equinox Defense Systems**
*Ready for Deployment.*
