
# 🦅 Equinox Deployment Container
# Lightweight, scalable, and ready for edge deployment.

# 1. Base Image (Slim Python for Efficiency)
FROM python:3.10-slim

# 2. System Dependencies (OpenCV requires these)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory
WORKDIR /app

# 4. Install Python Dependencies
# We leverage caching by copying requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics sahi openvino

# 5. Copy Source Code & Model
COPY src/ ./src/
COPY final_submission/best.pt ./model/best.pt

# 6. Expose Port for Streamlit
EXPOSE 8501

# 7. Healthcheck (For Kubernetes/Scalability)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 8. Launch Command
CMD ["streamlit", "run", "src/app_demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
