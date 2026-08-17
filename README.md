# Cloud-to-Edge YOLOv5 Object Detection Pipeline

##  Overview
This repository demonstrates a complete, end-to-end Computer Vision lifecycle. It begins with cloud-based dataset ingestion and model fine-tuning on a Tesla T4 GPU, and concludes with a hardware-agnostic local deployment using a custom OpenCV inference script. 

Instead of relying on YOLO's default global confidence settings, this project features dynamic, class-specific thresholding to optimize precision and recall for individual objects in real-time.

##  Key Features
* **Automated Data Engineering:** Utilizes the Roboflow API to programmatically download, extract, and clean datasets, including dynamic YAML restructuring for YOLOv5 compatibility.
* **Cloud GPU Training:** Fine-tunes a `yolov5s` model over 37 epochs with Early Stopping, achieving a mAP50 of 0.86. 
* **Edge Deployment:** Loads the optimized custom weights (`best.pt`) locally using PyTorch Hub for real-time webcam inference.
* **Custom Filtering Logic:** Bypasses standard Non-Maximum Suppression (NMS) confidence defaults by implementing strict, class-specific thresholds (e.g., `Banana: 0.7`, `Apple: 0.4`, `Orange: 0.01`) directly into the rendering loop.

##  Tech Stack
* **Deep Learning:** PyTorch, YOLOv5 (Ultralytics)
* **Computer Vision:** OpenCV, NumPy
* **Data Ops:** Roboflow API, YAML
* **Hardware:** Tesla T4 (Training), Local CPU (Inference)

##  How to Run Locally

### 1. Environment Setup
Create and activate a virtual environment (Python 3.10+ recommended):
```powershell
python -m venv yolov5env
yolov5env\Scripts\activate
```

### 2. Install Dependencies
Install the required packages (CPU-optimized PyTorch is sufficient for inference):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Real-Time Inference
Ensure `best.pt` is in your root directory and run the custom detection script. Press `q` to exit the webcam stream.
```bash
python inference.py
```
*(Note: If running inside a Jupyter Notebook, execute the inference cell directly).*

##  Author
**Mohammed Essalhi**
* [LinkedIn](https://linkedin.com/in/mohammed-essalhi-23794b24b)
