# 🚗 Real-Time Road Anomaly Detection on Raspberry Pi 4

### Bharat AI SoC Student Challenge – Problem Statement 3

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red)](https://www.raspberrypi.org/)
[![Model](https://img.shields.io/badge/Model-YOLO11n-blue)](https://github.com/ultralytics/ultralytics)
[![Runtime](https://img.shields.io/badge/Runtime-ONNX-green)](https://onnxruntime.ai/)
[![FPS](https://img.shields.io/badge/FPS-~5-orange)](https://github.com)

---

## 📌 Project Overview

This project implements a real-time **Edge AI system** on Raspberry Pi 4 for detecting road anomalies from recorded dashcam footage.

### Detected Objects:
- 🕳️ **Potholes** (with diameter estimation in pixels)
- 🚗 **Vehicles** (with motion classification: Moving / Stationary / Unknown)
- 🦌 **Animals**

The model is based on **YOLO11n**, exported to ONNX format, and deployed using **ONNX Runtime** on Raspberry Pi.

All detections are logged in a structured CSV file, and the annotated output video displays FPS and frame statistics.

---

## 🎯 Objective

- ✅ Achieve **≥5 FPS** inference on Raspberry Pi 4
- ✅ Detect and log road anomalies from recorded video
- ✅ Maintain high precision with reduced false positives
- ✅ Perform complete edge processing (no cloud dependency)

---

## 🛠 Hardware Used

| Component | Specification |
|-----------|---------------|
| **Board** | Raspberry Pi 4 |
| **OS** | 64-bit Raspberry Pi OS |
| **Storage** | High-speed microSD card |
| **Input Source** | Recorded MP4 dashcam footage |

---

## 💻 Software Stack

```
Python 3.9+
OpenCV
NumPy
Pandas
ONNX Runtime
YOLO11n (custom-trained for pothole detection)
```

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| **Base Model** | YOLO11n |
| **Training** | Custom-trained for pothole detection |
| **Exported Model** | `best.onnx` |
| **Inference Engine** | ONNX Runtime |
| **Deployment Mode** | CPU (ARM Cortex-A72) |
| **Average Performance** | ~5 FPS |

---

## ⚙️ System Architecture

```
+-----------------------------+
| Recorded Dashcam Video (MP4) |
+-----------------------------+
              |
              v
+-----------------------------+
| OpenCV Video Capture        |
+-----------------------------+
              |
              v
+-----------------------------+
| Frame Preprocessing         |
| (Resize → Normalize →       |
|  Format Conversion)         |
+-----------------------------+
              |
              v
+-----------------------------+
| ONNX Runtime Inference      |
| (YOLO11n - ARM CPU)         |
+-----------------------------+
              |
              v
+-----------------------------+
| Post-Processing             |
| (NMS + Confidence Filter)   |
+-----------------------------+
              |
              v
+-----------------------------+
| Object Classification       |
| (Potholes / Vehicles /      |
|  Animals / Obstacles)       |
+-----------------------------+
              |
              v
+-----------------------------+
| Feature Extraction          |
| - Diameter Estimation       |
| - Motion Classification     |
+-----------------------------+
              |
              v
+-----------------------------+
| Logging Module              |
| (CSV File Storage)          |
+-----------------------------+
              |
              v
+-----------------------------+
| Output Display              |
| (BBox + FPS + Frame Count)  |
+-----------------------------+
```

---

## 📊 Output Logging

All detections are saved in a structured CSV file (`detection_log.csv`).

### CSV Format

| Serial_Number | Frame_Number | Class | Confidence | BBox (x,y,w,h) | Diameter | Motion_Status |
|---------------|--------------|-------|------------|----------------|----------|---------------|
| 1 | 42 | pothole | 0.89 | (120,340,45,52) | 48.5 | N/A |
| 2 | 43 | vehicle | 0.92 | (300,200,80,60) | N/A | Moving |
| 3 | 45 | animal | 0.85 | (450,150,30,40) | N/A | N/A |

#### Column Descriptions:
- **Class** → pothole / vehicle / animal / obstacle
- **Diameter** → Calculated for potholes (in pixels)
- **Motion_Status** → Moving / Stationary / Unknown (for vehicles)

---

## 📂 Repository Structure

```
real-time-road-anomaly-raspberrypi/
│
├── README.md                        # Project documentation
├── main.py                          # Main detection script
├── best.onnx                        # YOLO11n ONNX model
├── yolo11n.pt                       # YOLO11n PyTorch model
│
├── demo1.mp4                        # Demo output video 1
├── demo2.mp4                        # Demo output video 2
├── demo3.mp4                        # Demo output video 3
│
├── demo1.csv                        # Detection results for video 1
├── demo2.csv                        # Detection results for video 2
└── demo3.csv                        # Detection results for video 3
```

---

## 💡 Code Structure

### 📝 [View Full Source Code - main.py](main.py)

Click above to view the complete implementation! The code includes:

#### **RoadAnomalyDetector Class**

```python
class RoadAnomalyDetector:
    """Real-time road anomaly detection using YOLO11n ONNX model"""
```

**Key Methods:**

1. **`__init__()`** - Initialize ONNX Runtime session and model parameters
2. **`preprocess()`** - Resize, normalize, and format input images
3. **`postprocess()`** - Parse YOLO outputs and apply NMS
4. **`estimate_pothole_diameter()`** - Calculate pothole size in pixels
5. **`classify_vehicle_motion()`** - Determine if vehicle is moving/stationary
6. **`log_detection()`** - Record detections to internal log
7. **`save_log_to_csv()`** - Export detection log to CSV
8. **`draw_detections()`** - Annotate frames with bounding boxes
9. **`process_video()`** - Main video processing pipeline

#### **Detection Pipeline**

```python
Frame Input → Preprocess → ONNX Inference → Postprocess → 
Feature Extraction → Logging → Annotation → Output
```

---

## ▶️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/gayatri719/real-time-road-anomaly-raspberrypi.git
cd real-time-road-anomaly-raspberrypi
```

### 2️⃣ Install Dependencies

```bash
pip install opencv-python numpy pandas onnxruntime
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python main.py
```

### 4️⃣ Output

The system will:
- ✅ Process recorded dashcam footage
- ✅ Perform real-time inference (~5 FPS)
- ✅ Display annotated output video
- ✅ Save detection results in CSV format
- ✅ Generate demo videos with corresponding CSV logs

---

## 📈 Performance Summary

| Parameter | Value |
|-----------|-------|
| **Platform** | Raspberry Pi 4 (ARM Cortex-A72) |
| **OS** | 64-bit Raspberry Pi OS |
| **Model Format** | ONNX |
| **Inference Engine** | ONNX Runtime |
| **Average FPS** | ~5 FPS |
| **Input Source** | Recorded MP4 Video |

---

## 🔬 Optimization Strategy

- 🎯 Lightweight **YOLO11n** architecture selected for edge deployment
- ⚡ **ONNX Runtime** used for efficient ARM CPU execution
- 🎚️ Confidence threshold tuning to reduce false positives
- 📐 Optimized input resolution for balanced speed and accuracy

---

## 📄 Project Documentation

### 📑 Detailed Technical Report

A comprehensive technical report is available that covers the complete implementation details, methodology, results, and analysis of this project.

**[📥 Download Full Project Report (PDF)](Document.pdf)**

### Report Highlights:

#### 🎯 **Executive Summary**
- Real-time edge AI system on Raspberry Pi 4
- 5 FPS processing with YOLOv11n models
- Dual-model architecture for potholes and obstacles
- Comprehensive logging and motion tracking

#### 🔬 **Technical Implementation**
- **Novel Contributions:**
  - Dual-model edge deployment (ONNX + PyTorch)
  - Intelligent multi-frame motion tracking
  - Automated pothole severity assessment
  - Adaptive confidence thresholding
  - Comprehensive edge logging system

#### 📊 **Key Results**
| Metric | Target | Achieved |
|--------|--------|----------|
| Inference FPS | ≥5 FPS | ✅ 5.0 FPS |
| Processing Time | <200 ms | ✅ ~200 ms |
| Memory Usage | <2 GB | ✅ ~800 MB |
| CPU Utilization | Optimized | ✅ 60-70% |



### 📋 Report Contents:
1. Executive Summary
2. Introduction & Problem Statement
3. Methodology & System Architecture
4. Technical Implementation
5. Hardware Utilization & Optimization
6. Results & Performance Analysis
7. Challenges & Solutions
8. Future Enhancements
9. Code Documentation
10. Installation Guide

---

## 🎬 Demo Videos

Three demonstration videos showing real-time detection, FPS display, and CSV logging. **Click on the video titles to play them directly!**

### 🎬 Demo Video 1 - Rural Road with Potholes

[▶️ **Click here to watch demo1.mp4**](https://github.com/gayatri719/real-time-road-anomaly-raspberrypi/blob/main/demo1.mp4)

Rural road scenario demonstrating pothole detection capabilities.

📊 **[View Detection Results (demo1.csv)](demo1.csv)**

---

### 🎬 Demo Video 2 - Highway with Vehicles and Potholes

[▶️ **Click here to watch demo2.mp4**](https://github.com/gayatri719/real-time-road-anomaly-raspberrypi/blob/main/demo2.mp4)

Highway scenario showing vehicle tracking and pothole detection.

📊 **[View Detection Results (demo2.csv)](demo2.csv)**

---

### 🎬 Demo Video 3 - Rural Road with Animals and Vehicles

[▶️ **Click here to watch demo3.mp4**](https://github.com/gayatri719/real-time-road-anomaly-raspberrypi/blob/main/demo3.mp4)

Complex scenario with animals, vehicles, and road anomalies.

📊 **[View Detection Results (demo3.csv)](demo3.csv)**

---

> **Note:** Click on any video link above, then click the "View raw" or download button on GitHub to play the video. You can also click on the CSV files to view the detailed detection logs.

---

## 🎓 Learning Outcomes

- ✅ Edge AI deployment on ARM architecture
- ✅ Neural network optimization for embedded systems
- ✅ Real-time computer vision pipeline development
- ✅ Understanding speed vs accuracy trade-offs

---

## 🚀 Future Improvements

- [ ] **INT8 quantization** for higher FPS
- [ ] **TensorFlow Lite** comparison
- [ ] **Distance estimation** using monocular depth
- [ ] **GPS-based anomaly tagging**
- [ ] Multi-threading for parallel processing
- [ ] Real-time streaming support

---



## 👩‍💻 Author

**OMKARA SRI HARSHA P**  
B.Tech Electronics & Communication Engineering  
Bharat AI SoC Student Challenge

---

---

<p align="center">Made with ❤️ for safer roads</p>
