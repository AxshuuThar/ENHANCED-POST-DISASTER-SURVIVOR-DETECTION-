# Enhanced Post-Disaster Survivor Detection and Analysis Using YOLOv8 and Transformer Models

![Project Banner](https://img.shields.io/badge/Project-AI%20for%20Disaster%20Response-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green)
![Transformers](https://img.shields.io/badge/Transformers-Scene%20Understanding-orange)

## 📋 Project Overview

This project implements an advanced AI system for post-disaster survivor detection and analysis by integrating **YOLOv8** for real-time object detection with **Transformer models** for enhanced scene understanding and error correction. The system is designed to improve search and rescue (SAR) operations in disaster scenarios.

## 🎯 Key Features

- **Real-time Survivor Detection**: YOLOv8-based detection with high accuracy
- **Scene Understanding**: Transformer models for contextual analysis
- **Error Correction**: Reduces false positives and negatives
- **Heatmap Generation**: Visualizes survivor concentration areas
- **Temporal Tracking**: DeepSORT with Kalman filtering for object tracking
- **Multimodal Analysis**: Integration of image and metadata

## 🏗️ System Architecture

### Core Components:
1. **YOLOv8 Detection Pipeline**
   - Real-time object detection
   - Non-Maximum Suppression (NMS)
   - Confidence-based scoring

2. **Transformer Integration (BERT)**
   - Scene context analysis
   - Error correction and validation
   - Multimodal data fusion

3. **Supporting Architectures**
   - DeepSORT for object tracking
   - Kalman Filter for motion prediction
   - DBSCAN for spatial clustering
   - KDE for heatmap generation

## 📊 Performance Metrics

- **High Precision & Recall**: Improved detection accuracy
- **Real-time Processing**: Optimized for search and rescue operations
- **Reduced False Detections**: Enhanced by transformer-based validation

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- Transformers library
- OpenCV

### Setup
```bash
# Clone the repository
git clone https://github.com/ebbran/Enhanced-Survivor-Detection.git
cd Enhanced-Survivor-Detection

# Install dependencies
pip install -r requirements.txt

# For YOLOv8 specific installation
pip install ultralytics
```
📈 Results
The system demonstrates:

Enhanced detection accuracy in post-disaster scenarios

Reduced false positives through transformer validation

Real-time processing capabilities suitable for UAV deployment

Improved scene understanding in complex environments

🔬 Research Contributions
Addressed Research Gaps:
Dataset Limitations: Integrated multiple disaster datasets

Advanced Detection Features: Dynamic heatmaps, crowd density analysis, temporal tracking

Transformer Integration: Scene understanding and error correction

🎯 Applications
Search and Rescue Operations: Real-time survivor detection

Disaster Management: Damage assessment and analysis

Infrastructure Planning: AI-powered reconstruction recommendations

UAV-based Surveillance: Autonomous disaster response
