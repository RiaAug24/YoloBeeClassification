
# Enhanced Bee Tracking and Classification Pipeline

A comprehensive and modular deep learning pipeline to detect and classify bee behaviors using YOLOv5, YOLOv7, or YOLOv8. This system facilitates setup, training, evaluation, benchmarking, and prediction from video files with robust logging and error handling.

---

## Pipeline Features

- Supports **YOLOv5**, **YOLOv7**, and **YOLOv8**
- Automatic environment setup and dataset validation
- Trains or loads existing models dynamically
- Predicts bee behaviors in video files
- Analyzes predictions and class distribution
- Benchmarks performance (FPS, inference time)
- Exports models to ONNX (for YOLOv8)
- Clean and extensible code with logging

---

##  Supported Bee Behaviors

The model is trained to classify the following bee activities:
- **foraging**
- **defense**
- **fanning**
- **washboarding**

---

## 📁 Project Structure

```
📦 your_project_directory/
 ┣ YoloBeeDetection&Classifier.py
 ┣ 📂 yolov5/ (auto-cloned if using YOLOv5)
 ┣ 📂 yolov7/ (auto-cloned if using YOLOv7)
 ┣ 📂 weights/           # Trained models will be saved here
 ┣ 📂 results/           # Training results and metrics
 ┣ 📂 output/            # Inference outputs
 ┣ 📂 runs/train/        # YOLO training logs and weights
 ┣ 📄 data.yaml          # Auto-generated dataset config
 ┗ 📁 BeeDataset/
     ┣ 📂 images/
     ┃ ┣ 📂 train/
     ┃ ┗ 📂 val/
     ┗ 📂 labels/
       ┣ 📂 train/
       ┗ 📂 val/
```

---

## ⚙️ Installation

```bash
# Clone this repo or use your own script
git clone https://github.com/<your-repo>/bee-tracking-pipeline.git
cd bee-tracking-pipeline

# Install dependencies (YOLOv5, YOLOv7, or Ultralytics YOLO)
pip install -r requirements.txt  # For YOLOv5/YOLOv7
pip install ultralytics          # For YOLOv8
```

---

## 🏁 Quick Start

```python
from YoloBeeDetection&Classifier import EnhancedBeeTrackingPipeline

pipeline = EnhancedBeeTrackingPipeline(
    base_path=".", 
    model_type="yolov8"  # or "yolov5", "yolov7"
)

pipeline.run_full_pipeline(
    video_path="your_video_or_directory_path",
    force_train=False,
    epochs=15,
    batch_size=4,
    conf_threshold=0.25
)

pipeline.cleanup()
```

---

## Benchmarking

To benchmark the model on a set of images:

```python
pipeline.benchmark_model("test_images/")
```

---

## Exporting Model

YOLOv8 models can be exported to formats like ONNX:

```python
pipeline.export_model(format="onnx")
```

---

## 🧹 Cleanup

To free GPU memory and clean temporary resources:

```python
pipeline.cleanup()
```

---

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (for video processing)
- YOLOv5 / YOLOv7 / Ultralytics (YOLOv8)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Ultralytics YOLOv5 & YOLOv8](https://github.com/ultralytics/yolov5)
- [WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7)

---
## Authors
Developed by: 
- Dr. Jason Elroy Martis - Associate Professor @NMAMIT, Nitte
- Riyaz Ahmed - Alumini @NMMAIT, Nitte
- Shrisha SK - Alumini @NMAMIT, Nitte
- Shreesha - Alumini @NMAMIT, Nitte

