# Install ultralytics package (if not already installed)
# Run this in your terminal or as a script before running the training:
# pip install ultralytics

from ultralytics import YOLO
import os

def main():
    # Define the path to your YAML file
    data_yaml = r"c:\Users\ebins\MiniPro\data\yolo_dataset\data.yaml"

    # Verify if the data.yaml file exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"YAML file not found at: {data_yaml}")

    # Load a pretrained YOLOv8 model
    try:
        model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt', 'yolov8m.pt', or 'yolov8l.pt' for larger models
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")

    # Train the model
    try:
        model.train(
            data=data_yaml,
            epochs=100,
            batch=8,          # Reduced batch size
            imgsz=640,        # Reduced image size
            optimizer="Adam",
            lr0=0.001,
            warmup_epochs=3,
            augment=True,
            workers=8,
            device=0,         # Use GPU
            amp=False,        # Disable AMP
            save=True,
            save_period=10,
            patience=10,
        )


    except Exception as e:
        raise RuntimeError(f"Error during training: {e}")

    # Evaluate the trained model
    try:
        results = model.val()  # Validation metrics after training
        print(results)
    except Exception as e:
        raise RuntimeError(f"Error during validation: {e}")

    # Export the model to ONNX (optional)
    try:
        # Export to ONNX format
        model.export(format="onnx", imgsz=640, dynamic=True)  # dynamic=True for dynamic axes if needed
        print(f"Model exported successfully to ONNX format")
    except Exception as e:
        raise RuntimeError(f"Error exporting model: {e}")

if __name__ == "__main__":
    main()

