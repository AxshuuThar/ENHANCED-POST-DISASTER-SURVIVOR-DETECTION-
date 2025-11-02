import os
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# List of dataset paths (modify as needed)
datasets = [
    r"C:\Users\ebins\MiniPro\data\uavhuman1\data.yaml",
    r"C:\Users\ebins\MiniPro\data\uavhuman2\data.yaml",
    r"C:\Users\ebins\MiniPro\data\uavhuman3\data.yaml",
    r"C:\Users\ebins\MiniPro\data\uavhuman4\data.yaml",
    r"C:\Users\ebins\MiniPro\data\yolo_dataset\data.yaml"
]

# Path to your best pre-trained YOLO model
best_model_path = r"C:\Users\ebins\MiniPro\code\runs\detect\train3\weights\bestt.pt"

# Ensure model exists
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Best model not found at: {best_model_path}")


def plot_graphs(training_metrics):
    """Generate graphs for training and validation metrics."""
    sns.set(style="darkgrid")
    epochs = list(range(1, len(training_metrics['train_loss']) + 1))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_metrics['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, training_metrics['val_loss'], label='Val Loss', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_metrics['map'], label='mAP', color='g', marker='D')
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.title("mAP Progression")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_and_evaluate():
    # Load the YOLOv8 model
    model = YOLO(best_model_path)
    print("✅ Loaded best model for fine-tuning.")

    # Store training metrics
    training_metrics = {'train_loss': [], 'val_loss': [], 'map': []}

    best_map = model.val().box.map  # Get mAP of the current best model
    best_model_found = best_model_path  # Track the best model file

    print(f"🔍 Initial Best Model: {best_model_path} | mAP: {best_map:.4f}")

    # Train the model on each dataset sequentially
    for dataset in datasets:
        if not os.path.exists(dataset):
            print(f"⚠️ Skipping missing dataset: {dataset}")
            continue

        print(f"\n🚀 Training started on: {dataset} ...")

        results = model.train(
            data=dataset,
            epochs=100,
            batch=8,
            imgsz=640,
            optimizer="Adam",
            lr0=0.0005,
            warmup_epochs=2,
            augment=True,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0,
            workers=2,
            device=0,
            save=True,
            save_period=5,
            patience=10,
        )

        print(f"✅ Finished training on {dataset}")

        # Store loss values
        training_metrics['train_loss'].append(results.box.loss)
        training_metrics['val_loss'].append(results.box.loss_val)

        # Validate after each dataset
        print(f"📊 Evaluating the model on {dataset} ...")
        val_results = model.val()
        new_map = val_results.box.map  # Get new mAP
        training_metrics['map'].append(new_map)

        print(f"📊 Validation mAP for {dataset}: {new_map:.4f}")

        # Check if the new model is better
        last_trained_model_path = model.ckpt_path  # Get latest trained model's path
        if new_map > best_map:
            print(f"🏆 New best model found! Updating from {best_map:.4f} to {new_map:.4f}")
            best_map = new_map
            best_model_found = last_trained_model_path

    # Display the best model found
    print("\n🎯 Final Best Model:")
    print(f"📍 Path: {best_model_found}")
    print(f"📊 Best mAP: {best_map:.4f}")

    # Generate training graphs
    plot_graphs(training_metrics)

    # Export the final best model to ONNX format
    print("📦 Exporting best model to ONNX format...")
    best_model = YOLO(best_model_found)
    best_model.export(format="onnx", imgsz=640, dynamic=True)
    print("✅ Model exported successfully to ONNX format.")

    print("🎉 Training completed on all datasets!")


if __name__ == "__main__":
    train_and_evaluate()