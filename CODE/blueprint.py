import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from typing import Union

# -------------------------------
# Configuration Constants
# -------------------------------
CONFIDENCE_THRESHOLD = 0.8
SEGMENTATION_OVERLAY_ALPHA = 0.5
DAMAGE_CLASS_ID = 1  # Note: This should be adjusted based on your model's classes
DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX


# -------------------------------
# Stage 1: Damage & Object Detection
# -------------------------------
class ObjectDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def detect(self, image: Image.Image) -> dict:
        img_tensor = self.transform(image).to(self.device)
        with torch.no_grad():
            predictions = self.model([img_tensor])
        return predictions[0]


# -------------------------------
# Stage 2: Scene Understanding
# -------------------------------
class SceneSegmenter:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def segment(self, image: Image.Image) -> np.ndarray:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)['out'][0]
        return output.argmax(0).cpu().numpy()


# -------------------------------
# Stage 3: Risk Assessment & Recommendations
# -------------------------------
class DisasterAnalyzer:
    def __init__(self):
        self.risk_weights = (0.6, 0.4)  # (damage_ratio_weight, detection_weight)

    def calculate_risk(self, detection_output: dict, segmentation_mask: np.ndarray) -> tuple:
        damage_area = np.sum(segmentation_mask == DAMAGE_CLASS_ID)
        damage_ratio = damage_area / segmentation_mask.size

        scores = detection_output['scores'].cpu().numpy()
        high_conf_detections = len(scores[scores > CONFIDENCE_THRESHOLD])

        risk_score = np.clip(
            self.risk_weights[0] * damage_ratio +
            self.risk_weights[1] * (high_conf_detections / 10.0),
            0, 1
        )
        return risk_score, high_conf_detections, damage_ratio

    def generate_recommendations(self, risk_score: float) -> str:
        if risk_score > 0.8:
            return ("🚨 Critical Damage: Immediate evacuation recommended. Prioritize structural stabilization "
                    "and emergency response before any reconstruction.")
        elif risk_score > 0.5:
            return ("⚠️ Significant Damage: Restricted access recommended. Focus on temporary stabilization "
                    "and detailed structural assessment.")
        else:
            return ("✅ Moderate Damage: Safety inspection recommended. Plan phased repairs with "
                    "continuous monitoring.")


# -------------------------------
# Visualization Utilities
# -------------------------------
class VisualizationEngine:
    @staticmethod
    def overlay_results(image: np.ndarray, detections: dict, segmentation: np.ndarray,
                        risk_data: tuple) -> np.ndarray:
        # Process detections
        boxes = detections['boxes'].cpu().numpy().astype(int)
        scores = detections['scores'].cpu().numpy()

        for box, score in zip(boxes, scores):
            if score > CONFIDENCE_THRESHOLD:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 0), 2)
                cv2.putText(image, f"{score:.2f}", (box[0], box[1] - 10),
                            DISPLAY_FONT, 0.5, (0, 255, 0), 2)

        # Process segmentation
        overlay = np.zeros_like(image)
        overlay[segmentation == DAMAGE_CLASS_ID] = (0, 0, 255)
        image = cv2.addWeighted(image, 1, overlay, SEGMENTATION_OVERLAY_ALPHA, 0)

        # Add text information
        risk_score, detections, damage_ratio = risk_data
        y_pos = 30
        cv2.putText(image, f"Risk Score: {risk_score:.2f}", (10, y_pos),
                    DISPLAY_FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Detections: {detections} | Damage Area: {damage_ratio:.1%}",
                    (10, y_pos + 30), DISPLAY_FONT, 0.6, (255, 255, 255), 2)
        return image


# -------------------------------
# Main Processing Functions
# -------------------------------
def process_image(image_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        image = Image.open(image_path).convert("RGB")
    except IOError:
        print(f"Error: Could not open image file {image_path}")
        return

    detector = ObjectDetector(device)
    segmenter = SceneSegmenter(device)
    analyzer = DisasterAnalyzer()
    visualizer = VisualizationEngine()

    detections = detector.detect(image)
    segmentation = segmenter.segment(image)
    risk_data = analyzer.calculate_risk(detections, segmentation)
    recommendations = analyzer.generate_recommendations(risk_data[0])

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_frame = visualizer.overlay_results(image_cv, detections, segmentation, risk_data)

    # Add recommendations
    y_start = processed_frame.shape[0] - 40
    cv2.putText(processed_frame, recommendations, (10, y_start),
                DISPLAY_FONT, 0.5, (255, 255, 255), 1)

    cv2.imshow("Disaster Assessment Results", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError
    except Exception as e:
        print(f"Error opening video file: {e}")
        return

    detector = ObjectDetector(device)
    segmenter = SceneSegmenter(device)
    analyzer = DisasterAnalyzer()
    visualizer = VisualizationEngine()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        detections = detector.detect(pil_image)
        segmentation = segmenter.segment(pil_image)
        risk_data = analyzer.calculate_risk(detections, segmentation)

        processed_frame = visualizer.overlay_results(frame.copy(), detections, segmentation, risk_data)
        cv2.imshow("Disaster Assessment Results", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# File Handling & UI
# -------------------------------
def get_file_path() -> Union[str, None]:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Media File",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path if file_path else None


def main():
    file_path = get_file_path()
    if not file_path:
        print("No file selected. Exiting.")
        return

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        process_image(file_path)
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(file_path)
    else:
        print("Unsupported file format. Please select an image or video file.")


if __name__ == "__main__":
    main()