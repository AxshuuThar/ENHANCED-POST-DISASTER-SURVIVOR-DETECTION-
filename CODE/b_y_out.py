import torch
from transformers import BertTokenizer, BertForSequenceClassification
import cv2
import os
from datetime import datetime
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from ultralytics import YOLO


class TextAnalyzer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=41
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define disaster-related category weights
        self.category_weights = {
            # Assuming these indices based on your training data
            # Modify these based on your actual category indices
            0: 1.2,  # Related to casualties
            1: 1.2,  # Infrastructure damage
            2: 1.1,  # Natural disasters
            3: 1.1,  # Emergency situations
            # Add more categories as needed
        }

    def analyze_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

            # Apply category weights to probabilities
            weighted_probs = np.array([
                prob * self.category_weights.get(i, 1.0)
                for i, prob in enumerate(probabilities)
            ])

            # Get top categories
            top_indices = np.argsort(weighted_probs)[-3:][::-1]
            top_probs = weighted_probs[top_indices]

            # Calculate disaster likelihood score
            disaster_score = np.clip(np.mean(top_probs), 0, 1)

            return {
                'disaster_score': disaster_score,
                'top_categories': list(zip(top_indices, top_probs))
            }


class DisasterDetector:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)

    def draw_analysis_overlay(self, img, text_analysis, detections):
        height, width = img.shape[:2]

        # Create transparent overlay for text
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Add disaster analysis information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Disaster Likelihood: {text_analysis['disaster_score']:.2f}",
                    (10, 30), font, 0.7, (255, 255, 255), 2)

        if detections:
            cv2.putText(img,
                        f"People Detected: {len(detections)} | Avg. Confidence: {np.mean([d['confidence'] for d in detections]):.2f}",
                        (10, 60), font, 0.7, (255, 255, 255), 2)

    def adjust_confidence(self, yolo_conf, disaster_score):
        # Adaptive confidence adjustment based on disaster context
        if disaster_score > 0.8:
            # High disaster likelihood - increase confidence for potential victims
            return np.clip(yolo_conf * 1.2, 0, 1)
        elif disaster_score < 0.3:
            # Low disaster likelihood - slightly decrease confidence
            return yolo_conf * 0.9
        else:
            # Moderate disaster likelihood - weighted combination
            return np.clip(yolo_conf * (0.7 + disaster_score * 0.3), 0, 1)

    def process_image(self, image_path, text_analysis):
        img = cv2.imread(image_path)
        results = self.model(img)
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                if box.cls[0] == 0:  # Person class
                    yolo_conf = float(box.conf[0])

                    # Adjust confidence based on disaster context
                    adjusted_conf = self.adjust_confidence(yolo_conf, text_analysis['disaster_score'])

                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': adjusted_conf
                    })

                    # Color based on adjusted confidence
                    color = (0, int(255 * adjusted_conf), int(255 * (1 - adjusted_conf)))

                    # Draw detection
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"Person {adjusted_conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add analysis overlay
        self.draw_analysis_overlay(img, text_analysis, detections)
        return img

    def process_video(self, video_path, text_analysis):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join('result', f'analyzed_{os.path.basename(video_path)}')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            detections = []

            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    if box.cls[0] == 0:
                        yolo_conf = float(box.conf[0])
                        adjusted_conf = self.adjust_confidence(yolo_conf, text_analysis['disaster_score'])

                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)

                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': adjusted_conf
                        })

                        color = (0, int(255 * adjusted_conf), int(255 * (1 - adjusted_conf)))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"Person {adjusted_conf:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.draw_analysis_overlay(frame, text_analysis, detections)
            out.write(frame)
            cv2.imshow('Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return output_path


class DisasterDetectionSystem:
    def __init__(self, bert_model_path, yolo_model_path):
        os.makedirs('result', exist_ok=True)
        self.text_analyzer = TextAnalyzer(bert_model_path)
        self.detector = DisasterDetector(yolo_model_path)

    def process_input(self):
        Tk().withdraw()
        text_input = simpledialog.askstring("Input",
                                            "Please describe the disaster situation:")

        if not text_input:
            print("No text input provided. Using default confidence.")
            text_analysis = {'disaster_score': 0.5, 'top_categories': [(0, 0.5)]}
        else:
            text_analysis = self.text_analyzer.analyze_text(text_input)
            print(f"\nDisaster Analysis Score: {text_analysis['disaster_score']:.2f}")
            print("Top Categories:")
            for idx, prob in text_analysis['top_categories']:
                print(f"Category {idx}: {prob:.2f}")

        file_path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=[
                ("All Supported Files", "*.jpg;*.jpeg;*.png;*.mp4;*.avi;*.mov"),
                ("Images", "*.jpg;*.jpeg;*.png"),
                ("Videos", "*.mp4;*.avi;*.mov")
            ]
        )

        if not file_path:
            print("No file selected.")
            return

        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            output_path = self.detector.process_video(file_path, text_analysis)
        else:
            result_img = self.detector.process_image(file_path, text_analysis)
            output_path = os.path.join('result', f'analyzed_{os.path.basename(file_path)}')
            cv2.imwrite(output_path, result_img)
            cv2.imshow('Analysis Results', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"Analysis complete. Results saved to: {output_path}")


def main():
    try:
        system = DisasterDetectionSystem(
            bert_model_path=r"C:\Users\ebins\MiniPro\code\bert_model.pth",
            yolo_model_path=r"C:\Users\ebins\MiniPro\code\runs\detect\train3\weights\best.pt"
        )
        system.process_input()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
