import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer, BertForSequenceClassification
import cv2
import os
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from ultralytics import YOLO
from scipy.stats import gaussian_kde
from deep_sort_realtime.deepsort_tracker import DeepSort
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import logging
import time
from datetime import datetime
from sklearn.cluster import DBSCAN
from torchvision.models import ResNet50_Weights


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('disaster_detection.log'), logging.StreamHandler()]
)
logger = logging.getLogger("DisasterDetection")

# Configuration Constants
CONFIG = {
    "yolo_model": r"C:\Users\ebins\MiniPro\code\runs\detect\train3\weights\bestt.pt",
    "bert_model": r"C:\Users\ebins\MiniPro\code\bert_model.pth",
    "detection_threshold": 0.4,
    "max_track_age": 30,
    "heatmap_sigma": 15,
    "density_grid": (50, 50),
    "output_dir": "results",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "temporal_heatmap_frames": 5  # number of recent frames to accumulate for temporal heatmap
}


class TextAnalyzer:
    def __init__(self, model_path):
        self.device = CONFIG["device"]
        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=41
            )
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                logger.warning(f"BERT model not found at {model_path}, using random initialization")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Text analyzer initialization failed: {str(e)}")
            raise

        self.category_weights = {0: 1.2, 1: 1.2, 2: 1.1, 3: 1.1}

    def analyze_text(self, text):
        try:
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                avg_attention = torch.mean(outputs.attentions[0]).item()
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                weighted_probs = np.array([
                    prob * self.category_weights.get(i, 1.0)
                    for i, prob in enumerate(probabilities)
                ])

                top_indices = np.argsort(weighted_probs)[-3:][::-1]
                top_probs = weighted_probs[top_indices]
                disaster_score = np.clip(np.mean(top_probs), 0, 1)

                return {
                    'disaster_score': disaster_score,
                    'top_categories': list(zip(top_indices, top_probs))
                }
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return {'disaster_score': 0.5, 'top_categories': []}


class FeatureExtractor:
    def __init__(self):
        self.device = CONFIG["device"]
        try:
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
            # Remove the classification head
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Feature extractor initialization failed: {str(e)}")
            raise

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, image_crop):
        try:
            if image_crop is None or image_crop.size == 0:
                return None
            if image_crop.shape[0] < 10 or image_crop.shape[1] < 10:
                return None

            pil_img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model(input_tensor)
                feature = feature.squeeze().cpu().numpy()
                norm = np.linalg.norm(feature)
                if norm == 0:
                    return feature
                return feature / norm
        except Exception as e:
            logger.warning(f"Feature extraction failed: {str(e)}")
            return None


class SceneUnderstanding:
    """
    Placeholder for a transformer-based scene understanding module.
    In a production system, integrate an image captioning or scene classification model.
    """
    def __init__(self):
        self.device = CONFIG["device"]
        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Scene understanding initialization failed: {str(e)}")
            raise

    def analyze_scene(self, image):
        # In a real implementation, use a vision transformer or image captioning model.
        # Here we return a dummy scene context.
        return "urban disaster"


class RobustAnalyzer:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=CONFIG["max_track_age"],
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100,
            override_track_class=None,
            half=True,
            bgr=True
        )
        self.feature_extractor = FeatureExtractor()
        self.scene_understander = SceneUnderstanding()
        self.frame_size = None
        self.temporal_heatmaps = []
        self.temporal_heatmap = None
        self.tracks_history = defaultdict(list)
        self.kalman_filters = {}
        self.density_history = []
        logger.info("Robust analyzer initialized")

    def initialize_kalman(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R *= 10
        kf.Q *= 0.1
        return kf

    def process_frame(self, frame, detections):
        try:
            self.frame_size = frame.shape[:2]
            # Update temporal heatmap (accumulate over recent frames)
            current_heatmap = self._create_heatmap(detections)
            self._accumulate_heatmap(current_heatmap)

            # Tracking with DeepSort
            tracks = self._update_tracker(frame, detections)

            # Density and clustering analysis
            density_map, clusters = self._analyze_density(detections)

            # Use the accumulated (averaged) heatmap for visualization
            frame = self._visualize_analysis(frame, tracks, self.temporal_heatmap, density_map, clusters)

            # Add scene context from visual analysis (placeholder)
            scene_context = self.scene_understander.analyze_scene(frame)
            cv2.putText(frame, f"Scene: {scene_context}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Overlay summary information
            cv2.putText(frame, f"Detections: {len(detections)} Clusters: {len(clusters)}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return frame, tracks, self.temporal_heatmap, clusters
        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return frame, [], np.zeros(self.frame_size, dtype=np.uint8), []

    def _accumulate_heatmap(self, current_heatmap):
        # Accumulate recent heatmaps to capture temporal changes
        self.temporal_heatmaps.append(current_heatmap)
        if len(self.temporal_heatmaps) > CONFIG["temporal_heatmap_frames"]:
            self.temporal_heatmaps.pop(0)
        # Compute the average of stored heatmaps
        self.temporal_heatmap = np.mean(np.stack(self.temporal_heatmaps, axis=0), axis=0).astype(np.uint8)

    def _update_tracker(self, frame, detections):
        try:
            bbs = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                crop = frame[y1:y2, x1:x2]
                embedding = self.feature_extractor.extract_embedding(crop)
                if embedding is not None:
                    bbs.append(([x1, y1, x2 - x1, y2 - y1], conf, embedding))

            tracks = self.tracker.update_tracks(bbs, frame=frame)
            return self._process_tracks(tracks)
        except Exception as e:
            logger.error(f"Tracking failed: {str(e)}")
            return []

    def _process_tracks(self, tracks):
        processed = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = tuple(map(int, ltrb))
            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = self.initialize_kalman()
            kf = self.kalman_filters[track_id]
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            self.tracks_history[track_id].append(center)
            if len(self.tracks_history[track_id]) == 1:
                kf.x = np.array([[center[0]], [center[1]], [0], [0]])
            else:
                kf.predict()
                kf.update(np.array([[center[0]], [center[1]]]))
            processed.append({
                'id': track_id,
                'bbox': bbox,
                'trajectory': self.tracks_history[track_id],
                'prediction': kf.x[:2].flatten()
            })
        return processed

    def _create_heatmap(self, detections):
        try:
            heatmap = np.zeros(self.frame_size, dtype=np.float32)
            indices = np.indices(self.frame_size)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                gaussian = np.exp(-(((indices[1] - center[0]) ** 2 + (indices[0] - center[1]) ** 2) /
                                    (2 * CONFIG["heatmap_sigma"] ** 2)))
                heatmap += gaussian * det['confidence']
            if heatmap.max() > 0:
                heatmap = (255 * heatmap / heatmap.max()).astype(np.uint8)
            return heatmap
        except Exception as e:
            logger.error(f"Heatmap creation failed: {str(e)}")
            return np.zeros(self.frame_size, dtype=np.uint8)

    def _analyze_density(self, detections):
        try:
            if not detections:
                return np.zeros(CONFIG["density_grid"]), []
                
            centers = np.array([[(d['bbox'][0] + d['bbox'][2]) / 2,
                                 (d['bbox'][1] + d['bbox'][3]) / 2] for d in detections])
            
            # Handle cases with insufficient data for KDE
            if centers.shape[0] < 2:
                return np.zeros(CONFIG["density_grid"]), []

            # Add small noise to prevent singular matrix
            centers += np.random.normal(0, 1e-6, centers.shape)
            
            try:
                kde = gaussian_kde(centers.T)
                xgrid = np.linspace(0, self.frame_size[1], CONFIG["density_grid"][0])
                ygrid = np.linspace(0, self.frame_size[0], CONFIG["density_grid"][1])
                xx, yy = np.meshgrid(xgrid, ygrid)
                density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            except np.linalg.LinAlgError:
                # Fallback to uniform distribution when KDE fails
                density = np.ones(CONFIG["density_grid"])

            # Improved clustering with adaptive parameters
            clustering = DBSCAN(eps=min(50, self.frame_size[1]/20), 
                              min_samples=max(2, len(centers)//10)).fit(centers)
            
            clusters = [{
                'center': np.mean(centers[clustering.labels_ == label], axis=0),
                'size': int(np.sum(clustering.labels_ == label))
            } for label in set(clustering.labels_) if label != -1]
            
            return density, clusters
        except Exception as e:
            logger.error(f"Density analysis failed: {str(e)}")
            return np.zeros(CONFIG["density_grid"]), []

    def _visualize_analysis(self, frame, tracks, heatmap, density_map, clusters):
        try:
            # Ensure heatmap is single-channel 8-bit for cv2.applyColorMap
            if len(heatmap.shape) != 2 or heatmap.dtype != np.uint8:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track['id']}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if len(track['trajectory']) > 1:
                    pts = np.array(track['trajectory'], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
            for cluster in clusters:
                cx, cy = map(int, cluster['center'])
                cv2.circle(frame, (cx, cy), 30, (0, 255, 255), 2)
                cv2.putText(frame, f"Group: {cluster['size']}", (cx - 30, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            return frame
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return frame


class DisasterDetector:
    def __init__(self):
        try:
            logger.info("Initializing YOLO model...")
            self.model = YOLO(CONFIG["yolo_model"])
            self.analyzer = RobustAnalyzer()
            logger.info("Disaster detector initialized")
            self.log_file = os.path.join(CONFIG["output_dir"], "performance_log.csv")
        except Exception as e:
            logger.error(f"Detector initialization failed: {str(e)}")
            raise

    def _log_performance_metrics(self, frame_data):
        """Log frame-wise metrics to a CSV file."""
        import csv
        headers = [
            "timestamp", "frame_number", "detection_count", "cluster_count",
            "disaster_score", "avg_confidence", "processing_time"
        ]
        file_exists = os.path.exists(self.log_file)

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(frame_data)

    def process_media(self, media_path, text_analysis):
        try:
            if media_path.lower().endswith(('.mp4', '.avi', '.mov')):
                return self._process_video(media_path, text_analysis)
            else:
                return self._process_image(media_path, text_analysis)
        except Exception as e:
            logger.error(f"Media processing failed: {str(e)}")
            return None

    def _process_image(self, image_path, text_analysis):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
            detections = self._detect_objects(frame, text_analysis)
            result, _, _, clusters = self.analyzer.process_frame(frame, detections)  # Changed to capture clusters

            # Performance logging (NEW)
            frame_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_number": frame_counter,
                "detection_count": len(detections),
                "cluster_count": len(clusters),
                "disaster_score": text_analysis['disaster_score'],
                "avg_confidence": np.mean([d['confidence'] for d in detections]) if detections else 0,
                "processing_time": processing_time,
                "attention_weight": avg_attention
            }
            self._log_performance_metrics(frame_data)

            return self._save_result(result, image_path)
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return None

    def _process_video(self, video_path, text_analysis):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video {video_path}")
            self._setup_output_dir()
            output_path = os.path.join(CONFIG["output_dir"],
                                       f"processed_{os.path.basename(video_path)}")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (frame_width, frame_height))

            frame_counter = 0  # NEW

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()  # NEW
                detections = self._detect_objects(frame, text_analysis)
                processed, _, _, clusters = self.analyzer.process_frame(frame,
                                                                        detections)  # Changed to capture clusters
                processing_time = time.time() - start_time  # NEW

                # Performance logging (NEW)
                frame_data = {
                    "timestamp": datetime.now().isoformat(),
                    "frame_number": frame_counter,
                    "detection_count": len(detections),
                    "cluster_count": len(clusters),
                    "disaster_score": text_analysis['disaster_score'],
                    "avg_confidence": np.mean([d['confidence'] for d in detections]) if detections else 0,
                    "processing_time": processing_time
                }
                self._log_performance_metrics(frame_data)
                frame_counter += 1  # NEW

                # Existing functionality remains unchanged
                cv2.imshow("Processed Video", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                writer.write(processed)

            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            return output_path
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            return None

    # Add this NEW method to the DisasterDetector class
    def _log_performance_metrics(self, frame_data):
        """Log metrics to CSV without affecting core functionality."""
        import csv
        log_path = os.path.join(CONFIG["output_dir"], "performance_metrics.csv")
        headers = list(frame_data.keys())

        try:
            file_exists = os.path.exists(log_path)
            with open(log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(frame_data)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {str(e)}")

    def _detect_objects(self, frame, text_analysis):
        try:
            results = self.model(frame)[0]
            detections = []
            for box in results.boxes:
                if box.cls == 0:  # Person class
                    conf = self._adjust_confidence(float(box.conf), text_analysis)
                    if conf >= CONFIG["detection_threshold"]:
                        detections.append({
                            'bbox': box.xyxy[0].cpu().numpy().astype(int),
                            'confidence': conf
                        })
            return detections
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            return []

    def _adjust_confidence(self, base_conf, text_analysis):
        disaster_score = text_analysis.get('disaster_score', 0.5)
        return np.clip(base_conf * (0.7 + disaster_score * 0.5), 0, 1)

    def _save_result(self, image, path):
        self._setup_output_dir()
        output_path = os.path.join(CONFIG["output_dir"], f"processed_{os.path.basename(path)}")
        cv2.imwrite(output_path, image)
        return output_path

    def _setup_output_dir(self):
        os.makedirs(CONFIG["output_dir"], exist_ok=True)


class DisasterDetectionSystem:
    def __init__(self):
        self.text_analyzer = TextAnalyzer(CONFIG["bert_model"])
        self.detector = DisasterDetector()
        logger.info("System initialized")

    def run(self):
        try:
            Tk().withdraw()
            text = simpledialog.askstring("Input", "Describe the disaster situation:")
            text_analysis = self.text_analyzer.analyze_text(text) if text else {
                'disaster_score': 0.5,
                'top_categories': []
            }
            path = filedialog.askopenfilename(
                title="Select Media File",
                filetypes=[("Media Files", "*.jpg;*.jpeg;*.png;*.mp4;*.avi;*.mov")]
            )
            if not path:
                logger.info("No file selected")
                return
            logger.info(f"Processing {path}...")
            result_path = self.detector.process_media(path, text_analysis)
            if result_path:
                logger.info(f"Analysis saved to: {result_path}")
                if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    result_img = cv2.imread(result_path)
                    cv2.imshow("Result", result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"System error: {str(e)}")


if __name__ == "__main__":
    try:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        system = DisasterDetectionSystem()
        system.run()
    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}")
