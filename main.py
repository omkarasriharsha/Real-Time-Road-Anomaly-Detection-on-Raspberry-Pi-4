#!/usr/bin/env python3
"""
Real-Time Road Anomaly Detection System
Bharat AI SoC Challenge - Problem Statement 3
YOLOv11 ONNX + PT Implementation for Raspberry Pi
Detects: Potholes + Obstacles (Humans, Animals, Vehicles)

CORRECTED VERSION:
- Improved obstacle detection (catches all vehicles)
- Better motion tracking with multi-frame comparison
- Motion status ONLY in CSV (not on video)
- Pothole diameter shown on video and in CSV
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
from datetime import datetime
import os
import argparse
from pathlib import Path
import csv
from collections import deque

# Try to import ultralytics for YOLO11 PT model
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] Ultralytics not found. Install with: pip install ultralytics")


class DualDetector:
    """
    Dual detection system: Potholes (ONNX) + Obstacles (YOLO11 PT)
    Enhanced with diameter calculation and robust motion detection
    """
    
    def __init__(self, pothole_model_path, obstacle_model_path=None, 
                 conf_threshold=0.75, iou_threshold=0.4, 
                 input_size=640, camera_mode=False):
        """
        Initialize the dual detector
        
        Args:
            pothole_model_path (str): Path to ONNX pothole model file
            obstacle_model_path (str): Path to YOLO11 PT model file (optional)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            input_size (int): Model input size (640 for YOLOv11)
            camera_mode (bool): Use camera instead of video file
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.camera_mode = camera_mode
        
        # Initialize ONNX Runtime session for pothole detection
        print("[INFO] Loading Pothole ONNX model...")
        self.pothole_session = ort.InferenceSession(
            pothole_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get pothole model input/output details
        self.pothole_input_name = self.pothole_session.get_inputs()[0].name
        self.pothole_output_names = [output.name for output in self.pothole_session.get_outputs()]
        
        # Get actual model input size from the model itself
        model_input_shape = self.pothole_session.get_inputs()[0].shape
        print(f"[INFO] Pothole model input shape: {model_input_shape}")
        
        # Extract input size from model (handle dynamic shapes)
        if len(model_input_shape) == 4:
            if isinstance(model_input_shape[2], int) and isinstance(model_input_shape[3], int):
                self.input_size = model_input_shape[2]
                print(f"[INFO] Using model's input size: {self.input_size}")
        
        print(f"[INFO] Pothole model loaded successfully!")
        
        # Initialize YOLO11 PT model for obstacle detection
        self.obstacle_model = None
        if obstacle_model_path and YOLO_AVAILABLE:
            print("[INFO] Loading Obstacle Detection YOLO11 PT model...")
            try:
                self.obstacle_model = YOLO(obstacle_model_path)
                print(f"[INFO] Obstacle model loaded successfully!")
            except Exception as e:
                print(f"[WARNING] Failed to load obstacle model: {e}")
        elif obstacle_model_path and not YOLO_AVAILABLE:
            print("[WARNING] Ultralytics not available. Obstacle detection disabled.")
        
        # COCO class names for obstacle detection
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
            23: 'giraffe'
        }
        
        # Vehicle classes for motion detection
        self.vehicle_classes = {2, 3, 5, 7, 1}  # car, motorcycle, bus, truck, bicycle
        
        # Classes to detect (humans, animals, vehicles)
        self.target_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'bus': 5,
            'truck': 7, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17,
            'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22,
            'giraffe': 23
        }
        
        # Statistics
        self.total_potholes_detected = 0
        self.total_obstacles_detected = 0
        self.obstacle_counts = {}  # Track count per obstacle type
        self.frame_count = 0
        self.fps_history = []
        
        # CSV logging
        self.csv_data = []
        
        # IMPROVED Motion detection tracking
        self.vehicle_tracks = {}  # Track vehicles across multiple frames
        self.motion_history_length = 5  # Track over 5 frames
        self.motion_threshold = 15  # Pixels movement threshold (increased for better detection)
        self.next_vehicle_id = 0
        
    def calculate_pothole_diameter(self, bbox):
        """
        Calculate approximate diameter of pothole from bounding box
        Uses average of width and height as diameter estimate
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            float: Diameter in pixels
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Use average of width and height as diameter
        diameter = (width + height) / 2
        
        return round(diameter, 2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou
    
    def track_and_calculate_motion(self, current_detections):
        """
        Track vehicles across frames and determine motion status
        Uses multi-frame tracking for robust motion detection
        
        Args:
            current_detections: List of current frame detections
        
        Returns:
            dict: Detection ID to motion status mapping
        """
        motion_statuses = {}
        matched_track_ids = set()
        
        # Match current detections to existing tracks
        for det_idx, detection in enumerate(current_detections):
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Only track vehicles
            if class_id not in self.vehicle_classes:
                motion_statuses[det_idx] = 'N/A'
                continue
            
            current_bbox = [x1, y1, x2, y2]
            current_center = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            # Find best matching track
            best_iou = 0
            best_track_id = None
            
            for track_id, track_data in self.vehicle_tracks.items():
                if track_id in matched_track_ids:
                    continue
                
                # Get last known position
                last_bbox = track_data['bbox_history'][-1]
                iou = self.calculate_iou(current_bbox, last_bbox)
                
                if iou > best_iou and iou > 0.3:  # IoU threshold for matching
                    best_iou = iou
                    best_track_id = track_id
            
            # Update existing track or create new one
            if best_track_id is not None:
                track_id = best_track_id
                matched_track_ids.add(track_id)
                
                # Update track
                self.vehicle_tracks[track_id]['bbox_history'].append(current_bbox)
                self.vehicle_tracks[track_id]['center_history'].append(current_center)
                self.vehicle_tracks[track_id]['frames_alive'] += 1
                
                # Calculate motion from history
                if len(self.vehicle_tracks[track_id]['center_history']) >= 3:
                    # Compare current position with position 3 frames ago
                    centers = self.vehicle_tracks[track_id]['center_history']
                    
                    # Calculate total displacement over last few frames
                    total_displacement = 0
                    for i in range(1, min(len(centers), 4)):
                        dx = centers[-1][0] - centers[-i][0]
                        dy = centers[-1][1] - centers[-i][1]
                        total_displacement += np.sqrt(dx**2 + dy**2)
                    
                    # Average displacement
                    avg_displacement = total_displacement / min(len(centers) - 1, 3)
                    
                    if avg_displacement > self.motion_threshold:
                        motion_statuses[det_idx] = 'Moving'
                    else:
                        motion_statuses[det_idx] = 'Stationary'
                else:
                    motion_statuses[det_idx] = 'Unknown'
            else:
                # Create new track
                track_id = self.next_vehicle_id
                self.next_vehicle_id += 1
                
                self.vehicle_tracks[track_id] = {
                    'bbox_history': deque([current_bbox], maxlen=self.motion_history_length),
                    'center_history': deque([current_center], maxlen=self.motion_history_length),
                    'frames_alive': 1,
                    'class_id': class_id
                }
                
                motion_statuses[det_idx] = 'Unknown'
        
        # Clean up old tracks (not seen for 10 frames)
        tracks_to_remove = []
        for track_id, track_data in self.vehicle_tracks.items():
            if track_id not in matched_track_ids:
                # Track not matched in this frame
                if track_data['frames_alive'] > 10:
                    tracks_to_remove.append(track_id)
                else:
                    track_data['frames_alive'] += 1
        
        for track_id in tracks_to_remove:
            del self.vehicle_tracks[track_id]
        
        return motion_statuses
    
    def preprocess_pothole(self, image):
        """
        Preprocess image for pothole ONNX model
        """
        original_height, original_width = image.shape[:2]
        
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        input_image = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format (channels first)
        input_image = np.transpose(input_image, (2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        # Calculate scale factors
        scale_x = original_width / self.input_size
        scale_y = original_height / self.input_size
        
        return input_image, scale_x, scale_y
    
    def postprocess_pothole(self, outputs, scale_x, scale_y, original_shape):
        """
        Postprocess pothole ONNX outputs
        """
        predictions = outputs[0]
        
        # Handle different output shapes
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        # Transpose if needed
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        boxes = []
        scores = []
        class_ids = []
        
        # Parse predictions
        for prediction in predictions:
            x_center, y_center, width, height = prediction[:4]
            class_scores = prediction[4:]
            
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence >= self.conf_threshold:
                # Convert from center format to corner format
                x1 = int((x_center - width / 2) * scale_x)
                y1 = int((y_center - height / 2) * scale_y)
                x2 = int((x_center + width / 2) * scale_x)
                y2 = int((y_center + height / 2) * scale_y)
                
                # Clip to image boundaries
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold
            )
            
            detections = []
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                indices_list = indices.flatten() if hasattr(indices, 'flatten') else indices
                
                for i in indices_list:
                    detections.append([
                        boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                        scores[i], class_ids[i]
                    ])
            return detections
        
        return []
    
    def detect_obstacles(self, frame):
        """
        Detect obstacles using YOLO11 PT model
        IMPROVED: Lower confidence for better detection
        """
        if self.obstacle_model is None:
            return []
        
        try:
            # Run YOLO detection with LOWER confidence for obstacles
            # This catches more vehicles including distant ones
            obstacle_conf = max(0.25, self.conf_threshold * 0.7)  # Use lower threshold
            
            results = self.obstacle_model(frame, conf=obstacle_conf, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only include target classes (humans, animals, vehicles)
                    if class_id in self.target_classes.values():
                        detections.append([
                            int(x1), int(y1), int(x2), int(y2),
                            confidence, class_id
                        ])
            
            return detections
        except Exception as e:
            print(f"[WARNING] Obstacle detection error: {e}")
            return []
    
    def log_detections_to_csv(self, frame_number, pothole_detections, 
                              obstacle_detections, motion_statuses):
        """
        Log detections to CSV data structure with diameter and motion status
        Motion status ONLY in CSV, not on video
        """
        # Prepare pothole data with diameter
        pothole_data = []
        for idx, detection in enumerate(pothole_detections, 1):
            x1, y1, x2, y2, confidence, class_id = detection
            bbox = [x1, y1, x2, y2]
            
            # Calculate diameter
            diameter = self.calculate_pothole_diameter(bbox)
            
            pothole_data.append({
                'type': 'Pothole',
                'class': 'Pothole',
                'conf': round(confidence, 4),
                'bbox': f"({x1},{y1},{x2},{y2})",
                'diameter': diameter
            })
        
        # Prepare obstacle data with motion status
        obstacle_data = []
        for idx, detection in enumerate(obstacle_detections):
            x1, y1, x2, y2, confidence, class_id = detection
            bbox = [x1, y1, x2, y2]
            class_name = self.coco_classes.get(class_id, f"Class_{class_id}")
            
            # Get motion status from tracking
            motion_status = motion_statuses.get(idx, 'Unknown')
            
            obstacle_data.append({
                'type': 'Obstacle',
                'class': class_name,
                'conf': round(confidence, 4),
                'bbox': f"({x1},{y1},{x2},{y2})",
                'motion': motion_status
            })
        
        # Create consolidated row for this frame
        # Format: class:conf=X.XX,bbox=(x1,y1,x2,y2),diameter=XX.XX
        pothole_details = ' | '.join([
            f"{d['class']}:conf={d['conf']},bbox={d['bbox']},diameter={d['diameter']}px" 
            for d in pothole_data
        ]) if pothole_data else 'None'
        
        # Format: class:conf=X.XX,bbox=(x1,y1,x2,y2),motion=Moving/Stationary/N/A
        obstacle_details = ' | '.join([
            f"{d['class']}:conf={d['conf']},bbox={d['bbox']},motion={d['motion']}" 
            for d in obstacle_data
        ]) if obstacle_data else 'None'
        
        row = {
            'Serial_Number': len(self.csv_data) + 1,
            'Frame_Number': frame_number,
            'Total_Potholes': len(pothole_detections),
            'Total_Obstacles': len(obstacle_detections),
            'Pothole_Details': pothole_details,
            'Obstacle_Details': obstacle_details
        }
        
        self.csv_data.append(row)
    
    def save_csv(self, csv_path):
        """
        Save CSV data to file
        """
        if not self.csv_data:
            print("[WARNING] No detection data to save to CSV")
            return
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Serial_Number',
                    'Frame_Number',
                    'Total_Potholes',
                    'Total_Obstacles',
                    'Pothole_Details',
                    'Obstacle_Details'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data)
            
            print(f"[INFO] CSV file saved: {csv_path}")
            print(f"[INFO] Total rows in CSV: {len(self.csv_data)}")
            print(f"[INFO] CSV includes:")
            print(f"       - Pothole diameter (in pixels)")
            print(f"       - Vehicle motion status (Moving/Stationary/Unknown/N/A)")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
    
    def draw_detections(self, image, pothole_detections, obstacle_detections, fps=0):
        """
        Draw all detections on image
        NOTE: Motion status NOT shown on video (only in CSV)
        """
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Count detections
        potholes_in_frame = len(pothole_detections)
        obstacles_in_frame = len(obstacle_detections)
        
        # Draw pothole detections (RED boxes) with diameter
        for detection in pothole_detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Calculate and show diameter
            diameter = self.calculate_pothole_diameter([x1, y1, x2, y2])
            label = f"Pothole: {confidence:.2f} | D:{diameter:.0f}px"
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                (0, 0, 255),
                -1
            )
            
            cv2.putText(
                annotated_image, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
        
        # Draw obstacle detections (GREEN boxes) WITHOUT motion status
        current_frame_obstacles = {}
        for detection in obstacle_detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Get class name
            class_name = self.coco_classes.get(class_id, f"Class_{class_id}")
            
            # Count obstacles by type in current frame
            current_frame_obstacles[class_name] = current_frame_obstacles.get(class_name, 0) + 1
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label WITHOUT motion status (only class and confidence)
            label = f"{class_name}: {confidence:.2f}"
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )
            
            cv2.putText(
                annotated_image, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
        
        # Update obstacle counts
        for obstacle_type, count in current_frame_obstacles.items():
            self.obstacle_counts[obstacle_type] = self.obstacle_counts.get(obstacle_type, 0) + count
        
        # Create info panel
        panel_height = 200
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)
        
        # Add statistics to panel
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"",
            f"POTHOLES - In Frame: {potholes_in_frame} | Total: {self.total_potholes_detected}",
            f"OBSTACLES - In Frame: {obstacles_in_frame} | Total: {self.total_obstacles_detected}",
        ]
        
        # Add obstacle breakdown
        if current_frame_obstacles:
            obstacle_summary = ", ".join([f"{k}: {v}" for k, v in current_frame_obstacles.items()])
            info_lines.append(f"Current: {obstacle_summary}")
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(
                panel, line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
            y_offset += 25
        
        # Combine panel with image
        result = np.vstack([panel, annotated_image])
        
        return result
    
    def process_video(self, video_source, output_path=None, display=True):
        """
        Process video with dual detection
        """
        # Open video source
        if self.camera_mode:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {video_source}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0 or original_fps > 120:
            original_fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.camera_mode else 0
        
        # Set target FPS to 5
        target_fps = 5.0
        frame_delay = 1.0 / target_fps
        
        print(f"[INFO] Video Properties:")
        print(f"       Resolution: {frame_width}x{frame_height}")
        print(f"       Original FPS: {original_fps}")
        print(f"       Target FPS: {target_fps}")
        if total_frames > 0:
            print(f"       Total Frames: {total_frames}")
        
        # Setup video writer and CSV path
        video_writer = None
        csv_path = None
        if output_path:
            output_height = frame_height + 200
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, int(target_fps),
                (frame_width, output_height)
            )
            print(f"[INFO] Saving output to: {output_path}")
            
            # Generate CSV path in same directory as output video
            output_dir = os.path.dirname(output_path)
            output_filename = os.path.splitext(os.path.basename(output_path))[0]
            csv_path = os.path.join(output_dir, f"{output_filename}_detections.csv")
            print(f"[INFO] CSV will be saved to: {csv_path}")
        
        # Calculate frame skip
        frame_skip = max(1, int(original_fps / target_fps))
        
        print("[INFO] Starting dual detection...")
        print("[INFO] Press 'q' to quit, 's' to save screenshot")
        
        frame_counter = 0
        last_process_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.camera_mode:
                        print("[WARNING] Failed to grab frame from camera")
                        continue
                    else:
                        print("[INFO] End of video reached")
                        break
                
                frame_counter += 1
                
                # Skip frames to achieve target FPS
                if frame_counter % frame_skip != 0:
                    continue
                
                # Wait to maintain exact FPS
                current_time = time.time()
                elapsed = current_time - last_process_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
                process_start = time.time()
                self.frame_count += 1
                
                # Detect potholes
                input_tensor, scale_x, scale_y = self.preprocess_pothole(frame)
                pothole_outputs = self.pothole_session.run(
                    self.pothole_output_names,
                    {self.pothole_input_name: input_tensor}
                )
                pothole_detections = self.postprocess_pothole(pothole_outputs, scale_x, scale_y, frame.shape)
                
                # Detect obstacles
                obstacle_detections = self.detect_obstacles(frame)
                
                # Track vehicles and calculate motion
                motion_statuses = self.track_and_calculate_motion(obstacle_detections)
                
                # Log to CSV (includes motion status)
                self.log_detections_to_csv(self.frame_count, pothole_detections, 
                                          obstacle_detections, motion_statuses)
                
                # Update statistics
                self.total_potholes_detected += len(pothole_detections)
                self.total_obstacles_detected += len(obstacle_detections)
                
                # Draw detections (motion status NOT shown)
                annotated_frame = self.draw_detections(frame, pothole_detections, 
                                                      obstacle_detections, target_fps)
                
                # Save frame
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Dual Detection - Potholes & Obstacles', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] User requested quit")
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        print(f"[INFO] Screenshot saved: {screenshot_path}")
                
                last_process_time = time.time()
                
                # Print progress
                if self.frame_count % 30 == 0:
                    print(f"[INFO] Frame {self.frame_count} | "
                          f"Potholes: {self.total_potholes_detected} | "
                          f"Obstacles: {self.total_obstacles_detected}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Save CSV file
            if csv_path:
                self.save_csv(csv_path)
            
            # Print final statistics
            print("\n" + "="*60)
            print("DUAL DETECTION SUMMARY")
            print("="*60)
            print(f"Total Frames Processed: {self.frame_count}")
            print(f"Total Potholes Detected: {self.total_potholes_detected}")
            print(f"Total Obstacles Detected: {self.total_obstacles_detected}")
            print("\nObstacle Breakdown:")
            for obstacle_type, count in sorted(self.obstacle_counts.items()):
                print(f"  {obstacle_type}: {count}")
            if self.frame_count > 0:
                print(f"\nAverage Potholes per Frame: {self.total_potholes_detected/self.frame_count:.2f}")
                print(f"Average Obstacles per Frame: {self.total_obstacles_detected/self.frame_count:.2f}")
            print("="*60)


def main():
    """
    Main function with argument parser
    """
    parser = argparse.ArgumentParser(
        description='Dual Detection System - Corrected Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dual_detector.py --pothole-model pothole.onnx --obstacle-model yolo11.pt --video input.mp4 --output result.mp4

Corrections:
  ✓ Improved obstacle detection (lower confidence threshold)
  ✓ Better motion tracking (multi-frame comparison)
  ✓ Motion status ONLY in CSV (not shown on video)
  ✓ Detects stationary vehicles properly
        """
    )
    
    # Required arguments
    parser.add_argument('--pothole-model', type=str, required=True,
                       help='Path to pothole ONNX model file')
    parser.add_argument('--obstacle-model', type=str,
                       help='Path to YOLO11 PT model file (optional)')
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str,
                            help='Path to input video file')
    input_group.add_argument('--camera', action='store_true',
                            help='Use camera as input')
    
    # Optional arguments
    parser.add_argument('--output', type=str,
                       help='Path to output video file (optional)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--iou', type=float, default=0.4,
                       help='IOU threshold for NMS (default: 0.4)')
    parser.add_argument('--input-size', type=int, default=640,
                       help='Model input size (default: 640)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    
    args = parser.parse_args()
    
    # Validate files
    if not os.path.exists(args.pothole_model):
        print(f"[ERROR] Pothole model file not found: {args.pothole_model}")
        return
    
    if args.obstacle_model and not os.path.exists(args.obstacle_model):
        print(f"[ERROR] Obstacle model file not found: {args.obstacle_model}")
        return
    
    if args.video and not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        return
    
    # Create output directory
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Print configuration
    print("\n" + "="*60)
    print("DUAL DETECTION SYSTEM - CORRECTED VERSION")
    print("="*60)
    print(f"Pothole Model: {args.pothole_model}")
    print(f"Obstacle Model: {args.obstacle_model if args.obstacle_model else 'None'}")
    print(f"Input: {'Camera' if args.camera else args.video}")
    print(f"Output: {args.output if args.output else 'Display only'}")
    print(f"Confidence: {args.conf}")
    print(f"\nFEATURES:")
    print(f"  ✓ Pothole diameter on video + CSV")
    print(f"  ✓ Vehicle motion ONLY in CSV")
    print(f"  ✓ Improved obstacle detection")
    print(f"  ✓ Multi-frame motion tracking")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = DualDetector(
        pothole_model_path=args.pothole_model,
        obstacle_model_path=args.obstacle_model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.input_size,
        camera_mode=args.camera
    )
    
    # Process video
    video_source = 0 if args.camera else args.video
    detector.process_video(
        video_source=video_source,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
