"""OpenPose-like pose estimation for behavioral analysis."""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass 
class PoseKeypoints:
    """Pose keypoints detection result."""
    keypoints: np.ndarray  # Shape: (num_people, num_keypoints, 3) [x, y, confidence]
    person_boxes: List[Tuple[int, int, int, int]]
    skeleton_connections: List[Tuple[int, int]]
    processing_time: float

class OpenPoseEstimator:
    """Pose estimation using MediaPipe (33 keypoints) ou MoveNet (17 keypoints)."""
    
    def __init__(self, model_type: str = "mediapipe", device: str = "auto", lazy_loading: bool = False):
        self.device = self._select_device(device)
        # Deux modèles possibles : MediaPipe OU MoveNet (pas de duplication)
        self.model_type = model_type  # "mediapipe" ou "movenet"
        self.pose_detector = None
        self.mp_pose = None
        self.keypoint_names = []
        self.skeleton_connections = []
        self.lazy_loading = lazy_loading
        self._model_loaded = False
        
        # Chargement immédiat ou lazy
        if not lazy_loading:
            self._load_model()
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (lazy loading)."""
        if not self._model_loaded:
            self._load_model()
    
    def _load_model(self):
        """Load pose estimation model (MediaPipe OU MoveNet)."""
        if self._model_loaded:
            return
            
        if self.model_type == "mediapipe":
            self._load_mediapipe()
        elif self.model_type == "movenet":
            self._load_movenet()
        else:
            logger.warning(f"Unknown pose model type: {self.model_type}")
            self.pose_detector = None
    
    def _load_mediapipe(self):
        """Load MediaPipe pose model."""
        try:
            import mediapipe as mp
            
            logger.info("Loading MediaPipe Pose model (modèle physique)...")
            
            self.mp_pose = mp.solutions.pose
            # Configuration selon documentation officielle
            self.pose_detector = self.mp_pose.Pose(
                min_detection_confidence=0.7,  # Plus strict selon doc
                min_tracking_confidence=0.7    # Plus strict selon doc
            )
            
            # Pour drawing (selon doc officielle)
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Define keypoint names (MediaPipe format)
            self.keypoint_names = [
                "nose", "left_eye_inner", "left_eye", "left_eye_outer",
                "right_eye_inner", "right_eye", "right_eye_outer",
                "left_ear", "right_ear", "mouth_left", "mouth_right",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_pinky", "right_pinky",
                "left_index", "right_index", "left_thumb", "right_thumb",
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle", "left_heel", "right_heel",
                "left_foot_index", "right_foot_index"
            ]
            
            # Define skeleton connections
            self.skeleton_connections = [
                (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
            ]
            
            self._model_loaded = True
            logger.info("MediaPipe pose model loaded successfully (33 keypoints)")
            
        except ImportError as e:
            logger.error(f"MediaPipe not available: {e}")
            self.pose_detector = None
            self._model_loaded = False
    
    def _load_movenet(self):
        """Load MoveNet model (17 keypoints)."""
        try:
            import tensorflow as tf
            
            logger.info("Loading MoveNet model (modèle physique)...")
            
            # Load MoveNet model - téléchargé et sauvegardé localement
            self.pose_detector = tf.keras.utils.get_file(
                'movenet_thunder.tflite',
                'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite'
            )
            
            # MoveNet keypoint names (17 keypoints)
            self.keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            
            # MoveNet skeleton connections
            self.skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
                (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16)
            ]
            
            self._model_loaded = True
            logger.info("MoveNet model loaded successfully (17 keypoints)")
            
        except Exception as e:
            logger.error(f"Could not load MoveNet: {e}")
            self.pose_detector = None
            self._model_loaded = False
    
    def estimate_poses(self, frame: np.ndarray, person_boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> PoseKeypoints:
        """Estimate poses in frame using MediaPipe OU MoveNet."""
        start_time = time.perf_counter()
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        if self.pose_detector is None:
            return self._fallback_pose_estimation(frame, person_boxes, start_time)
        
        try:
            if self.model_type == "mediapipe":
                return self._estimate_mediapipe_poses(frame, person_boxes, start_time)
            elif self.model_type == "movenet":
                return self._estimate_movenet_poses(frame, person_boxes, start_time)
        except Exception as e:
            logger.error(f"Pose estimation ({self.model_type}) failed: {e}")
            return self._fallback_pose_estimation(frame, person_boxes, start_time)
    
    def _estimate_mediapipe_poses(self, frame: np.ndarray, person_boxes: Optional[List[Tuple[int, int, int, int]]],
                                start_time: float) -> PoseKeypoints:
        """Estimate poses using MediaPipe."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if person_boxes is None:
            # Process entire frame
            results = self.pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                keypoints = self._mediapipe_to_keypoints(results.pose_landmarks, frame.shape)
                return PoseKeypoints(
                    keypoints=keypoints.reshape(1, -1, 3),  # Single person
                    person_boxes=[(0, 0, frame.shape[1], frame.shape[0])],
                    skeleton_connections=self.skeleton_connections,
                    processing_time=time.perf_counter() - start_time
                )
            else:
                return PoseKeypoints(
                    keypoints=np.array([]),
                    person_boxes=[],
                    skeleton_connections=self.skeleton_connections,
                    processing_time=time.perf_counter() - start_time
                )
        else:
            # Process each person box separately
            all_keypoints = []
            valid_boxes = []
            
            for box in person_boxes:
                x1, y1, x2, y2 = box
                # Convert to integers to avoid slice index issues
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi = frame_rgb[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                results = self.pose_detector.process(roi)
                
                if results.pose_landmarks:
                    keypoints = self._mediapipe_to_keypoints(results.pose_landmarks, roi.shape)
                    # Adjust coordinates to global frame
                    keypoints[:, 0] += x1
                    keypoints[:, 1] += y1
                    
                    all_keypoints.append(keypoints)
                    valid_boxes.append(box)
            
            keypoints_array = np.array(all_keypoints) if all_keypoints else np.array([])
            
            return PoseKeypoints(
                keypoints=keypoints_array,
                person_boxes=valid_boxes,
                skeleton_connections=self.skeleton_connections,
                processing_time=time.perf_counter() - start_time
            )
    
    def _mediapipe_to_keypoints(self, landmarks, image_shape) -> np.ndarray:
        """Convert MediaPipe landmarks to keypoints array."""
        h, w = image_shape[:2]
        keypoints = np.zeros((33, 3))  # MediaPipe has 33 keypoints
        
        for i, landmark in enumerate(landmarks.landmark):
            keypoints[i] = [
                landmark.x * w,
                landmark.y * h,
                landmark.visibility
            ]
        
        return keypoints
    
    def _estimate_movenet_poses(self, frame: np.ndarray, person_boxes: Optional[List[Tuple[int, int, int, int]]],
                              start_time: float) -> PoseKeypoints:
        """Estimate poses using MoveNet (simplified implementation)."""
        # Note: MoveNet est plus complexe à implémenter avec TensorFlow Lite
        # Pour cette version, on utilise le fallback
        logger.info("MoveNet estimation - using fallback for now")
        return self._fallback_pose_estimation(frame, person_boxes, start_time)
    
    def _fallback_pose_estimation(self, frame: np.ndarray, person_boxes: Optional[List[Tuple[int, int, int, int]]],
                                start_time: float) -> PoseKeypoints:
        """Fallback pose estimation using simple heuristics."""
        if person_boxes is None:
            return PoseKeypoints(
                keypoints=np.array([]),
                person_boxes=[],
                skeleton_connections=[],
                processing_time=time.perf_counter() - start_time
            )
        
        # Simple heuristic: estimate rough body parts based on bounding box
        estimated_keypoints = []
        
        for box in person_boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Rough estimates for key body parts
            keypoints = np.array([
                [x1 + w*0.5, y1 + h*0.15, 0.5],  # Head
                [x1 + w*0.3, y1 + h*0.4, 0.3],   # Left shoulder
                [x1 + w*0.7, y1 + h*0.4, 0.3],   # Right shoulder
                [x1 + w*0.2, y1 + h*0.6, 0.3],   # Left elbow
                [x1 + w*0.8, y1 + h*0.6, 0.3],   # Right elbow
                [x1 + w*0.1, y1 + h*0.8, 0.3],   # Left wrist
                [x1 + w*0.9, y1 + h*0.8, 0.3],   # Right wrist
                [x1 + w*0.4, y1 + h*0.6, 0.3],   # Left hip
                [x1 + w*0.6, y1 + h*0.6, 0.3],   # Right hip
                [x1 + w*0.3, y1 + h*0.85, 0.2],  # Left knee
                [x1 + w*0.7, y1 + h*0.85, 0.2],  # Right knee
                [x1 + w*0.3, y1 + h*0.98, 0.2],  # Left ankle
                [x1 + w*0.7, y1 + h*0.98, 0.2],  # Right ankle
            ])
            
            estimated_keypoints.append(keypoints)
        
        keypoints_array = np.array(estimated_keypoints) if estimated_keypoints else np.array([])
        
        return PoseKeypoints(
            keypoints=keypoints_array,
            person_boxes=person_boxes,
            skeleton_connections=[(0,1), (0,2), (1,3), (3,5), (2,4), (4,6), (1,7), (2,8), (7,8), (7,9), (9,11), (8,10), (10,12)],
            processing_time=time.perf_counter() - start_time
        )
    
    def analyze_pose_behavior(self, keypoints: np.ndarray, previous_keypoints: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze pose for behavioral indicators."""
        if keypoints.size == 0:
            return {"behavior_score": 0.0, "indicators": []}
        
        indicators = []
        behavior_scores = []
        
        # Analyze each person
        for person_idx in range(keypoints.shape[0]):
            person_kpts = keypoints[person_idx]
            
            # Filter valid keypoints (confidence > 0.3)
            valid_kpts = person_kpts[person_kpts[:, 2] > 0.3]
            
            if len(valid_kpts) < 5:  # Need minimum keypoints
                continue
            
            # Analyze pose characteristics
            pose_analysis = self._analyze_single_pose(person_kpts)
            
            # Analyze movement if previous keypoints available
            if previous_keypoints is not None and person_idx < previous_keypoints.shape[0]:
                movement_analysis = self._analyze_movement(person_kpts, previous_keypoints[person_idx])
                pose_analysis.update(movement_analysis)
            
            behavior_scores.append(pose_analysis.get('suspicion_score', 0.0))
            indicators.extend(pose_analysis.get('indicators', []))
        
        overall_score = np.mean(behavior_scores) if behavior_scores else 0.0
        
        return {
            "behavior_score": float(overall_score),
            "indicators": indicators,
            "num_people": len(behavior_scores),
            "individual_scores": behavior_scores
        }
    
    def _analyze_single_pose(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """Analyze single person pose for suspicious behavior."""
        indicators = []
        suspicion_score = 0.0
        
        # Extract key body parts (assuming MediaPipe format)
        try:
            head = keypoints[0] if len(keypoints) > 0 else None
            left_shoulder = keypoints[11] if len(keypoints) > 11 else None
            right_shoulder = keypoints[12] if len(keypoints) > 12 else None
            left_wrist = keypoints[15] if len(keypoints) > 15 else None
            right_wrist = keypoints[16] if len(keypoints) > 16 else None
            left_hip = keypoints[23] if len(keypoints) > 23 else None
            right_hip = keypoints[24] if len(keypoints) > 24 else None
            
            # Crouching/bending detection
            if (head is not None and left_hip is not None and 
                head[2] > 0.3 and left_hip[2] > 0.3):
                
                head_hip_distance = abs(head[1] - left_hip[1])
                if head_hip_distance < 100:  # Very low pose
                    indicators.append("crouching_detected")
                    suspicion_score += 0.3
            
            # Hand near waist/pocket area
            if (left_wrist is not None and left_hip is not None and
                left_wrist[2] > 0.3 and left_hip[2] > 0.3):
                
                wrist_hip_distance = np.linalg.norm([
                    left_wrist[0] - left_hip[0],
                    left_wrist[1] - left_hip[1]
                ])
                
                if wrist_hip_distance < 50:
                    indicators.append("hand_near_waist")
                    suspicion_score += 0.2
            
            # Similar check for right hand
            if (right_wrist is not None and right_hip is not None and
                right_wrist[2] > 0.3 and right_hip[2] > 0.3):
                
                wrist_hip_distance = np.linalg.norm([
                    right_wrist[0] - right_hip[0],
                    right_wrist[1] - right_hip[1]
                ])
                
                if wrist_hip_distance < 50:
                    indicators.append("hand_near_waist")
                    suspicion_score += 0.2
            
            # Unusual arm positioning
            if (left_shoulder is not None and right_shoulder is not None and
                left_wrist is not None and right_wrist is not None):
                
                # Arms crossed or very close to body
                shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                arm_span = abs(left_wrist[0] - right_wrist[0])
                
                if arm_span < shoulder_width * 0.5:
                    indicators.append("arms_close_to_body")
                    suspicion_score += 0.1
                    
        except (IndexError, TypeError) as e:
            logger.debug(f"Pose analysis error: {e}")
        
        return {
            "suspicion_score": min(suspicion_score, 1.0),
            "indicators": indicators
        }
    
    def _analyze_movement(self, current_kpts: np.ndarray, previous_kpts: np.ndarray) -> Dict[str, Any]:
        """Analyze movement patterns between frames."""
        indicators = []
        movement_score = 0.0
        
        # Calculate overall movement
        valid_current = current_kpts[current_kpts[:, 2] > 0.3]
        valid_previous = previous_kpts[previous_kpts[:, 2] > 0.3]
        
        if len(valid_current) > 0 and len(valid_previous) > 0:
            # Calculate movement magnitude
            movement = np.linalg.norm(valid_current[:, :2] - valid_previous[:len(valid_current), :2], axis=1)
            avg_movement = np.mean(movement)
            
            # Sudden movements
            if avg_movement > 20:
                indicators.append("sudden_movement")
                movement_score += 0.2
            
            # Very slow/suspicious movement
            if avg_movement < 2:
                indicators.append("very_slow_movement")
                movement_score += 0.1
        
        return {
            "movement_indicators": indicators,
            "movement_score": movement_score
        }
    
    def draw_pose_landmarks(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw pose landmarks on image (selon documentation officielle MediaPipe)."""
        if not self._model_loaded or self.pose_detector is None:
            return image.copy()
        
        # Convert keypoints to MediaPipe format for drawing
        if len(keypoints) == 0:
            return image.copy()
        
        try:
            # Créer un objet landmarks MediaPipe pour drawing
            from mediapipe.python.solutions import pose as mp_pose_module
            
            # Simuler un résultat MediaPipe pour le drawing
            image_copy = image.copy()
            
            # Dessiner manuellement si nécessaire  
            for person_keypoints in keypoints:
                if len(person_keypoints) >= 33:  # 33 keypoints MediaPipe
                    # Dessiner les points
                    for i, (x, y, confidence) in enumerate(person_keypoints):
                        if confidence > 0.5:  # Seuil de confiance
                            cv2.circle(image_copy, (int(x), int(y)), 4, (255, 0, 0), -1)
                    
                    # Dessiner les connexions
                    for connection in self.skeleton_connections:
                        start_idx, end_idx = connection
                        if (start_idx < len(person_keypoints) and 
                            end_idx < len(person_keypoints) and
                            person_keypoints[start_idx][2] > 0.5 and 
                            person_keypoints[end_idx][2] > 0.5):
                            
                            start_point = (int(person_keypoints[start_idx][0]), 
                                         int(person_keypoints[start_idx][1]))
                            end_point = (int(person_keypoints[end_idx][0]), 
                                       int(person_keypoints[end_idx][1]))
                            cv2.line(image_copy, start_point, end_point, (255, 0, 0), 6)
            
            return image_copy
            
        except Exception as e:
            logger.warning(f"Could not draw pose landmarks: {e}")
            return image.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded pose model."""
        keypoints_count = 33 if self.model_type == "mediapipe" else 17
        
        return {
            "model_type": self.model_type,
            "keypoints_count": keypoints_count,
            "model_loaded": self._model_loaded,
            "lazy_loading": self.lazy_loading,
            "detection_confidence": 0.7 if self.model_type == "mediapipe" else "N/A",
            "tracking_confidence": 0.7 if self.model_type == "mediapipe" else "N/A",
            "model_available": self.pose_detector is not None
        }