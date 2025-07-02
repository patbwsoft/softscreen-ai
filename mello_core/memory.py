import numpy as np
import cv2
import yaml
import os
from datetime import datetime
from mello_core.reflection import MelloReflection

class TemporalMemory:
    def __init__(self, config_path=os.path.join(os.path.dirname(__file__), "settings.yaml")):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.blur_labels = self.config.get("blur_labels", [])
        self.conf_threshold = self.config.get("confidence_threshold", 0.3)
        self.blur_strength = self.config.get("blur_strength_multiplier", 1.0)
        self.min_object_area = self.config.get("min_object_area", 2500)
        self.label_weights = self.config.get("label_weights", {})
        self.reattach_distance_thresh = self.config.get("reattach_distance_thresh", 40)

        self.memories = {}
        self.next_id = 0
        self.frame_counter = 0

        # === Tracking Stats for Reflection ===
        self.total_tracked = 0
        self.confidence_dips = []
        self.blur_areas = []
        self.label_log = []

    def get_centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def box_area(self, box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    def process_frame(self, frame, detections):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        updated_ids = set()
        self.frame_counter += 1

        for det in detections:
            label = det['label']
            conf = det['confidence']
            box = det['bbox']
            area = self.box_area(box)

            if label not in self.blur_labels or conf < self.conf_threshold or area < self.min_object_area:
                continue

            self.total_tracked += 1
            self.label_log.append(label)
            self.blur_areas.append(area)
            if conf < self.conf_threshold:
                self.confidence_dips.append(self.frame_counter)

            centroid = self.get_centroid(box)
            matched = False

            for mem_id, memory in self.memories.items():
                if memory['label'] == label:
                    old_centroid = self.get_centroid(memory['bbox'])
                    dist = np.linalg.norm(np.array(centroid) - np.array(old_centroid))

                    if dist < self.reattach_distance_thresh:
                        memory['bbox'] = box
                        memory['confidence'] = conf
                        memory['age'] = 0
                        updated_ids.add(mem_id)
                        matched = True
                        break

            if not matched:
                self.memories[self.next_id] = {
                    'label': label,
                    'confidence': conf,
                    'bbox': box,
                    'age': 0
                }
                updated_ids.add(self.next_id)
                self.next_id += 1

        # Clean and fade out memories
        for mem_id in list(self.memories.keys()):
            memory = self.memories[mem_id]
            memory['age'] += 1
            if mem_id not in updated_ids:
                memory['confidence'] *= 0.8
                if memory['confidence'] < 0.2:
                    del self.memories[mem_id]
                    continue

            box = memory['bbox']
            x1, y1, x2, y2 = [int(v) for v in box]
            mask[y1:y2, x1:x2] = 255

        return self.apply_blur(frame, mask)

    def apply_blur(self, frame, mask):
        if np.count_nonzero(mask) == 0:
            return frame

        kernel_size = 191
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        mask_3ch = cv2.merge([mask, mask, mask])
        mask_norm = (mask_3ch / 255.0).astype(np.float32)
        result = (frame.astype(np.float32) * (1 - mask_norm) + blurred.astype(np.float32) * mask_norm).astype(np.uint8)

        return result

    def write_reflection(self):
        session_data = {
            'total_objects': self.total_tracked,
            'avg_confidence': self._avg_confidence(),
            'unique_labels': self.label_log,
            'confidence_dips': self.confidence_dips,
            'frame_count': self.frame_counter,
            'blur_areas': self.blur_areas
        }
        reflection = MelloReflection(session_data, self.memories, self.config)
        reflection.generate_reflection()

    def _avg_confidence(self):
        if not self.memories:
            return 0.0
        return sum([m['confidence'] for m in self.memories.values()]) / len(self.memories)