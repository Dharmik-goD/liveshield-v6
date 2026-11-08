from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Track:
    track_id: int
    label: str
    centroid: tuple[int, int]
    box: list[int]
    confidence: float
    missed_frames: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)


class ObjectTracker:
    """In-memory centroid tracker for consecutive YOLO detections."""

    def __init__(
        self,
        max_distance: float = 100.0,
        max_missed_frames: int = 12,
        history_limit: int = 30,
    ):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.history_limit = history_limit
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0

    @staticmethod
    def _centroid(box: list[int]) -> tuple[int, int]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.dist(a, b)

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: list[dict]) -> dict:
        self.frame_count += 1

        unmatched = set(self.tracks.keys())
        assignments = []

        candidates = []

        for index, detection in enumerate(detections):
            centroid = self._centroid(detection["box"])

            for track_id in unmatched:
                track = self.tracks[track_id]

                if track.label != detection["label"]:
                    continue

                distance = self._distance(
                    centroid,
                    track.centroid,
                )

                if distance <= self.max_distance:
                    candidates.append((distance, index, track_id))

        used_detections = set()

        for distance, index, track_id in sorted(candidates):
            if index in used_detections or track_id not in unmatched:
                continue

            detection = detections[index]
            centroid = self._centroid(detection["box"])
            track = self.tracks[track_id]

            track.centroid = centroid
            track.box = detection["box"]
            track.confidence = detection["confidence"]
            track.missed_frames = 0
            track.history.append(centroid)
            track.history = track.history[-self.history_limit:]

            assignments.append((index, track_id))
            used_detections.add(index)
            unmatched.remove(track_id)

        for index, detection in enumerate(detections):
            if index in used_detections:
                continue

            centroid = self._centroid(detection["box"])
            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = Track(
                track_id=track_id,
                label=detection["label"],
                centroid=centroid,
                box=detection["box"],
                confidence=detection["confidence"],
                history=[centroid],
            )

            assignments.append((index, track_id))

        for track_id in list(unmatched):
            track = self.tracks[track_id]
            track.missed_frames += 1

            if track.missed_frames > self.max_missed_frames:
                del self.tracks[track_id]

        results = []

        for index, track_id in assignments:
            track = self.tracks[track_id]

            results.append({
                "track_id": track.track_id,
                "label": track.label,
                "confidence": track.confidence,
                "box": track.box,
                "centroid": list(track.centroid),
                "trajectory": [
                    list(point) for point in track.history
                ],
            })

        return {
            "frame": self.frame_count,
            "active_count": len(self.tracks),
            "tracked_objects": results,
            "storage": "memory_only",
        }
