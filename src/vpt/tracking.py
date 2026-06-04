"""Lightweight centroid tracker.

YOLO emits a fresh set of detections per frame with no identity across frames.
To compute (vx, vy) we need to associate detections to tracks. This is a simple
class-aware nearest-centroid matcher: good enough when objects are sparse,
move smoothly, and YOLO is reasonably stable. Swap in a real tracker
(ByteTrack, BoT-SORT, Ultralytics' built-in `model.track`) for production.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float
    vy: float
    age: int  # frames since last seen (0 when matched this frame)


class CentroidTracker:
    """Greedy class-aware nearest-centroid tracker on normalized xywh detections."""

    def __init__(self, max_age: int = 5, max_distance: float = 0.08) -> None:
        self.max_age = max_age
        self.max_distance = max_distance
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0

    def update(self, detections: list[tuple[int, float, float, float, float]]
               ) -> list[tuple[int, Track]]:
        """detections: list of (cls, cx, cy, w, h) with normalized coordinates.

        Returns the list of (track_id, Track) detected this frame, with vx/vy
        derived from each track's previous centroid (0 for newly-spawned tracks).
        Stale tracks beyond `max_age` are pruned.
        """
        unmatched_track_ids = set(self.tracks.keys())
        emitted: list[tuple[int, Track]] = []

        for cls, cx, cy, w, h in detections:
            best_id, best_d2 = None, self.max_distance ** 2
            for tid in unmatched_track_ids:
                tr = self.tracks[tid]
                if tr.cls != cls:
                    continue
                d2 = (tr.cx - cx) ** 2 + (tr.cy - cy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = tid

            if best_id is not None:
                prev = self.tracks[best_id]
                vx = cx - prev.cx
                vy = cy - prev.cy
                tr = Track(cls=cls, cx=cx, cy=cy, w=w, h=h, vx=vx, vy=vy, age=0)
                self.tracks[best_id] = tr
                unmatched_track_ids.discard(best_id)
                emitted.append((best_id, tr))
            else:
                tid = self._next_id
                self._next_id += 1
                tr = Track(cls=cls, cx=cx, cy=cy, w=w, h=h, vx=0.0, vy=0.0, age=0)
                self.tracks[tid] = tr
                emitted.append((tid, tr))

        for tid in list(unmatched_track_ids):
            self.tracks[tid].age += 1
            if self.tracks[tid].age > self.max_age:
                del self.tracks[tid]

        return emitted
