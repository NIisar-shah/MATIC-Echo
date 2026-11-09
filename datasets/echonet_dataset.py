import os
import collections
from random import sample
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon


def _defaultdict_of_lists():
    return collections.defaultdict(list)


def _polygon_area(trace: np.ndarray) -> float:
    """Approximate traced-region area from VolumeTracings chord endpoints."""
    x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
    return float(np.sum(np.hypot(x2 - x1, y2 - y1)))


class EchoNetDynamicDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_length: int = 16,
        period: int = 2,
        resize: Tuple[int, int] = (112, 112),
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        random_clip: bool = True,
    ):
        self.root = root
        self.video_dir = os.path.join(root, "Videos")
        self.filelist_path = os.path.join(root, "FileList.csv")
        self.tracings_path = os.path.join(root, "VolumeTracings.csv")

        self.split = split.lower()
        self.clip_length = clip_length
        self.period = period
        self.resize = resize
        self.mean = mean
        self.std = std
        self.random_clip = random_clip and self.split == "train"

        if not os.path.exists(self.filelist_path):
            raise FileNotFoundError(f"Missing FileList.csv: {self.filelist_path}")
        if not os.path.exists(self.tracings_path):
            raise FileNotFoundError(f"Missing VolumeTracings.csv: {self.tracings_path}")
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Missing Videos directory: {self.video_dir}")

        self.metadata = pd.read_csv(self.filelist_path)
        self.metadata["Split"] = self.metadata["Split"].str.lower()

        if self.split != "all":
            self.metadata = self.metadata[self.metadata["Split"] == self.split].reset_index(drop=True)

        self.frames = collections.defaultdict(list)
        self.traces = collections.defaultdict(_defaultdict_of_lists)

        self._load_traces()
        self.samples = self._build_samples()

    def _load_traces(self):
        with open(self.tracings_path, "r") as f:
            header = f.readline().strip().split(",")
            expected = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
            if header != expected:
                raise ValueError(f"Unexpected VolumeTracings header: {header}")

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(",")
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                frame = int(frame)

                if frame not in self.traces[filename]:
                    self.frames[filename].append(frame)

                self.traces[filename][frame].append((x1, y1, x2, y2))

        for filename in self.frames:
            for frame in self.frames[filename]:
                self.traces[filename][frame] = np.array(self.traces[filename][frame], dtype=np.float32)

    def _build_samples(self):
        samples = []

        for _, row in self.metadata.iterrows():
            filename = row["FileName"]
            if os.path.splitext(filename)[1] == "":
                filename = filename + ".avi"

            video_path = os.path.join(self.video_dir, filename)
            if not os.path.exists(video_path):
                continue

            trace_key = filename
            if trace_key not in self.frames or len(self.frames[trace_key]) < 2:
                continue

            frame_ids = sorted(self.frames[trace_key])
            # Assign phases by cavity area rather than by frame order:
            # VolumeTracings.csv lists traced frames in temporal order, which
            # does not determine which is end-diastolic. ED is the larger cavity.
            _f0, _f1 = frame_ids[0], frame_ids[-1]
            _a0 = _polygon_area(self.traces[trace_key][_f0])
            _a1 = _polygon_area(self.traces[trace_key][_f1])
            if _a0 >= _a1:
                ed_index, es_index = _f0, _f1
            else:
                ed_index, es_index = _f1, _f0

            sample = {
                "filename": filename,
                "video_path": video_path,
                "ef": float(row["EF"]) if "EF" in row else 0.0,
                "ed_index": int(ed_index),
                "es_index": int(es_index),
                "ed_trace": self.traces[trace_key][ed_index],
                "es_trace": self.traces[trace_key][es_index],
            }
            samples.append(sample)

        if len(samples) == 0:
            raise RuntimeError("No valid samples found. Check dataset structure and split names.")

        return samples

    def _load_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        frames = []
        orig_h, orig_w = None, None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if orig_h is None:
                orig_h, orig_w = frame.shape[:2]

            if self.resize is not None:
                frame = cv2.resize(frame, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_LINEAR)

            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Empty video: {path}")

        video = np.stack(frames, axis=0)           # [T, H, W, C]
        video = np.transpose(video, (3, 0, 1, 2))  # [C, T, H, W]

        if self.mean is not None:
            mean = np.array(self.mean, dtype=np.float32).reshape(3, 1, 1, 1)
            video = video - mean

        if self.std is not None:
            std = np.array(self.std, dtype=np.float32).reshape(3, 1, 1, 1)
            video = video / std

        return video, orig_h, orig_w

    def _trace_to_mask(
        self,
        trace: np.ndarray,
        orig_h: int,
        orig_w: int,
        out_h: int,
        out_w: int,
    ) -> np.ndarray:
        scale_x = out_w / float(orig_w)
        scale_y = out_h / float(orig_h)

        x1 = trace[:, 0] * scale_x
        y1 = trace[:, 1] * scale_y
        x2 = trace[:, 2] * scale_x
        y2 = trace[:, 3] * scale_y

        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))

        rr, cc = polygon(
            np.rint(y).astype(int),
            np.rint(x).astype(int),
            shape=(out_h, out_w)
        )

        mask = np.zeros((out_h, out_w), dtype=np.float32)
        mask[rr, cc] = 1.0
        return mask

    def _sample_clip(self, num_frames: int, ed_index: int, es_index: int,
                     center_on: Optional[int] = None):
        """Choose the clip start index and the frame indices it contains.

        Args:
            num_frames: number of frames in the decoded video.
            ed_index:   raw frame index of the end-diastolic annotation.
            es_index:   raw frame index of the end-systolic annotation.
            center_on:  when given, the clip is centred on this raw frame
                        index so that it lands exactly on the sampling grid.
                        Used by the phase-centred evaluation protocol.

        Returns:
            (start, indices)
        """
        needed = self.clip_length * self.period

        if center_on is not None:
            # Place `center_on` at clip position clip_length // 2. Because the
            # offset from start is (clip_length // 2) * period, the annotated
            # frame is always on the sampling grid regardless of its parity.
            half = (self.clip_length // 2) * self.period
            start = int(center_on) - half
            max_start = max(0, num_frames - needed)
            start = int(np.clip(start, 0, max_start))
            indices = start + np.arange(self.clip_length) * self.period
            return start, indices

        if num_frames < needed:
            start = 0
        else:
            if self.random_clip:
                midpoint = (ed_index + es_index) // 2
                low = max(0, midpoint - needed)
                high = min(midpoint, num_frames - needed)
                if high <= low:
                    start = max(0, min(midpoint - needed // 2, num_frames - needed))
                else:
                    start = np.random.randint(low, high + 1)
            else:
                midpoint = (ed_index + es_index) // 2
                start = max(0, min(midpoint - needed // 2, num_frames - needed))

        indices = start + np.arange(self.clip_length) * self.period
        return start, indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        video, orig_h, orig_w = self._load_video(sample["video_path"])
        _, total_frames, out_h, out_w = video.shape
        ed_frame = video[:, sample["ed_index"], :, :]
        es_frame = video[:, sample["es_index"], :, :]
        ed_mask = self._trace_to_mask(sample["ed_trace"], orig_h, orig_w, out_h, out_w)
        es_mask = self._trace_to_mask(sample["es_trace"], orig_h, orig_w, out_h, out_w)

        start, clip_indices = self._sample_clip(total_frames, sample["ed_index"], sample["es_index"])

        valid_indices = clip_indices[clip_indices < total_frames]
        clip = video[:, valid_indices, :, :]

        if clip.shape[1] < self.clip_length:
            pad_len = self.clip_length - clip.shape[1]
            pad = np.zeros((clip.shape[0], pad_len, clip.shape[2], clip.shape[3]), dtype=clip.dtype)
            clip = np.concatenate([clip, pad], axis=1)

        ed_in_clip = -1
        es_in_clip = -1

        for i, frame_id in enumerate(valid_indices):
            if int(frame_id) == sample["ed_index"]:
                ed_in_clip = i
            if int(frame_id) == sample["es_index"]:
                es_in_clip = i

        phase_labels = -1 * np.ones((self.clip_length,), dtype=np.int64)
        if ed_in_clip >= 0:
            phase_labels[ed_in_clip] = 1   # ED
        if es_in_clip >= 0:
            phase_labels[es_in_clip] = 2   # ES

        output = {
            "video": torch.from_numpy(clip).float(),                 # [C, T, H, W]
            "ef": torch.tensor(sample["ef"], dtype=torch.float32),
            "ed_frame": torch.from_numpy(ed_frame).float(),
            "es_frame": torch.from_numpy(es_frame).float(),
            "ed_mask": torch.from_numpy(ed_mask).float(),            # [H, W]
            "es_mask": torch.from_numpy(es_mask).float(),            # [H, W]
            "ed_index": torch.tensor(sample["ed_index"], dtype=torch.long),
            "es_index": torch.tensor(sample["es_index"], dtype=torch.long),
            "ed_in_clip": torch.tensor(ed_in_clip, dtype=torch.long),
            "es_in_clip": torch.tensor(es_in_clip, dtype=torch.long),
            "phase_labels": torch.from_numpy(phase_labels),          # [T]
            "clip_start": torch.tensor(start, dtype=torch.long),
            "filename": sample["filename"],
        }

        return output

    # ------------------------------------------------------------------ #
    # Phase-centred evaluation protocol
    # ------------------------------------------------------------------ #
    def get_phase_centered_clips(self, index: int) -> Dict[str, Dict]:
        """Return two clips for one video, each centred on an annotated phase.

        This is the evaluation-time counterpart to __getitem__. It guarantees
        that both the ED and the ES annotation are represented on the sampling
        grid, removing the parity restriction that otherwise discards roughly
        half of the annotated frames when period > 1.

        Returns:
            {
              "ed": {"video": [C,T,H,W], "phase_in_clip": int, "mask": [H,W], ...},
              "es": {...},
              "filename": str,
              "ef": float,
            }
        """
        sample = self.samples[index]

        video, orig_h, orig_w = self._load_video(sample["video_path"])
        _, total_frames, out_h, out_w = video.shape

        ed_mask = self._trace_to_mask(sample["ed_trace"], orig_h, orig_w, out_h, out_w)
        es_mask = self._trace_to_mask(sample["es_trace"], orig_h, orig_w, out_h, out_w)

        out = {
            "filename": sample["filename"],
            "ef": float(sample["ef"]),
            "ed_index": int(sample["ed_index"]),
            "es_index": int(sample["es_index"]),
        }

        for phase, phase_index, phase_mask in (
            ("ed", sample["ed_index"], ed_mask),
            ("es", sample["es_index"], es_mask),
        ):
            start, clip_indices = self._sample_clip(
                total_frames,
                sample["ed_index"],
                sample["es_index"],
                center_on=phase_index,
            )

            valid_indices = clip_indices[clip_indices < total_frames]
            clip = video[:, valid_indices, :, :]

            if clip.shape[1] < self.clip_length:
                pad_len = self.clip_length - clip.shape[1]
                pad = np.zeros(
                    (clip.shape[0], pad_len, clip.shape[2], clip.shape[3]),
                    dtype=clip.dtype,
                )
                clip = np.concatenate([clip, pad], axis=1)

            # Recompute the true in-clip position after boundary clamping.
            phase_in_clip = -1
            for i, frame_id in enumerate(valid_indices):
                if int(frame_id) == int(phase_index):
                    phase_in_clip = i
                    break

            out[phase] = {
                "video": torch.from_numpy(clip).float(),
                "phase_in_clip": torch.tensor(phase_in_clip, dtype=torch.long),
                "mask": torch.from_numpy(phase_mask).float(),
                "phase_index": torch.tensor(int(phase_index), dtype=torch.long),
                "clip_start": torch.tensor(int(start), dtype=torch.long),
            }

        return out
