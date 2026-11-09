"""Uniform-stride dual-phase sampling for sparse EchoNet supervision."""

from typing import Dict

import numpy as np
import torch

from datasets.echonet_dataset import EchoNetDynamicDataset


class DualPhaseUniformEchoNetDataset(EchoNetDynamicDataset):
    """Expose one ED-centred and one ES-centred uniform clip per video.

    Dataset length is twice the number of videos.  Even items are centred on
    ED and odd items on ES.  Every clip has constant temporal stride and at
    least its selected phase is exactly present on the sampling grid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        required = (self.clip_length - 1) * self.period + 1
        if "NumberOfFrames" not in self.metadata.columns:
            raise KeyError("FileList.csv must contain NumberOfFrames")

        frame_counts = {}
        for _, row in self.metadata.iterrows():
            filename = str(row["FileName"])
            if not filename.lower().endswith(".avi"):
                filename += ".avi"
            frame_counts[filename] = int(row["NumberOfFrames"])

        before = len(self.samples)
        self.samples = [
            sample for sample in self.samples
            if frame_counts.get(sample["filename"], 0) >= required
        ]
        self.excluded_short_videos = before - len(self.samples)
        print(
            f"[DualPhaseUniformEchoNetDataset] split={self.split} "
            f"videos={len(self.samples)} excluded_short={self.excluded_short_videos} "
            f"minimum_frames={required}"
        )

    def __len__(self):
        return 2 * len(self.samples)

    def _uniform_phase_indices(self, n_frames: int, anchor: int):
        span = (self.clip_length - 1) * self.period
        if n_frames <= span:
            raise ValueError(
                f"Video has {n_frames} frames but uniform clip requires at least {span + 1}"
            )
        candidates = []
        for slot in range(self.clip_length):
            start = int(anchor) - slot * self.period
            if start >= 0 and start + span < n_frames:
                candidates.append((abs(slot - self.clip_length // 2), slot, start))
        if not candidates:
            raise RuntimeError(f"Cannot place annotated frame {anchor} on uniform grid")
        _, slot, start = min(candidates)
        indices = start + np.arange(self.clip_length, dtype=np.int64) * self.period
        return start, slot, indices

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base_index, phase_code = divmod(index, 2)  # 0=ED, 1=ES
        sample = self.samples[base_index]
        video, orig_h, orig_w = self._load_video(sample["video_path"])
        _, n_frames, out_h, out_w = video.shape
        ed, es = int(sample["ed_index"]), int(sample["es_index"])
        anchor = ed if phase_code == 0 else es
        start, anchor_slot, indices = self._uniform_phase_indices(n_frames, anchor)

        ed_mask = self._trace_to_mask(sample["ed_trace"], orig_h, orig_w, out_h, out_w)
        es_mask = self._trace_to_mask(sample["es_trace"], orig_h, orig_w, out_h, out_w)
        ed_hits = np.flatnonzero(indices == ed)
        es_hits = np.flatnonzero(indices == es)
        ed_slot = int(ed_hits[0]) if len(ed_hits) else -1
        es_slot = int(es_hits[0]) if len(es_hits) else -1
        labels = -np.ones(self.clip_length, dtype=np.int64)
        if ed_slot >= 0:
            labels[ed_slot] = 1
        if es_slot >= 0:
            labels[es_slot] = 2

        return {
            "video": torch.from_numpy(video[:, indices]).float(),
            "ef": torch.tensor(sample["ef"], dtype=torch.float32),
            "ed_frame": torch.from_numpy(video[:, ed]).float(),
            "es_frame": torch.from_numpy(video[:, es]).float(),
            "ed_mask": torch.from_numpy(ed_mask).float(),
            "es_mask": torch.from_numpy(es_mask).float(),
            "ed_index": torch.tensor(ed), "es_index": torch.tensor(es),
            "ed_in_clip": torch.tensor(ed_slot), "es_in_clip": torch.tensor(es_slot),
            "phase_labels": torch.from_numpy(labels),
            "clip_start": torch.tensor(start),
            "source_indices": torch.from_numpy(indices.copy()).long(),
            "anchor_phase": torch.tensor(phase_code),
            "anchor_in_clip": torch.tensor(anchor_slot),
            "filename": sample["filename"],
        }

