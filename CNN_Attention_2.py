#!/usr/bin/env python3
"""
Memory-efficient high-performance model for 90%+ apnea detection accuracy.
Includes full preprocessing pipeline from the paper.
Optimized for Tesla P100 16GB GPU with advanced techniques.
"""
!pip install wfdb
import argparse
import os
import time
from pathlib import Path
from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

try:
    import wfdb
except Exception:
    wfdb = None

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# ----------------------------- Preprocessing Pipeline ---------------------------------

class ApneaECGPreprocessor:
    """
    Preprocessor for Apnea-ECG database following the paper's methodology.
    """

    def __init__(self, data_path: str, sampling_rate: int = 100):
        """
        Args:
            data_path: Path to the Apnea-ECG database directory
            sampling_rate: Sampling rate of ECG (100 Hz for Apnea-ECG)
        """
        self.data_path = data_path
        self.fs = sampling_rate
        self.segment_length = 60  # seconds
        self.target_fs = 3  # Hz for interpolation
        self.samples_per_segment = self.segment_length * self.fs  # 6000 samples

    def load_recording(self, record_name: str):
        """
        Load ECG signal and apnea annotations for a single recording.

        Args:
            record_name: Name of the recording (e.g., 'a01', 'a02', etc.)

        Returns:
            ecg_signal: ECG signal array
            apnea_labels: Per-minute apnea labels (0=normal, 1=apnea)
        """
        record_path = os.path.join(self.data_path, record_name)

        # Load ECG signal
        record = wfdb.rdrecord(record_path)
        ecg_signal = record.p_signal[:, 0]  # First channel

        # Load apnea annotations
        annotation = wfdb.rdann(record_path, 'apn')

        # Convert annotations to per-minute labels
        n_minutes = len(ecg_signal) // (self.fs * 60)
        apnea_labels = np.zeros(n_minutes, dtype=int)

        for i, symbol in enumerate(annotation.symbol):
            minute_idx = annotation.sample[i] // (self.fs * 60)
            if minute_idx < n_minutes:
                # 'A' = apnea, 'N' = normal
                apnea_labels[minute_idx] = 1 if symbol == 'A' else 0

        return ecg_signal, apnea_labels

    def segment_ecg(self, ecg_signal: np.ndarray, labels: np.ndarray):
        """
        Step 1: Segment ECG into consecutive 60-second windows.

        Args:
            ecg_signal: Full ECG signal
            labels: Per-minute apnea labels

        Returns:
            segments: List of 60-second ECG segments
            segment_labels: Corresponding labels for each segment
        """
        n_segments = len(ecg_signal) // self.samples_per_segment
        segments = []
        segment_labels = []

        for i in range(n_segments):
            start_idx = i * self.samples_per_segment
            end_idx = start_idx + self.samples_per_segment

            if end_idx <= len(ecg_signal) and i < len(labels):
                segments.append(ecg_signal[start_idx:end_idx])
                segment_labels.append(labels[i])

        return segments, segment_labels

    def detect_r_peaks(self, ecg_segment: np.ndarray) -> np.ndarray:
        """
        Step 2: Detect R-peaks using Hamilton algorithm.
        Uses the biosppy implementation of Hamilton-Tompkins.

        Args:
            ecg_segment: 60-second ECG segment

        Returns:
            r_peaks: Array of R-peak indices
        """
        try:
            from biosppy.signals import ecg as ecg_module
            out = ecg_module.hamilton_segmenter(signal=ecg_segment, sampling_rate=self.fs)
            # biosppy returns tuple-like in some versions; try to handle both
            if isinstance(out, dict):
                r_peaks = out.get('rpeaks', np.array([]))
            elif isinstance(out, (list, tuple)):
                # some versions return an array directly
                r_peaks = out[0] if len(out) > 0 else np.array([])
            else:
                r_peaks = np.array(out)
            return np.array(r_peaks, dtype=int)
        except Exception:
            # Fallback: simple peak detection
            return self._simple_peak_detection(ecg_segment)

    def _simple_peak_detection(self, ecg_segment: np.ndarray) -> np.ndarray:
        """
        Simple R-peak detection fallback if biosppy is not available.
        """
        from scipy.signal import find_peaks

        # Normalize
        if np.std(ecg_segment) == 0:
            ecg_normalized = ecg_segment - np.mean(ecg_segment)
        else:
            ecg_normalized = (ecg_segment - np.mean(ecg_segment)) / np.std(ecg_segment)

        # Find peaks with minimum distance corresponding to physiological heart rate
        min_distance = int(0.4 * self.fs)  # ~150 bpm max
        peaks, _ = find_peaks(ecg_normalized, distance=min_distance, height=0.5)

        return peaks.astype(int)

    def compute_rr_intervals(self, r_peaks: np.ndarray):
        """
        Step 2 (continued): Compute RR intervals from R-peaks.

        Args:
            r_peaks: Array of R-peak indices

        Returns:
            rr_intervals: RR intervals in seconds
            rr_times: Time points (in seconds) for each RR interval
        """
        if len(r_peaks) < 2:
            return np.array([]), np.array([])

        # RR intervals in samples
        rr_samples = np.diff(r_peaks)

        # Convert to seconds
        rr_intervals = rr_samples / self.fs

        # Time points: midpoint between consecutive R-peaks
        rr_times = (r_peaks[:-1] + rr_samples / 2) / self.fs

        return rr_intervals, rr_times

    def clean_rr_intervals(self, rr_intervals: np.ndarray, rr_times: np.ndarray,
                          r_peak_amplitudes: np.ndarray):
        """
        Step 3: Apply median filter to clean RR intervals (Chen et al. method).

        Args:
            rr_intervals: Raw RR intervals
            rr_times: Time points for RR intervals
            r_peak_amplitudes: Amplitudes at R-peaks

        Returns:
            clean_rr: Cleaned RR intervals
            clean_times: Corresponding time points
            clean_amps: Cleaned amplitudes
        """
        if len(rr_intervals) < 5:
            return rr_intervals, rr_times, r_peak_amplitudes

        # Physiological bounds: 0.3s to 2.0s (30-200 bpm)
        valid_mask = (rr_intervals >= 0.3) & (rr_intervals <= 2.0)

        # Apply median filter (window size 5 as commonly used)
        kernel_size = 5
        # medfilt requires odd kernel and kernel <= len(rr_intervals)
        if kernel_size >= len(rr_intervals):
            rr_filtered = rr_intervals
        else:
            rr_filtered = medfilt(rr_intervals, kernel_size=kernel_size)

        # Remove outliers: values that differ too much from filtered version
        threshold = 0.2  # 200ms threshold
        outlier_mask = np.abs(rr_intervals - rr_filtered) < threshold

        # Combine masks
        final_mask = valid_mask & outlier_mask

        clean_rr = rr_intervals[final_mask]
        clean_times = rr_times[final_mask]
        clean_amps = r_peak_amplitudes[final_mask] if len(r_peak_amplitudes) == len(rr_intervals) else r_peak_amplitudes

        return clean_rr, clean_times, clean_amps

    def extract_r_peak_amplitudes(self, ecg_segment: np.ndarray, r_peaks: np.ndarray) -> np.ndarray:
        """
        Step 4: Extract R-peak amplitudes from ECG.

        Args:
            ecg_segment: ECG segment
            r_peaks: R-peak indices

        Returns:
            amplitudes: ECG amplitudes at R-peak locations
        """
        if len(r_peaks) == 0:
            return np.array([])
        # Exclude last peak (used for last RR)
        indices = r_peaks[:-1] if len(r_peaks) > 1 else r_peaks
        return ecg_segment[indices]

    def interpolate_to_3hz(self, values: np.ndarray, times: np.ndarray,
                          segment_duration: float = 60.0) -> np.ndarray:
        """
        Step 5: Cubic interpolation to regular 3 Hz grid.

        Args:
            values: Irregularly sampled values (RR intervals or amplitudes)
            times: Time points for the values
            segment_duration: Duration of segment in seconds

        Returns:
            interpolated: Values interpolated to 3 Hz grid (180 samples)
        """
        n_samples = int(segment_duration * self.target_fs)
        target_times = np.linspace(0, segment_duration, n_samples, endpoint=False)

        if len(values) < 2 or len(times) < 2:
            # Not enough data points for interpolation
            return np.zeros(n_samples, dtype=np.float32)

        # Ensure times are within bounds
        times = np.clip(times, 0, segment_duration - 0.01)

        # Cubic interpolation
        try:
            interp_func = interp1d(times, values, kind='cubic', bounds_error=False, fill_value='extrapolate')
            interpolated = interp_func(target_times)
        except Exception:
            # Fallback to linear if cubic fails
            interp_func = interp1d(times, values, kind='linear', bounds_error=False, fill_value='extrapolate')
            interpolated = interp_func(target_times)

        return np.asarray(interpolated, dtype=np.float32)

    def process_segment(self, ecg_segment: np.ndarray) -> np.ndarray:
        """
        Process a single 60-second ECG segment through the full pipeline.

        Args:
            ecg_segment: 60-second ECG segment

        Returns:
            features: 2-channel time series (RR intervals and amplitudes) at 3 Hz
                     Shape: (180, 2)
        """
        # Step 2: Detect R-peaks
        r_peaks = self.detect_r_peaks(ecg_segment)

        if len(r_peaks) < 2:
            # Not enough peaks, return zeros
            return np.zeros((180, 2), dtype=np.float32)

        # Step 2: Compute RR intervals
        rr_intervals, rr_times = self.compute_rr_intervals(r_peaks)

        # Step 4: Extract R-peak amplitudes
        r_peak_amps = self.extract_r_peak_amplitudes(ecg_segment, r_peaks)

        # Step 3: Clean RR intervals and amplitudes
        clean_rr, clean_times, clean_amps = self.clean_rr_intervals(rr_intervals, rr_times, r_peak_amps)

        if len(clean_rr) < 2:
            # Not enough valid data
            return np.zeros((180, 2), dtype=np.float32)

        # Step 5: Interpolate to 3 Hz
        rr_interp = self.interpolate_to_3hz(clean_rr, clean_times)
        amp_interp = self.interpolate_to_3hz(clean_amps, clean_times)

        # Step 6: Stack into 2-channel features
        features = np.stack([rr_interp, amp_interp], axis=1)

        return features

# ----------------------------- Utilities ---------------------------------

def set_seed(seed=42):
    import random as _rnd
    _rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------- Focal Loss ---------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none',
            weight=self.weight, label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ----------------------------- Efficient Model ---------------------------

class EfficientResBlock(nn.Module):
    """Efficient residual block with depthwise separable convolutions."""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        # Depthwise
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        # Pointwise
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.dropout(x)
        return F.gelu(residual + x)


class EfficientApneaNet(nn.Module):
    """Memory-efficient architecture with multi-branch inputs for high accuracy."""
    def __init__(self, d_model=128, n_blocks=8, dropout=0.2, input_channels=2):
        super().__init__()

        # Multi-branch input: time domain + frequency domain
        self.time_stem = nn.Sequential(
            nn.Conv1d(input_channels, d_model//2, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
        )

        # Frequency features via learnable filters (larger kernel for low freq)
        self.freq_stem = nn.Sequential(
            nn.Conv1d(input_channels, d_model//2, kernel_size=51, padding=25, stride=2),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
        )

        # Combine branches
        self.combine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Efficient residual blocks with varying receptive fields
        self.blocks = nn.ModuleList([
            EfficientResBlock(d_model, kernel_size=7 if i % 2 == 0 else 11)
            for i in range(n_blocks)
        ])

        # Add skip connections every 2 blocks
        self.skip_connections = n_blocks > 4

        # Lightweight channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model//4, 1),
            nn.GELU(),
            nn.Conv1d(d_model//4, d_model, 1),
            nn.Sigmoid()
        )

        # Temporal attention (memory efficient)
        self.temp_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.temp_norm = nn.LayerNorm(d_model)

        # Classification head with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        # Dual pathway for time and frequency features
        x_time = self.time_stem(x)
        x_freq = self.freq_stem(x)
        x = torch.cat([x_time, x_freq], dim=1)
        x = self.combine(x)  # (B, d_model, L/4)

        # Residual blocks with skip connections
        for i, block in enumerate(self.blocks):
            if self.skip_connections and i > 0 and i % 2 == 0 and i < len(self.blocks) - 1:
                identity = x
                x = block(x)
                if i + 2 < len(self.blocks):
                    x = x + identity  # Skip connection
            else:
                x = block(x)

        # Channel attention
        attn_weights = self.channel_attn(x)
        x = x * attn_weights

        # Global features
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, d_model)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, d_model)

        # Temporal attention on downsampled sequence
        # Ensure we have at least some length to pool to 50; if not, adaptively pool to min length
        pool_len = min(50, x.size(-1))
        x_seq = F.adaptive_avg_pool1d(x, pool_len).transpose(1, 2)  # (B, pool_len, d_model)
        x_attn, _ = self.temp_attn(x_seq, x_seq, x_seq)
        x_attn = self.temp_norm(x_attn + x_seq)
        x_attn = x_attn.mean(dim=1)  # (B, d_model)

        # Combine features
        x_combined = torch.cat([x_avg, x_max, x_attn], dim=1)

        # Classify
        logits = self.classifier(x_combined)
        return logits

# --------------------------- Dataset ---------------------------

class ApneaDataset(Dataset):
    """Optimized dataset with enhanced data augmentation and preprocessing."""

    def __init__(self, data_dir: str, record_names: list = None, cache_dir: str = None,
                 segment_length: int = 6000, stride: int = 3000, split='train',
                 augment=True, use_preprocessing=False):
        super().__init__()
        self.segment_length = int(segment_length)
        self.stride = int(stride)
        self.split = split
        self.augment = augment and (split == 'train')
        self.use_preprocessing = use_preprocessing

        cache_dir = Path(cache_dir) if cache_dir else Path(data_dir)
        preprocess_suffix = '_preprocessed' if use_preprocessing else ''
        cache_file = cache_dir / f'apnea_{split}_{segment_length}_{stride}{preprocess_suffix}.pt'

        if cache_file.exists():
            print(f"Loading cached {split} from {cache_file}")
            data = torch.load(cache_file)
            self.segments = data['segments']
            self.labels = data['labels']
        else:
            assert wfdb is not None, "wfdb required"
            assert record_names is not None, "record_names required"

            self.segments = []
            self.labels = []
            self.data_dir = Path(data_dir)

            if use_preprocessing:
                self.preprocessor = ApneaECGPreprocessor(str(data_dir))

            print(f"Processing {len(record_names)} records for {split}...")
            for i, rec in enumerate(record_names):
                print(f"  [{i+1}/{len(record_names)}] {rec}...", end='\r')
                if use_preprocessing:
                    self._load_record_with_preprocessing(rec)
                else:
                    self._load_record(rec)

            if len(self.segments) == 0:
                raise RuntimeError("No segments loaded")

            # if segments are list of arrays of possibly different shapes, stack may fail.
            self.segments = torch.tensor(np.stack(self.segments, axis=0), dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)

            print(f"\nSaving cache to {cache_file}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'segments': self.segments, 'labels': self.labels}, cache_file)

        # Handle dimensions
        if use_preprocessing:
            # Shape should be (N, 180, 2) for preprocessed data
            if self.segments.ndim == 2:
                self.segments = self.segments.unsqueeze(-1)
        else:
            # Shape should be (N, 6000, 1) for raw data
            if self.segments.ndim == 2:
                self.segments = self.segments.unsqueeze(-1)

        print(f"{split.capitalize()}: {len(self.segments)} segments, "
              f"Shape: {self.segments.shape}, Class: {Counter(self.labels.tolist())}")

    def _load_record_with_preprocessing(self, record_name: str):
        """Load record using preprocessing pipeline."""
        try:
            ecg_signal, apnea_labels = self.preprocessor.load_recording(record_name)
            segments, segment_labels = self.preprocessor.segment_ecg(ecg_signal, apnea_labels)

            for seg, label in zip(segments, segment_labels):
                features = self.preprocessor.process_segment(seg)  # Shape: (180, 2)

                # Only keep segments with valid features
                if not np.all(features == 0):
                    self.segments.append(features)
                    self.labels.append(int(label))

        except Exception as e:
            print(f"\nError loading {record_name}: {e}")

    def _load_record(self, record_name: str):
        """Load record without preprocessing (raw ECG)."""
        try:
            record = wfdb.rdrecord(str(self.data_dir / record_name))
            signal = record.p_signal[:, 0].astype(np.float32)

            if np.isnan(signal).any():
                nans = np.isnan(signal)
                not_nans = ~nans
                if not_nans.sum() > 0:
                    signal[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), signal[not_nans])
                else:
                    signal = np.zeros_like(signal)

            annotation = wfdb.rdann(str(self.data_dir / record_name), 'apn')

            n_minutes = len(signal) // 6000
            minute_labels = np.zeros(n_minutes, dtype=int)

            for i, symbol in enumerate(annotation.symbol):
                if symbol == 'A':
                    sample = annotation.sample[i]
                    minute = sample // 6000
                    if minute < n_minutes:
                        minute_labels[minute] = 1

            n_samples = len(signal)
            for start in range(0, n_samples - self.segment_length + 1, self.stride):
                end = start + self.segment_length
                seg = signal[start:end].astype(np.float32)

                seg_mean = np.nanmean(seg)
                seg_std = np.nanstd(seg)
                if np.isnan(seg_std) or seg_std < 1e-8:
                    seg = seg - seg_mean
                else:
                    seg = (seg - seg_mean) / (seg_std + 1e-8)

                seg = np.clip(seg, -10, 10)

                minute = start // 6000
                if minute < len(minute_labels):
                    label = minute_labels[minute]
                    self.segments.append(seg)
                    self.labels.append(int(label))

        except Exception as e:
            print(f"\nError loading {record_name}: {e}")

    def _augment(self, seg):
        """Enhanced augmentation for better generalization."""
        # Convert to numpy and ensure correct shape
        if torch.is_tensor(seg):
            seg = seg.numpy()

        seg = seg.copy()

        # Handle 2D preprocessed data (180, 2) differently from 1D raw data
        if seg.ndim == 2 and seg.shape[1] == 2:
            # Apply augmentation to each channel separately
            for ch in range(2):
                channel_data = seg[:, ch].astype(np.float32)

                # Time warping
                if np.random.random() < 0.3:
                    warp_factor = np.random.uniform(0.92, 1.08)
                    indices = np.arange(len(channel_data))
                    new_len = max(1, int(len(channel_data) * warp_factor))
                    new_indices = np.linspace(0, len(channel_data)-1, new_len)
                    channel_data = np.interp(new_indices, indices, channel_data)
                    if len(channel_data) > seg.shape[0]:
                        channel_data = channel_data[:seg.shape[0]]
                    else:
                        channel_data = np.pad(channel_data, (0, seg.shape[0] - len(channel_data)), 'edge')

                # Gaussian noise
                if np.random.random() < 0.5:
                    channel_data = channel_data + np.random.normal(0, 0.05, channel_data.shape).astype(np.float32)

                # Random scaling
                if np.random.random() < 0.3:
                    channel_data = channel_data * np.random.uniform(0.9, 1.1)

                seg[:, ch] = channel_data
        else:
            # Raw 1D data augmentation
            seg = seg.squeeze().astype(np.float32)

            # Time warping
            if np.random.random() < 0.3:
                warp_factor = np.random.uniform(0.92, 1.08)
                indices = np.arange(len(seg))
                new_len = max(1, int(len(seg) * warp_factor))
                new_indices = np.linspace(0, len(seg)-1, new_len)
                seg = np.interp(new_indices, indices, seg)
                if len(seg) > self.segment_length:
                    seg = seg[:self.segment_length]
                else:
                    seg = np.pad(seg, (0, self.segment_length - len(seg)), 'edge')

            # Gaussian noise
            if np.random.random() < 0.5:
                seg = seg + np.random.normal(0, 0.05, seg.shape).astype(np.float32)

            # Random scaling
            if np.random.random() < 0.3:
                seg = seg * np.random.uniform(0.9, 1.1)

            # Random shift
            if np.random.random() < 0.3:
                shift = np.random.randint(-300, 300)
                seg = np.roll(seg, shift)

            # Cutout
            if np.random.random() < 0.2:
                cutout_len = np.random.randint(100, 500)
                start_cut = np.random.randint(0, max(1, len(seg) - cutout_len))
                seg[start_cut:start_cut+cutout_len] = 0

            # Random baseline shift
            if np.random.random() < 0.2:
                seg = seg + np.random.uniform(-0.1, 0.1)

        return torch.from_numpy(seg.astype(np.float32))

    def __len__(self):
        return self.segments.shape[0]

    def __getitem__(self, idx):
        seg = self.segments[idx]
        label = self.labels[idx]

        if self.augment:
            seg = self._augment(seg)

        # Ensure proper dimensions
        if seg.ndim == 1:
            seg = seg.unsqueeze(-1)
        elif seg.ndim == 3:
            seg = seg.squeeze(0)

        return seg, label

# -------------------------- Training ------------------------

def compute_class_weights(labels_tensor):
    counts = Counter(labels_tensor.tolist())
    total = sum(counts.values())
    num_classes = len(counts)
    # Avoid division by zero
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(dataloader)
    print_freq = max(1, num_batches // 15)
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(dataloader, 1):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use autocast for mixed precision if CUDA available
        if device.type == 'cuda':
            autocast = torch.cuda.amp.autocast
        else:
            # cpu autocast exists in newer torch; fall back to no-op context if not
            try:
                autocast = torch.cpu.amp.autocast  # type: ignore
            except Exception:
                class _noop:
                    def __init__(self, *a, **k): pass
                    def __enter__(self): pass
                    def __exit__(self, *a): pass
                autocast = _noop

        with autocast(enabled=(device.type == 'cuda')):
            output = model(data)
            loss = criterion(output, target)

        if torch.isnan(loss):
            print(f"\nWARNING: NaN loss, skipping batch {batch_idx}")
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += int(pred.eq(target).sum().item())
        total += int(target.size(0))

        if batch_idx % print_freq == 0 or batch_idx == num_batches:
            curr_acc = 100.0 * correct / total if total > 0 else 0.0
            curr_loss = total_loss / batch_idx
            speed = batch_idx / (time.time() - start_time + 1e-8)
            eta = (num_batches - batch_idx) / speed if speed > 0 else 0
            print(f"  Ep {epoch} [{batch_idx:4d}/{num_batches}] Loss: {curr_loss:.4f} Acc: {curr_acc:.2f}% "
                  f"({speed:.1f} b/s, ETA: {eta:.0f}s)", end='\r')
    print()
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device, use_tta=False, n_tta=5):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if use_tta:
                # Test-Time Augmentation
                probs_sum = torch.zeros((data.size(0), 2), device=device)

                # Original
                output = model(data)
                probs_sum += F.softmax(output, dim=1)

                # Augmented versions
                for _ in range(n_tta - 1):
                    # Light noise augmentation
                    noise = torch.randn_like(data) * 0.03
                    data_aug = data + noise
                    output_aug = model(data_aug)
                    probs_sum += F.softmax(output_aug, dim=1)

                probs = probs_sum / n_tta
                # Convert back to logits for loss calculation safely
                output_for_loss = torch.log(probs + 1e-10)
            else:
                output = model(data)
                probs = F.softmax(output, dim=1)
                output_for_loss = output

            loss = criterion(output_for_loss, target)

            total_loss += loss.item()
            pred = probs.argmax(dim=1)  # use probabilities for final decision

            correct += int(pred.eq(target).sum().item())
            total += int(target.size(0))

            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(target.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs), precision, recall, f1

# ------------------------------ Main ------------------------------------

def main(args):
    set_seed(args.seed)
    DATA_DIR = Path(args.data_dir)
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    record_files = list(DATA_DIR.glob('*.hea'))
    all_records = [f.stem for f in record_files]
    valid_records = [rec for rec in all_records
                    if (DATA_DIR / (rec + '.apn')).exists() and not rec.endswith('er')]

    if len(valid_records) == 0:
        raise RuntimeError("No valid records found")

    print(f"Found {len(valid_records)} valid records")

    valid_records_shuffled = valid_records.copy()
    random.Random(args.seed).shuffle(valid_records_shuffled)
    split_idx = int(len(valid_records_shuffled) * args.train_split)
    train_records = valid_records_shuffled[:split_idx]
    val_records = valid_records_shuffled[split_idx:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}\n")

    cache_dir = args.cache_dir if args.cache_dir else str(DATA_DIR)

    # Determine input channels based on preprocessing
    input_channels = 2 if args.use_preprocessing else 1
    segment_length = 180 if args.use_preprocessing else args.segment_length
    stride = 90 if args.use_preprocessing else args.stride

    train_dataset = ApneaDataset(
        str(DATA_DIR), train_records, cache_dir,
        segment_length, stride, 'train', augment=True,
        use_preprocessing=args.use_preprocessing
    )
    val_dataset = ApneaDataset(
        str(DATA_DIR), val_records, cache_dir,
        segment_length, stride, 'val', augment=False,
        use_preprocessing=args.use_preprocessing
    )

    num_workers = 2 if str(DATA_DIR).startswith('/kaggle') else 4

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        torch.cuda.empty_cache()

    model = EfficientApneaNet(
        d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout,
        input_channels=input_channels
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    class_weights = compute_class_weights(train_dataset.labels).to(device)
    print(f"Class weights: {class_weights}")

    # Use Focal Loss instead of CrossEntropyLoss
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Stochastic Weight Averaging setup
    swa_model = AveragedModel(model)
    swa_start = int(args.epochs * 0.75)  # Start SWA at 75% of training
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    no_improve = 0

    print("\nStarting training...")
    print("="*80)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler
        )

        # Use TTA for validation after epoch 40
        use_tta = epoch > 40 and args.use_tta
        val_loss, val_acc, _, val_targets, val_probs, precision, recall, f1 = validate(
            model, val_loader, criterion, device, use_tta=use_tta, n_tta=args.n_tta
        )

        # Update SWA model
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            try:
                swa_scheduler.step()
            except Exception:
                pass

        try:
            auc = roc_auc_score(val_targets, val_probs)
        except Exception:
            auc = 0.0

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={auc:.4f}")
        print(f"         Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
        if use_tta:
            print(f"         (with TTA x{args.n_tta})")
        if epoch >= swa_start:
            print(f"         (SWA active)")

        if val_acc > best_val_acc or (val_acc >= best_val_acc and f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = f1
            no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_acc': val_acc, 'val_auc': auc, 'val_f1': f1
            }, args.best_model_path)
            print(f"  ✓ Best! (Acc={val_acc:.2f}%, F1={f1:.3f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{args.patience})")

        print("-"*80)

        if no_improve >= args.patience:
            print(f"\nEarly stop at epoch {epoch}")
            break

    # Final evaluation with SWA model
    print("\n" + "="*80)
    print("Evaluating SWA model...")
    # update_bn expects a model and the loader; pass swa_model if acceptable
    try:
        model_for_bn = swa_model.module if hasattr(swa_model, 'module') else swa_model
        torch.optim.swa_utils.update_bn(train_loader, model_for_bn)
    except Exception:
        pass

    swa_val_loss, swa_val_acc, _, swa_val_targets, swa_val_probs, swa_precision, swa_recall, swa_f1 = validate(
        swa_model, val_loader, criterion, device, use_tta=args.use_tta, n_tta=args.n_tta
    )

    try:
        swa_auc = roc_auc_score(swa_val_targets, swa_val_probs)
    except Exception:
        swa_auc = 0.0

    print(f"SWA Model: Acc={swa_val_acc:.2f}%, AUC={swa_auc:.4f}, F1={swa_f1:.3f}")

    # Save SWA model if better
    try:
        swa_save_path = args.best_model_path.replace('.pth', '_swa.pth')
    except Exception:
        swa_save_path = args.best_model_path + '_swa.pth'

    if swa_val_acc > best_val_acc or (swa_val_acc >= best_val_acc and swa_f1 > best_val_f1):
        try:
            state_dict = swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'val_acc': swa_val_acc, 'val_auc': swa_auc, 'val_f1': swa_f1
            }, swa_save_path)
            print(f"✓ SWA model saved (better than regular best)")
            best_val_acc = swa_val_acc
            best_val_f1 = swa_f1
        except Exception:
            pass

    print(f"\n{'='*80}")
    print(f"BEST - Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.3f}")
    print(f"{'='*80}")

if __name__ == '__main__':
    # Auto-detect environment - FIXED PATHS
    kaggle_data = '/kaggle/input/vincent1/apnea-ecg-database-1.0.0'
    colab_data = '/content/apnea-ecg/1.0.0'
    if Path(kaggle_data).exists():
        default_data_dir = kaggle_data
        default_cache_dir = '/kaggle/working'
        default_model_path = '/kaggle/working/best_model.pth'
    elif Path(colab_data).exists():
        default_data_dir = colab_data
        default_cache_dir = '/content'
        default_model_path = '/content/best_model.pth'
    else:
        default_data_dir = None
        default_cache_dir = None
        default_model_path = 'best_model.pth'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=default_data_dir)
    parser.add_argument('--cache-dir', type=str, default=default_cache_dir)
    parser.add_argument('--segment-length', type=int, default=6000)  # 60s for raw ECG
    parser.add_argument('--stride', type=int, default=3000)  # 50% overlap
    parser.add_argument('--batch-size', type=int, default=48)  # Optimized for P100
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)  # Efficient size
    parser.add_argument('--n-blocks', type=int, default=10)  # Deeper for better accuracy
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--best-model-path', type=str, default=default_model_path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-tta', action='store_true', default=True, help='Use test-time augmentation')
    parser.add_argument('--n-tta', type=int, default=5, help='Number of TTA iterations')
    parser.add_argument('--use-preprocessing', action='store_true', default=False,
                        help='Use paper preprocessing (RR intervals + R-peak amplitudes at 3Hz)')

    args, _ = parser.parse_known_args()

    if args.data_dir is None:
        raise SystemExit("ERROR: Dataset not found")

    print("="*80)
    print("MEMORY-EFFICIENT MODEL (Target: 90%+ Accuracy)")
    print("="*80)
    print(f"  Data:       {args.data_dir}")
    print(f"  Preprocessing: {'Paper method (RR+Amp @ 3Hz)' if args.use_preprocessing else 'Raw ECG'}")
    if args.use_preprocessing:
        print(f"  Segment:    180 samples (60s @ 3Hz), stride=90")
    else:
        print(f"  Segment:    {args.segment_length} samples (60s @ 100Hz), stride={args.stride}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Model:      d_model={args.d_model}, blocks={args.n_blocks}")
    print(f"  Optimizer:  AdamW (lr={args.lr}, wd={args.weight_decay})")
    print("="*80 + "\n")

    main(args)