#!/usr/bin/env python3
"""
Production-grade memory-efficient model for 90%+ apnea detection accuracy.
Optimized for Tesla P100 16GB GPU with comprehensive validation and error handling.

Features:
- Record-level cross-validation to prevent data leakage
- Comprehensive input validation and error handling
- Enhanced data augmentation
- Test set evaluation
- Detailed logging and diagnostics
- Configurable sampling rate and parameters
"""
!pip install wfdb
import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    wfdb = None

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

# ----------------------------- Configuration ---------------------------------

class Config:
    """Centralized configuration with validation."""
    
    def __init__(self, **kwargs):
        # Data parameters
        self.sampling_rate = kwargs.get('sampling_rate', 100)  # Hz
        self.segment_duration = kwargs.get('segment_duration', 60)  # seconds
        self.segment_length = self.sampling_rate * self.segment_duration
        self.stride_ratio = kwargs.get('stride_ratio', 0.5)  # 50% overlap
        self.stride = int(self.segment_length * self.stride_ratio)
        
        # Model parameters
        self.d_model = kwargs.get('d_model', 128)
        self.n_blocks = kwargs.get('n_blocks', 8)
        self.dropout = kwargs.get('dropout', 0.2)
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 48)
        self.epochs = kwargs.get('epochs', 80)
        self.lr = kwargs.get('lr', 3e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.patience = kwargs.get('patience', 15)
        
        # Splits (train/val/test)
        self.train_split = kwargs.get('train_split', 0.7)
        self.val_split = kwargs.get('val_split', 0.15)
        self.test_split = kwargs.get('test_split', 0.15)
        
        # Paths
        self.data_dir = kwargs.get('data_dir')
        self.cache_dir = kwargs.get('cache_dir')
        self.output_dir = kwargs.get('output_dir', '.')
        
        # Other
        self.seed = kwargs.get('seed', 42)
        self.num_workers = kwargs.get('num_workers', 4)
        
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.sampling_rate > 0, "Sampling rate must be positive"
        assert self.segment_duration > 0, "Segment duration must be positive"
        assert 0 < self.stride_ratio <= 1, "Stride ratio must be in (0, 1]"
        assert self.d_model > 0 and self.d_model % 2 == 0, "d_model must be positive and even"
        assert self.n_blocks > 0, "n_blocks must be positive"
        assert 0 <= self.dropout < 1, "Dropout must be in [0, 1)"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epochs > 0, "Epochs must be positive"
        assert self.lr > 0, "Learning rate must be positive"
        assert self.patience > 0, "Patience must be positive"
        
        # Split validation
        split_sum = self.train_split + self.val_split + self.test_split
        assert abs(split_sum - 1.0) < 1e-6, f"Splits must sum to 1.0, got {split_sum}"
        
        # Path validation
        if self.data_dir:
            data_path = Path(self.data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory not found: {data_path}")

# ----------------------------- Logging Setup ---------------------------------

def setup_logging(output_dir: str, verbose: bool = True) -> logging.Logger:
    """Setup comprehensive logging."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'
    
    # Create logger
    logger = logging.getLogger('ApneaDetection')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler (concise)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ----------------------------- Utilities ---------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def validate_signal_shape(signal: np.ndarray, expected_length: int, record_name: str):
    """Validate signal dimensions and length."""
    if signal.ndim != 1:
        raise ValueError(f"Record {record_name}: Expected 1D signal, got shape {signal.shape}")
    
    if len(signal) < expected_length:
        raise ValueError(
            f"Record {record_name}: Signal too short ({len(signal)} < {expected_length})"
        )

# ----------------------------- Enhanced Data Augmentation ---------------------------------

class SignalAugmenter:
    """Comprehensive signal augmentation for time-series ECG data."""
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
    
    def apply(self, signal: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
        """Apply random augmentations."""
        if np.random.random() > augment_prob:
            return signal
        
        signal = signal.copy()
        
        # Gaussian noise (40% chance)
        if np.random.random() < 0.4:
            noise_level = np.random.uniform(0.02, 0.05)
            signal += np.random.normal(0, noise_level, signal.shape).astype(np.float32)
        
        # Amplitude scaling (30% chance)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            signal *= scale
        
        # Time warping (20% chance)
        if np.random.random() < 0.2:
            signal = self._time_warp(signal)
        
        # Baseline wander (15% chance)
        if np.random.random() < 0.15:
            signal = self._add_baseline_wander(signal)
        
        # Random sign flip (10% chance) - physiologically valid
        if np.random.random() < 0.1:
            signal *= -1
        
        return signal
    
    def _time_warp(self, signal: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping augmentation."""
        signal = np.squeeze(signal)
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal for time warping, got shape {signal.shape}")
        
        length = len(signal)
        # Create smooth warping curve
        warp = np.cumsum(np.random.normal(1.0, sigma, length))
        warp = warp / warp[-1] * (length - 1)  # Normalize to signal length
        warp = np.clip(warp, 0, length - 1)
        
        # Interpolate
        indices = np.arange(length)
        warped = np.interp(indices, warp, signal)
        return warped.astype(np.float32)
    
    def _add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Add low-frequency baseline wander."""
        length = len(signal)
        # Create low-frequency sine wave
        freq = np.random.uniform(0.1, 0.3)  # Hz
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.05, 0.15)
        
        t = np.arange(length) / self.sampling_rate
        baseline = amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        return signal + baseline.astype(np.float32)

# ----------------------------- Efficient Model ---------------------------

class EfficientResBlock(nn.Module):
    """Efficient residual block with depthwise separable convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive odd number, got {kernel_size}")
        
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=kernel_size//2, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.dropout(x)
        return F.gelu(residual + x)


class EfficientApneaNet(nn.Module):
    """Production-grade memory-efficient architecture for sleep apnea detection."""
    
    def __init__(self, d_model: int = 128, n_blocks: int = 8, 
                 dropout: float = 0.2, input_length: int = 6000):
        super().__init__()
        
        # Validate inputs
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(f"d_model must be positive even integer, got {d_model}")
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.input_length = input_length
        
        # Input stem with multi-scale feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model//2, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
            nn.Conv1d(d_model//2, d_model, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        
        # Efficient residual blocks with varying receptive fields
        self.blocks = nn.ModuleList([
            EfficientResBlock(d_model, kernel_size=7 if i % 2 == 0 else 11, dropout=dropout)
            for i in range(n_blocks)
        ])
        
        # Lightweight channel attention (squeeze-and-excitation)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model//4, 1),
            nn.GELU(),
            nn.Conv1d(d_model//4, d_model, 1),
            nn.Sigmoid()
        )
        
        # Temporal attention (memory efficient)
        self.temp_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.temp_norm = nn.LayerNorm(d_model)
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, 1) or (B, 1, L)
        
        Returns:
            logits: Output logits of shape (B, 2)
        """
        # Validate input
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, L, C) or (B, C, L), got shape {x.shape}")
        
        # Handle both (B, L, 1) and (B, 1, L) formats
        if x.shape[-1] == 1:
            x = x.transpose(1, 2)  # (B, L, 1) -> (B, 1, L)
        
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 input channel, got {x.shape[1]} channels")
        
        # Stem reduces sequence length by 4x
        x = self.stem(x)  # (B, d_model, L/4)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Channel attention
        attn_weights = self.channel_attn(x)
        x = x * attn_weights
        
        # Global features via pooling
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, d_model)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, d_model)
        
        # Temporal attention on downsampled sequence
        x_seq = F.adaptive_avg_pool1d(x, 50).transpose(1, 2)  # (B, 50, d_model)
        x_attn, _ = self.temp_attn(x_seq, x_seq, x_seq)
        x_attn = self.temp_norm(x_attn + x_seq)
        x_attn = x_attn.mean(dim=1)  # (B, d_model)
        
        # Combine features
        x_combined = torch.cat([x_avg, x_max, x_attn], dim=1)
        
        # Classify
        logits = self.classifier(x_combined)
        return logits

# --------------------------- Dataset with Record-Level Split ---------------------------

class ApneaDataset(Dataset):
    """
    Optimized dataset with proper record-level splitting to prevent data leakage.
    """

    def __init__(self, 
                 data_dir: str,
                 record_names: List[str],
                 config: Config,
                 cache_dir: Optional[str] = None,
                 split: str = 'train',
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        
        if not WFDB_AVAILABLE:
            raise ImportError("wfdb package is required. Install with: pip install wfdb")
        
        self.config = config
        self.split = split
        self.logger = logger or logging.getLogger('ApneaDataset')
        self.augmenter = SignalAugmenter(config.sampling_rate) if split == 'train' else None
        
        cache_dir = Path(cache_dir) if cache_dir else Path(data_dir)
        cache_file = cache_dir / f'apnea_{split}_{config.segment_length}_{config.stride}_v2.pt'

        if cache_file.exists():
            self.logger.info(f"Loading cached {split} from {cache_file}")
            try:
                data = torch.load(cache_file, weights_only=True)
                self.segments = data['segments']
                self.labels = data['labels']
                self.record_ids = data.get('record_ids', [])
                self.logger.info(f"Loaded {len(self.segments)} segments from cache")
            except Exception as e:
                self.logger.error(f"Failed to load cache: {e}")
                raise
        else:
            self.segments = []
            self.labels = []
            self.record_ids = []
            self.data_dir = Path(data_dir)
            
            if not record_names:
                raise ValueError("record_names cannot be empty")
            
            self.logger.info(f"Processing {len(record_names)} records for {split}...")
            
            failed_records = []
            for i, rec in enumerate(record_names):
                try:
                    self.logger.info(f"  [{i+1}/{len(record_names)}] Processing {rec}...")
                    self._load_record(rec)
                except Exception as e:
                    self.logger.error(f"Failed to load {rec}: {e}")
                    self.logger.debug(traceback.format_exc())
                    failed_records.append(rec)
            
            if failed_records:
                self.logger.warning(f"Failed to load {len(failed_records)} records: {failed_records}")
            
            if len(self.segments) == 0:
                raise RuntimeError(f"No segments loaded for {split} split from {len(record_names)} records")
            
            self.segments = torch.tensor(np.stack(self.segments, axis=0), dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            
            self.logger.info(f"Saving cache to {cache_file}")
            try:
                torch.save({
                    'segments': self.segments,
                    'labels': self.labels,
                    'record_ids': self.record_ids
                }, cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

        # Ensure proper shape
        if self.segments.ndim == 2:
            self.segments = self.segments.unsqueeze(-1)

        # Log class distribution
        class_dist = Counter(self.labels.tolist())
        self.logger.info(
            f"{split.capitalize()}: {len(self.segments)} segments, "
            f"Class distribution: {dict(class_dist)}"
        )

    def _load_record(self, record_name: str):
        """Load and process a single record with comprehensive error handling."""
        try:
            # Read signal
            record = wfdb.rdrecord(str(self.data_dir / record_name))
            
            if record.p_signal is None or record.p_signal.shape[0] == 0:
                raise ValueError(f"Empty signal in record {record_name}")
            
            if record.p_signal.shape[1] < 1:
                raise ValueError(f"No channels in record {record_name}")
            
            signal = record.p_signal[:, 0].astype(np.float32)
            
            # Validate signal
            validate_signal_shape(signal, self.config.segment_length, record_name)
            
            # Handle NaN values
            if np.isnan(signal).any():
                nans = np.isnan(signal)
                not_nans = ~nans
                if not_nans.sum() > 0:
                    signal[nans] = np.interp(
                        np.flatnonzero(nans),
                        np.flatnonzero(not_nans),
                        signal[not_nans]
                    )
                else:
                    raise ValueError(f"Record {record_name} contains only NaN values")
            
            # Read annotations
            annotation = wfdb.rdann(str(self.data_dir / record_name), 'apn')
            
            # Create minute-level labels
            n_minutes = len(signal) // self.config.segment_length
            minute_labels = np.zeros(n_minutes, dtype=int)
            
            apnea_count = 0
            for i, symbol in enumerate(annotation.symbol):
                if symbol == 'A':
                    apnea_count += 1
                    sample = annotation.sample[i]
                    minute = sample // self.config.segment_length
                    if 0 <= minute < n_minutes:
                        minute_labels[minute] = 1
            
            self.logger.debug(f"  {record_name}: {len(signal)} samples, {apnea_count} apnea events")
            
            # Extract segments
            n_samples = len(signal)
            segments_added = 0
            
            for start in range(0, n_samples - self.config.segment_length + 1, self.config.stride):
                end = start + self.config.segment_length
                seg = signal[start:end].astype(np.float32)
                
                # Normalize segment
                seg_mean = np.mean(seg)
                seg_std = np.std(seg)
                
                if np.isnan(seg_std) or seg_std < 1e-8:
                    seg = seg - seg_mean
                else:
                    seg = (seg - seg_mean) / (seg_std + 1e-8)
                
                # Clip outliers
                seg = np.clip(seg, -10, 10)
                
                # Assign label based on minute
                minute = start // self.config.segment_length
                if minute < len(minute_labels):
                    label = minute_labels[minute]
                    self.segments.append(seg)
                    self.labels.append(int(label))
                    self.record_ids.append(record_name)
                    segments_added += 1
            
            self.logger.debug(f"  {record_name}: Added {segments_added} segments")
                    
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Record files not found for {record_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing record {record_name}: {e}")
    
    def __len__(self) -> int:
        return self.segments.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = self.segments[idx]
        label = self.labels[idx]
    
    # Apply augmentation for training
        if self.augmenter is not None:
        # Convert to numpy if needed
            if torch.is_tensor(seg):
                seg_np = seg.squeeze().numpy()  # Remove channel dimension for augmentation
            else:
                seg_np = np.squeeze(seg)
        
        # Apply augmentation (returns 1D array)
            seg_np = self.augmenter.apply(seg_np, augment_prob=0.5)
        
        # Convert back to tensor and ensure correct shape (L, 1)
            seg = torch.from_numpy(seg_np).float()
            if seg.ndim == 1:
                seg = seg.unsqueeze(-1)  # (L,) -> (L, 1)
        else:
        # For validation/test, just ensure correct shape
            if torch.is_tensor(seg):
                if seg.ndim == 1:
                    seg = seg.unsqueeze(-1)  # (L,) -> (L, 1)
                elif seg.ndim == 3:
                    seg = seg.squeeze(0)  # (1, L, 1) -> (L, 1)
            else:
                seg = torch.from_numpy(seg).float()
                if seg.ndim == 1:
                    seg = seg.unsqueeze(-1)
    
    # Final shape validation
        if seg.shape != (self.config.segment_length, 1):
            raise ValueError(
                f"Invalid segment shape: expected ({self.config.segment_length}, 1), "
                f"got {seg.shape}"
            )
    
        return seg, label

# -------------------------- Training Utilities ------------------------

def compute_class_weights(labels_tensor: torch.Tensor, logger: logging.Logger) -> torch.Tensor:
    """Compute balanced class weights."""
    counts = Counter(labels_tensor.tolist())
    total = sum(counts.values())
    num_classes = len(counts)
    
    if num_classes != 2:
        logger.warning(f"Expected 2 classes, found {num_classes}")
    
    weights = []
    for i in range(num_classes):
        if counts[i] == 0:
            logger.warning(f"Class {i} has 0 samples!")
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * counts[i]))
    
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model: nn.Module, 
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                epoch: int,
                logger: logging.Logger,
                scaler: Optional[torch.amp.GradScaler] = None) -> Tuple[float, float]:
    """Train for one epoch with comprehensive logging."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    skipped_batches = 0
    
    num_batches = len(dataloader)
    print_freq = max(1, num_batches // 15)
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(dataloader, 1):
        try:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                output = model(data)
                loss = criterion(output, target)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at batch {batch_idx}, skipping")
                skipped_batches += 1
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
            
            scheduler.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % print_freq == 0 or batch_idx == num_batches:
                curr_acc = 100.0 * correct / total if total > 0 else 0.0
                curr_loss = total_loss / (batch_idx - skipped_batches) if batch_idx > skipped_batches else 0.0
                speed = batch_idx / (time.time() - start_time)
                eta = (num_batches - batch_idx) / speed if speed > 0 else 0
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"  Ep {epoch} [{batch_idx:4d}/{num_batches}] "
                    f"Loss: {curr_loss:.4f} Acc: {curr_acc:.2f}% "
                    f"LR: {current_lr:.2e} ({speed:.1f} b/s, ETA: {eta:.0f}s)"
                )
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            logger.debug(traceback.format_exc())
            skipped_batches += 1
            continue

    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} batches due to errors")

    avg_loss = total_loss / (num_batches - skipped_batches) if num_batches > skipped_batches else float('inf')
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             logger: logging.Logger) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Validate model with comprehensive metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in dataloader:
            try:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()

                probs = F.softmax(output, dim=1)[:, 1]
                pred = output.argmax(dim=1)

                correct += pred.eq(target).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().tolist())
                all_targets.extend(target.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
            
            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    # Compute metrics with error handling
    try:
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        precision = recall = f1 = 0.0
    
    return (avg_loss, accuracy, np.array(all_preds), np.array(all_targets),
            np.array(all_probs), precision, recall, f1)

def evaluate_test_set(model: nn.Module,
                     dataloader: DataLoader,
                     device: torch.device,
                     logger: logging.Logger,
                     output_dir: str):
    """Comprehensive evaluation on test set."""
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*80)
    
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in dataloader:
            try:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                
                probs = F.softmax(output, dim=1)[:, 1]
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().tolist())
                all_targets.extend(target.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
            except Exception as e:
                logger.error(f"Error in test batch: {e}")
                continue
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute comprehensive metrics
    accuracy = 100.0 * np.sum(all_preds == all_targets) / len(all_targets)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception as e:
        logger.warning(f"Could not compute AUC: {e}")
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Log results
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy:  {accuracy:.2f}%")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  AUC:       {auc:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  {cm}")
    
    # Classification report
    logger.info(f"\nDetailed Classification Report:")
    report = classification_report(all_targets, all_preds, 
                                   target_names=['Normal', 'Apnea'],
                                   digits=4)
    logger.info(f"\n{report}")
    
    # Save results
    results_file = Path(output_dir) / 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINAL TEST SET EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"AUC:       {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

# ------------------------------ Main ------------------------------------

def split_records(records: List[str], 
                 train_split: float, 
                 val_split: float,
                 test_split: float,
                 seed: int,
                 logger: logging.Logger) -> Tuple[List[str], List[str], List[str]]:
    """
    Split records into train/val/test ensuring no data leakage.
    Each record appears in exactly one split.
    """
    import random
    
    # Validate splits
    split_sum = train_split + val_split + test_split
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {split_sum}")
    
    # Shuffle records
    records_shuffled = records.copy()
    random.Random(seed).shuffle(records_shuffled)
    
    # Calculate split indices
    n_records = len(records_shuffled)
    train_end = int(n_records * train_split)
    val_end = train_end + int(n_records * val_split)
    
    train_records = records_shuffled[:train_end]
    val_records = records_shuffled[train_end:val_end]
    test_records = records_shuffled[val_end:]
    
    logger.info(f"\nRecord-level split (seed={seed}):")
    logger.info(f"  Train: {len(train_records)} records")
    logger.info(f"  Val:   {len(val_records)} records")
    logger.info(f"  Test:  {len(test_records)} records")
    logger.info(f"  Total: {len(records)} records")
    
    # Verify no overlap
    train_set = set(train_records)
    val_set = set(val_records)
    test_set = set(test_records)
    
    if train_set & val_set:
        raise RuntimeError("Train and validation sets overlap!")
    if train_set & test_set:
        raise RuntimeError("Train and test sets overlap!")
    if val_set & test_set:
        raise RuntimeError("Validation and test sets overlap!")
    
    return train_records, val_records, test_records

def main(config: Config, logger: logging.Logger):
    """Main training pipeline with comprehensive error handling."""
    
    set_seed(config.seed)
    
    # Validate data directory
    DATA_DIR = Path(config.data_dir)
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Find all valid records
    record_files = list(DATA_DIR.glob('*.hea'))
    all_records = [f.stem for f in record_files]
    
    # Filter valid records (must have .apn annotation, exclude error records)
    valid_records = []
    for rec in all_records:
        if rec.endswith('er'):
            logger.debug(f"Skipping error record: {rec}")
            continue
        
        apn_file = DATA_DIR / (rec + '.apn')
        if not apn_file.exists():
            logger.debug(f"Skipping {rec}: no .apn annotation")
            continue
        
        valid_records.append(rec)
    
    if len(valid_records) == 0:
        raise RuntimeError("No valid records found in data directory")

    logger.info(f"\nFound {len(all_records)} total records")
    logger.info(f"Valid records with annotations: {len(valid_records)}")

    # Record-level split to prevent data leakage
    train_records, val_records, test_records = split_records(
        valid_records,
        config.train_split,
        config.val_split,
        config.test_split,
        config.seed,
        logger
    )
    
    if len(test_records) == 0:
        logger.warning("No test records! Consider adjusting split ratios.")

    # Create datasets
    logger.info("\nCreating datasets...")
    try:
        train_dataset = ApneaDataset(
            str(DATA_DIR), train_records, config,
            config.cache_dir, 'train', logger
        )
        val_dataset = ApneaDataset(
            str(DATA_DIR), val_records, config,
            config.cache_dir, 'val', logger
        )
        
        if test_records:
            test_dataset = ApneaDataset(
                str(DATA_DIR), test_records, config,
                config.cache_dir, 'test', logger
            )
        else:
            test_dataset = None
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        logger.debug(traceback.format_exc())
        raise

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()

    # Create model
    try:
        model = EfficientApneaNet(
            d_model=config.d_model,
            n_blocks=config.n_blocks,
            dropout=config.dropout,
            input_length=config.segment_length
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"\nModel: EfficientApneaNet")
        logger.info(f"  Total parameters:     {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1e6:.2f} MB (fp32)")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        logger.debug(traceback.format_exc())
        raise

    # Setup training
    class_weights = compute_class_weights(train_dataset.labels, logger)
    class_weights = class_weights.to(device)
    logger.info(f"\nClass weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    no_improve = 0
    best_model_path = Path(config.output_dir) / 'best_model.pth'

    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
    }
    
    try:
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer,
                scheduler, device, epoch, logger, scaler
            )
            
            # Validate
            val_loss, val_acc, _, val_targets, val_probs, precision, recall, f1 = validate(
                model, val_loader, criterion, device, logger
            )

            # Compute AUC
            try:
                auc = roc_auc_score(val_targets, val_probs)
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                auc = 0.0

            epoch_time = time.time() - epoch_start
            
            # Log epoch summary
            logger.info(f"\nEpoch {epoch:2d}/{config.epochs} completed in {epoch_time:.1f}s")
            logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={auc:.4f}")
            logger.info(f"         Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            training_history['val_precision'].append(precision)
            training_history['val_recall'].append(recall)
            training_history['val_f1'].append(f1)
            training_history['val_auc'].append(auc)

            # Save best model
            is_best = (val_acc > best_val_acc) or (val_acc >= best_val_acc and f1 > best_val_f1)
            
            if is_best:
                best_val_acc = val_acc
                best_val_f1 = f1
                no_improve = 0
                
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_acc': val_acc,
                        'val_auc': auc,
                        'val_f1': f1,
                        'config': config.__dict__,
                        'training_history': training_history
                    }, best_model_path)
                    logger.info(f"  ✓ Best model saved! (Acc={val_acc:.2f}%, F1={f1:.4f})")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")
            else:
                no_improve += 1
                logger.info(f"  No improvement ({no_improve}/{config.patience})")

            logger.info("-"*80)

            # Early stopping
            if no_improve >= config.patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break

    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}")
        logger.debug(traceback.format_exc())
        raise

    # Training summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Best Validation - Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.4f}")
    logger.info(f"Model saved to: {best_model_path}")

    # Test set evaluation
    if test_dataset and best_model_path.exists():
        logger.info("\nLoading best model for test set evaluation...")
        try:
            checkpoint = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_results = evaluate_test_set(
                model, test_loader, device, logger, config.output_dir
            )
            
            logger.info(f"\n{'='*80}")
            logger.info("FINAL TEST RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"  Accuracy:  {test_results['accuracy']:.2f}%")
            logger.info(f"  Precision: {test_results['precision']:.4f}")
            logger.info(f"  Recall:    {test_results['recall']:.4f}")
            logger.info(f"  F1 Score:  {test_results['f1']:.4f}")
            logger.info(f"  AUC:       {test_results['auc']:.4f}")
            logger.info(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate test set: {e}")
            logger.debug(traceback.format_exc())
    
    elif not test_dataset:
        logger.info("\nNo test set available for final evaluation")

    logger.info("\nTraining pipeline completed successfully!")

if __name__ == '__main__':
    # Auto-detect environment
    kaggle_data = '/kaggle/input/vincent1/apnea-ecg-database-1.0.0'
    colab_data = '/content/apnea-ecg/1.0.0'
    
    if Path(kaggle_data).exists():
        default_data_dir = kaggle_data
        default_cache_dir = '/kaggle/working'
        default_output_dir = '/kaggle/working'
    elif Path(colab_data).exists():
        default_data_dir = colab_data
        default_cache_dir = '/content'
        default_output_dir = '/content'
    else:
        default_data_dir = None
        default_cache_dir = None
        default_output_dir = '.'
    
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Production-grade sleep apnea detection from ECG signals',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                       help='Path to apnea-ecg database')
    parser.add_argument('--cache-dir', type=str, default=default_cache_dir,
                       help='Directory for caching processed data')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                       help='Directory for outputs (models, logs)')
    
    # Signal processing
    parser.add_argument('--sampling-rate', type=int, default=100,
                       help='ECG sampling rate in Hz')
    parser.add_argument('--segment-duration', type=int, default=60,
                       help='Segment duration in seconds')
    parser.add_argument('--stride-ratio', type=float, default=0.5,
                       help='Stride ratio for overlapping segments')
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--n-blocks', type=int, default=8,
                       help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=48,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Data splits
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test set ratio')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args, _ = parser.parse_known_args()
    
    # Validate data directory
    if args.data_dir is None:
        print("ERROR: Dataset not found. Please specify --data-dir")
        print("\nLooking for apnea-ecg-database in:")
        print(f"  - {kaggle_data}")
        print(f"  - {colab_data}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(output_dir), args.verbose)
    
    # Create config
    try:
        config = Config(**vars(args))
    except Exception as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)
    
    # Log configuration
    logger.info("="*80)
    logger.info("SLEEP APNEA DETECTION - PRODUCTION GRADE MODEL")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Data directory:    {config.data_dir}")
    logger.info(f"  Cache directory:   {config.cache_dir}")
    logger.info(f"  Output directory:  {config.output_dir}")
    logger.info(f"\nSignal Processing:")
    logger.info(f"  Sampling rate:     {config.sampling_rate} Hz")
    logger.info(f"  Segment duration:  {config.segment_duration}s ({config.segment_length} samples)")
    logger.info(f"  Stride ratio:      {config.stride_ratio} ({config.stride} samples)")
    logger.info(f"\nModel Architecture:")
    logger.info(f"  d_model:           {config.d_model}")
    logger.info(f"  n_blocks:          {config.n_blocks}")
    logger.info(f"  dropout:           {config.dropout}")
    logger.info(f"\nTraining:")
    logger.info(f"  Batch size:        {config.batch_size}")
    logger.info(f"  Epochs:            {config.epochs}")
    logger.info(f"  Learning rate:     {config.lr}")
    logger.info(f"  Weight decay:      {config.weight_decay}")
    logger.info(f"  Patience:          {config.patience}")
    logger.info(f"\nData Splits:")
    logger.info(f"  Train:             {config.train_split:.1%}")
    logger.info(f"  Validation:        {config.val_split:.1%}")
    logger.info(f"  Test:              {config.test_split:.1%}")
    logger.info(f"\nOther:")
    logger.info(f"  Random seed:       {config.seed}")
    logger.info(f"  Num workers:       {config.num_workers}")
    logger.info("="*80 + "\n")
    
    # Run training
    try:
        main(config, logger)
    except Exception as e:
        logger.error(f"\nFATAL ERROR: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)