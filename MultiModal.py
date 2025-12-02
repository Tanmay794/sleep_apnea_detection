#!/usr/bin/env python3
"""
High-performance apnea detection with R-R interval extraction (Target: 90%+ accuracy)
Based on PhysioNet Apnea-ECG Database methodology
"""
!pip install wfdb
import argparse
import os
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

try:
    import wfdb
except Exception:
    wfdb = None

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------- Utilities ---------------------------------

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------- R-Peak Detection & R-R Interval Extraction ---------------------------------

def detect_r_peaks_hamilton(ecg_signal, fs=100):
    """
    Hamilton R-peak detection algorithm
    Returns indices of R-peaks
    """
    # Bandpass filter (5-15 Hz)
    b, a = scipy_signal.butter(2, [5, 15], btype='band', fs=fs)
    filtered = scipy_signal.filtfilt(b, a, ecg_signal)
    
    # Derivative
    diff_signal = np.diff(filtered)
    
    # Squaring
    squared = diff_signal ** 2
    
    # Moving average integration (150ms window)
    window_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks = []
    refractory = int(0.2 * fs)  # 200ms refractory period
    
    for i in range(1, len(integrated) - 1):
        if integrated[i] > threshold:
            if integrated[i] > integrated[i-1] and integrated[i] > integrated[i+1]:
                if not peaks or (i - peaks[-1]) > refractory:
                    peaks.append(i)
    
    return np.array(peaks)

def median_filter_rr(rr_intervals, window=5):
    """
    Median filter for removing physiologically uninterpretable R-R intervals
    Based on Chen et al. methodology
    """
    if len(rr_intervals) < window:
        return rr_intervals
    
    filtered = rr_intervals.copy()
    median_rr = np.median(rr_intervals)
    
    for i in range(len(rr_intervals)):
        # Check if RR interval is physiologically valid (300ms - 2000ms)
        if rr_intervals[i] < 0.3 or rr_intervals[i] > 2.0:
            filtered[i] = median_rr
            continue
        
        # Median filter
        start = max(0, i - window//2)
        end = min(len(rr_intervals), i + window//2 + 1)
        window_vals = rr_intervals[start:end]
        local_median = np.median(window_vals)
        
        # Replace outliers (> 20% deviation from local median)
        if abs(rr_intervals[i] - local_median) > 0.2 * local_median:
            filtered[i] = local_median
    
    return filtered

def extract_rr_features(ecg_segment, fs=100):
    """
    NaN-safe R-peak detection + 3 Hz interpolation.
    Returns:
        rr_interp  : length 180  (seconds * 3 Hz)
        ramp_interp: length 180
    """
    # ---- 1. R-peak detection -------------------------------------------------
    r_peaks = detect_r_peaks_hamilton(ecg_segment, fs)
    if len(r_peaks) < 3:                       # not enough peaks → dummy
        return np.zeros(180, dtype=np.float32), np.zeros(180, dtype=np.float32)

    # ---- 2. RR intervals -----------------------------------------------------
    rr_sec = np.diff(r_peaks) / fs
    rr_sec = median_filter_rr(rr_sec)          # outlier removal
    rr_times = r_peaks[1:] / fs                # time stamp of each RR

    # ---- 3. R-peak amplitudes ------------------------------------------------
    ramp = ecg_segment[r_peaks[1:]]

    # ---- 4. Cubic interpolation to 3 Hz -------------------------------------
    targ_t = np.linspace(0, 60, 180)
    kind = 'cubic' if len(rr_sec) >= 4 else 'linear'

    f_rr   = interp1d(rr_times, rr_sec,   kind=kind, bounds_error=False, fill_value=(rr_sec[0], rr_sec[-1]))
    f_amp  = interp1d(rr_times, ramp,     kind=kind, bounds_error=False, fill_value=(ramp[0],   ramp[-1]))

    rr_out   = np.clip(f_rr(targ_t), 0.3, 2.0).astype(np.float32)
    ramp_out = f_amp(targ_t).astype(np.float32)

    # ---- 5. Normalise with safety checks -------------------------------------
    rr_std = rr_out.std()
    ramp_std = ramp_out.std()
    
    if rr_std > 1e-6:
        rr_out = (rr_out - rr_out.mean()) / rr_std
    else:
        rr_out = rr_out - rr_out.mean()
    
    if ramp_std > 1e-6:
        ramp_out = (ramp_out - ramp_out.mean()) / ramp_std
    else:
        ramp_out = ramp_out - ramp_out.mean()
    
    # Clip to prevent extreme values
    rr_out = np.clip(rr_out, -10, 10)
    ramp_out = np.clip(ramp_out, -10, 10)
    
    return rr_out, ramp_out

# ----------------------------- Improved Model ---------------------------------

class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ensure total output channels == out_channels
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2  # remainder so c1 + c2 + c3 == out_channels

        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, c2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, c3, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return F.gelu(self.bn(out))


class EnhancedResBlock(nn.Module):
    """Enhanced residual block with squeeze-excitation"""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        
        # Squeeze-Excitation with stability improvements
        se_channels = max(8, channels // 8)  # Ensure at least 8 channels
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, se_channels, 1),
            nn.GELU(),
            nn.Conv1d(se_channels, channels, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        
        # Apply SE attention with safety check
        se_weight = self.se(x)
        # Clamp to prevent extreme values
        se_weight = torch.clamp(se_weight, 0.0, 1.0)
        x = x * se_weight
        
        x = self.dropout(x)
        return F.gelu(residual + x)

class ImprovedApneaNet(nn.Module):
    def __init__(self, d_model=256, n_blocks=10, dropout=0.15):
        super().__init__()
        
        # Ensure the three modality-channel outputs sum to d_model
        c1 = d_model // 3
        c2 = d_model // 3
        c3 = d_model - c1 - c2

        # ECG pathway (6000 samples) -> c1 channels
        self.ecg_stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=15, padding=7, stride=4),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Conv1d(c1, c1, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(c1),
            nn.GELU(),
        )

        # RR interval pathway (180 samples @ 3Hz) -> c2 channels
        self.rr_stem = nn.Sequential(
            nn.Conv1d(1, c2, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
        )

        # R-amplitude pathway (180 samples @ 3Hz) -> c3 channels
        self.ramp_stem = nn.Sequential(
            nn.Conv1d(1, c3, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.Conv1d(c3, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3),
            nn.GELU(),
        )
        
        # Multi-scale fusion
        self.fusion = MultiScaleBlock(d_model, d_model)
        
        # Enhanced residual blocks
        self.blocks = nn.ModuleList([
            EnhancedResBlock(d_model, kernel_size=7 if i % 2 == 0 else 11)
            for i in range(n_blocks)
        ])
        
        # Temporal attention with larger context
        self.temp_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
        self.temp_norm = nn.LayerNorm(d_model)
        self.temp_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Feature aggregation
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Enhanced classifier with multiple paths
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 2)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, ecg, rr, ramp):
        # Validate inputs
        if torch.isnan(ecg).any() or torch.isinf(ecg).any():
            ecg = torch.nan_to_num(ecg, nan=0.0, posinf=10.0, neginf=-10.0)
        if torch.isnan(rr).any() or torch.isinf(rr).any():
            rr = torch.nan_to_num(rr, nan=0.0, posinf=10.0, neginf=-10.0)
        if torch.isnan(ramp).any() or torch.isinf(ramp).any():
            ramp = torch.nan_to_num(ramp, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Process each modality
        ecg = ecg.transpose(1, 2)
        rr = rr.transpose(1, 2)
        ramp = ramp.transpose(1, 2)
        
        ecg_feat = self.ecg_stem(ecg)
        rr_feat = self.rr_stem(rr)
        ramp_feat = self.ramp_stem(ramp)
        
        # Align sequence lengths and concatenate
        target_len = min(ecg_feat.size(2), rr_feat.size(2), ramp_feat.size(2))
        ecg_feat = F.adaptive_avg_pool1d(ecg_feat, target_len)
        rr_feat = F.adaptive_avg_pool1d(rr_feat, target_len)
        ramp_feat = F.adaptive_avg_pool1d(ramp_feat, target_len)
        
        x = torch.cat([ecg_feat, rr_feat, ramp_feat], dim=1)
        
        # Multi-scale fusion
        x = self.fusion(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Multiple pooling strategies with safety
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x_std = torch.std(x, dim=2) + 1e-8  # Add epsilon for stability
        
        # Temporal attention
        x_seq = x.transpose(1, 2)
        x_attn, _ = self.temp_attn(x_seq, x_seq, x_seq)
        x_attn = self.temp_norm(x_attn + x_seq)
        x_attn = x_attn + self.temp_ffn(x_attn)
        x_attn = x_attn.mean(dim=1)
        
        # Combine all features
        x_combined = torch.cat([x_avg, x_max, x_std, x_attn], dim=1)
        
        # Final NaN check before classifier
        if torch.isnan(x_combined).any() or torch.isinf(x_combined).any():
            x_combined = torch.nan_to_num(x_combined, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Classify
        logits = self.classifier(x_combined)
        return logits

# --------------------------- Enhanced Dataset ---------------------------

class EnhancedApneaDataset(Dataset):
    """Dataset with R-R interval and R-amplitude extraction"""

    def __init__(self, data_dir: str, record_names: list = None, cache_dir: str = None,
                 segment_length: int = 6000, stride: int = 3000, split='train', augment=True):
        super().__init__()
        self.segment_length = int(segment_length)
        self.stride = int(stride)
        self.split = split
        self.augment = augment and (split == 'train')
        
        cache_dir = Path(cache_dir) if cache_dir else Path(data_dir)
        cache_file = cache_dir / f'apnea_enhanced_{split}_{segment_length}_{stride}.pt'

        if cache_file.exists():
            print(f"Loading cached {split} from {cache_file}")
            data = torch.load(cache_file)
            self.ecg_segments = data['ecg_segments']
            self.rr_segments = data['rr_segments']
            self.ramp_segments = data['ramp_segments']
            self.labels = data['labels']
        else:
            assert wfdb is not None, "wfdb required"
            assert record_names is not None, "record_names required"
            
            self.ecg_segments = []
            self.rr_segments = []
            self.ramp_segments = []
            self.labels = []
            self.data_dir = Path(data_dir)
            
            print(f"Processing {len(record_names)} records for {split} (with R-R extraction)...")
            for i, rec in enumerate(record_names):
                print(f"  [{i+1}/{len(record_names)}] {rec}...", end='\r')
                self._load_record(rec)
            
            if len(self.ecg_segments) == 0:
                raise RuntimeError("No segments loaded")
            
            self.ecg_segments = torch.tensor(np.stack(self.ecg_segments, axis=0), dtype=torch.float32)
            self.rr_segments = torch.tensor(np.stack(self.rr_segments, axis=0), dtype=torch.float32)
            self.ramp_segments = torch.tensor(np.stack(self.ramp_segments, axis=0), dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            
            print(f"\nSaving cache to {cache_file}")
            torch.save({
                'ecg_segments': self.ecg_segments,
                'rr_segments': self.rr_segments,
                'ramp_segments': self.ramp_segments,
                'labels': self.labels
            }, cache_file)

        if self.ecg_segments.ndim == 2:
            self.ecg_segments = self.ecg_segments.unsqueeze(-1)
        if self.rr_segments.ndim == 2:
            self.rr_segments = self.rr_segments.unsqueeze(-1)
        if self.ramp_segments.ndim == 2:
            self.ramp_segments = self.ramp_segments.unsqueeze(-1)

        print(f"{split.capitalize()}: {len(self.ecg_segments)} segments, "
              f"Class: {Counter(self.labels.tolist())}")

    def _load_record(self, record_name: str):
        try:
            rec = wfdb.rdrecord(str(self.data_dir / record_name))
            sig = rec.p_signal[:, 0].astype(np.float32)
            if np.isnan(sig).any():
                nans = np.isnan(sig)
                sig[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), sig[~nans])

            ann = wfdb.rdann(str(self.data_dir / record_name), 'apn')
            n_min = len(sig) // 6000
            mins = np.zeros(n_min, dtype=int)
            for samp, sym in zip(ann.sample, ann.symbol):
                if sym == 'A':
                    m = samp // 6000
                    if m < n_min:
                        mins[m] = 1

            for start in range(0, len(sig) - self.segment_length + 1, self.stride):
                seg = sig[start:start + self.segment_length]
                
                # Normalise ECG with safety
                seg_std = seg.std()
                if seg_std > 1e-6:
                    seg = (seg - seg.mean()) / seg_std
                else:
                    seg = seg - seg.mean()
                seg = np.clip(seg, -10, 10)

                rr, ramp = extract_rr_features(seg, fs=100)

                minute = start // 6000
                if minute < len(mins):
                    self.ecg_segments.append(seg)
                    self.rr_segments.append(rr)
                    self.ramp_segments.append(ramp)
                    self.labels.append(int(mins[minute]))
        except Exception as e:
            print(f'\nSkip {record_name}: {e}')
    
    def _augment(self, ecg, rr, ramp):
        ecg, rr, ramp = map(lambda x: x.numpy() if torch.is_tensor(x) else x, (ecg, rr, ramp))

        if np.random.rand() < 0.5:
            ecg += np.random.normal(0, 0.02, ecg.shape).astype(np.float32)
            rr += np.random.normal(0, 0.01, rr.shape).astype(np.float32)
            ramp += np.random.normal(0, 0.01, ramp.shape).astype(np.float32)

        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            ecg *= scale
            ramp *= scale

        if np.random.rand() < 0.2:
            shift = np.random.randint(-150, 150)
            ecg = np.roll(ecg, shift)

        # Clip after augmentation
        ecg = np.clip(ecg, -10, 10)
        rr = np.clip(rr, -10, 10)
        ramp = np.clip(ramp, -10, 10)

        return tuple(map(torch.from_numpy, (ecg, rr, ramp)))
            
    def __len__(self):
        return self.ecg_segments.shape[0]

    def __getitem__(self, idx):
        ecg = self.ecg_segments[idx]
        rr = self.rr_segments[idx]
        ramp = self.ramp_segments[idx]
        label = self.labels[idx]
        
        if self.augment:
            ecg, rr, ramp = self._augment(ecg, rr, ramp)
        
        # Ensure correct shape
        if ecg.ndim == 1:
            ecg = ecg.unsqueeze(-1)
        if rr.ndim == 1:
            rr = rr.unsqueeze(-1)
        if ramp.ndim == 1:
            ramp = ramp.unsqueeze(-1)
        
        return ecg, rr, ramp, label

# -------------------------- Training ------------------------

def compute_class_weights(labels_tensor):
    counts = Counter(labels_tensor.tolist())
    total = sum(counts.values())
    num_classes = len(counts)
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

    for batch_idx, (ecg, rr, ramp, target) in enumerate(dataloader, 1):
        ecg = ecg.to(device, non_blocking=True)
        rr = rr.to(device, non_blocking=True)
        ramp = ramp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            output = model(ecg, rr, ramp)
            loss = criterion(output, target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWARNING: NaN/Inf loss, skipping batch {batch_idx}")
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
            curr_acc = 100.0 * correct / total
            curr_loss = total_loss / batch_idx
            speed = batch_idx / (time.time() - start_time)
            eta = (num_batches - batch_idx) / speed if speed > 0 else 0
            
            print(f"  Ep {epoch} [{batch_idx:4d}/{num_batches}] "
                  f"Loss: {curr_loss:.4f} Acc: {curr_acc:.2f}% "
                  f"({speed:.1f} b/s, ETA: {eta:.0f}s)", end='\r')

    print()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for ecg, rr, ramp, target in dataloader:
            ecg = ecg.to(device, non_blocking=True)
            rr = rr.to(device, non_blocking=True)
            ramp = ramp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = model(ecg, rr, ramp)
            loss = criterion(output, target)

            total_loss += loss.item()
            probs = F.softmax(output, dim=1)[:, 1]
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(target.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Calculate specificity and sensitivity
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs), precision, recall, f1, sensitivity, specificity

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

    import random
    valid_records_shuffled = valid_records.copy()
    random.Random(args.seed).shuffle(valid_records_shuffled)
    split_idx = int(len(valid_records_shuffled) * args.train_split)
    train_records = valid_records_shuffled[:split_idx]
    val_records = valid_records_shuffled[split_idx:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}\n")

    cache_dir = args.cache_dir if args.cache_dir else str(DATA_DIR)
    train_dataset = EnhancedApneaDataset(
        str(DATA_DIR), train_records, cache_dir,
        args.segment_length, args.stride, 'train', augment=True
    )
    val_dataset = EnhancedApneaDataset(
        str(DATA_DIR), val_records, cache_dir,
        args.segment_length, args.stride, 'val', augment=False
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

    model = ImprovedApneaNet(
        d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    class_weights = compute_class_weights(train_dataset.labels).to(device)
    print(f"Class weights: {class_weights}")

    # Use Focal Loss for better class imbalance handling
        # stable, NaN-free loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.25
    )

    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_val_f1 = 0.0
    no_improve = 0

    print("\nStarting training...")
    print("="*100)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler
        )

        val_loss, val_acc, _, val_targets, val_probs, precision, recall, f1, sensitivity, specificity = validate(
            model, val_loader, criterion, device
        )

        try:
            auc = roc_auc_score(val_targets, val_probs)
        except Exception:
            auc = 0.0

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={auc:.4f}")
        print(f"         Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
        print(f"         Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}")

        if val_acc > best_val_acc or (val_acc >= best_val_acc and f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = f1
            no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_acc': val_acc, 'val_auc': auc, 'val_f1': f1,
                'sensitivity': sensitivity, 'specificity': specificity
            }, args.best_model_path)
            print(f"  ✓ Best! (Acc={val_acc:.2f}%, F1={f1:.3f}, Sens={sensitivity:.3f}, Spec={specificity:.3f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{args.patience})")

        print("-"*100)

        if no_improve >= args.patience:
            print(f"\nEarly stop at epoch {epoch}")
            break

    print(f"\n{'='*100}")
    print(f"BEST - Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.3f}")
    print(f"{'='*100}")


if __name__ == '__main__':
    kaggle_data = '/kaggle/input/vincent2/apnea-ecg-database-1.0.0'
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
    parser.add_argument('--segment-length', type=int, default=6000)
    parser.add_argument('--stride', type=int, default=2400)  # 60% overlap for more data
    parser.add_argument('--batch-size', type=int, default=48)  # Adjusted for multi-modal
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-blocks', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--best-model-path', type=str, default=default_model_path)
    parser.add_argument('--seed', type=int, default=42)

    args, _ = parser.parse_known_args()

    if args.data_dir is None:
        raise SystemExit("ERROR: Dataset not found")

    print("="*100)
    print("ENHANCED MODEL WITH R-R INTERVALS (Target: 90%+ Accuracy)")
    print("="*100)
    print(f"  Data:       {args.data_dir}")
    print(f"  Segment:    {args.segment_length} samples (60s), stride={args.stride}")
    print(f"  Features:   ECG + R-R Intervals + R-peak Amplitudes")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Model:      d_model={args.d_model}, blocks={args.n_blocks}")
    print(f"  Optimizer:  AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Loss:       Focal Loss (alpha=0.25, gamma=2.0)")
    print("="*100 + "\n")

    main(args)