#!/usr/bin/env python3
"""
Stable and high-performance CNN-Transformer hybrid for apnea detection.
Designed for 90%+ accuracy with numerical stability.
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

try:
    import wfdb
except Exception:
    wfdb = None

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# ----------------------------- Utilities ---------------------------------

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------- Stable Model ------------------------------

class ResidualBlock(nn.Module):
    """Stable residual block with layer normalization."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: (B, C, L)
        residual = x
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.norm1(x)
        x = x.transpose(1, 2)  # (B, C, L)
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        
        return F.gelu(residual + x)


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN with residual connections."""
    def __init__(self, d_model=128, n_layers=4, dropout=0.2):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Multi-scale feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(d_model) for _ in range(n_layers)
        ])
        
        # Multi-scale pooling
        self.pool_short = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_medium = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
    def forward(self, x):
        # x: (B, L, 1)
        x = x.transpose(1, 2)  # (B, 1, L)
        
        # Initial projection
        x = self.input_proj(x)  # (B, d_model, L)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Multi-scale features
        x_short = self.pool_short(x)
        x_medium = self.pool_medium(x)
        
        # Global pooling
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, d_model)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, d_model)
        
        # Attention on downsampled sequence
        x_seq = F.adaptive_avg_pool1d(x, 100).transpose(1, 2)  # (B, 100, d_model)
        x_attn, _ = self.attention(x_seq, x_seq, x_seq)
        x_attn = self.attn_norm(x_attn + x_seq)
        x_attn = x_attn.mean(dim=1)  # (B, d_model)
        
        # Concatenate features
        x_combined = torch.cat([x_max, x_avg, x_attn], dim=-1)  # (B, d_model*3)
        
        # Classification
        logits = self.classifier(x_combined)
        return logits


# --------------------------- Dataset & Caching ---------------------------

class ApneaECGDataset(Dataset):
    """Optimized dataset with proper cache handling."""

    def __init__(self, data_dir: str, record_names: list = None, cache_dir: str = None,
                 segment_length: int = 3000, stride: int = 3000, split='train'):
        super().__init__()
        self.segment_length = int(segment_length)
        self.stride = int(stride)
        self.split = split
        
        cache_dir = Path(cache_dir) if cache_dir else Path(data_dir)
        cache_file = cache_dir / f'apnea_cache_{split}.pt'

        if cache_file.exists():
            print(f"Loading cached {split} dataset from {cache_file}")
            data = torch.load(cache_file)
            self.segments = data['segments']
            self.labels = data['labels']
        else:
            assert wfdb is not None, "wfdb not available"
            assert record_names is not None, "record_names required"
            
            self.segments = []
            self.labels = []
            self.data_dir = Path(data_dir)
            
            print(f"Processing {len(record_names)} records for {split}...")
            for i, rec in enumerate(record_names):
                print(f"  [{i+1}/{len(record_names)}] {rec}...", end='\r')
                self._load_record(rec)
            
            if len(self.segments) == 0:
                raise RuntimeError("No segments loaded")
            
            self.segments = torch.tensor(np.stack(self.segments, axis=0), dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            
            print(f"\nSaving {split} cache to {cache_file}")
            torch.save({'segments': self.segments, 'labels': self.labels}, cache_file)

        if self.segments.ndim == 2:
            self.segments = self.segments.unsqueeze(-1)

        print(f"{split.capitalize()}: {len(self.segments)} segments. "
              f"Class dist: {Counter(self.labels.tolist())}")

    def _load_record(self, record_name: str):
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
    
                # Robust normalization
                seg_mean = np.nanmean(seg)
                seg_std = np.nanstd(seg)
                if np.isnan(seg_std) or seg_std < 1e-8:
                    seg = seg - seg_mean
                else:
                    seg = (seg - seg_mean) / (seg_std + 1e-8)
                
                # Clip extreme values for stability
                seg = np.clip(seg, -10, 10)
    
                minute = start // 6000
                if minute < len(minute_labels):
                    label = minute_labels[minute]
                    self.segments.append(seg)
                    self.labels.append(int(label))
                    
        except Exception as e:
            print(f"\nError loading {record_name}: {e}")
            
    def __len__(self):
        return self.segments.shape[0]

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

# -------------------------- Training / Validation ------------------------

def compute_class_weights(labels_tensor):
    counts = Counter(labels_tensor.tolist())
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    num_batches = len(dataloader)
    print_freq = max(1, num_batches // 20)
    
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(dataloader, 1):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            output = model(data)
            loss = criterion(output, target)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nWARNING: NaN loss at batch {batch_idx}, skipping...")
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

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % print_freq == 0 or batch_idx == num_batches:
            curr_acc = 100.0 * correct / total
            curr_loss = total_loss / batch_idx
            elapsed = time.time() - start_time
            speed = batch_idx / elapsed
            eta = (num_batches - batch_idx) / speed if speed > 0 else 0
            
            print(f"  Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                  f"Loss: {curr_loss:.4f} Acc: {curr_acc:.2f}% "
                  f"Speed: {speed:.1f} b/s ETA: {eta:.0f}s", end='\r')

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
        for data, target in dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            probs = F.softmax(output, dim=1)[:, 1]
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    # Additional metrics
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

    # Find valid records
    record_files = list(DATA_DIR.glob('*.hea'))
    all_records = [f.stem for f in record_files]
    valid_records = [rec for rec in all_records 
                    if (DATA_DIR / (rec + '.apn')).exists() and not rec.endswith('er')]
    
    if len(valid_records) == 0:
        raise RuntimeError(f"No valid records found")

    print(f"Found {len(valid_records)} valid records")

    # Split records
    import random
    valid_records_shuffled = valid_records.copy()
    random.Random(args.seed).shuffle(valid_records_shuffled)
    split_idx = int(len(valid_records_shuffled) * args.train_split)
    train_records = valid_records_shuffled[:split_idx]
    val_records = valid_records_shuffled[split_idx:]
    print(f"Train: {len(train_records)} records, Val: {len(val_records)} records\n")

    # Create datasets
    cache_dir = args.cache_dir if args.cache_dir else str(DATA_DIR)
    train_dataset = ApneaECGDataset(
        str(DATA_DIR), record_names=train_records, cache_dir=cache_dir,
        segment_length=args.segment_length, stride=args.stride, split='train'
    )
    val_dataset = ApneaECGDataset(
        str(DATA_DIR), record_names=val_records, cache_dir=cache_dir,
        segment_length=args.segment_length, stride=args.stride, split='val'
    )

    # DataLoaders
    num_workers = 2 if str(DATA_DIR).startswith('/kaggle') else 4
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = MultiScaleCNN(
        d_model=args.d_model, n_layers=args.n_layers, dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Loss and optimizer
    class_weights = compute_class_weights(train_dataset.labels).to(device)
    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_val_f1 = 0.0
    no_improve = 0

    # Training loop
    print("\nStarting training...")
    print("="*90)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler=scaler
        )
        
        val_loss, val_acc, val_preds, val_targets, val_probs, precision, recall, f1 = validate(
            model, val_loader, criterion, device
        )

        try:
            auc = roc_auc_score(val_targets, val_probs)
        except Exception:
            auc = 0.0

        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:2d}/{args.epochs} - Time: {epoch_time:.1f}s")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {auc:.4f}")
        print(f"  Val   - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        improved = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = f1
            no_improve = 0
            improved = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': auc,
                'val_f1': f1
            }, args.best_model_path)
            print(f"  ✓ New best! (Acc: {val_acc:.2f}%, F1: {f1:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{args.patience})")

        print("-"*90)

        if no_improve >= args.patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    print(f"\n{'='*90}")
    print(f"Training finished!")
    print(f"Best validation - Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.4f}")
    print(f"{'='*90}")

if __name__ == '__main__':
    # Auto-detect environment
    kaggle_data = '/kaggle/input/vincent/apnea-ecg-database-1.0.0'
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
    
    parser = argparse.ArgumentParser(description='High-performance apnea detection')
    parser.add_argument('--data-dir', type=str, default=default_data_dir)
    parser.add_argument('--cache-dir', type=str, default=default_cache_dir)
    parser.add_argument('--segment-length', type=int, default=3000)
    parser.add_argument('--stride', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--best-model-path', type=str, default=default_model_path)
    parser.add_argument('--seed', type=int, default=42)

    args, _ = parser.parse_known_args()
    
    if args.data_dir is None:
        raise SystemExit(f"\nERROR: Dataset not found at {kaggle_data} or {colab_data}\n")
    
    print("="*90)
    print("CONFIGURATION")
    print("="*90)
    print(f"  Data:          {args.data_dir}")
    print(f"  Cache:         {args.cache_dir}")
    print(f"  Model save:    {args.best_model_path}")
    print(f"  Segment:       {args.segment_length} samples (30s)")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model:         d_model={args.d_model}, n_layers={args.n_layers}, dropout={args.dropout}")
    print("="*90 + "\n")
    
    main(args)