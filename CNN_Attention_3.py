#!/usr/bin/env python3
"""
Memory-efficient high-performance model for 90%+ apnea detection accuracy.
Optimized for Tesla P100 16GB GPU with advanced techniques.
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
from torch.optim.swa_utils import AveragedModel, SWALR

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
    def __init__(self, d_model=128, n_blocks=8, dropout=0.2):
        super().__init__()
        
        # Multi-branch input: time domain + frequency domain
        self.time_stem = nn.Sequential(
            nn.Conv1d(1, d_model//2, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
        )
        
        # Frequency features via learnable filters (larger kernel for low freq)
        self.freq_stem = nn.Sequential(
            nn.Conv1d(1, d_model//2, kernel_size=51, padding=25, stride=2),
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
        # x: (B, L, 1) -> (B, 1, L)
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
        x_seq = F.adaptive_avg_pool1d(x, 50).transpose(1, 2)  # (B, 50, d_model)
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
    """Optimized dataset with enhanced data augmentation."""

    def __init__(self, data_dir: str, record_names: list = None, cache_dir: str = None,
                 segment_length: int = 6000, stride: int = 3000, split='train', augment=True):
        super().__init__()
        self.segment_length = int(segment_length)
        self.stride = int(stride)
        self.split = split
        self.augment = augment and (split == 'train')
        
        cache_dir = Path(cache_dir) if cache_dir else Path(data_dir)
        cache_file = cache_dir / f'apnea_{split}_{segment_length}_{stride}.pt'

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
            
            print(f"Processing {len(record_names)} records for {split}...")
            for i, rec in enumerate(record_names):
                print(f"  [{i+1}/{len(record_names)}] {rec}...", end='\r')
                self._load_record(rec)
            
            if len(self.segments) == 0:
                raise RuntimeError("No segments loaded")
            
            self.segments = torch.tensor(np.stack(self.segments, axis=0), dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            
            print(f"\nSaving cache to {cache_file}")
            torch.save({'segments': self.segments, 'labels': self.labels}, cache_file)

        if self.segments.ndim == 2:
            self.segments = self.segments.unsqueeze(-1)

        print(f"{split.capitalize()}: {len(self.segments)} segments, "
              f"Class: {Counter(self.labels.tolist())}")

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
    # Convert to numpy and ensure 1D
        if torch.is_tensor(seg):
            seg = seg.squeeze().numpy()  # Remove all dimensions of size 1
        else:
            seg = np.squeeze(seg)  # Ensure 1D array
    
        seg = seg.copy()
    
    # Time warping (stretch/compress)
        if np.random.random() < 0.3:
            warp_factor = np.random.uniform(0.92, 1.08)
            indices = np.arange(len(seg))
            new_indices = np.linspace(0, len(seg)-1, int(len(seg) * warp_factor))
            seg = np.interp(new_indices, indices, seg)
            if len(seg) > self.segment_length:
                seg = seg[:self.segment_length]
            else:
                seg = np.pad(seg, (0, self.segment_length - len(seg)), 'edge')
    
    # Gaussian noise (stronger)
        if np.random.random() < 0.5:
            seg = seg + np.random.normal(0, 0.05, seg.shape).astype(np.float32)
    
    # Random scaling
        if np.random.random() < 0.3:
            seg = seg * np.random.uniform(0.9, 1.1)
    
    # Random shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-300, 300)
            seg = np.roll(seg, shift)
    
    # Cutout (zero out random segments)
        if np.random.random() < 0.2:
            cutout_len = np.random.randint(100, 500)
            start_cut = np.random.randint(0, max(1, len(seg) - cutout_len))
            seg[start_cut:start_cut+cutout_len] = 0
    
    # Random baseline shift
        if np.random.random() < 0.2:
            seg = seg + np.random.uniform(-0.1, 0.1)
    
        return torch.from_numpy(seg.astype(np.float32))

    # def _augment(self, seg):
    #     """Enhanced augmentation for better generalization."""
    #     seg = seg.numpy() if torch.is_tensor(seg) else seg.copy()
        
    #     # Time warping (stretch/compress)
    #     if np.random.random() < 0.3:
    #         warp_factor = np.random.uniform(0.92, 1.08)
    #         indices = np.arange(len(seg))
    #         new_indices = np.linspace(0, len(seg)-1, int(len(seg) * warp_factor))
    #         seg = np.interp(new_indices, indices, seg)
    #         if len(seg) > self.segment_length:
    #             seg = seg[:self.segment_length]
    #         else:
    #             seg = np.pad(seg, (0, self.segment_length - len(seg)), 'edge')
        
    #     # Gaussian noise (stronger)
    #     if np.random.random() < 0.5:
    #         seg = seg + np.random.normal(0, 0.05, seg.shape).astype(np.float32)
        
    #     # Random scaling
    #     if np.random.random() < 0.3:
    #         seg = seg * np.random.uniform(0.9, 1.1)
        
    #     # Random shift
    #     if np.random.random() < 0.3:
    #         shift = np.random.randint(-300, 300)
    #         seg = np.roll(seg, shift)
        
    #     # Cutout (zero out random segments)
    #     if np.random.random() < 0.2:
    #         cutout_len = np.random.randint(100, 500)
    #         start_cut = np.random.randint(0, max(1, len(seg) - cutout_len))
    #         seg[start_cut:start_cut+cutout_len] = 0
        
    #     # Random baseline shift
    #     if np.random.random() < 0.2:
    #         seg = seg + np.random.uniform(-0.1, 0.1)
        
    #     return torch.from_numpy(seg.astype(np.float32))
            
    def __len__(self):
        return self.segments.shape[0]

    def __getitem__(self, idx):
        seg = self.segments[idx]
        label = self.labels[idx]
        
        if self.augment:
            seg = self._augment(seg)
        
        if seg.ndim == 1:
            seg = seg.unsqueeze(-1)
        elif seg.ndim == 3:
            seg = seg.squeeze(1)
        
        return seg, label

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

    for batch_idx, (data, target) in enumerate(dataloader, 1):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
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
                output = torch.log(probs + 1e-10)  # Convert back to logits for loss
            else:
                output = model(data)
                probs = F.softmax(output, dim=1)
            
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(target.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    avg_loss = total_loss / len(dataloader)
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

    import random
    valid_records_shuffled = valid_records.copy()
    random.Random(args.seed).shuffle(valid_records_shuffled)
    split_idx = int(len(valid_records_shuffled) * args.train_split)
    train_records = valid_records_shuffled[:split_idx]
    val_records = valid_records_shuffled[split_idx:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}\n")

    cache_dir = args.cache_dir if args.cache_dir else str(DATA_DIR)
    train_dataset = ApneaDataset(
        str(DATA_DIR), train_records, cache_dir,
        args.segment_length, args.stride, 'train', augment=True
    )
    val_dataset = ApneaDataset(
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
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    model = EfficientApneaNet(
        d_model=args.d_model, n_blocks=args.n_blocks, dropout=args.dropout
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

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

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
            swa_scheduler.step()

        try:
            auc = roc_auc_score(val_targets, val_probs)
        except:
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
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    
    swa_val_loss, swa_val_acc, _, swa_val_targets, swa_val_probs, swa_precision, swa_recall, swa_f1 = validate(
        swa_model, val_loader, criterion, device, use_tta=args.use_tta, n_tta=args.n_tta
    )
    
    try:
        swa_auc = roc_auc_score(swa_val_targets, swa_val_probs)
    except:
        swa_auc = 0.0
    
    print(f"SWA Model: Acc={swa_val_acc:.2f}%, AUC={swa_auc:.4f}, F1={swa_f1:.3f}")
    
    # Save SWA model if better
    if swa_val_acc > best_val_acc or (swa_val_acc >= best_val_acc and swa_f1 > best_val_f1):
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'val_acc': swa_val_acc, 'val_auc': swa_auc, 'val_f1': swa_f1
        }, args.best_model_path.replace('.pth', '_swa.pth'))
        print(f"✓ SWA model saved (better than regular best)")
        best_val_acc = swa_val_acc
        best_val_f1 = swa_f1

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
    parser.add_argument('--segment-length', type=int, default=6000)  # 60s
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

    args, _ = parser.parse_known_args()
    
    if args.data_dir is None:
        raise SystemExit("ERROR: Dataset not found")
    
    print("="*80)
    print("MEMORY-EFFICIENT MODEL (Target: 90%+ Accuracy)")
    print("="*80)
    print(f"  Data:       {args.data_dir}")
    print(f"  Segment:    {args.segment_length} samples (60s), stride={args.stride}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Model:      d_model={args.d_model}, blocks={args.n_blocks}")
    print(f"  Optimizer:  AdamW (lr={args.lr}, wd={args.weight_decay})")
    print("="*80 + "\n")
    
    main(args)