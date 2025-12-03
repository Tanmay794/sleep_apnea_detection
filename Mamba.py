#!/usr/bin/env python3
"""
Mamba-based apnea detection.
Includes SSM shape fixes, NaN-safe normalization, parse_known_args for notebooks,
and Kaggle-friendly defaults.
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

# Optional: wfdb for reading PhysioNet records
try:
    import wfdb
except Exception:
    wfdb = None

from sklearn.metrics import roc_auc_score

# ----------------------------- Utilities ---------------------------------


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- Mamba Blocks ------------------------------


class MambaBlock(nn.Module):
    """Mamba block with a corrected selective SSM implementation."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters: produce per-inner-channel B/C for each state dimension
        # x_proj now outputs d_inner * d_state * 2 values per time-step so we can reshape to (B,L,d_inner,d_state*2)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner * d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # State space matrices (learnable logs)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # depthwise conv (causal)
        x_conv = x_in.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # ensure causal length L
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM using x_conv as input
        y = self.ssm(x_conv)

        # gating
        y = y * F.silu(res)

        out = self.out_proj(y)
        return out

    def ssm(self, x):
        """
        x: (B, L, d_inner)
        We'll produce:
          delta: (B, L, d_inner)
          x_dbl -> reshape to (B, L, d_inner, d_state*2)
          Bmat, Cmat -> (B, L, d_inner, d_state)
        """
        B, L, D = x.shape  # D == d_inner
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)

        x_dbl = self.x_proj(x)  # (B, L, d_inner * d_state * 2)
        # reshape to (B, L, d_inner, d_state*2)
        x_dbl = x_dbl.view(B, L, self.d_inner, self.d_state * 2)
        Bmat, Cmat = x_dbl.split([self.d_state, self.d_state], dim=-1)  # each (B,L,d_inner,d_state)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        y = self.selective_scan(x, delta, A, Bmat, Cmat, self.D)
        return y

    def selective_scan(self, u, delta, A, Bmat, Cmat, D):
        """
        Vectorized selective scan (much faster than loop)
        u: (B, L, d_inner)
        delta: (B, L, d_inner)
        A: (d_inner, d_state)
        Bmat: (B, L, d_inner, d_state)
        Cmat: (B, L, d_inner, d_state)
        D: (d_inner,)
        """
        B_batch, L, d_inner = u.shape
        d_state = A.shape[1]
        A = A.to(u.device)
        
        # Expand dimensions for broadcasting
        delta_expanded = delta.unsqueeze(-1)  # (B, L, d_inner, 1)
        
        # Discretization - all timesteps at once
        deltaA = torch.exp(delta_expanded * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB = delta_expanded * Bmat  # (B, L, d_inner, d_state)
        
        # Prepare input
        u_expanded = u.unsqueeze(-1)  # (B, L, d_inner, 1)
        
        # Associative scan approximation (parallel cumulative sum trick)
        # This is an approximation but much faster
        x_state = torch.zeros((B_batch, d_inner, d_state), device=u.device, dtype=u.dtype)
        
        # Chunk processing for memory efficiency
        chunk_size = 50  # Process 50 timesteps at a time
        ys = []
        
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            chunk_len = chunk_end - chunk_start
            
            for i in range(chunk_start, chunk_end):
                x_state = deltaA[:, i] * x_state + deltaB[:, i] * u[:, i].unsqueeze(-1)
                y_i = torch.sum(x_state * Cmat[:, i], dim=-1)
                ys.append(y_i)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + u * D.to(u.device)
        return y

class MambaModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_layers=3, d_state=8, d_conv=4, expand=2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, L, 1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


# --------------------------- Dataset & Caching ---------------------------


class ApneaECGDataset(Dataset):
    """Dataset that loads from cache if present, else reads raw records."""

    def __init__(self, data_dir: str, record_names: list = None, cache_path: str = None,
                 segment_length: int = 1000, stride: int = 1000, use_cache=True):
        super().__init__()
        self.segment_length = int(segment_length)
        self.stride = int(stride)
        self.cache_path = Path(cache_path) if cache_path else None

        if use_cache and self.cache_path and self.cache_path.exists():
            print(f"Loading cached dataset from {self.cache_path}")
            data = torch.load(self.cache_path)
            self.segments = data['segments']  # (N, L)
            self.labels = data['labels']
        else:
            assert wfdb is not None, "wfdb not available. Install wfdb or create cache first."
            assert record_names is not None, "record_names must be provided if not using cache"
            self.segments = []
            self.labels = []
            self.data_dir = Path(data_dir)
            for rec in record_names:
                self._load_record(rec)
            # convert to tensors
            if len(self.segments) == 0:
                raise RuntimeError("No segments loaded. Check records and segment parameters.")
            self.segments = torch.tensor(np.stack(self.segments, axis=0), dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            if self.cache_path:
                print(f"Saving cache to {self.cache_path} (this may take a moment)")
                torch.save({'segments': self.segments, 'labels': self.labels}, self.cache_path)

        # ensure shape (N, L, 1)
        if self.segments.ndim == 2:
            self.segments = self.segments.unsqueeze(-1)

        print(f"Dataset ready: {len(self.segments)} segments. Class dist: {Counter(self.labels.tolist())}")

    def _load_record(self, record_name: str):
        try:
            record = wfdb.rdrecord(str(self.data_dir / record_name))
            signal = record.p_signal[:, 0].astype(np.float32)
    
            # Remove NaNs in signal (if any) by interpolation or zero-fill
            if np.isnan(signal).any():
                nans = np.isnan(signal)
                not_nans = ~nans
                if not_nans.sum() > 0:
                    signal[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), signal[not_nans])
                else:
                    signal = np.zeros_like(signal)
    
            annotation = wfdb.rdann(str(self.data_dir / record_name), 'apn')
            
            # Create minute-by-minute labels (100 Hz, so 6000 samples = 1 minute)
            n_minutes = len(signal) // 6000
            minute_labels = np.zeros(n_minutes, dtype=int)
            
            # Mark apnea minutes based on annotations
            for i, symbol in enumerate(annotation.symbol):
                if symbol == 'A':  # Apnea event
                    sample = annotation.sample[i]
                    minute = sample // 6000
                    if minute < n_minutes:
                        minute_labels[minute] = 1
    
            n_samples = len(signal)
            for start in range(0, n_samples - self.segment_length + 1, self.stride):
                end = start + self.segment_length
                seg = signal[start:end].astype(np.float32)
    
                # NaN-safe mean/std
                seg_mean = np.nanmean(seg)
                seg_std = np.nanstd(seg)
                if np.isnan(seg_std) or seg_std < 1e-8:
                    seg = seg - seg_mean
                else:
                    seg = (seg - seg_mean) / (seg_std + 1e-8)
    
                # Use minute-level label
                minute = start // 6000
                if minute < len(minute_labels):
                    label = minute_labels[minute]
                    self.segments.append(seg)
                    self.labels.append(int(label))
                    
        except Exception as e:
            print(f"Error loading {record_name}: {e}")
            
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num, scaler=None, accum_steps=1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader, 1):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            output = model(data)
            loss = criterion(output, target) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if batch_idx % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

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
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs)


# ------------------------------ Main ------------------------------------


def create_cache(data_dir: str, record_names: list, out_path: str, segment_length=1000, stride=1000):
    """Precompute and save segments/labels to a torch file (NaN-safe)."""
    assert wfdb is not None, "wfdb is required to create cache. Install wfdb."
    data_dir = Path(data_dir)
    segments = []
    labels = []
    for rec in record_names:
        try:
            record = wfdb.rdrecord(str(data_dir / rec))
            signal = record.p_signal[:, 0].astype(np.float32)

            if np.isnan(signal).any():
                nans = np.isnan(signal)
                not_nans = ~nans
                if not_nans.sum() > 0:
                    signal[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), signal[not_nans])
                else:
                    signal = np.zeros_like(signal)

            annotation = wfdb.rdann(str(data_dir / rec), 'apn')
            
            # Create minute-by-minute labels
            n_minutes = len(signal) // 6000
            minute_labels = np.zeros(n_minutes, dtype=int)
            
            for i, symbol in enumerate(annotation.symbol):
                if symbol == 'A':
                    sample = annotation.sample[i]
                    minute = sample // 6000
                    if minute < n_minutes:
                        minute_labels[minute] = 1

            n_samples = len(signal)
            for start in range(0, n_samples - segment_length + 1, stride):
                end = start + segment_length
                seg = signal[start:end].astype(np.float32)
                mean = np.nanmean(seg)
                std = np.nanstd(seg)
                if np.isnan(std) or std < 1e-8:
                    seg = seg - mean
                else:
                    seg = (seg - mean) / (std + 1e-8)
                
                minute = start // 6000
                if minute < len(minute_labels):
                    label = minute_labels[minute]
                    segments.append(seg)
                    labels.append(int(label))
                    
        except Exception as e:
            print(f"Error caching {rec}: {e}")

    if len(segments) == 0:
        raise RuntimeError("No segments created while caching. Check dataset and params.")

    segments = torch.tensor(np.stack(segments, axis=0), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    torch.save({'segments': segments, 'labels': labels}, str(out_path))
    print(f"Cache saved to {out_path}. Total segments: {len(labels)}")
    print(f"Class distribution: {Counter(labels.tolist())}")

def main(args):
    set_seed(args.seed)

    DATA_DIR = Path(args.data_dir)
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Find records
    record_files = list(DATA_DIR.glob('*.hea'))
    all_records = [f.stem for f in record_files]
    valid_records = [rec for rec in all_records if (DATA_DIR / (rec + '.apn')).exists() and not rec.endswith('er')]
    if len(valid_records) == 0:
        raise RuntimeError(f"No valid records found in {DATA_DIR} (check .apn files)")

    print(f"Found {len(valid_records)} valid records")

    # cache path default handled outside
    cache_path = Path(args.cache_path) if args.cache_path else DATA_DIR / 'apnea_segments.pt'
    if args.create_cache:
        print("Creating cache (this may take a while)...")
        create_cache(DATA_DIR, valid_records, cache_path, segment_length=args.segment_length, stride=args.stride)
        print("Cache creation done. Re-run without --create-cache to train using cache.")
        return

    # split
    # split - shuffle first for better distribution
    import random
    valid_records_shuffled = valid_records.copy()
    random.Random(args.seed).shuffle(valid_records_shuffled)  # deterministic shuffle
    split_idx = int(len(valid_records_shuffled) * args.train_split)
    train_records = valid_records_shuffled[:split_idx]
    val_records = valid_records_shuffled[split_idx:]
    print(f"Train records: {len(train_records)}, Val records: {len(val_records)}")

    # datasets
    use_cache = cache_path.exists()
    if use_cache:
        print(f"Using cache at {cache_path}")
    train_dataset = ApneaECGDataset(str(DATA_DIR), record_names=train_records, cache_path=str(cache_path),
                                    segment_length=args.segment_length, stride=args.stride, use_cache=use_cache)
    val_dataset = ApneaECGDataset(str(DATA_DIR), record_names=val_records, cache_path=str(cache_path),
                                  segment_length=args.segment_length, stride=args.stride, use_cache=use_cache)

    # dataloaders
    # determine num_workers (already computed earlier)
    num_workers = min(max(0, (os.cpu_count() or 4) - 1), args.num_workers)
    print(f"Using dataloader num_workers={num_workers}, persistent_workers=False")
    
    # only apply prefetch_factor when num_workers>0, else None
    pf = args.prefetch_factor if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=pf
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=pf
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # model
    model = MambaModel(input_dim=1, d_model=args.d_model, n_layers=args.n_layers,
                       d_state=args.d_state, d_conv=args.d_conv, expand=args.expand).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # loss/optimizer
    class_weights = compute_class_weights(train_dataset.labels).to(device)
    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    # epochs
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        try:
            it = iter(train_loader)
            t_fetch_start = time.time()
            _ = next(it)
            t_fetch_end = time.time()
            fetch_time = t_fetch_end - t_fetch_start
        except Exception:
            fetch_time = float('nan')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            scaler=scaler, accum_steps=args.accum_steps)
        val_loss, val_acc, val_preds, val_targets, val_probs = validate(model, val_loader, criterion, device)

        try:
            auc = roc_auc_score(val_targets, val_probs)
        except Exception:
            auc = float('nan')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(auc)

        scheduler.step()

        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs}  time={epoch_time:.1f}s  fetch_time={fetch_time:.3f}s  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  val_auc={auc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc, 'history': history}
            torch.save(save_dict, args.best_model_path)
            print(f"Saved best model to {args.best_model_path} (val_acc={val_acc:.2f}%)")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    print("Training finished. Best val acc: {:.2f}%".format(best_val_acc))


if __name__ == '__main__':
    # notebook-safe parsing: ignore unknown args like -f from Jupyter
    kaggle_default = '/kaggle/input/apnea-data/apnea-ecg-database-1.0.0'
    colab_default = '/content/apnea-ecg/1.0.0'
    data_dir_default = kaggle_default if Path(kaggle_default).exists() else (colab_default if Path(colab_default).exists() else None)

    parser = argparse.ArgumentParser(description='Mamba apnea training (fixed)')
    parser.add_argument('--data-dir', type=str, default=data_dir_default,
                        help='Path to Apnea-ECG folder (contains .hea/.dat/.apn).')
    parser.add_argument('--create-cache', action='store_true', help='Create cache and exit')
    parser.add_argument('--cache-path', type=str, default=None, help='Cache path (defaults to /kaggle/working if Kaggle)')
    parser.add_argument('--segment-length', type=int, default=500)
    parser.add_argument('--stride', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--d-state', type=int, default=8)
    parser.add_argument('--d-conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--best-model-path', type=str, default='best_mamba_apnea.pth')
    parser.add_argument('--seed', type=int, default=42)

    # use parse_known_args so Jupyter -f won't break us
    args, _unknown = parser.parse_known_args()

    if args.data_dir is None:
        raise SystemExit(
            "\nERROR: --data-dir not provided and no default dataset found.\n"
            "Please provide the Apnea-ECG dataset path or place it at /kaggle/input/apnea-data/apnea-ecg-database-1.0.0"
        )

    # apply Kaggle defaults
    if args.cache_path is None:
        if str(args.data_dir).startswith('/kaggle/input'):
            args.cache_path = '/kaggle/working/apnea_segments.pt'
            if args.best_model_path == 'best_mamba_apnea.pth':
                args.best_model_path = '/kaggle/working/best_mamba_apnea.pth'
            # reduce default workers on Kaggle
            try:
                default_workers = parser.get_default('num_workers')
            except Exception:
                default_workers = 8
            if args.num_workers == default_workers:
                args.num_workers = 2
        else:
            args.cache_path = str(Path(args.data_dir) / 'apnea_segments.pt')
    else:
        if str(args.cache_path).startswith('/kaggle/input') and str(args.data_dir).startswith('/kaggle/input'):
            args.cache_path = '/kaggle/working/apnea_segments.pt'
        if str(args.data_dir).startswith('/kaggle/input') and str(args.best_model_path).startswith('/kaggle/input'):
            args.best_model_path = '/kaggle/working/best_mamba_apnea.pth'

    try:
        default_workers = parser.get_default('num_workers')
    except Exception:
        default_workers = 8
    if str(args.data_dir).startswith('/kaggle/input') and args.num_workers == default_workers:
        args.num_workers = 2

    print("\nResolved paths and settings:")
    print(f"  data_dir:      {args.data_dir}")
    print(f"  cache_path:    {args.cache_path}")
    print(f"  best_model:    {args.best_model_path}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  epochs:        {args.epochs}")
    print(f"  num_workers:   {args.num_workers}\n")

    main(args)
