# 🎓 Training Best Practices: Qwen3-Omni + Modal + Audio Data

Comprehensive guide based on research of Qwen, Modal, and audio ML best practices.

## 🎯 Qwen Fine-Tuning Best Practices

### Recommended Hyperparameters

Based on successful Qwen2/Qwen-Audio fine-tuning:

```python
# LoRA Configuration
LORA_CONFIG = {
    "r": 16,                    # Rank (try: 8, 16, 24, 32)
    "lora_alpha": 32,           # Usually 2*r
    "lora_dropout": 0.1,        # Dropout rate
    "target_modules": [         # Which layers to adapt
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training Arguments
TRAINING_ARGS = {
    "learning_rate": 1e-5,                   # Conservative for audio
    "per_device_train_batch_size": 2,       # Adjust based on GPU memory
    "gradient_accumulation_steps": 16,      # Effective batch = 2 * 16 = 32
    "num_train_epochs": 3,
    "warmup_ratio": 0.03,                   # 3% warmup
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "fp16": False,                          # Use bf16 instead
    "bf16": True,                           # Better for audio
    "gradient_checkpointing": True,         # Save memory
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 2
}
```

### Learning Rate Guidelines

| Model Size | Initial LR | With LoRA |
|------------|-----------|-----------|
| 1.5B-3B | 1e-5 to 5e-5 | 1e-4 to 5e-4 |
| 7B | 5e-6 to 1e-5 | 5e-5 to 1e-4 |
| 14B+ | 1e-6 to 5e-6 | 1e-5 to 5e-5 |

**Rule of thumb**: Start conservative, increase if loss plateaus early

### LoRA Rank Selection

```python
# Task complexity determines rank
TASK_TO_RANK = {
    "simple_classification": 8,
    "moderate_reasoning": 16,      # ← Our use case
    "complex_generation": 32,
    "very_complex": 64
}
```

**Our choice**: `r=16` (transition detection is moderate complexity)

---

## ☁️ Modal Best Practices

### GPU Selection

```python
# Cost vs Performance
GPU_CONFIGS = {
    "development": gpu.T4(),              # $0.60/hr, 16GB
    "small_model": gpu.A10G(),            # $1.10/hr, 24GB
    "production": gpu.A100(size="40GB"),  # $3.50/hr, 40GB
    "large_scale": gpu.A100(size="80GB", count=2)  # $14/hr, 160GB
}

# For Qwen3-Omni audio fine-tuning (7B model):
# Recommended: A100-40GB (or 2x A10G)
```

### Memory Optimization

```python
@app.function(
    gpu=gpu.A100(size="40GB"),
    memory=16384,  # 16GB RAM (in addition to GPU memory)
    timeout=3600,  # 1 hour
    volumes={VOLUME_PATH: volume}
)
def train():
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Use mixed precision
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    # Accumulate gradients
    for i, batch in enumerate(dataloader):
        with autocast():
            loss = model(**batch).loss / accumulation_steps
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

### Batch Processing

```python
# Process data in parallel
@app.function(
    cpu=4,
    memory=8192,
    volumes={VOLUME_PATH: volume}
)
def preprocess_segment(segment_data):
    """Process one audio segment."""
    return processed_data

# Use .map() for parallel execution
results = list(preprocess_segment.map(segment_list))
```

### Volume Best Practices

```python
# Create persisted volume
volume = modal.Volume.from_name("dj-mix-data", create_if_missing=True)

# IMPORTANT: Commit after writes!
@app.function(volumes={"/data": volume})
def save_data(data, path):
    with open(path, 'w') as f:
        f.write(data)

    volume.commit()  # ← CRITICAL: Persist changes

    return path

# Read without commit (faster)
@app.function(volumes={"/data": volume})
def load_data(path):
    with open(path, 'r') as f:
        return f.read()
    # No commit needed for reads
```

### Long-Running Jobs

```python
# Use --detach for background execution
# modal run --detach modal_finetune_qwen.py

@app.function(
    gpu=gpu.A100(),
    timeout=14400,  # 4 hours max
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def train_model():
    """Long-running training job."""
    import wandb
    wandb.init(project="dj-mix-ai")

    # Training loop with checkpointing
    for epoch in range(num_epochs):
        train_epoch()

        # Save checkpoint
        checkpoint_path = f"/data/checkpoint-{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        volume.commit()
```

---

## 🎵 Audio Data Quality Best Practices

### Sample Rate Standardization

```python
# CRITICAL: All audio must have same sample rate
TARGET_SR = 22050  # Or 16000 for speech-focused models

def validate_sample_rate(audio_path):
    """Ensure consistent sample rate."""
    import librosa

    # Load without resampling to check
    y, sr = librosa.load(audio_path, sr=None)

    if sr != TARGET_SR:
        # Resample
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return y, sr
```

### Audio Normalization Pipeline

```python
def normalize_audio_segment(audio, sr):
    """Complete normalization pipeline."""
    import librosa
    import numpy as np

    # 1. Convert to mono
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    # 2. Normalize volume (peak normalization)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95  # Leave 5% headroom

    # 3. Remove DC offset
    audio = audio - np.mean(audio)

    # 4. Optional: High-pass filter (remove rumble)
    audio = librosa.effects.preemphasis(audio, coef=0.97)

    return audio
```

### Quality Checks

```python
def validate_audio_quality(audio, sr, min_duration=10, max_duration=120):
    """Comprehensive quality validation."""
    import numpy as np

    checks = {
        "duration": None,
        "rms_energy": None,
        "clipping": None,
        "silence": None,
        "valid": True,
        "errors": []
    }

    # Duration check
    duration = len(audio) / sr
    checks["duration"] = duration
    if duration < min_duration:
        checks["valid"] = False
        checks["errors"].append(f"Too short: {duration:.1f}s")
    if duration > max_duration:
        checks["valid"] = False
        checks["errors"].append(f"Too long: {duration:.1f}s")

    # RMS energy (detect silence)
    rms = np.sqrt(np.mean(audio**2))
    checks["rms_energy"] = float(rms)
    if rms < 0.001:
        checks["valid"] = False
        checks["errors"].append("Mostly silent")

    # Clipping detection
    clipped_samples = np.sum(np.abs(audio) >= 0.99)
    clipping_rate = clipped_samples / len(audio)
    checks["clipping"] = float(clipping_rate)
    if clipping_rate > 0.01:  # >1% clipped
        checks["valid"] = False
        checks["errors"].append(f"Clipping: {clipping_rate*100:.1f}%")

    # Silence detection (consecutive zeros)
    zero_runs = np.diff(np.where(np.abs(audio) < 1e-6)[0])
    max_silence = np.max(zero_runs) if len(zero_runs) > 0 else 0
    silence_duration = max_silence / sr
    checks["silence"] = float(silence_duration)
    if silence_duration > 1.0:  # >1s silence
        checks["valid"] = False
        checks["errors"].append(f"Long silence: {silence_duration:.1f}s")

    return checks
```

### Data Augmentation

```python
def augment_audio(audio, sr, augmentation_type='light'):
    """Apply audio augmentation for training robustness."""
    import librosa
    import numpy as np

    if augmentation_type == 'none':
        return audio

    # Light augmentation (recommended for DJ mixes)
    if augmentation_type == 'light':
        # Slight tempo variation (±3%)
        tempo_factor = np.random.uniform(0.97, 1.03)
        audio = librosa.effects.time_stretch(audio, rate=tempo_factor)

        # Slight pitch shift (±0.5 semitones)
        pitch_shift = np.random.uniform(-0.5, 0.5)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)

        # Add subtle noise (SNR 40-50dB)
        noise_level = np.random.uniform(0.001, 0.003)
        noise = np.random.normal(0, noise_level, len(audio))
        audio = audio + noise

    # Heavy augmentation (if you have limited data)
    elif augmentation_type == 'heavy':
        # Tempo: ±6%
        tempo_factor = np.random.uniform(0.94, 1.06)
        audio = librosa.effects.time_stretch(audio, rate=tempo_factor)

        # Pitch: ±1 semitone
        pitch_shift = np.random.uniform(-1.0, 1.0)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)

        # EQ simulation (bass/treble cut)
        if np.random.random() < 0.5:
            from scipy import signal
            # Random EQ frequency
            cutoff = np.random.uniform(200, 800)
            sos = signal.butter(2, cutoff/(sr/2), btype='highpass', output='sos')
            audio = signal.sosfilt(sos, audio)

    return audio
```

---

## 📊 Training Data Format

### JSONL Structure

```jsonl
{"messages": [{"role": "user", "content": "Analyze this DJ mix transition."}, {"role": "assistant", "content": "Transition from Track A to Track B at 3:24."}], "audio_file": "/data/segment_001.wav", "metadata": {"from_track": "Track A", "to_track": "Track B", "midpoint": 204.5}}
{"messages": [{"role": "user", "content": "Analyze this DJ mix transition."}, {"role": "assistant", "content": "Transition from Track B to Track C at 7:12."}], "audio_file": "/data/segment_002.wav", "metadata": {"from_track": "Track B", "to_track": "Track C", "midpoint": 432.0}}
```

### Train/Val/Test Split

```python
def create_splits(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create stratified splits by DJ set."""
    assert train_ratio + val_ratio + test_ratio == 1.0

    # Group by DJ set
    dj_sets = {}
    for item in data:
        dj_name = item['metadata']['dj_name']
        if dj_name not in dj_sets:
            dj_sets[dj_name] = []
        dj_sets[dj_name].append(item)

    # Stratify: each split gets data from each DJ
    train, val, test = [], [], []

    for dj_name, items in dj_sets.items():
        n = len(items)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        train.extend(items[:train_n])
        val.extend(items[train_n:train_n+val_n])
        test.extend(items[train_n+val_n:])

    return train, val, test
```

---

## ⚡ Performance Optimization

### Dataloader Optimization

```python
from torch.utils.data import DataLoader

# Optimal settings
dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    prefetch_factor=2,      # Prefetch 2 batches
    persistent_workers=True # Keep workers alive
)
```

### Mixed Precision Training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    bf16=True,              # BF16 > FP16 for audio
    tf32=True,              # Enable TF32 on Ampere GPUs
    gradient_checkpointing=True,
    optim="adamw_torch_fused",  # Faster AdamW
)
```

### Gradient Accumulation

```python
# If GPU memory is limited
# Effective batch = per_device_batch * accumulation_steps * num_gpus

training_args = TrainingArguments(
    per_device_train_batch_size=1,     # Small batch
    gradient_accumulation_steps=32,    # Accumulate 32 steps
    # Effective batch size = 1 * 32 = 32
)
```

---

## 🎯 Our Specific Configuration

### For DJ Mix Transition Detection

```python
# Model: Qwen3-Omni (7B parameters)
# Task: Transition detection (moderate complexity)
# Data: 126 transitions, augmented to 1000+
# GPU: A100-40GB on Modal

CONFIG = {
    "model_name": "Qwen/Qwen-Audio",

    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },

    "training": {
        "learning_rate": 5e-5,  # Higher for LoRA
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 50
    },

    "audio": {
        "sample_rate": 22050,
        "max_duration": 60,  # seconds
        "augmentation": "light"
    },

    "modal": {
        "gpu": "A100-40GB",
        "timeout": 14400,  # 4 hours
        "memory": 16384
    }
}
```

---

## 📈 Monitoring & Debugging

### W&B Integration

```python
import wandb

wandb.init(
    project="dj-mix-ai",
    config=CONFIG,
    tags=["qwen3-omni", "transition-detection", "lora"]
)

# Log custom metrics
wandb.log({
    "transition_accuracy": accuracy,
    "mean_time_error": mean_error,
    "gpu_memory": gpu_memory_mb
})
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| OOM | "CUDA out of memory" | Reduce batch size, enable gradient checkpointing |
| Slow training | <5 steps/sec | Increase num_workers, use pin_memory |
| NaN loss | Loss becomes NaN | Lower learning rate, check for inf values in data |
| No learning | Loss plateaus | Increase learning rate, check data quality |
| Overfitting | Train loss << val loss | Add dropout, reduce epochs, augment data |

---

## ✅ Pre-Flight Checklist

Before starting training:

- [ ] All audio resampled to 22050Hz
- [ ] Audio normalized (peak < 0.99)
- [ ] No silent segments
- [ ] No clipping (>1%)
- [ ] Duration range: 30-120 seconds
- [ ] JSONL format validated
- [ ] Train/val/test split created (80/10/10)
- [ ] Modal volume created and mounted
- [ ] GPU config tested
- [ ] Batch size fits in memory
- [ ] Learning rate validated
- [ ] Logging configured (W&B or TensorBoard)

---

## 🔗 References

- [Qwen Official Training Guide](https://github.com/QwenLM/Qwen)
- [Modal GPU Best Practices](https://modal.com/docs/guide/gpu)
- [HuggingFace Audio Course](https://huggingface.co/learn/audio-course)
- [PEFT LoRA Documentation](https://huggingface.co/docs/peft/conceptual_guides/lora)

---

**Last Updated**: 2025-01-13
**Status**: Ready for training
