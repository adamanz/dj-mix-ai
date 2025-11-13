# Data Cleaning Best Practices for Audio LLM Training

Based on research of HuggingFace Transformers, Qwen-Audio, and music AI projects.

## 🎯 Overview

This guide documents best practices for preparing audio data to fine-tune Qwen3-Omni for DJ mix transition detection and song identification.

## 📊 Data Format Requirements

### 1. Audio Files

**Specifications:**
- **Format**: WAV (recommended) or MP3
- **Sample Rate**: 16kHz or 22kHz (consistent across dataset)
- **Channels**: Mono (stereo will be converted)
- **Duration**: 30-60 seconds per segment (Qwen3-Omni supports up to 40 minutes)
- **Bit Depth**: 16-bit or 24-bit

**Why these specs?**
- Lower sample rates (16kHz) reduce computational cost while preserving speech/music content
- Mono audio simplifies processing and reduces file size
- Segment duration balances context (transition window) with processing efficiency

### 2. Training Data Format (JSONL)

Each line is a complete training example:

```jsonl
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze this DJ mix audio segment and identify the track transition. Audio file: /data/segments/transition_001.wav"
    },
    {
      "role": "assistant",
      "content": "This audio segment contains a DJ transition from \"Frankey & Sandrino - Acamar\" to \"Guy J - Another Feeling\". The transition midpoint occurs at 1503.0 seconds (25:03). This is a 60.0-second crossfade/blend between the tracks."
    }
  ],
  "audio_file": "/data/segments/guy_j/transition_001.wav",
  "metadata": {
    "transition_index": 1,
    "from_track": "Frankey & Sandrino - Acamar",
    "to_track": "Guy J - Another Feeling",
    "midpoint": 1503.0,
    "dj_name": "Guy J",
    "set_name": "Lost & Found"
  }
}
```

**Key Fields:**
- `messages`: ChatML format conversation (user prompt + assistant response)
- `audio_file`: Absolute path to audio segment
- `metadata`: Additional context for evaluation and analysis

## 🔧 Preprocessing Pipeline

### Stage 1: Audio Normalization

```python
import librosa
import soundfile as sf

def normalize_audio(input_path, output_path, target_sr=22050):
    """Normalize audio: resample, convert to mono, normalize volume."""
    # Load audio
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # Normalize volume to [-1.0, 1.0] range
    audio = librosa.util.normalize(audio)

    # Save
    sf.write(output_path, audio, target_sr)

    return len(audio) / target_sr  # duration in seconds
```

**Steps:**
1. **Resample** to consistent sample rate (e.g., 22kHz)
2. **Convert to mono** (stereo → mono)
3. **Normalize volume** to standard range
4. **Trim silence** (optional, for cleaner segments)

### Stage 2: Segment Extraction

```python
def extract_transition_segment(
    audio_path,
    start_time,
    end_time,
    output_path,
    sr=22050
):
    """Extract specific time range from audio file."""
    # Load only the required segment
    duration = end_time - start_time
    audio, sr = librosa.load(
        audio_path,
        sr=sr,
        offset=start_time,
        duration=duration,
        mono=True
    )

    # Save segment
    sf.write(output_path, audio, sr)

    return {
        'duration': duration,
        'sample_rate': sr,
        'samples': len(audio)
    }
```

**Transition Zone Calculation:**
```python
def calculate_transition_zone(
    track_start_time,
    next_track_start_time,
    transition_window=30  # seconds
):
    """Calculate the audio segment containing a transition."""
    # Last 30s of current track + first 30s of next track
    transition_start = max(0, next_track_start_time - transition_window)
    transition_end = next_track_start_time + transition_window

    return {
        'start': transition_start,
        'end': transition_end,
        'midpoint': next_track_start_time,
        'duration': transition_end - transition_start
    }
```

### Stage 3: Quality Checks

```python
def validate_audio_segment(audio_path, min_duration=10, max_duration=120):
    """Validate audio segment meets quality requirements."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr

        # Check duration
        if duration < min_duration:
            return False, f"Too short: {duration:.1f}s"
        if duration > max_duration:
            return False, f"Too long: {duration:.1f}s"

        # Check for silence (RMS energy)
        rms = librosa.feature.rms(y=audio)[0]
        avg_rms = np.mean(rms)
        if avg_rms < 0.001:
            return False, "Mostly silent"

        # Check for clipping
        if np.max(np.abs(audio)) >= 0.99:
            return False, "Audio clipping detected"

        return True, "OK"

    except Exception as e:
        return False, f"Error: {str(e)}"
```

## 📝 ChatML Format Guidelines

### User Prompts (Consistent Templates)

**For Transition Detection:**
```
"Analyze this DJ mix audio segment and identify the track transition."
```

**For Song Identification:**
```
"Identify the track playing in this audio segment from a DJ mix."
```

### Assistant Responses (Structured Format)

**For Transition Detection:**
```
"This audio segment contains a DJ transition from \"{TRACK_A}\" to \"{TRACK_B}\". The transition midpoint occurs at {TIMESTAMP} seconds ({MM:SS}). This is a {DURATION}-second crossfade/blend between the tracks."
```

**For Song Identification:**
```
"The track playing is \"{ARTIST} - {TITLE}\" from the album \"{ALBUM}\". Identified with {CONFIDENCE}% confidence."
```

## 🎵 Audio-Specific Considerations

### 1. Handle Variable Lengths

DJ transitions vary in duration:
- **Short crossfades**: 15-30 seconds
- **Long blends**: 60-120 seconds
- **Cut transitions**: <5 seconds

**Solution**: Use consistent window (e.g., 60s) centered on transition midpoint.

### 2. Address Tempo Variations

DJs often speed up/slow down tracks (±6-12% typical).

**Consideration**: Don't normalize tempo during preprocessing - this is part of the learning task!

### 3. EQ and Filter Effects

DJs apply:
- High-pass/low-pass filters
- EQ adjustments (bass/mid/treble)
- Reverb, delay, echo

**Strategy**: Include diverse examples with various effects in training data.

## 📊 Dataset Statistics to Track

Monitor these metrics during preprocessing:

```python
stats = {
    'total_segments': 126,
    'total_duration': 42300,  # seconds
    'avg_segment_duration': 60.0,
    'sample_rate': 22050,
    'audio_format': 'wav',
    'num_dj_sets': 4,
    'num_unique_tracks': 130,
    'num_transitions': 126
}
```

## 🔍 Validation Checklist

Before training:

- [ ] All audio files resample to same rate
- [ ] All segments are mono
- [ ] No audio clipping (max amplitude < 0.99)
- [ ] No silent segments (RMS > threshold)
- [ ] Duration range reasonable (30-120s)
- [ ] JSONL format validated (all fields present)
- [ ] Audio file paths are absolute and exist
- [ ] Metadata matches audio content
- [ ] Train/validation/test split defined
- [ ] File sizes reasonable (<50MB per segment)

## 🚀 Optimization Tips

### Memory Efficiency

```python
# Load audio in chunks for large files
def process_large_audio_file(path, chunk_duration=60):
    """Process audio file in chunks to save memory."""
    total_duration = librosa.get_duration(filename=path)

    for start in range(0, int(total_duration), chunk_duration):
        audio, sr = librosa.load(
            path,
            offset=start,
            duration=chunk_duration,
            sr=22050,
            mono=True
        )

        yield audio, sr, start
```

### Parallel Processing

```python
from multiprocessing import Pool

def process_audio_files(file_list, num_workers=4):
    """Process multiple audio files in parallel."""
    with Pool(num_workers) as pool:
        results = pool.map(normalize_audio, file_list)
    return results
```

## 📚 Reference Implementations

### HuggingFace Transformers Pattern

```python
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("Qwen/Qwen-Audio")

def preprocess_function(examples):
    """HuggingFace-style preprocessing."""
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=22050,
        max_length=int(22050 * 60),  # 60 seconds
        truncation=True,
        return_attention_mask=True,
    )
    return inputs
```

### Qwen-Audio Specific

```python
from transformers import Qwen2AudioProcessor

processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen-Audio")

def prepare_dataset(example):
    """Qwen-Audio preprocessing."""
    audio = example["audio"]

    # Process audio + text together
    example_processed = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["text"],
    )

    # Calculate input length for batching
    example_processed["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example_processed
```

## 🎯 Expected Results

After proper data cleaning:

- **Consistent format** across all samples
- **No preprocessing errors** during training
- **Balanced dataset** (similar durations, diverse examples)
- **Clean audio quality** (no clipping, silence, or artifacts)
- **Accurate labels** matching audio content

## 🔗 Additional Resources

- [HuggingFace Audio Preprocessing](https://huggingface.co/docs/transformers/preprocessing)
- [Qwen-Audio Documentation](https://github.com/QwenLM/Qwen-Audio)
- [Librosa Audio Processing](https://librosa.org/doc/latest/index.html)
- [Audio Transformers Course](https://huggingface.co/learn/audio-course)

---

**Next Steps**:
1. Run preprocessing pipeline on all training data
2. Validate output using checklist above
3. Create train/val/test splits (80/10/10)
4. Begin fine-tuning with clean data
