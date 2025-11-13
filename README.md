# DJ Mix AI - Transition Detection & Song Identification

Fine-tuning [Qwen3-Omni](https://huggingface.co/Qwen/Qwen-Audio) to detect track transitions and identify songs in DJ mixes, even with time-stretching, EQ, and effects applied.

## 🎯 Goals

1. **Transition Detection**: Identify where tracks blend/crossfade in DJ mixes
2. **Song Identification**: Recognize tracks despite DJ transformations (beatmatching, EQ, effects)
3. **SOTA Performance**: Achieve state-of-the-art results for DJ-specific audio analysis

## 🚀 Approach

- **Model**: Fine-tune Qwen3-Omni (multimodal LLM trained on 20M hours of audio)
- **Training Data**: 130+ labeled DJ mix tracks with precise timestamps
- **Infrastructure**: Modal for serverless GPU training
- **Timeline**: 2-4 weeks for full implementation

## 📊 Training Data

| DJ Set | Tracks | Transitions | Duration |
|--------|--------|-------------|----------|
| Guy J - Lost & Found | 40 | 39 | ~4 hours |
| Verdikt - Progressive House | 45 | 44 | ~5 hours |
| Bonobo - Live Set | 13 | 12 | ~1 hour |
| Bonobo - DJ Mix | 32 | 31 | ~2 hours |
| **Total** | **130** | **126** | **~12 hours** |

## 🏗️ Architecture

### Phase 1: Data Preparation (Current)
1. ✅ Clean tracklists from YouTube DJ sets
2. ✅ Calculate transition zones (30s crossfade windows)
3. ⏳ Download audio in Modal cloud storage
4. ⏳ Extract transition segments
5. ⏳ Create Qwen3-Omni training format (ChatML)

### Phase 2: Fine-tuning
1. Fine-tune for transition detection
2. Fine-tune for song identification
3. Evaluate on held-out test sets

### Phase 3: Deployment
1. Create inference API
2. Test on full DJ mixes
3. Build web interface

## 🛠️ Setup

### Prerequisites
- Python 3.11+
- Modal account (for GPU training)
- yt-dlp (for YouTube downloads)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dj-mix-ai.git
cd dj-mix-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Modal
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

### Quick Start

```bash
# 1. Clean tracklists
python prepare_training_data.py

# 2. Prepare training data on Modal (downloads + extracts segments)
modal run modal_prepare_training.py

# 3. Fine-tune Qwen3-Omni (coming soon)
# modal run modal_finetune_qwen.py
```

## 📁 Project Structure

```
dj-mix-ai/
├── prepare_training_data.py      # Clean tracklists, calculate transitions
├── modal_prepare_training.py     # Cloud-based data prep (download + extract)
├── modal_finetune_qwen.py        # Fine-tuning script (coming soon)
├── training_data/                # Cleaned tracklists (JSON)
│   ├── guy_j_cleaned.json
│   ├── verdikt_cleaned.json
│   ├── bonobo_set1_cleaned.json
│   └── bonobo_set2_cleaned.json
└── README.md
```

## 🔬 Technical Details

### Transition Detection Method

For each track boundary, we create a transition zone:
- **Zone**: Last 30s of Track A + First 30s of Track B
- **Total Duration**: 60 seconds of audio
- **Label**: Start/end tracks, exact transition timestamp

### Training Data Format

```jsonl
{
  "messages": [
    {
      "role": "user",
      "content": "Analyze this DJ mix audio segment and identify the track transition."
    },
    {
      "role": "assistant",
      "content": "This is a transition from \"Track A\" to \"Track B\". The midpoint occurs at 3:24 (204 seconds)."
    }
  ],
  "audio_file": "/path/to/transition_segment.wav",
  "metadata": {
    "from_track": "Track A",
    "to_track": "Track B",
    "midpoint": 204.5
  }
}
```

### Why Qwen3-Omni?

- **Trained on 20M hours** of audio data
- **Multimodal understanding** (audio + text)
- **Up to 40 minutes** of audio input
- **Custom AuT encoder** for audio features
- **SOTA performance** on audio reasoning tasks

## 📈 Expected Outcomes

1. **Transition Detection**: ±2 second accuracy for transition midpoints
2. **Song Identification**: >85% accuracy on DJ-transformed tracks
3. **Robustness**: Handles tempo changes (±20%), EQ, filters, reverb

## 🤝 Contributing

Contributions welcome! Areas of focus:
- Additional training data (more DJ sets with tracklists)
- Improved preprocessing techniques
- Alternative model architectures
- Evaluation metrics

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{djmixai2025,
  title={DJ Mix AI: Fine-tuning Multimodal LLMs for Transition Detection},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/dj-mix-ai}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- [Qwen-Audio Paper](https://arxiv.org/abs/2311.07919)
- [Modal Docs](https://modal.com/docs)
- [Training Data Sources](./DATASETS.md)

## 📧 Contact

Questions? Open an issue or reach out on Twitter [@yourusername](https://twitter.com/yourusername)

---

**Status**: 🏗️ In Development | **Phase**: Data Preparation | **Progress**: 35% Complete
