"""
Modal-based training data preparation for Qwen3-Omni.

Downloads YouTube DJ sets, extracts transition segments, and prepares
training data entirely in Modal's cloud storage.
"""
import modal
import json
from pathlib import Path

# Create Modal app
app = modal.App("dj-mix-training-prep")

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "yt-dlp",
        "librosa",
        "soundfile",
        "numpy",
        "scipy"
    )
)

# Create Modal volume for persistent storage
volume = modal.Volume.from_name("dj-mix-data", create_if_missing=True)

VOLUME_PATH = "/data"


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,  # 1 hour
    memory=8192,  # 8GB RAM
)
def download_youtube_audio(youtube_url: str, output_name: str):
    """Download audio from YouTube in the cloud."""
    import subprocess

    output_path = f"{VOLUME_PATH}/audio/{output_name}.wav"
    Path(f"{VOLUME_PATH}/audio").mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if Path(output_path).exists():
        print(f"✓ Already downloaded: {output_name}")
        return output_path

    print(f"Downloading {output_name}...")

    # Download using yt-dlp
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'wav',
        '--audio-quality', '0',  # Best quality
        '-o', f'{VOLUME_PATH}/audio/{output_name}.%(ext)s',
        youtube_url
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        volume.commit()
        print(f"✓ Downloaded: {output_name}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"STDERR: {e.stderr}")
        return None


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    memory=8192,
)
def extract_transition_segments(
    audio_path: str,
    tracklist_data: dict,
    transition_duration: int = 30
):
    """Extract transition audio segments from full DJ set."""
    import librosa
    import soundfile as sf
    import numpy as np

    print(f"\nExtracting transitions from: {tracklist_data['dj_name']}")

    # Create output directory
    segments_dir = Path(f"{VOLUME_PATH}/segments/{tracklist_data['dj_name'].replace(' ', '_')}")
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    print(f"  Loading audio...")
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(audio) / sr
    print(f"  ✓ Loaded {duration/60:.1f} minutes")

    # Extract each transition
    segments = []
    for i, transition in enumerate(tracklist_data['transitions']):
        start_time = transition['transition_start_seconds']
        end_time = transition['transition_end_seconds']

        # Ensure we don't go beyond audio length
        end_time = min(end_time, duration)

        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = audio[start_sample:end_sample]

        # Save segment
        segment_filename = f"transition_{i:03d}.wav"
        segment_path = segments_dir / segment_filename
        sf.write(segment_path, segment_audio, sr)

        segments.append({
            'transition_index': i,
            'from_track': transition['from_title'],
            'to_track': transition['to_title'],
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'midpoint': transition['transition_midpoint'],
            'segment_file': str(segment_path)
        })

        print(f"  ✓ Transition {i+1}/{len(tracklist_data['transitions'])}: "
              f"{segment_filename} ({end_time - start_time:.1f}s)")

    # Save segment metadata
    metadata_path = segments_dir / "segments_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'dj_name': tracklist_data['dj_name'],
            'set_name': tracklist_data['set_name'],
            'num_segments': len(segments),
            'segments': segments
        }, f, indent=2)

    volume.commit()

    print(f"\n  ✓ Extracted {len(segments)} transition segments")
    return segments


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    memory=8192,
)
def create_qwen_training_data(segments_metadata: dict):
    """
    Create Qwen3-Omni training data in ChatML format.

    Format for transition detection:
    User: <audio_segment>
    Assistant: This is a transition from [Track A] to [Track B].
               The transition occurs at [timestamp].
    """
    print(f"\nCreating Qwen3-Omni training data...")

    training_examples = []

    for segment in segments_metadata['segments']:
        # Create training example
        example = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this DJ mix audio segment and identify the track transition. Audio file: {segment['segment_file']}"
                },
                {
                    "role": "assistant",
                    "content": f"This audio segment contains a DJ transition from \"{segment['from_track']}\" to \"{segment['to_track']}\". "
                               f"The transition midpoint occurs at {segment['midpoint']:.1f} seconds ({segment['midpoint']//60:.0f}:{segment['midpoint']%60:02.0f}). "
                               f"This is a {segment['duration']:.1f}-second crossfade/blend between the tracks."
                }
            ],
            "audio_file": segment['segment_file'],
            "metadata": {
                "transition_index": segment['transition_index'],
                "from_track": segment['from_track'],
                "to_track": segment['to_track'],
                "midpoint": segment['midpoint']
            }
        }

        training_examples.append(example)

    # Save training data
    output_dir = Path(f"{VOLUME_PATH}/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    dj_name_slug = segments_metadata['dj_name'].replace(' ', '_').lower()
    training_file = output_dir / f"{dj_name_slug}_training.jsonl"

    with open(training_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    volume.commit()

    print(f"  ✓ Created {len(training_examples)} training examples")
    print(f"  ✓ Saved to: {training_file}")

    return str(training_file)


@app.local_entrypoint()
def main():
    """Prepare all training data in the cloud."""
    import json
    from pathlib import Path

    print("="*60)
    print("DJ MIX TRAINING DATA PREPARATION (Modal)")
    print("="*60)

    # Load tracklist metadata from local files
    tracklists = []
    local_data_dir = Path("training_data")

    for json_file in local_data_dir.glob("*_cleaned.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            tracklists.append(data)

    print(f"\nFound {len(tracklists)} DJ sets to process")

    # Process each set
    for tracklist in tracklists:
        print(f"\n{'='*60}")
        print(f"Processing: {tracklist['dj_name']} - {tracklist['set_name']}")
        print(f"{'='*60}")

        # Step 1: Download audio in cloud
        audio_path = download_youtube_audio.remote(
            tracklist['youtube_url'],
            tracklist['dj_name'].replace(' ', '_').lower()
        )

        if not audio_path:
            print(f"  ⚠️  Skipping {tracklist['dj_name']} - download failed")
            continue

        # Step 2: Extract transition segments
        segments = extract_transition_segments.remote(
            audio_path,
            tracklist
        )

        # Step 3: Create training data
        training_file = create_qwen_training_data.remote({
            'dj_name': tracklist['dj_name'],
            'set_name': tracklist['set_name'],
            'segments': segments
        })

        print(f"\n  ✓ Complete: {tracklist['dj_name']}")

    print("\n" + "="*60)
    print("✓ ALL TRAINING DATA PREPARED!")
    print("="*60)
    print(f"\nData stored in Modal volume: dj-mix-data")
    print(f"Location: {VOLUME_PATH}/")
    print("\nNext steps:")
    print("  1. Review training data format")
    print("  2. Start Qwen3-Omni fine-tuning")
    print("  3. Evaluate on test sets")
