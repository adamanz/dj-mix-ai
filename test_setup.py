#!/usr/bin/env python3
"""
Test script to verify MUSDB18 download and beat tracking capabilities.
"""
import musdb
import librosa
import numpy as np
from pathlib import Path

# Note: madmom has compatibility issues with Python 3.11
# Using librosa for beat tracking instead

def test_musdb_download():
    """Download and verify MUSDB18 dataset."""
    print("="*60)
    print("Testing MUSDB18 Download")
    print("="*60)

    # Download 7-second clips
    print("\nDownloading MUSDB18 clips (this may take a few minutes)...")
    db = musdb.DB(download=True, subsets=['train'])

    print(f"✓ Downloaded {len(db)} tracks")
    print(f"  - Database location: {db.root}")

    return db

def test_beat_tracking(db):
    """Test beat tracking on first track."""
    print("\n" + "="*60)
    print("Testing Beat Tracking")
    print("="*60)

    # Get first track
    track = db[0]
    audio = track.audio.T  # shape: (2, samples) stereo -> (samples,)
    sr = track.rate

    print(f"\nTrack: {track.name}")
    print(f"  - Duration: {len(audio[0])/sr:.2f} seconds")
    print(f"  - Sample rate: {sr} Hz")
    print(f"  - Channels: {audio.shape[0]}")

    # Convert stereo to mono
    audio_mono = audio.mean(axis=0)

    # Beat tracking with librosa
    print("\n  Running beat detection...")
    tempo, beat_frames = librosa.beat.beat_track(y=audio_mono, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"  ✓ Found {len(beat_times)} beats")
    # Convert tempo to scalar if it's an array
    tempo_val = float(tempo) if hasattr(tempo, '__iter__') else tempo
    print(f"  ✓ Estimated tempo: {tempo_val:.1f} BPM")
    beats = beat_times  # For consistency with return value

    # Onset detection with librosa
    print("\n  Running onset detection...")
    onset_frames = librosa.onset.onset_detect(
        y=audio_mono,
        sr=sr,
        units='time'
    )
    print(f"  ✓ Found {len(onset_frames)} onsets")

    return beats, onset_frames

def print_summary():
    """Print setup summary."""
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\n✓ All dependencies installed")
    print("✓ MUSDB18 dataset downloaded")
    print("✓ Beat tracking tested successfully")
    print("\nNext steps:")
    print("  1. Run 'python analyze_track.py' to analyze a full track")
    print("  2. Run 'python create_synthetic_mix.py' to create training data")
    print("\nProject directory: ~/dj-mix-analysis")

if __name__ == "__main__":
    try:
        # Test download
        db = test_musdb_download()

        # Test beat tracking
        beats, onsets = test_beat_tracking(db)

        # Print summary
        print_summary()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have internet connection (for download)")
        print("  2. Check disk space (~500MB needed for clips)")
        print("  3. Try running again - downloads will resume")
        raise
