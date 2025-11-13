#!/usr/bin/env python3
"""
Create synthetic DJ mixes from MUSDB18 tracks with beat-matched transitions.
This generates training data for the DJ mix analysis model.
"""
import musdb
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import json

class DJMixGenerator:
    """Generate synthetic DJ mixes with beat-matched crossfades."""

    def __init__(self, db, output_dir="synthetic_mixes"):
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def beat_match_tracks(self, track1, track2, crossfade_beats=32):
        """
        Beat-match two tracks and create a crossfade transition.

        Args:
            track1: First track audio and metadata
            track2: Second track audio and metadata
            crossfade_beats: Number of beats to crossfade over

        Returns:
            Mixed audio, transition metadata
        """
        # Get audio
        audio1 = track1['audio'].mean(axis=0)  # Stereo to mono
        audio2 = track2['audio'].mean(axis=0)
        sr = track1['sr']

        # Detect beats and tempo
        tempo1, beats1 = librosa.beat.beat_track(y=audio1, sr=sr)
        tempo2, beats2 = librosa.beat.beat_track(y=audio2, sr=sr)

        beat_times1 = librosa.frames_to_time(beats1, sr=sr)
        beat_times2 = librosa.frames_to_time(beats2, sr=sr)

        # Time-stretch track2 to match track1's tempo
        stretch_factor = float(tempo1) / float(tempo2)
        audio2_stretched = librosa.effects.time_stretch(audio2, rate=stretch_factor)

        # Calculate crossfade region
        # Start crossfade at the last N beats of track1
        if len(beat_times1) > crossfade_beats:
            crossfade_start = beat_times1[-crossfade_beats]
            crossfade_start_samples = int(crossfade_start * sr)
        else:
            # If track is too short, crossfade over half the track
            crossfade_start_samples = len(audio1) // 2

        # Trim track1 to crossfade point
        track1_body = audio1[:crossfade_start_samples]
        track1_tail = audio1[crossfade_start_samples:]

        # Trim track2 to match crossfade length and continue
        crossfade_len = len(track1_tail)
        track2_head = audio2_stretched[:crossfade_len]
        track2_body = audio2_stretched[crossfade_len:]

        # Create crossfade with equal power curve
        t = np.linspace(0, 1, crossfade_len)
        fade_out = np.cos(t * np.pi / 2)  # Smooth fade out
        fade_in = np.sin(t * np.pi / 2)   # Smooth fade in

        # Ensure arrays are same length
        min_len = min(len(track1_tail), len(track2_head), len(fade_out))
        crossfade = (track1_tail[:min_len] * fade_out[:min_len] +
                     track2_head[:min_len] * fade_in[:min_len])

        # Combine everything
        mixed_audio = np.concatenate([track1_body, crossfade, track2_body])

        # Create metadata
        transition_metadata = {
            'track1_duration': len(track1_body) / sr,
            'transition_start': len(track1_body) / sr,
            'transition_end': (len(track1_body) + len(crossfade)) / sr,
            'track2_start': (len(track1_body) + len(crossfade)) / sr,
            'total_duration': len(mixed_audio) / sr,
            'tempo1': float(tempo1),
            'tempo2': float(tempo2),
            'stretch_factor': stretch_factor,
            'crossfade_beats': crossfade_beats,
        }

        return mixed_audio, transition_metadata

    def create_mix(self, num_tracks=3, crossfade_beats=32, mix_name="mix_001"):
        """
        Create a complete DJ mix from multiple tracks.

        Args:
            num_tracks: Number of tracks to mix together
            crossfade_beats: Beats to crossfade between tracks
            mix_name: Name for this mix
        """
        print(f"\nCreating mix: {mix_name}")
        print(f"  - Using {num_tracks} tracks")
        print(f"  - Crossfade: {crossfade_beats} beats")

        # Select random tracks
        track_indices = np.random.choice(len(self.db), size=num_tracks, replace=False)

        tracks_data = []
        for idx in track_indices:
            track = self.db[int(idx)]
            audio = track.audio.T
            sr = track.rate

            tracks_data.append({
                'name': track.name,
                'audio': audio,
                'sr': sr,
                'index': int(idx)
            })
            print(f"  + Track {len(tracks_data)}: {track.name}")

        # Mix tracks sequentially
        current_audio = tracks_data[0]['audio'].mean(axis=0)
        current_sr = tracks_data[0]['sr']

        mix_metadata = {
            'mix_name': mix_name,
            'num_tracks': num_tracks,
            'tracks': [{'name': tracks_data[0]['name'], 'index': tracks_data[0]['index']}],
            'transitions': []
        }

        for i in range(1, num_tracks):
            print(f"  → Mixing track {i} into track {i+1}...")

            # Prepare current and next track
            track1 = {'audio': np.stack([current_audio, current_audio]), 'sr': current_sr}
            track2 = tracks_data[i]

            # Beat-match and crossfade
            mixed, transition = self.beat_match_tracks(track1, track2, crossfade_beats)

            # Update for next iteration
            current_audio = mixed
            current_sr = track2['sr']

            # Store metadata
            mix_metadata['tracks'].append({
                'name': tracks_data[i]['name'],
                'index': tracks_data[i]['index']
            })
            mix_metadata['transitions'].append(transition)

        # Save mix
        mix_path = self.output_dir / f"{mix_name}.wav"
        sf.write(mix_path, current_audio, current_sr)

        # Save metadata
        metadata_path = self.output_dir / f"{mix_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(mix_metadata, f, indent=2)

        print(f"\n✓ Mix saved: {mix_path}")
        print(f"  - Duration: {len(current_audio)/current_sr:.1f} seconds")
        print(f"  - Metadata: {metadata_path}")

        return str(mix_path), mix_metadata

def main():
    """Create example synthetic DJ mixes."""
    print("="*60)
    print("DJ Mix Generator")
    print("="*60)

    # Load database
    print("\nLoading MUSDB18 database...")
    import os
    # Set the path to the downloaded dataset
    musdb_path = os.path.expanduser("~/MUSDB18/MUSDB18-7")
    db = musdb.DB(root=musdb_path, subsets=['train'])
    print(f"  ✓ Loaded {len(db)} tracks")

    # Create generator
    generator = DJMixGenerator(db)

    # Create a few example mixes
    print("\nGenerating synthetic DJ mixes...")

    # Mix 1: Short mix with 3 tracks
    generator.create_mix(num_tracks=3, crossfade_beats=16, mix_name="short_mix_001")

    # Mix 2: Longer crossfades
    generator.create_mix(num_tracks=3, crossfade_beats=32, mix_name="long_fade_001")

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print(f"\nMixes saved to: {generator.output_dir}")
    print("\nNext steps:")
    print("  1. Listen to the mixes to verify quality")
    print("  2. Examine the metadata JSON files")
    print("  3. Generate more mixes for training")
    print("  4. Build the track boundary detection model")

if __name__ == "__main__":
    main()
