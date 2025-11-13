#!/usr/bin/env python3
"""
Complete DJ set analysis pipeline:
1. Download audio from YouTube
2. Detect track boundaries
3. Identify tracks
4. Generate tracklist
"""
import os
import sys
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from datetime import timedelta
import subprocess

class DJSetAnalyzer:
    """Analyze DJ sets and extract tracklists."""

    def __init__(self, output_dir="dj_set_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audio_path = None
        self.audio = None
        self.sr = None

    def download_from_youtube(self, youtube_url, output_name="dj_set"):
        """Download audio from YouTube."""
        print("="*60)
        print("Step 1: Downloading Audio from YouTube")
        print("="*60)
        print(f"\nURL: {youtube_url}")

        output_path = self.output_dir / f"{output_name}.wav"

        # Check if already downloaded
        if output_path.exists():
            print(f"\n✓ Audio already exists: {output_path}")
            self.audio_path = str(output_path)
            return str(output_path)

        print("\nDownloading... (this may take a few minutes)")

        # Download using yt-dlp
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',
            '--audio-quality', '0',  # Best quality
            '-o', str(self.output_dir / f"{output_name}.%(ext)s"),
            youtube_url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"\n✓ Download complete: {output_path}")
            self.audio_path = str(output_path)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Download failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise

    def load_audio(self, audio_path=None):
        """Load audio file."""
        if audio_path:
            self.audio_path = audio_path

        print("\n" + "="*60)
        print("Step 2: Loading Audio")
        print("="*60)
        print(f"\nLoading: {self.audio_path}")

        # Load audio (convert to mono)
        self.audio, self.sr = librosa.load(self.audio_path, sr=22050, mono=True)

        duration = len(self.audio) / self.sr
        print(f"  ✓ Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"  ✓ Sample rate: {self.sr} Hz")
        print(f"  ✓ Samples: {len(self.audio):,}")

        return self.audio, self.sr

    def detect_transitions(self, sensitivity='medium'):
        """
        Detect track transitions using multiple methods.

        Strategy:
        1. Spectral flux (energy changes)
        2. Beat tracking discontinuities
        3. Tempo changes
        4. Harmonic/percussive separation changes
        """
        print("\n" + "="*60)
        print("Step 3: Detecting Track Boundaries")
        print("="*60)

        print("\n  Analyzing spectral features...")

        # Method 1: Spectral flux (detect major energy changes)
        hop_length = 512
        spectral_flux = librosa.onset.onset_strength(
            y=self.audio,
            sr=self.sr,
            hop_length=hop_length
        )

        # Smooth it to find larger-scale changes
        from scipy.ndimage import gaussian_filter1d
        smoothed_flux = gaussian_filter1d(spectral_flux, sigma=50)

        # Find peaks in smoothed flux (potential transitions)
        from scipy.signal import find_peaks

        # Adjust sensitivity
        if sensitivity == 'high':
            height = np.percentile(smoothed_flux, 75)
            distance = self.sr // hop_length * 15  # At least 15 seconds apart
        elif sensitivity == 'low':
            height = np.percentile(smoothed_flux, 90)
            distance = self.sr // hop_length * 45  # At least 45 seconds apart
        else:  # medium
            height = np.percentile(smoothed_flux, 85)
            distance = self.sr // hop_length * 30  # At least 30 seconds apart

        peaks, properties = find_peaks(
            smoothed_flux,
            height=height,
            distance=distance,
            prominence=np.std(smoothed_flux) * 0.5
        )

        # Convert to time
        boundary_times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=hop_length)

        print(f"  ✓ Found {len(boundary_times)} potential track boundaries")

        # Method 2: Tempo analysis
        print("\n  Analyzing tempo changes...")

        # Analyze tempo in windows
        window_size = 30  # 30 seconds
        hop_size = 15  # 15 seconds

        tempos = []
        tempo_times = []

        for start in range(0, int(len(self.audio) / self.sr) - window_size, hop_size):
            start_sample = int(start * self.sr)
            end_sample = int((start + window_size) * self.sr)
            window_audio = self.audio[start_sample:end_sample]

            tempo, _ = librosa.beat.beat_track(y=window_audio, sr=self.sr)
            tempos.append(float(tempo))
            tempo_times.append(start + window_size / 2)

        tempos = np.array(tempos)
        tempo_times = np.array(tempo_times)

        # Find tempo changes (new tracks often have tempo changes)
        tempo_changes = np.abs(np.diff(tempos))
        tempo_change_threshold = np.percentile(tempo_changes, 75)
        tempo_transition_indices = np.where(tempo_changes > tempo_change_threshold)[0]
        tempo_transitions = tempo_times[tempo_transition_indices]

        print(f"  ✓ Found {len(tempo_transitions)} tempo changes")

        # Combine both methods
        all_boundaries = np.sort(np.concatenate([boundary_times, tempo_transitions]))

        # Remove duplicates (boundaries within 10 seconds of each other)
        final_boundaries = [0.0]  # Start of mix
        for boundary in all_boundaries:
            if boundary - final_boundaries[-1] > 10.0:  # At least 10 seconds apart
                final_boundaries.append(boundary)

        # Add end of mix
        duration = len(self.audio) / self.sr
        if duration - final_boundaries[-1] > 20.0:  # If last segment is > 20 seconds
            final_boundaries.append(duration)

        self.boundaries = np.array(final_boundaries)

        print(f"\n  ✓ Final: {len(self.boundaries) - 1} tracks detected")
        print("\n  Track boundaries (timestamps):")
        for i, boundary in enumerate(self.boundaries):
            time_str = str(timedelta(seconds=int(boundary)))
            print(f"    {i}: {time_str} ({boundary:.1f}s)")

        return self.boundaries

    def extract_segments(self):
        """Extract audio segments for each detected track."""
        print("\n" + "="*60)
        print("Step 4: Extracting Track Segments")
        print("="*60)

        segments_dir = self.output_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        segments = []

        for i in range(len(self.boundaries) - 1):
            start_time = self.boundaries[i]
            end_time = self.boundaries[i + 1]

            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)

            segment_audio = self.audio[start_sample:end_sample]

            # Save segment
            segment_path = segments_dir / f"track_{i:02d}.wav"
            sf.write(segment_path, segment_audio, self.sr)

            duration = (end_time - start_time)
            print(f"  ✓ Track {i+1}: {duration:.1f}s saved to {segment_path.name}")

            segments.append({
                'track_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'file': str(segment_path)
            })

        self.segments = segments
        return segments

    def identify_tracks(self):
        """
        Attempt to identify tracks using multiple methods.
        """
        print("\n" + "="*60)
        print("Step 5: Identifying Tracks")
        print("="*60)
        print("\nNote: Track identification requires additional services.")
        print("We'll create fingerprints and attempt basic identification.\n")

        identified_tracks = []

        for segment in self.segments:
            track_num = segment['track_number']
            print(f"\n  Analyzing Track {track_num}...")

            # Load segment
            audio, sr = librosa.load(segment['file'], sr=22050, mono=True)

            # Extract features for identification
            # Method 1: Tempo
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            tempo_val = float(tempo)

            # Method 2: Key (using chroma)
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            estimated_key = np.argmax(key_profile)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_name = keys[estimated_key]

            # Method 3: Spectral centroid (brightness)
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

            print(f"    - Estimated tempo: {tempo_val:.1f} BPM")
            print(f"    - Estimated key: {key_name}")
            print(f"    - Spectral centroid: {centroid:.1f} Hz")
            print(f"    - Duration: {segment['duration']:.1f}s")

            identified_tracks.append({
                'track_number': track_num,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment['duration'],
                'tempo': tempo_val,
                'key': key_name,
                'spectral_centroid': float(centroid),
                'identified_as': 'Unknown',  # Would need external API
                'confidence': 0.0
            })

        self.identified_tracks = identified_tracks
        return identified_tracks

    def generate_tracklist(self):
        """Generate and save final tracklist."""
        print("\n" + "="*60)
        print("Step 6: Generating Tracklist")
        print("="*60)

        tracklist = {
            'source': 'YouTube DJ Set',
            'audio_file': self.audio_path,
            'total_duration': float(len(self.audio) / self.sr),
            'num_tracks': len(self.identified_tracks),
            'tracks': self.identified_tracks
        }

        # Save JSON
        tracklist_path = self.output_dir / "tracklist.json"
        with open(tracklist_path, 'w') as f:
            json.dump(tracklist, f, indent=2)

        print(f"\n✓ Tracklist saved: {tracklist_path}")

        # Save human-readable format
        txt_path = self.output_dir / "tracklist.txt"
        with open(txt_path, 'w') as f:
            f.write("DJ SET TRACKLIST\n")
            f.write("="*60 + "\n\n")

            for track in self.identified_tracks:
                start = str(timedelta(seconds=int(track['start_time'])))
                duration_str = f"{track['duration']:.1f}s"
                tempo_str = f"{track['tempo']:.0f} BPM"
                key_str = track['key']

                f.write(f"Track {track['track_number']}:\n")
                f.write(f"  Time: {start}\n")
                f.write(f"  Duration: {duration_str}\n")
                f.write(f"  Tempo: {tempo_str}\n")
                f.write(f"  Key: {key_str}\n")
                f.write(f"  Title: {track['identified_as']}\n")
                f.write("\n")

        print(f"✓ Text tracklist saved: {txt_path}")

        # Display tracklist
        print("\n" + "="*60)
        print("FINAL TRACKLIST")
        print("="*60 + "\n")

        for track in self.identified_tracks:
            start = str(timedelta(seconds=int(track['start_time'])))
            print(f"{track['track_number']:2d}. [{start}] "
                  f"{track['tempo']:.0f} BPM | Key: {track['key']} | "
                  f"{track['duration']:.1f}s")

        return tracklist

def main():
    """Main analysis pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_dj_set.py <youtube_url>")
        print("\nExample:")
        print("  python analyze_dj_set.py 'https://www.youtube.com/watch?v=93ZGx5wjRdo'")
        sys.exit(1)

    youtube_url = sys.argv[1]

    print("="*60)
    print("DJ SET ANALYZER")
    print("="*60)
    print(f"\nAnalyzing: {youtube_url}\n")

    # Create analyzer
    analyzer = DJSetAnalyzer()

    try:
        # Step 1: Download
        audio_path = analyzer.download_from_youtube(youtube_url)

        # Step 2: Load audio
        analyzer.load_audio(audio_path)

        # Step 3: Detect boundaries
        boundaries = analyzer.detect_transitions(sensitivity='medium')

        # Step 4: Extract segments
        segments = analyzer.extract_segments()

        # Step 5: Identify tracks
        identified = analyzer.identify_tracks()

        # Step 6: Generate tracklist
        tracklist = analyzer.generate_tracklist()

        print("\n" + "="*60)
        print("✓ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {analyzer.output_dir}")
        print(f"  - Audio: {audio_path}")
        print(f"  - Segments: {analyzer.output_dir}/segments/")
        print(f"  - Tracklist: {analyzer.output_dir}/tracklist.txt")
        print(f"  - Metadata: {analyzer.output_dir}/tracklist.json")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
