#!/usr/bin/env python3
"""
Download audio from YouTube DJ sets for training.
"""
import json
import subprocess
from pathlib import Path


class AudioDownloader:
    """Download training audio from YouTube."""

    def __init__(self, data_dir="training_data", audio_dir="training_audio"):
        self.data_dir = Path(data_dir)
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(exist_ok=True)

    def download_audio(self, youtube_url, output_name):
        """Download audio from YouTube."""
        output_path = self.audio_dir / f"{output_name}.wav"

        # Check if already downloaded
        if output_path.exists():
            print(f"  ✓ Already exists: {output_path.name}")
            return str(output_path)

        print(f"  Downloading... (this may take a few minutes)")

        # Download using yt-dlp
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',
            '--audio-quality', '0',  # Best quality
            '-o', str(self.audio_dir / f"{output_name}.%(ext)s"),
            youtube_url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✓ Downloaded: {output_path.name}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Download failed: {e}")
            print("  STDERR:", e.stderr)
            return None

    def download_all_sets(self):
        """Download all training sets."""
        print("="*60)
        print("DOWNLOADING TRAINING AUDIO")
        print("="*60)

        sets = [
            ('guy_j_cleaned.json', 'guy_j_set'),
            ('verdikt_cleaned.json', 'verdikt_set'),
            ('bonobo_set1_cleaned.json', 'bonobo_set1'),
            ('bonobo_set2_cleaned.json', 'bonobo_set2')
        ]

        downloaded = []

        for json_file, output_name in sets:
            json_path = self.data_dir / json_file

            # Load metadata
            with open(json_path, 'r') as f:
                data = json.load(f)

            print(f"\n{data['dj_name']} - {data['set_name']}")
            print(f"  URL: {data['youtube_url']}")

            # Download
            audio_path = self.download_audio(data['youtube_url'], output_name)

            if audio_path:
                # Update JSON with audio path
                data['audio_file'] = audio_path
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)

                downloaded.append(output_name)

        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"\n  Downloaded: {len(downloaded)}/{len(sets)} sets")
        print(f"  Audio Directory: {self.audio_dir}")

        if len(downloaded) < len(sets):
            print("\n  ⚠️  Some downloads failed. Check yt-dlp installation.")


def main():
    """Main execution."""
    downloader = AudioDownloader()
    downloader.download_all_sets()

    print("\n" + "="*60)
    print("✓ AUDIO DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Extract transition audio segments")
    print("  2. Create Qwen3-Omni training dataset")
    print("  3. Start fine-tuning on Modal")


if __name__ == "__main__":
    main()
