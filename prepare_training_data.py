#!/usr/bin/env python3
"""
Prepare training data for Qwen3-Omni fine-tuning.

Takes DJ set tracklists and converts them to training format:
1. Clean timestamp formats
2. Calculate transition zones (last 30s of track A + first 30s of track B)
3. Structure in ChatML conversation format for Qwen3-Omni
"""
import json
import re
from datetime import timedelta
from pathlib import Path


class TrainingDataPreparator:
    """Prepare DJ set tracklists for Qwen3-Omni training."""

    def __init__(self, output_dir="training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def parse_timestamp(self, timestamp_str):
        """Convert various timestamp formats to seconds."""
        timestamp_str = timestamp_str.strip()

        # Format: HH:MM:SS or MM:SS or SS
        parts = timestamp_str.split(':')

        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 1:  # SS
            return int(parts[0])
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")

    def clean_guy_j_tracklist(self):
        """Clean Guy J tracklist."""
        print("\n" + "="*60)
        print("Cleaning Guy J Tracklist")
        print("="*60)

        tracklist = """
00    0:00:00    Guy J - Forja 2022 Intro ID
02    0:04:41    Cornucopia - TithoreaID
03    0:11:14    Cornucopia - Other Side of the World [unreleased]
04    0:18:13    Frankey & Sandrino - Acamar (Guy J Remix)
05    0:25:06    Guy J - Another Feeling [unreleased]
06    0:30:53    Guy J - Eternal Now ID
07    0:36:15    Frankey & Sandrino - Epsilon ID
08    0:41:59    Guy J - Departure 2022
09    0:47:28    Guy J - Lamur (Robert Babicz Remix)
10    0:53:41    Guy J - Moog and the City
11    0:59:02    Max Cooper feat. Kathrin deBoer - Aleph 2 (Guy J Remix)
12    1:05:31    Guy J - Sunny Tales
13    1:11:22    Guy J - Night Drive ID
14    1:16:26    Julian Wassermann - Hades
15    1:21:58    Guy J - Voyage
16    1:28:05    Marino Canal - Meraviglia
17    1:33:23    Guy J - Palms ID
18    1:38:12    Hernan Cattaneo & Soundexile - Into the Dusk
19    1:43:59    Guy J - Lamur (Be Svendsen Remix)
20    1:50:03    8kays - Indigo
21    1:55:50    Barry Jamieson - Seneca Falls
22    2:01:15    &ME - White Coats ID
23    2:06:40    Guy J & Davi - Make You Better
24    2:12:25    Stylo - Departure ID
25    2:17:59    Guy J - I know It
26    2:23:35    Tim Engelhardt - Trust
27    2:28:51    Guy J - Seemless
28    2:34:42    Victor Ruiz - Simile
29    2:41:10    Tale Of Us - Ricordi
30    2:47:12    Guy J - Orion 2022 ID
31    2:52:39    Nick Warren & Nicolas Rada - Nostalgic
32    2:59:09    Guy J - Aurora
33    3:06:35    Worakls - Salzburg (Guy J Remix)
34    3:12:32    Guy J - Her (Henrik Schwarz Remix)
35    3:18:13    Guy J - Shiny Light Purple
36    3:24:21    Ordonez - Bagatelle
37    3:28:53    Tom Day - Who We Want To Be
38    3:34:25    Guy J - Lamur (Henry Saiz Remix) ID
39    3:40:40    Guy J - Nymphaea
40    3:47:05    Guy J - I Tell You ID (edited) (full version on next podcast)
"""

        tracks = []
        lines = tracklist.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Parse: track_num    timestamp    title
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 3:
                track_num = parts[0]
                timestamp = parts[1]
                title = parts[2]

                try:
                    seconds = self.parse_timestamp(timestamp)
                    tracks.append({
                        'track_number': int(track_num) if track_num.isdigit() else track_num,
                        'timestamp_seconds': seconds,
                        'timestamp_str': timestamp,
                        'title': title.strip()
                    })
                except Exception as e:
                    print(f"  ⚠️  Skipping line: {line[:50]}... Error: {e}")

        print(f"\n  ✓ Parsed {len(tracks)} tracks")

        # Save cleaned tracklist
        output_file = self.output_dir / "guy_j_cleaned.json"
        with open(output_file, 'w') as f:
            json.dump({
                'youtube_url': 'https://www.youtube.com/watch?v=jJZKoJv33Hs',
                'dj_name': 'Guy J',
                'set_name': 'Lost & Found',
                'num_tracks': len(tracks),
                'tracks': tracks
            }, f, indent=2)

        print(f"  ✓ Saved to: {output_file}")
        return tracks

    def clean_verdikt_tracklist(self):
        """Clean Verdikt tracklist."""
        print("\n" + "="*60)
        print("Cleaning Verdikt Tracklist")
        print("="*60)

        tracklist = """
01    00:01    SYML - The Bird (Sasha Remix) NETTWERK
02    08:00    Craig Pruess & Ananda - Devi Prayer (Juan Sapia Private Bootleg)
03    15:00    Kamilo Sanclemente - Odin HOPE
04    23:00    JakoJako & Mauro Ferreira - Ashes in The Rain LUMP
05    30:00    Ramses - Inside (Marc DePulse Remix) STIL VOR TALENT
06    37:00    Matt Lange - Vast ANJUNADEEP
07    44:00    Frankey & Sandrino - Pantha INNERVISIONS
08    50:00    Verðikt - Timeless MOTEK
09    56:00    Rodrigo Deem - Hera (10 Years Mix)
10    1:03:00    Mees Salomé - Light Into Dark (Einmusik Remix) MEES SALOMÉ
11    1:10:00    Jan Blomqvist - Maybe Not (Anyma Remix) ARMADA
12    1:16:00    André Sobota - Limen REBELLION DER TRAUMER
13    1:23:00    Goom Gum - Kaya (Khen Remix) MOTEK
14    1:31:00    Budakid & Innellea - Prometeo STILL VOR TALENT
15    1:38:00    Nico Stojan - Sorrow feat. APARDE (Nick Devon Remix) MULTINOTES
16    1:45:00    Lost Desert - Saha (Adrian Roman Remix) MOTEK
17    1:51:00    Lost Desert & JakoJako - La Lumiere ENCANTA
18    1:59:00    Cassian - Stronger RÜFÜS
19    2:06:00    Roy Rosenfeld & Verðikt - Eris ASTRAL BAZAAR
20    2:14:00    Adam Port & Matteo Milleri - The Wizard KEINEMUSIK
21    2:20:00    Nick Devon - My Everything STEYOYOKE
22    2:27:00    Nick Devon - Holy feat. APARDE STEYOYOKE
23    2:33:00    Sascha Braemer - Whisper SUPDUB
24    2:38:00    Parallels - Falling feat. Salma Halabi (Froidz Remix) HEIMLICH MUSIK
25    2:45:00    Stylo - Gypsy CROSSTOWN REBELS
26    2:52:00    WhoMadeWho - Silence & Secrets (Innellea Remix)
27    2:58:00    Olivier Giacomotto - Soy SAPIENS
28    3:05:00    Atish - Mirare MANJUMASI
29    3:10:00    Yotto & Verðikt - Out Of Reach ANJUNADEEP
30    3:17:00    Ziger - Taro Cosmic Awakenings RITTER BUTZKE STUDIO
31    3:24:00    Sebastien Leger - Chameleon ARMADA
32    3:30:00    Nihil Young - Alhanna MOTEK
33    3:36:00    François Dubois - Aera SHANTI MOSCOW
34    3:43:00    Marco Resmann & Dahu - Maya EEYORE
35    3:50:00    Monkey Safari - Boulogne Billancourt (Bunte Bummler Remix) HOMMAGE
36    3:57:00    Tim Engelhardt - Goodbye STIL VOR TALENT
37    4:03:00    Hollt - Takar DIYNAMIC
38    4:10:00    Artbat - Tabu DIYNAMIC
39    4:16:00    Sebastien Leger - Libellule ARMADA
40    4:23:00    JakoJako - Phoenix LUMP
41    4:28:00    Einmusik - Still (JakoJako Remix) KATERMUKKE
42    4:36:00    Chicola - Obsession RADIKON
43    4:43:00    Massano - Don't Wake Me Up AFTERLIFE
44    4:48:00    Verðikt - Essence MOTEK
45    4:53:00    Khen - Beit Lid KHEN MUSIC
"""

        tracks = []
        lines = tracklist.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Parse: track_num    timestamp    title
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 3:
                track_num = parts[0]
                timestamp = parts[1]
                title = ' '.join(parts[2:])  # Join remaining parts

                try:
                    seconds = self.parse_timestamp(timestamp)
                    tracks.append({
                        'track_number': int(track_num) if track_num.isdigit() else track_num,
                        'timestamp_seconds': seconds,
                        'timestamp_str': timestamp,
                        'title': title.strip()
                    })
                except Exception as e:
                    print(f"  ⚠️  Skipping line: {line[:50]}... Error: {e}")

        print(f"\n  ✓ Parsed {len(tracks)} tracks")

        # Save cleaned tracklist
        output_file = self.output_dir / "verdikt_cleaned.json"
        with open(output_file, 'w') as f:
            json.dump({
                'youtube_url': 'https://www.youtube.com/watch?v=qmrn2QJwwaI',
                'dj_name': 'Verdikt',
                'set_name': 'Progressive House Mix',
                'num_tracks': len(tracks),
                'tracks': tracks
            }, f, indent=2)

        print(f"  ✓ Saved to: {output_file}")
        return tracks

    def clean_bonobo_set1_tracklist(self):
        """Clean Bonobo Set 1 tracklist."""
        print("\n" + "="*60)
        print("Cleaning Bonobo Set 1 Tracklist")
        print("="*60)

        tracklist = """
00:01 Polyghost (full extended version)
01:00 Flicker (Fabric dub)
07:48 Linked (Strings version)
13:16 Ten Tigers
19:30 Animals
23:45 Otomo
29:14 Eyesdown (Machinedrum Remix)
34:40 Recurring
40:00 All in Forms
44:49 Jets
50:00 Ketto (Dub)
56:23 Pick Up
61:00 Sapphire
"""

        tracks = []
        lines = tracklist.strip().split('\n')

        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue

            # Parse: timestamp title
            match = re.match(r'(\d+:\d+)\s+(.+)', line.strip())
            if match:
                timestamp = match.group(1)
                title = match.group(2)

                try:
                    seconds = self.parse_timestamp(timestamp)
                    tracks.append({
                        'track_number': i,
                        'timestamp_seconds': seconds,
                        'timestamp_str': timestamp,
                        'title': title.strip()
                    })
                except Exception as e:
                    print(f"  ⚠️  Skipping line: {line[:50]}... Error: {e}")

        print(f"\n  ✓ Parsed {len(tracks)} tracks")

        # Save cleaned tracklist
        output_file = self.output_dir / "bonobo_set1_cleaned.json"
        with open(output_file, 'w') as f:
            json.dump({
                'youtube_url': 'https://www.youtube.com/watch?v=VM2dxRnhq7g',
                'dj_name': 'Bonobo',
                'set_name': 'Live Set',
                'num_tracks': len(tracks),
                'tracks': tracks
            }, f, indent=2)

        print(f"  ✓ Saved to: {output_file}")
        return tracks

    def clean_bonobo_set2_tracklist(self):
        """Clean Bonobo Set 2 tracklist."""
        print("\n" + "="*60)
        print("Cleaning Bonobo Set 2 Tracklist")
        print("="*60)

        tracklist = """
00:01 Bonobo - Linked
04:00 2000 and One - Wan Poku Moro
08:00 Mark Knight - Second Story
11:30 Justin Martin - The Feels
16:00 Purple Disco Machine - Body Funk
20:00 Frankey & Sandrino - Acamar
24:30 Bonobo - Bambro Koyo Ganda
29:00 Dusky - Stick By This
33:00 Kiasmos - Looped
37:00 Bonobo - Otomo
42:00 Ben Böhmer - Breathing
46:30 Lane 8 - Fingerprint
51:00 Yotto - Hyperfall
55:30 Bonobo - Sapphire
60:00 Bonobo - Migration
64:30 Jon Hopkins - Emerald Rush
69:00 Bonobo - Kerala
73:30 Bonobo - Cirrus
78:00 Bonobo - Jets
82:30 Bonobo - Ten Tigers
87:00 Kiasmos - Burnt
91:30 Bonobo - Eyesdown
96:00 Bonobo - Break Apart
100:30 Bonobo - The Keeper
105:00 Bonobo - First Fires
109:30 Bonobo - Towers
114:00 Bonobo - Figures
118:30 Bonobo - Surface
123:00 Bonobo - Nothing Owed
127:30 Bonobo - Transits
132:00 Bonobo - Polyghost
136:30 Bonobo - Flicker
"""

        tracks = []
        lines = tracklist.strip().split('\n')

        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue

            # Parse: timestamp title
            match = re.match(r'(\d+:\d+)\s+(.+)', line.strip())
            if match:
                timestamp = match.group(1)
                title = match.group(2)

                try:
                    seconds = self.parse_timestamp(timestamp)
                    tracks.append({
                        'track_number': i,
                        'timestamp_seconds': seconds,
                        'timestamp_str': timestamp,
                        'title': title.strip()
                    })
                except Exception as e:
                    print(f"  ⚠️  Skipping line: {line[:50]}... Error: {e}")

        print(f"\n  ✓ Parsed {len(tracks)} tracks")

        # Save cleaned tracklist
        output_file = self.output_dir / "bonobo_set2_cleaned.json"
        with open(output_file, 'w') as f:
            json.dump({
                'youtube_url': 'https://www.youtube.com/watch?v=0F6KyA4g71g',
                'dj_name': 'Bonobo',
                'set_name': 'DJ Mix',
                'num_tracks': len(tracks),
                'tracks': tracks
            }, f, indent=2)

        print(f"  ✓ Saved to: {output_file}")
        return tracks

    def calculate_transition_zones(self, tracks, transition_duration=30):
        """
        Calculate transition zones between tracks.

        Transition zone = last N seconds of track A + first N seconds of track B
        """
        transitions = []

        for i in range(len(tracks) - 1):
            current_track = tracks[i]
            next_track = tracks[i + 1]

            # Calculate track duration
            track_duration = next_track['timestamp_seconds'] - current_track['timestamp_seconds']

            # Transition zone
            transition_start = current_track['timestamp_seconds'] + max(0, track_duration - transition_duration)
            transition_end = next_track['timestamp_seconds'] + transition_duration

            transitions.append({
                'from_track': current_track['track_number'],
                'to_track': next_track['track_number'],
                'from_title': current_track['title'],
                'to_title': next_track['title'],
                'transition_start_seconds': transition_start,
                'transition_end_seconds': transition_end,
                'transition_midpoint': next_track['timestamp_seconds'],
                'transition_duration': transition_duration * 2
            })

        return transitions

    def prepare_all_tracklists(self):
        """Clean all tracklists and calculate transitions."""
        print("="*60)
        print("PREPARING ALL TRAINING DATA")
        print("="*60)

        # Clean all tracklists
        guy_j_tracks = self.clean_guy_j_tracklist()
        verdikt_tracks = self.clean_verdikt_tracklist()
        bonobo1_tracks = self.clean_bonobo_set1_tracklist()
        bonobo2_tracks = self.clean_bonobo_set2_tracklist()

        # Calculate transitions for each set
        print("\n" + "="*60)
        print("Calculating Transition Zones")
        print("="*60)

        all_data = [
            ('guy_j_cleaned.json', guy_j_tracks),
            ('verdikt_cleaned.json', verdikt_tracks),
            ('bonobo_set1_cleaned.json', bonobo1_tracks),
            ('bonobo_set2_cleaned.json', bonobo2_tracks)
        ]

        for filename, tracks in all_data:
            transitions = self.calculate_transition_zones(tracks, transition_duration=30)

            # Load existing data
            with open(self.output_dir / filename, 'r') as f:
                data = json.load(f)

            # Add transitions
            data['transitions'] = transitions
            data['num_transitions'] = len(transitions)

            # Save updated data
            with open(self.output_dir / filename, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"\n  ✓ {filename}: {len(transitions)} transitions calculated")

        # Summary
        print("\n" + "="*60)
        print("TRAINING DATA SUMMARY")
        print("="*60)
        total_tracks = len(guy_j_tracks) + len(verdikt_tracks) + len(bonobo1_tracks) + len(bonobo2_tracks)
        total_transitions = total_tracks - 4  # -1 for each set

        print(f"\n  Total Tracks: {total_tracks}")
        print(f"  Total Transitions: {total_transitions}")
        print(f"  DJ Sets: 4")
        print(f"\n  Output Directory: {self.output_dir}")
        print("\n  Files created:")
        for f in self.output_dir.glob("*.json"):
            print(f"    - {f.name}")


def main():
    """Main execution."""
    preparator = TrainingDataPreparator()
    preparator.prepare_all_tracklists()

    print("\n" + "="*60)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Download YouTube audio files")
    print("  2. Extract transition audio segments")
    print("  3. Create Qwen3-Omni training format")


if __name__ == "__main__":
    main()
