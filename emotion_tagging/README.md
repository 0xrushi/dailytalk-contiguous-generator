# Emotion-Tagged Transcript Generator

A Python CLI tool that generates emotion-tagged transcripts from audio files. Each utterance is annotated with a Fish Audio compatible emotion/style tag using a deterministic pipeline based on SpeechBrain SER and prosody features.

## Features

- **Automatic Speech Recognition** using Whisper (medium model) via faster-whisper
- **Emotion/Style Tagging** using SpeechBrain SER (Wav2Vec2-IEMOCAP) + Prosody Analysis
- **Voice Activity Detection** using Silero VAD for accurate speech segmentation
- **Fish Audio Compatible Tags** - deterministic mapping to supported tags
- **JSONL Output** with timestamps, segments, and metadata for TTS stitching
- **Deterministic Punctuation** based on assigned emotion tags

## Installation

1. Clone or download this repository:
```bash
cd emotion_tagging
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Batch Processing (Directory)

Process all WAV files in a directory and generate a single `transcript.jsonl`. This pipeline computes prosody percentiles per file to assign relative tags (e.g. "high energy" relative to the speaker's baseline).

```bash
python process_all.py --input-dir /path/to/wavs --output transcript.jsonl --channel left
```

## Command Line Arguments (`process_all.py`)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | - | Directory containing WAV files to process |
| `--output` | Yes | - | Output JSONL file path |
| `--channel` | No | `left` | Audio channel: `left`, `right`, or `mix` |
| `--language` | No | `en` | Whisper language hint (e.g., `en`, `es`, `fr`) |
| `--device` | No | `auto` | Device: `cpu`, `cuda`, or `auto` |

## Output Format

### JSONL Format

The tool generates a JSONL file with one JSON object per line:

```json
{"segment_id":0,"audio_file":"0.wav","channel":"left","start":12.43,"end":14.02,"duration":1.59,"ser_label":"happy","features":{...},"tag":"excited","text_raw":"its been a while","text":"It's been a while!","tagged_text":"(excited) It's been a while!"}
```

### JSON Fields

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | int | Global 0-indexed segment identifier |
| `audio_file` | string | Filename of the source audio |
| `channel` | string | Audio channel used |
| `start` | float | Segment start time in seconds |
| `end` | float | Segment end time in seconds |
| `duration` | float | Segment duration in seconds |
| `ser_label` | string | Raw emotion label from SpeechBrain SER |
| `features` | dict | Prosody features (rms, zcr, f0, speech_rate) |
| `tag` | string | Mapped Fish Audio emotion/style tag |
| `text_raw` | string | Raw transcript from Whisper |
| `text` | string | Punctuated transcript (deterministic) |
| `tagged_text` | string | Fish format string: `({tag}) {text}` |

## Processing Pipeline

1. **Load Audio**: Load audio file and convert to mono at 16kHz
2. **VAD Segmentation**: Use Silero VAD to detect speech segments
3. **Transcription**: Transcribe each segment using Whisper to get `text_raw`
4. **Feature Extraction**: 
   - **SER**: Classify segment using `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`
   - **Prosody**: Compute RMS energy, Zero-Crossing Rate, Pitch (F0) mean/std, and Speech Rate
5. **Tag Mapping**: 
   - Compute percentiles (p20, p80) for prosody features across the file
   - Bucket features into LOW/MID/HIGH
   - Apply deterministic rules to map SER label + buckets to Fish Audio tags (e.g., High Energy + Angry -> Shouting)
6. **Punctuation**: 
   - `excited`, `delighted`, `shouting`, `screaming` → `!`
   - `curious`, `uncertain`, `doubtful`, `confused` → `?`
   - Others → `.`
7. **Output**: Write JSONL file

## Models Used

- **ASR**: Whisper medium via `faster-whisper`
- **SER**: `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`
- **VAD**: Silero VAD

## Requirements

- Python 3.8+
- `speechbrain`
- `faster-whisper`
- `librosa`, `torchaudio`, `numpy`, `soundfile`

## License

This tool is provided as-is for research and personal use.