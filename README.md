# DailyTalkContiguous Generator

A pipeline for generating emotion-tagged conversational audio datasets suitable for fine-tuning modern LLMs like Moshi. This project processes stereo audio conversations, automatically transcribes them, tags each utterance with emotion/style labels, and reconstructs them with Text-to-Speech (TTS) to create a contiguous conversation dataset.

## What This Project Does

This project transforms raw conversational audio into **DailyTalkContiguous** - a dataset format where:

1. **Stereo Audio Processing**: Processes stereo audio files where two speakers are separated into left and right channels
2. **Automatic Transcription**: Uses Whisper (medium model) to transcribe speech segments
3. **Emotion Tagging**: Automatically tags each utterance with Fish Audio-compatible emotion/style tags using:
   - **VAD (Valence, Arousal, Dominance)** emotion prediction via audEERING models
   - **Prosody Analysis**: RMS energy, zero-crossing rate, pitch (F0) statistics, and speech rate
   - **Deterministic Mapping**: Converts VAD + prosody features into emotion tags (e.g., "excited", "calm", "shouting", "whispering")
4. **TTS Reconstruction**: Replaces one speaker's audio (left channel) with TTS-generated audio using Fish Audio API, while preserving the other speaker's original audio (right channel)
5. **Timeline Reconstruction**: Dynamically adjusts conversation timing to accommodate TTS duration changes, ensuring natural turn-taking

## What is DailyTalkContiguous?

**DailyTalkContiguous** is a reformatted version of the [DailyTalk dataset](https://github.com/keonlee9420/DailyTalk) that:

- **Stores conversations as stereo files**: Each conversation is a single stereo WAV file where:
  - **Left channel**: One speaker's audio
  - **Right channel**: The other speaker's audio
- **Contains word-level timestamps**: Each utterance is annotated with precise timing information
- **Is optimized for TTS training**: The format allows for easy replacement of one speaker with TTS-generated audio while preserving the other speaker's natural speech

This format is ideal for training conversational AI models like Moshi because:
- It provides natural conversation flow and turn-taking patterns
- Emotion tags enable expressive speech synthesis
- The stereo format allows for speaker separation and replacement
- Timestamps enable precise alignment for training

## How This Generates DailyTalkContiguous

The pipeline consists of two main stages:

### Stage 1: Emotion Tagging (`process_all.py`)

Processes raw stereo audio files and generates emotion-tagged transcripts:

```bash
cd emotion_tagging
python process_all.py \
    --input-dir /path/to/stereo/wavs \
    --output transcript.jsonl \
    --channel left \
    --language en
```

**Processing Pipeline:**

1. **Audio Loading**: Loads stereo WAV files and extracts the specified channel (left/right/mix)
2. **Voice Activity Detection (VAD)**: Uses Silero VAD to detect speech segments (0.8s - 12s duration)
3. **Transcription**: Transcribes each segment using Whisper medium model
4. **Emotion Analysis**:
   - Predicts VAD scores (Valence, Arousal, Dominance) using audEERING models
   - Computes prosody features: RMS energy, zero-crossing rate, pitch mean/std, speech rate
5. **Tag Assignment**:
   - Computes percentiles (33rd, 66th) for all features per file
   - Buckets features into LOW/MID/HIGH categories
   - Maps VAD + prosody buckets to Fish Audio tags using deterministic rules
6. **Punctuation**: Adds punctuation based on emotion tags:
   - `excited`, `delighted`, `shouting`, `screaming` → `!`
   - `curious`, `uncertain`, `doubtful`, `confused` → `?`
   - Others → `.`
7. **Output**: Generates `transcript.jsonl` with all segments and metadata

**Output Format (`transcript.jsonl`):**
```json
{
  "segment_id": 0,
  "audio_file": "0.wav",
  "channel": "left",
  "start": 12.43,
  "end": 14.02,
  "duration": 1.59,
  "vad": {"arousal": 0.65, "dominance": 0.72, "valence": 0.58},
  "features": {"rms": 0.12, "zcr": 0.08, "f0_mean": 180.5, "f0_std": 25.3, "speech_rate": 2.1},
  "vad_bucket": {"arousal": "HIGH", "dominance": "HIGH", "valence": "MID"},
  "prosody_bucket": {"energy": "MID", "rate": "HIGH", "zcr": "LOW", "pitch_var": "MID"},
  "tag": "excited",
  "text_raw": "its been a while",
  "text": "It's been a while!",
  "tagged_text": "(excited) It's been a while!"
}
```

### Stage 2: TTS Reconstruction (`generate_tts.py`)

Reconstructs conversations by replacing left channel audio with TTS-generated audio:

```bash
cd emotion_tagging
python generate_tts.py \
    --transcript transcript.jsonl \
    --output-dir /path/to/output \
    --token YOUR_FISH_AUDIO_API_TOKEN \
    --output-transcript transcript_shifted.jsonl \
    --temperature 0.7 \
    --top_p 0.7
```

**Reconstruction Pipeline:**

1. **Load Transcript**: Reads emotion-tagged transcript from Stage 1
2. **Right Channel Detection**: Uses VAD to detect speech segments in the right channel
3. **Event Merging**: Combines left channel (TTS) and right channel (original) events chronologically
4. **TTS Generation**: For each left channel segment:
   - Generates TTS audio using Fish Audio API with emotion tags: `(tag) text`
   - Resamples to match original audio sample rate
5. **Timeline Reconstruction**:
   - Preserves gaps between events
   - Handles overlaps by enforcing sequentiality
   - Adjusts timestamps to match new TTS durations
6. **Output**: 
   - Generates `*_edited.wav` stereo files with reconstructed conversations
   - Creates `transcript_shifted.jsonl` with updated timestamps

**Final Output:**
- **Stereo WAV files**: Left channel = TTS-generated, Right channel = original speaker
- **Updated transcript**: JSONL with corrected timestamps matching TTS durations
- **Format compatible with DailyTalkContiguous**: Ready for dataset creation

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- Fish Audio API token with access to premium model **s1** (for TTS generation)

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd /path/to/kyutai-data
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
cd emotion_tagging
pip install -r requirements.txt
```

### Required Dependencies

- `faster-whisper>=1.0.0` - Whisper ASR
- `transformers>=4.35.0` - Hugging Face models
- `torch>=2.0.0` - PyTorch
- `torchaudio>=2.0.0` - Audio processing
- `soundfile>=0.12.0` - Audio I/O
- `librosa>=0.10.0` - Audio analysis
- `numpy>=1.24.0` - Numerical operations
- `omegaconf>=2.3.0` - Configuration
- `accelerate` - Model acceleration
- `requests` - API calls

### Configuration

**Set up Fish Audio API Key:**

1. Copy the example environment file:
```bash
cp .envrc.example .envrc
```

2. Edit `.envrc` and add your Fish Audio API token:
```bash
export BEARER_TOKEN=your_fish_audio_api_token_here
```

3. Source the environment file (if using direnv, it will load automatically):
```bash
source .envrc
```

**Configure Fish Audio Reference ID:**

Before running the TTS reconstruction step, you also need to configure the reference ID in `emotion_tagging/generate_tts.py`:

1. Open `emotion_tagging/generate_tts.py`
2. Find line 61: `"reference_id": "<add referennce id here>",`
3. Replace `<add referennce id here>` with your Fish Audio reference ID

The reference ID is used by the Fish Audio API to determine the voice characteristics for TTS generation. You can obtain this from your Fish Audio account or API documentation.

**Note**: This project uses the premium Fish Audio model **s1** for high-quality emotion-conditioned TTS generation. Ensure your Fish Audio API token has access to the s1 model.

## Usage

### Complete Pipeline

**Step 1: Process audio files and generate emotion-tagged transcripts**

```bash
cd emotion_tagging
python process_all.py \
    --input-dir /path/to/stereo/wavs \
    --output transcript.jsonl \
    --channel left \
    --language en \
    --device auto
```

**Arguments:**
- `--input-dir`: Directory containing stereo WAV files (required)
- `--output`: Output JSONL file path (required)
- `--channel`: Audio channel to process - `left`, `right`, or `mix` (default: `left`)
- `--language`: Whisper language hint - `en`, `es`, `fr`, etc. (default: `en`)
- `--device`: Processing device - `cpu`, `cuda`, or `auto` (default: `auto`)

**Step 2: Reconstruct conversations with TTS**

```bash
cd emotion_tagging
python generate_tts.py \
    --transcript transcript.jsonl \
    --output-dir /path/to/output/directory \
    --token YOUR_FISH_AUDIO_API_TOKEN \
    --output-transcript transcript_shifted.jsonl \
    --temperature 0.7 \
    --top_p 0.7
```

**Arguments:**
- `--transcript`: Path to transcript.jsonl from Step 1 (required)
- `--output-dir`: Directory containing original WAV files and where output will be saved (required)
- `--token`: Fish Audio API token (required)
- `--output-transcript`: Output transcript with shifted timestamps (default: `transcript_shifted.jsonl`)
- `--temperature`: TTS temperature parameter (default: 0.7)
- `--top_p`: TTS top-p parameter (default: 0.7)

### Example Workflow

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Process stereo audio files
cd emotion_tagging
python process_all.py \
    --input-dir ../DailyTalkContiguous/data_stereo \
    --output ../transcript.jsonl \
    --channel left

# 3. Generate TTS-reconstructed conversations
# (Make sure .envrc is sourced to load BEARER_TOKEN)
python generate_tts.py \
    --transcript ../transcript.jsonl \
    --output-dir ../DailyTalkContiguous/data_stereo \
    --token $BEARER_TOKEN \
    --output-transcript ../transcript_shifted.jsonl
```

## How to Fine-tune Modern LLMs like Moshi

The generated DailyTalkContiguous dataset is designed for fine-tuning conversational AI models like Moshi. Here's how to use it:

### Dataset Format

The final dataset consists of:

1. **Stereo Audio Files** (`*_edited.wav`):
   - Left channel: TTS-generated speech with emotion tags
   - Right channel: Natural human speech
   - Format: 16kHz (or original sample rate) stereo WAV

2. **Transcript File** (`transcript_shifted.jsonl`):
   - One JSON object per line
   - Contains timestamps, emotion tags, and text
   - Format compatible with training pipelines

### Fine-tuning Process

1. **Prepare Training Data**:
   - Use `transcript_shifted.jsonl` for text annotations
   - Use `*_edited.wav` files for audio
   - Extract features: audio embeddings, text tokens, emotion tags

2. **Model Architecture Considerations**:
   - **Input**: Audio waveform or features + text tokens
   - **Emotion Tags**: Use as conditioning signals for expressive generation
   - **Turn-taking**: Model learns from natural conversation flow in stereo format

3. **Training Configuration**:
   - **Loss Functions**: Combine audio reconstruction loss with emotion classification loss
   - **Conditioning**: Use emotion tags as conditional inputs
   - **Multi-speaker**: Leverage stereo format for speaker separation learning

4. **Example Training Setup** (conceptual):
```python
# Pseudo-code for training setup
for entry in transcript_shifted.jsonl:
    audio_left = load_audio(entry['audio_file'] + '_edited.wav', channel='left')
    audio_right = load_audio(entry['audio_file'] + '_edited.wav', channel='right')
    text = entry['tagged_text']  # "(excited) It's been a while!"
    emotion_tag = entry['tag']
    
    # Train model to generate audio_left from text + emotion_tag
    # while learning from audio_right as natural reference
```

### Key Advantages for LLM Fine-tuning

1. **Emotion-Aware Training**: Emotion tags provide explicit conditioning for expressive speech
2. **Natural Conversation Flow**: Stereo format preserves turn-taking and timing
3. **Speaker Diversity**: Right channel provides natural speech references
4. **Scalable Generation**: TTS-generated left channel enables large-scale dataset creation
5. **Precise Alignment**: Timestamps enable accurate audio-text alignment for training

### Integration with Moshi

Moshi and similar models can benefit from:
- **Emotion-conditioned generation**: Use emotion tags as input tokens
- **Conversational context**: Learn from stereo conversation format
- **Expressive synthesis**: Train on emotion-tagged audio-text pairs
- **Multi-modal learning**: Combine audio, text, and emotion signals

## Project Structure

```
kyutai-data/
├── README.md                    # This file
├── emotion_tagging/             # Main processing pipeline
│   ├── process_all.py          # Stage 1: Emotion tagging
│   ├── generate_tts.py         # Stage 2: TTS reconstruction
│   ├── tag_emotions.py         # Audio processing utilities
│   ├── vad_emotion_mapper.py   # VAD + prosody → emotion tags
│   ├── vad_model_audeering.py # audEERING VAD model wrapper
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # Detailed emotion tagging docs
├── DailyTalkContiguous/         # Output dataset directory
│   ├── data_stereo/            # Stereo audio files
│   ├── dailytalk.jsonl         # Dataset manifest
│   └── README.md               # Dataset documentation
└── transcript.jsonl            # Intermediate transcript (Stage 1 output)
```

## Models Used

- **ASR**: Whisper Medium via `faster-whisper`
- **VAD (Speech Detection)**: Silero VAD
- **Emotion Prediction**: audEERING VAD (Valence, Arousal, Dominance)
- **TTS**: Fish Audio API premium model **s1** (emotion-conditioned synthesis)
- **Prosody Analysis**: librosa (RMS, ZCR, F0, speech rate)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Note**: This project uses the DailyTalk dataset which is licensed under CC-BY-SA 4.0. Please refer to the [original DailyTalk repository](https://github.com/keonlee9420/DailyTalk) for dataset license details.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Use `--device cpu` or process fewer files at once
   - Reduce batch sizes in model loading

2. **Fish Audio API Errors**:
   - Verify API token is valid
   - Check network connectivity
   - Review API rate limits

3. **Audio Format Issues**:
   - Ensure WAV files are stereo (2 channels)
   - Check sample rate compatibility (16kHz recommended)

4. **Missing Dependencies**:
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt` again

## Contributing

This is a research tool. Contributions and improvements are welcome!

## References

- [DailyTalk Dataset](https://github.com/keonlee9420/DailyTalk)
- [Fish Audio](https://fish.audio/)
- [Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
