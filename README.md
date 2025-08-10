# AI Speaking Evaluator

An AI-powered English speaking evaluator that analyzes pronunciation, fluency, and speech rate.  
Built with:
- **OpenAI Whisper** (speech-to-text)
- **Montreal Forced Aligner (MFA)** (forced alignment for phoneme timing)
- Custom scoring pipeline for pronunciation & fluency evaluation

---

## Features
- **Speech-to-Text Transcription** via Whisper
- **Forced Alignment** with MFA for phoneme-level timing
- **Pronunciation Scoring** using phoneme accuracy
- **Fluency Metrics**: speech rate, pauses, word timing
- Modular design for future extension (vocabulary, grammar, interaction feedback)

---

## Project Structure
src/
pipeline.py # Main entry point for evaluation
utility/
mfa_wrapper.py # Helper functions to run MFA alignment
scoring.py # Pronunciation and fluency scoring
data/
test/ # Sample audio + transcript


---

## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Eloquency-Platform/eloquency-ai-service.git
```

### 2️⃣ Create and activate Conda environment
We use Conda to avoid _kalpy missing errors in MFA on Windows.
```bash
conda create -n elo-ai python=3.11
conda activate elo-ai
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner
```

## MFA Setup
Download pretrained MFA models for English (only needed once) from the MFA website: https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-english_uk_mfa-v2.0.0

## Usage
Run the evaluation pipeline on a sample file:

```bash
python speaking-eval/pipeline.py
```

Example output:
* Transcribed text
* Pronunciation accuracy score
* Speech rate (words per minute)
* Detected pauses and fluency metrics

## Notes
* Works with .wav files (mono, 16kHz recommended)
* MFA output TextGrid files are stored in data/test/mfa_output
* Whisper runs in CPU mode unless GPU is available