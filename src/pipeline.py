# We are using conda environment instead of virtualenv as mfa module should be installed in a conda environment.
# Start conda environment with:
# conda activate aligner
#
# Make sure you have installed all dependencies in the environment.
# If you have not installed the dependencies, you can install them with:
# pip install -r requirements.txt

from asr.whisper_infer import WhisperASR
from pronunciation.pronunciation_metrics import analyze_pronunciation
from utility.cmu_dict import transcript_to_phonemes
from utility.mfa_wrapper import run_mfa_alignment
import soundfile as sf
import os
import json

class SpeakingEvaluatorPipeline:
    def __init__(self, asr_model_size="base"):
        self.asr = WhisperASR(model_size=asr_model_size)

    def evaluate(self, audio_path, mfa_dict, mfa_model, mfa_output_dir):
        # 1. ASR Transcription
        asr_result = self.asr.transcribe(audio_path)
        transcript = asr_result["transcript"]
        segments = asr_result["segments"]

        # Save transcript
        transcript_path = os.path.splitext(audio_path)[0] + ".txt"
        self.asr.save_transcript(transcript, transcript_path)

        # 2. Generate expected phonemes
        expected_phonemes = transcript_to_phonemes(transcript)

        # 3. Run MFA alignment
        data_dir = os.path.dirname(audio_path)
        textgrid_path = run_mfa_alignment(data_dir, mfa_dict, mfa_model, mfa_output_dir)

        # 4. Get pronunciation analysis results
        pronunciation_results = analyze_pronunciation(textgrid_path, expected_phonemes)
        actual_phonemes = pronunciation_results["pred_phonemes"]
        score = pronunciation_results["score"]

        return {
            "transcript": transcript,
            "expected_phonemes": expected_phonemes,
            "actual_phonemes": actual_phonemes,
            "pronunciation_score": score,
            "details": pronunciation_results
        }

    def save_results(self, results: dict, output_path: str):
        """Save evaluation results to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Get paths to downloaded MFA models
    mfa_models_dir = "/Users/truonghaidang/Documents/MFA/pretrained_models"
    dict_path = os.path.join(mfa_models_dir, "dictionary", "english_mfa.dict")
    model_path = os.path.join(mfa_models_dir, "acoustic", "english_mfa.zip")
    
    pipeline = SpeakingEvaluatorPipeline(asr_model_size="base")
    result = pipeline.evaluate(
        "data/test/arctic_a0003.wav",
        dict_path,                    # path to MFA dictionary
        model_path,                   # path to MFA acoustic model
        "data/test/mfa_output"       # output directory for MFA
    )

    # Save results to JSON file
    output_path = "data/test/evaluation_results.json"
    pipeline.save_results(result, output_path)

    print(result)