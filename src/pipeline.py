# Start conda environment with:
# conda activate aligner

from asr.whisper_infer import WhisperASR
from pronunciation.pronunciation_metrics import analyze_pronunciation
from utility.cmu_dict import transcript_to_phonemes
from utility.mfa_wrapper import run_mfa_alignment
import soundfile as sf
import os

class SpeakingEvaluatorPipeline:
    def __init__(self, asr_model_size="large-v3"):
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

        # 4. Extract actual phonemes from MFA TextGrid
        actual_phonemes = analyze_pronunciation(textgrid_path, textgrid_path)["pred_phonemes"]

        # 5. Compare and score pronunciation
        score_result = analyze_pronunciation(textgrid_path, textgrid_path)
        score = score_result["score"]

        return {
            "transcript": transcript,
            "expected_phonemes": expected_phonemes,
            "actual_phonemes": actual_phonemes,
            "pronunciation_score": score,
            "details": score_result
        }

# Example usage
if __name__ == "__main__":
    pipeline = SpeakingEvaluatorPipeline(asr_model_size="base")
    result = pipeline.evaluate(
        "data/test/arctic_a0003.wav",
        "src/english.dict",           # path to MFA dictionary
        "src/english_mfa",            # path to MFA acoustic model
        "data/test/mfa_output"  # output directory for MFA
    )
    print(result)