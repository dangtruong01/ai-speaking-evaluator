from asr.whisper_infer import WhisperASR
# from fluency.fluency_metrics import analyze_fluency
# from pronunciation.pronunciation_metrics import analyze_pronunciation
# from grammar_vocab.grammar_vocab import analyze_grammar_vocab
# from scoring.scoring import compute_score
# from feedback.feedback import generate_feedback
import soundfile as sf
import os

class SpeakingEvaluatorPipeline:
    def __init__(self, asr_model_size="base"):
        self.asr = WhisperASR(model_size=asr_model_size)

    def evaluate(self, audio_path):
        # 1. ASR Transcription
        asr_result = self.asr.transcribe(audio_path)
        transcript = asr_result["transcript"]
        segments = asr_result["segments"]

        # Optionally save TextGrid
        audio_info = sf.info(audio_path)
        duration = audio_info.duration
        textgrid_path = os.path.splitext(audio_path)[0] + ".TextGrid"
        self.asr.save_textgrid(segments, textgrid_path, duration)
        
        # 2. Save transcript
        transcript_path = os.path.splitext(audio_path)[0] + ".txt"
        self.asr.save_transcript(transcript, transcript_path)

        # # 2. Fluency & Speech Rate
        # fluency_metrics = analyze_fluency(segments, transcript)

        # # 3. Pronunciation (optional: can use forced alignment or compare to expected transcript if available)
        # pronunciation_metrics = analyze_pronunciation(segments)

        # # 4. Grammar & Vocabulary
        # grammar_vocab_metrics = analyze_grammar_vocab(transcript)

        # # 5. Scoring
        # score = compute_score(fluency_metrics, pronunciation_metrics, grammar_vocab_metrics)

        # # 6. Feedback
        # feedback = generate_feedback(score, fluency_metrics, pronunciation_metrics, grammar_vocab_metrics)

        # return {
        #     "transcript": transcript,
        #     "fluency": fluency_metrics,
        #     "pronunciation": pronunciation_metrics,
        #     "grammar_vocab": grammar_vocab_metrics,
        #     "score": score,
        #     "feedback": feedback
        # }

# Example usage
if __name__ == "__main__":
    pipeline = SpeakingEvaluatorPipeline(asr_model_size="base")
    result = pipeline.evaluate("data/test/3/arctic_a0003.wav")