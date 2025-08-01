import whisper

class WhisperASR:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path, language="en"):
        result = self.model.transcribe(audio_path, language=language, word_timestamps=True)
        transcript = result.get("text", "")
        segments = result.get("segments", [])
        return {
            "transcript": transcript,
            "segments": segments  # Contains timestamps for words/phrases
        }

    def save_textgrid(self, segments, output_path, audio_duration):
        """
        Save a Praat TextGrid file with word-level intervals from Whisper segments.
        """
        words = []
        for seg in segments:
            if "words" in seg:
                words.extend(seg["words"])

        with open(output_path, "w") as f:
            f.write('File type = "ooTextFile"\n')
            f.write('Object class = "TextGrid"\n\n')
            f.write(f"xmin = 0\nxmax = {audio_duration}\n")
            f.write("tiers? <exists>\nsize = 1\nitem []:\n")
            f.write('\titem [1]:\n\t\tclass = "IntervalTier"\n\t\tname = "words"\n')
            f.write(f"\t\txmin = 0\n\t\txmax = {audio_duration}\n")
            f.write(f"\t\tintervals: size = {len(words)}\n")
            for i, word in enumerate(words, 1):
                xmin = word['start']
                xmax = word['end']
                text = word['word']
                f.write(f"\t\tintervals [{i}]:\n")
                f.write(f"\t\t\txmin = {xmin}\n")
                f.write(f"\t\t\txmax = {xmax}\n")
                f.write(f'\t\t\ttext = "{text}"\n')

    def save_transcript(self, transcript, output_path):
        """
        Save the transcript as a plain text file.
        """
        with open(output_path, "w") as f:
            f.write(transcript)

# # Example usage
# if __name__ == "__main__":
#     import soundfile as sf
#     asr = WhisperASR(model_size="base")
#     audio_path = "path_to_audio_file.wav"
#     output = asr.transcribe(audio_path)
#     print("Transcript:", output["transcript"])
#     print("Segments:", output["segments"])
#
#     # Get audio duration for TextGrid
#     audio_info = sf.info(audio_path)
#     duration = audio_info.duration
#
#     # Save TextGrid
#     asr.save_textgrid(output["segments"], "output.TextGrid", duration)
#
#     # Save transcript
#