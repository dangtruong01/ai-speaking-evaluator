import parselmouth
import Levenshtein

def extract_phonemes_from_textgrid(textgrid_path, tier_name="phones"):
    tg = parselmouth.Data.read(textgrid_path)
    phonemes = []
    for interval in tg.to_textgrid().get_tier_by_name(tier_name).intervals:
        text = interval.text.strip()
        if text and text.lower() != "sil":
            phonemes.append(text)
    return phonemes

def compare_phoneme_sequences(ref_phonemes, pred_phonemes):
    distance = Levenshtein.distance(" ".join(ref_phonemes), " ".join(pred_phonemes))
    max_len = max(len(ref_phonemes), len(pred_phonemes))
    accuracy = 1 - distance / max_len if max_len > 0 else 0
    score = int(accuracy * 100)
    return {
        "levenshtein_distance": distance,
        "accuracy": accuracy,
        "score": score,
        "ref_phonemes": ref_phonemes,
        "pred_phonemes": pred_phonemes
    }

def analyze_pronunciation(textgrid_path_ref, textgrid_path_pred):
    ref_phonemes = extract_phonemes_from_textgrid(textgrid_path_ref)
    pred_phonemes = extract_phonemes_from_textgrid(textgrid_path_pred)
    return compare_phoneme_sequences(ref_phonemes, pred_phonemes)