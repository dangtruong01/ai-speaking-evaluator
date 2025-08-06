# This module provides functions to analyze pronunciation metrics from TextGrid files.
# It extracts phonemes, converts them to CMU notation (as it's in IPA format initially), and compares them against reference phonemes (in CMU format).
# It uses the `textgrid` library to read TextGrid files and processes phonemes accordingly.
# It includes functions to convert IPA to CMU notation, strip stress markers, and analyze pronunciation.
# The module is designed to be used in a pipeline for evaluating pronunciation metrics.

import textgrid
from typing import Dict, List, Any

IPA_TO_CMU = {
    'f': 'F',
    'ð': 'DH',    # \u00f0
    'æ': 'AE',    # \u00e6
    'ʈ': 'T',     # \u0288
    'ɛ': 'EH',    # \u025b
    'ɲ': 'N',     # \u0272
    'ŋ': 'NG',    # \u014b
    'ɡ': 'G',     # \u0261
    'ʃ': 'SH',    # \u0283
    'ʊ': 'UH',    # \u028a
    'ʉː': 'UW',   # \u0289\u02d0
    'aj': 'AY',
    'spn': 'UNK',  # special token for unknown sounds
    'a': 'AA',
    'i': 'IY',
    'h': 'HH',
    'n': 'N',
    'd': 'D',
    'z': 'Z',
    't': 'T',
    'm': 'M',
    'v': 'V'
}

def extract_phonemes_from_textgrid(textgrid_path: str, tier_name: str = "phones") -> List[Dict[str, Any]]:
    """Extract phonemes from a TextGrid file."""
    # Read the TextGrid file
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    phonemes = []
    
    try:
        # Find the phones tier
        tier = [t for t in tg.tiers if t.name == tier_name][0]
        
        # Extract phonemes with timing information
        for interval in tier:
            if interval.mark.strip():  # Skip empty intervals
                phonemes.append({
                    'text': interval.mark,
                    'start': interval.minTime,
                    'end': interval.maxTime
                })
                
    except Exception as e:
        print(f"Error processing TextGrid: {str(e)}")
        return []
    
    return phonemes

def convert_ipa_to_cmu(ipa_phoneme: str) -> str:
    """Convert IPA phoneme to CMU notation."""
    return IPA_TO_CMU.get(ipa_phoneme, ipa_phoneme.upper())

def strip_stress_markers(phoneme: str) -> str:
    """Remove stress markers (0,1,2) from phonemes."""
    return ''.join(c for c in phoneme if not c.isdigit())

def analyze_pronunciation(textgrid_path: str, reference_phonemes: List[str]) -> Dict[str, Any]:
    """Compare actual pronunciation against reference phonemes."""
    actual_phonemes = extract_phonemes_from_textgrid(textgrid_path)
    actual_phones = [convert_ipa_to_cmu(p['text']) for p in actual_phonemes]
    
    # Compare phonemes ignoring stress markers
    matches = [
        {
            "actual": a,
            "reference": r,
            "match": strip_stress_markers(a.upper()) == strip_stress_markers(r.upper())
        }
        for a, r in zip(actual_phones, reference_phonemes)
    ]
    
    # Calculate score based on matches
    score = sum(1 for m in matches if m["match"]) / len(matches) if matches else 0
    
    return {
        "pred_phonemes": actual_phones,
        "score": score,
        "reference": reference_phonemes,
        "actual": actual_phones,
        "phoneme_matches": matches
    }