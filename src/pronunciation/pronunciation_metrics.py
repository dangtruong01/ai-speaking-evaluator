# This module provides functions to analyze pronunciation metrics from TextGrid files.
# It extracts phonemes, converts them to CMU notation (as it's in IPA format initially), and compares them against reference phonemes (in CMU format).
# It uses the `textgrid` library to read TextGrid files and processes phonemes accordingly.
# It includes functions to convert IPA to CMU notation, strip stress markers, and analyze pronunciation.
# The module is designed to be used in a pipeline for evaluating pronunciation metrics.

import textgrid
import numpy as np
from typing import Dict, List, Any, Tuple

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

def get_alignment_score(p1: str, p2: str) -> float:
    """Get similarity score between two phonemes."""
    if strip_stress_markers(p1.upper()) == strip_stress_markers(p2.upper()):
        return 1.0
    # Partial matches for similar phonemes
    if p1 in ['AA', 'AO', 'AH'] and p2 in ['AA', 'AO', 'AH']:
        return 0.8
    return 0.0

def align_sequences(actual: List[str], reference: List[str]) -> List[Tuple[str, str, float]]:
    """
    Align two phoneme sequences using dynamic programming with gap insertions.
    Returns list of (actual, reference, score) tuples.
    """
    GAP = "GAP"  # Token to represent gaps in either sequence
    n, m = len(actual), len(reference)
    
    # Initialize matrices
    dp = np.zeros((n + 1, m + 1))  # Scores
    bp = np.zeros((n + 1, m + 1), dtype=int)  # Backpointers: 0=match/mismatch, 1=gap in ref, 2=gap in actual
    
    # Initialize first row and column
    for i in range(n + 1):
        dp[i, 0] = i * 0.8  # Gap penalty
    for j in range(m + 1):
        dp[0, j] = j * 0.8
        
    # Fill matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = get_alignment_score(actual[i-1], reference[j-1])
            diag = dp[i-1, j-1] + (0 if match_score > 0.7 else 1)  # Match/mismatch
            up = dp[i-1, j] + 0.8  # Gap in reference
            left = dp[i, j-1] + 0.8  # Gap in actual
            
            # Choose minimum and store backpointer
            if diag <= up and diag <= left:
                dp[i, j] = diag
                bp[i, j] = 0
            elif up <= left:
                dp[i, j] = up
                bp[i, j] = 1
            else:
                dp[i, j] = left
                bp[i, j] = 2
    
    # Backtrack to get alignment
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bp[i, j] == 0:
            # Match/mismatch
            score = get_alignment_score(actual[i-1], reference[j-1])
            aligned.append((actual[i-1], reference[j-1], score))
            i -= 1
            j -= 1
        elif i > 0 and bp[i, j] == 1:
            # Gap in reference
            aligned.append((actual[i-1], GAP, 0.0))
            i -= 1
        else:
            # Gap in actual
            aligned.append((GAP, reference[j-1], 0.0))
            j -= 1
            
    return list(reversed(aligned))

def analyze_pronunciation(textgrid_path: str, reference_phonemes: List[str]) -> Dict[str, Any]:
    """Compare actual pronunciation against reference phonemes with gap-based alignment."""
    actual_phonemes = extract_phonemes_from_textgrid(textgrid_path)
    actual_phones = [convert_ipa_to_cmu(p['text']) for p in actual_phonemes]
    
    # Align sequences
    aligned_phonemes = align_sequences(actual_phones, reference_phonemes)
    
    # Create detailed matches
    matches = [
        {
            "actual": actual if actual != "GAP" else "---",
            "reference": ref if ref != "GAP" else "---",
            "match": actual != "GAP" and ref != "GAP" and score > 0.7,
            "score": score
        }
        for actual, ref, score in aligned_phonemes
    ]
    
    # Calculate overall score, ignoring gaps
    valid_matches = [m for m in matches if m["actual"] != "---" and m["reference"] != "---"]
    score = sum(m["score"] for m in valid_matches) / len(valid_matches) if valid_matches else 0
    
    return {
        "pred_phonemes": actual_phones,
        "score": score,
        "reference": reference_phonemes,
        "actual": actual_phones,
        "phoneme_matches": matches,
        "alignment": aligned_phonemes
    }