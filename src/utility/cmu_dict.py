import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

def transcript_to_phonemes(transcript):
    cmu = cmudict.dict()
    words = transcript.lower().split()
    phonemes = []
    for word in words:
        # Remove punctuation
        word_clean = ''.join([c for c in word if c.isalpha()])
        if word_clean in cmu:
            # Use first pronunciation variant
            phonemes.extend(cmu[word_clean][0])
        else:
            phonemes.append('UNK')
    return phonemes