
def compute_wer(c, s, d, i):
    n_words = c + s + d
    return (s + d + i) / n_words

def compute_nwer(c, s, d, i):
    # normalized wer, i thought it can make sense
    n_words = c + s + i
    return n_words / (n_words + (s + d + i))

def compute_wa(c, s, i, d):


