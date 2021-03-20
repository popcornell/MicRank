

def compute_wer(s, d, i, n_words):

    return (s + d + i) / n_words

def compute_nwer(s, d, i, n_words):

    return n_words / (n_words + (s + d + i))

def compute_wa(c, i, n_words):

    return (c - i) /  n_words
