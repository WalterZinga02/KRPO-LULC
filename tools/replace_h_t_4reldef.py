import re
from rapidfuzz import fuzz


def find_best_span(sentence, target):
    tokens = re.findall(r"\w+|[^\w\s]", sentence)
    best_span = None
    best_score = -1
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, n + 1):
            span = " ".join(tokens[i:j])
            score = fuzz.ratio(target.lower(), span.lower())
            if score > best_score:
                best_score = score
                best_span = (i, j, span)
    return best_span


def replace_entities(sentence, _h_, _t_):
    _h = _h_.replace("_", " ")
    _t = _t_.replace("_", " ")
    tokens = re.findall(r"\w+|[^\w\s]", sentence)

    h_start, h_end, _ = find_best_span(sentence, _h)
    tokens[h_start:h_end] = ["the subject entity"]

    t_sentence = " ".join(tokens)
    t_tokens = re.findall(r"\w+|[^\w\s]", t_sentence)
    t_start, t_end, _ = find_best_span(t_sentence, _t)
    t_tokens[t_start:t_end] = ["the object entity"]

    return " ".join(t_tokens)
