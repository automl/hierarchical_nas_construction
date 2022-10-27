import os
import re
import string

import numpy as np
from sklearn.model_selection import train_test_split as tts


def is_all_upper(l):
    return all([c.isupper() or c == " " for c in l])


def expand(word):
    return word + "_" * (phrase_word_bounds[1] - len(word))


phrase_size = 3
phrase_word_bounds = 3, 6
encoding = "abcdefghijklmnopqrstuvwxyz_"


def encode(phrase):
    out = np.zeros([1, 27, phrase_size * phrase_word_bounds[1]], dtype=np.float32)
    for i, word in enumerate(phrase):
        for j, letter in enumerate(word):
            out[0, encoding.index(letter), j + (phrase_word_bounds[1] * i)] = 1.0
    return out


def test_train_split(author_data, author_idx, return_metainfo=False):
    author_splits = {}
    for author in author_data.keys():
        idxs = np.arange(len(author_data[author]["encodings"]))
        train, test = tts(idxs, train_size=11100, test_size=1000)
        author_splits[author] = train, test

    train_xs, train_ys, test_xs, test_ys = [], [], [], []
    metainfo = []
    for k, v in author_splits.items():
        train_embeddings = [author_data[k]["encodings"][i] for i in v[0]]
        train_phrases = [author_data[k]["phrases"][i] for i in v[0]]
        test_embeddings = [author_data[k]["encodings"][i] for i in v[1]]

        train_xs += train_embeddings
        train_ys += [author_idx[k]] * len(train_embeddings)
        metainfo += train_phrases
        test_xs += test_embeddings
        test_ys += [author_idx[k]] * len(test_embeddings)

    train_xs = np.array(train_xs).astype(np.float32)
    train_ys = np.array(train_ys).astype(np.long)
    test_xs = np.array(test_xs).astype(np.float32)
    test_ys = np.array(test_ys).astype(np.long)

    if return_metainfo:
        return (train_xs, train_ys), (test_xs, test_ys), metainfo
    else:
        return (train_xs, train_ys), (test_xs, test_ys)


def load_gutenberg(metainfo=False):
    texts = {}
    authors = {}
    for text in os.listdir("texts"):
        with open("texts/" + text, "r") as f:
            author = text.split(".")[0][:-1]
            if author in [
                "aquinas",
                "confucius",
                "hawthorne",
                "plato",
                "shakespeare",
                "tolstoy",
            ]:
                texts[text] = f.readlines()[95:]
                authors[text] = author
    all_authors = sorted(list(set(authors.values())))
    author_idx = {a: float(i) for i, a in enumerate(all_authors)}
    lat_letters = "abcdefghijklmnopqrstuvwxyz "
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    punc = string.punctuation
    invalids = [
        "prologue",
        "epilogue",
        "chapter",
        "scene",
        "act",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
        "xi",
        "xii",
        "xiii",
        "xiv",
        "xv",
        "xvi",
        "xvii",
        "xviii",
        "xix",
        "xx",
    ]

    text_proc = {}
    for k, v in texts.items():
        t = [l.strip().translate(str.maketrans(punc, " " * len(punc))) for l in v]
        t = [l for l in t if l and not is_all_upper(l)]
        t = " ".join(t).lower()
        t = "".join([c for c in t if c in lat_letters])
        t = _RE_COMBINE_WHITESPACE.sub(" ", t).strip()
        t = [w for w in t.split() if w not in invalids]
        text_proc[k] = t

    # extract matching phrases
    author_phrase_set = {}
    for k, v in text_proc.items():
        author = authors[k]

        if author not in author_phrase_set:
            author_phrase_set[author] = set()

        for word in range(0, len(v) - phrase_size):
            phrase = [w for w in v[word : word + phrase_size]]
            if all(
                [phrase_word_bounds[0] <= len(w) <= phrase_word_bounds[1] for w in phrase]
            ):
                exp_phrase = tuple([expand(w) for w in phrase])
                author_phrase_set[author].add(exp_phrase)

    # filter overlapping phrases
    author_unique_phrases = {}
    for a, ws in author_phrase_set.items():
        no_overlap = ws
        for other_a, other_ws in author_phrase_set.items():
            if a != other_a:
                no_overlap = no_overlap.difference(other_ws)
        author_unique_phrases[a] = no_overlap

    # encode phrases
    author_data = {}
    for k, v in author_unique_phrases.items():
        if k not in author_data:
            author_data[k] = {"encodings": [], "phrases": []}
        for phrase in v:
            author_data[k]["encodings"].append(encode(phrase))
            author_data[k]["phrases"].append(phrase)

    return test_train_split(author_data, author_idx, metainfo)
