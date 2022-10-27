import subprocess

import numpy as np
from sklearn.model_selection import train_test_split as tts


def get_lang(lang):
    dump = subprocess.Popen(
        ("aspell", "-d", lang, "dump", "master"), stdout=subprocess.PIPE
    )
    expand = subprocess.check_output(("aspell", "-l", lang, "expand"), stdin=dump.stdout)
    dump.wait()
    word_exp = [x.split() for x in expand.decode("utf-8").split("\n")]
    return [word for words in word_exp for word in words]


def retrieve_langs(lang_list):
    langs = {}
    for lang in lang_list:
        langs[lang] = get_lang(lang)
    return langs


def filter_extra_chars(lang_dict):
    filters = "'- "
    for lang, words in lang_dict.items():
        filt_words = []
        for word in words:
            if not any([f in word for f in filters]):
                filt_words.append(word)
        lang_dict[lang] = filt_words
    return lang_dict


lat_letters = "abcdefghijklmnopqrstuvwx"


def latin_filter(lang_dict, verbose=True):
    lang_latin = {}
    for lang, words in lang_dict.items():
        lang_latin[lang] = set()
        for word in words:
            if len([l for l in word if l not in lat_letters]) == 0 and len(word) == 6:
                lang_latin[lang].add(word)
        if verbose:
            print(lang, len(lang_latin[lang]))
    return lang_latin


def overlap_filter(lang_dict):
    lang_no_overlap = {}
    for lang, words in lang_dict.items():
        no_overlap = words
        for other_lang in lang_dict.keys():
            if other_lang != lang:
                no_overlap = no_overlap.difference(lang_dict[other_lang])
        lang_no_overlap[lang] = no_overlap
    return lang_no_overlap


def convert(words):
    one_hot = True

    if one_hot:
        out = np.zeros([24, 24])
        for i, word in enumerate(words):
            for j, letter in enumerate(word):
                out[i * 6 + j, lat_letters.index(letter)] = 1.0
    else:
        out = np.zeros([6, 6])
        for i, word in enumerate(words):
            for j, letter in enumerate(word):
                out[i, j] = lat_letters.index(letter) / len(lat_letters)

    return out


def test_train_split(lang_dict, return_metainfo=False):
    lang_splits = {}
    for lang, words in lang_dict.items():
        train, test = tts(list(words), train_size=1500, test_size=700)
        lang_splits[lang] = train, test

    lang_groups = {}
    n = 4
    n_train = 6000
    n_test = 1000
    for lang, (train, test) in lang_splits.items():
        train_groups = list(
            set(zip(*[np.random.choice(train, n_train + 500) for _ in range(n)]))
        )[:n_train]
        test_groups = list(
            set(zip(*[np.random.choice(test, n_test + 500) for _ in range(n)]))
        )[:n_test]
        lang_groups[lang] = train_groups, test_groups

    train_xs, train_ys = [], []
    test_xs, test_ys = [], []
    metainfo = []
    lang_idxs = {l: i for i, l in enumerate(lang_groups.keys())}

    for lang, (train, test) in lang_groups.items():
        train_xs += [convert(ws) for ws in train]
        train_ys += [lang_idxs[lang] for _ in train]
        metainfo += [ws for ws in train]
        test_xs += [convert(ws) for ws in test]
        test_ys += [lang_idxs[lang] for _ in test]

    train_xs = np.expand_dims(np.array(train_xs), axis=1).astype(np.float32)
    test_xs = np.expand_dims(np.array(test_xs), axis=1).astype(np.float32)
    train_ys = np.array(train_ys).astype(np.long)
    test_ys = np.array(test_ys).astype(np.long)

    train_shuff = np.arange(len(train_ys))
    np.random.shuffle(train_shuff)
    test_shuff = np.arange(len(test_ys))
    np.random.shuffle(test_shuff)

    train_xs = train_xs[train_shuff]
    train_ys = train_ys[train_shuff]
    metainfo = [metainfo[i] for i in train_shuff]
    test_xs = test_xs[test_shuff]
    test_ys = test_ys[test_shuff]

    if return_metainfo:
        return (train_xs, train_ys), (test_xs, test_ys), metainfo, lang_idxs
    else:
        return (train_xs, train_ys), (test_xs, test_ys)


def load_language_data(metainfo=False, verbose=True):
    lang_dict = retrieve_langs(
        ["en", "nl", "de", "es", "fr", "pt_PT", "sw", "zu", "fi", "sv"]
    )
    lang_dict = filter_extra_chars(lang_dict)
    lang_dict = latin_filter(lang_dict, verbose=verbose)
    return test_train_split(lang_dict, return_metainfo=metainfo)
