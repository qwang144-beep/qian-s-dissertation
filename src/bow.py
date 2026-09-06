"""
musiXmatch Bag-of-Words parser.

The mxm dataset ships as `mxm_dataset_train.txt` / `mxm_dataset_test.txt`:
    # ... comment lines start with '#'
    %word1,word2,...,word5000        <- the top-5000 vocabulary, rank-ordered
    TRACKID,MXMID,idx:cnt,idx:cnt,... <- one line per track (idx is 1-based
                                         into the % list; cnt is the count)

IMPORTANT replication note
--------------------------
The mxm vocabulary is PORTER-STEMMED and lower-cased (e.g. "love"->"love",
"running"->"run", "beautiful"->"beauti"). The LyCon paper feeds this stemmed
vocabulary straight into the LLM ("concatenation of vocabulary list ... sorted
in descending order of frequency"). We reproduce that by default. If you prefer
to un-stem, pass a reverse map built from mxm_reverse_mapping.txt (stem -> most
common surface form); note that mapping is partial, so document whichever choice
you make in your methodology.
"""
from typing import Dict, List, Optional, Iterator, Tuple


def load_vocab_and_tracks(path: str) -> Tuple[List[str], Iterator]:
    """Not used directly; see stream_tracks. Kept for clarity of the format."""
    raise NotImplementedError


def _parse(path: str):
    vocab: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("%"):
                vocab = line[1:].split(",")
                continue
            parts = line.split(",")
            track_id, _mxm_id = parts[0], parts[1]
            pairs = []
            for tok in parts[2:]:
                idx, cnt = tok.split(":")
                pairs.append((int(idx), int(cnt)))
            yield track_id, pairs, vocab


def load_bow(paths: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Return {track_id: [(stemmed_word, count), ...]} across the given files.

    Words within a track are returned SORTED BY COUNT DESCENDING (ties broken by
    the word's overall rank, i.e. its index in the top-5000 list), matching the
    paper's "sorted in descending order of frequency".
    """
    out: Dict[str, List[Tuple[str, int]]] = {}
    for path in paths:
        for track_id, pairs, vocab in _parse(path):
            # (word, count, rank) then sort by (-count, rank)
            triples = [(vocab[idx - 1], cnt, idx) for idx, cnt in pairs]
            triples.sort(key=lambda t: (-t[1], t[2]))
            out[track_id] = [(w, c) for w, c, _r in triples]
    return out


def load_reverse_map(path: str) -> Dict[str, str]:
    """mxm_reverse_mapping.txt : 'stem<TAB>surface' -> {stem: surface}."""
    m: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            stem, surface = line.split("\t")[:2] if "\t" in line else line.split(",")[:2]
            m[stem] = surface
    return m


def vocabulary_string(bow_row: List[Tuple[str, int]],
                      reverse_map: Optional[Dict[str, str]] = None,
                      sep: str = ", ") -> str:
    """Build the [VOCABULARY] fill: frequency-descending words, joined by `sep`.

    If `reverse_map` is given, stems are un-stemmed where a surface form exists.
    """
    words = []
    for stem, _cnt in bow_row:
        words.append(reverse_map.get(stem, stem) if reverse_map else stem)
    return sep.join(words)


if __name__ == "__main__":
    # smoke test on a synthetic mxm-format file
    import tempfile, os
    demo = "# comment\n%love,run,fire,beauti,night\nTRXXX,123,1:5,3:9,4:2,5:7\n"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(demo); tmp = fh.name
    bow = load_bow([tmp])
    print("parsed:", bow)                       # counts: love5 run? ...
    row = bow["TRXXX"]
    assert row[0] == ("fire", 9)                # highest count first
    assert row[-1] == ("beauti", 2)             # lowest count last
    print("vocab string:", vocabulary_string(row))
    print("un-stemmed  :", vocabulary_string(row, {"beauti": "beautiful", "run": "running"}))
    os.unlink(tmp)
    print("bow.py smoke test passed.")
