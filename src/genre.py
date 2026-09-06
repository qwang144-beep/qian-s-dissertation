"""
AllMusic Style / Genre loader (MSD-MASD style assignment).

The paper's [GENRE] is "determined by concatenating the genres listed in the
Allmusic Style Dataset". The annotation file (msd-MASD-styleAssignment.cls,
linked from the LyCon repo) is a plain text assignment. Its exact layout can
vary between the .cls partitions, so this loader accepts the two common forms
and is easy to adjust:

    Form A (one line per track):   TRACKID<TAB or space>Style_Label
    Form B (%class blocks):        %Style_Label
                                   TRACKID
                                   TRACKID
                                   ...

Either way we return {track_id: "Genre1 Genre2 ..."} with labels concatenated
if a track carries more than one. Underscores in AllMusic labels (e.g.
"Alternative_Indie_Rock") are turned into spaces for the prompt.
"""
from typing import Dict, List


def _clean_label(lbl: str) -> str:
    return lbl.strip().replace("_", " ")


def load_styles(path: str) -> Dict[str, List[str]]:
    per_track: Dict[str, List[str]] = {}
    current_class = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("%"):                       # Form B header
                current_class = _clean_label(line[1:])
                continue
            # split on tab first, then any whitespace
            parts = line.split("\t") if "\t" in line else line.split()
            tid = parts[0].strip()
            if len(parts) >= 2:                            # Form A: id + label
                label = _clean_label(" ".join(parts[1:]))
            elif current_class is not None:                # Form B: id under %class
                label = current_class
            else:
                continue
            per_track.setdefault(tid, [])
            if label not in per_track[tid]:
                per_track[tid].append(label)
    return per_track


def genre_string(styles: List[str], sep: str = " ") -> str:
    """Concatenate a track's style labels into the [GENRE] fill."""
    return sep.join(styles)


if __name__ == "__main__":
    import tempfile, os
    # Form A demo
    a = "TRSEKGD128F42B654D\tExperimental\nTRAAA\tAlternative_Indie_Rock\nTRAAA\tPop_Rock\n"
    with tempfile.NamedTemporaryFile("w", suffix=".cls", delete=False) as fh:
        fh.write(a); tmp = fh.name
    s = load_styles(tmp)
    print("Form A:", s)
    assert s["TRSEKGD128F42B654D"] == ["Experimental"]
    assert genre_string(s["TRAAA"]) == "Alternative Indie Rock Pop Rock"
    os.unlink(tmp)

    # Form B demo
    b = "%Experimental\nTRSEKGD128F42B654D\n%Blues Contemporary\nTRZZZ\n"
    with tempfile.NamedTemporaryFile("w", suffix=".cls", delete=False) as fh:
        fh.write(b); tmp = fh.name
    s = load_styles(tmp)
    print("Form B:", s)
    assert s["TRSEKGD128F42B654D"] == ["Experimental"]
    os.unlink(tmp)
    print("genre.py smoke test passed.")
