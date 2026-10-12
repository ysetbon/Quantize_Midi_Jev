from nextharmony.model import _parallels, label_of, name_of, voicings, words_of

C, A = 0, 9  # tonic pitch classes


def test_labels_and_names():
    assert name_of(label_of([48, 52, 55, 60], C)) == "I"            # C3 E3 G3 C4
    assert name_of(label_of([52, 55, 60, 64], C)) == "I6"           # first inversion
    assert name_of(label_of([43, 59, 62, 65], C)) == "V7"           # G2 B3 D4 F4
    assert name_of(label_of([47, 55, 62, 65], C)) == "V65"          # B2 in the bass
    assert name_of(label_of([45, 57, 60, 64], C)) == "vi"           # A minor in C major
    assert name_of(label_of([57, 64, 69, 73], A)) == "I"            # A major in A major


def test_words_name_the_bass():
    assert words_of(label_of([47, 55, 62, 65], C), C) == (
        "dominant seventh chord on the dominant (G), first inversion, B in the bass")


def test_parallels():
    assert _parallels([48, 55, 64, 72], [50, 57, 65, 74])          # C-G up to D-A: parallel fifths and octaves
    assert not _parallels([43, 59, 62, 65], [48, 60, 64, 64])      # V7 -> I with contrary/oblique motion


def test_voicings_are_legal_and_ranked():
    prev = [43, 59, 62, 65]                                        # V7 in C: G2 B3 D4 F4
    vs = voicings(label_of([48, 52, 55, 60], C), C, prev)          # next chord: I
    assert vs and all(v[0] % 12 == 0 for v, _ in vs)               # C in the bass
    assert all({m % 12 for m in v} == {0, 4, 7} for v, _ in vs)    # every chord tone present
    assert all(not _parallels(prev, v) for v, _ in vs)
    assert [c for _, c in vs] == sorted(c for _, c in vs)          # smoothest first
