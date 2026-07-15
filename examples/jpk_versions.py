"""JPK / Bruker command dialects — one profile per software generation.

JPK's scripting grammar differs across software versions. Each generation is a
swappable ``JPKDialect``; the adapter auto-detects the version from the handshake
and selects the matching profile, or you can force one. The concrete verb
strings are placeholders (``verified=False``) — paste the real grammar for your
version into ``jpk_dialects.py``.

    python examples/jpk_versions.py
"""

from nanoindent.adapters import (BackendSimulator, JPKAdapter, MockJPKConnection,
                                 DIALECTS, select_dialect)


def main():
    print("Registered dialects (one per JPK/Bruker software generation):")
    for name, d in DIALECTS.items():
        flag = "verified" if d.verified else "PLACEHOLDER — fill from manual"
        print(f"  {name:16s} {d.versions:52s} [{flag}]")

    print("\nVersion → dialect mapping (heuristic, VERIFY):")
    for v in ["NanoWizard 3.2 / JPK SPM 3.4", "NanoWizard 4.2 (JPK SPM 5.1)",
              "Bruker BioAFM 2.1 / NanoRacer", "some unknown instrument"]:
        print(f"  {v:36s} -> {select_dialect(v).name}")

    print("\nAuto-detection through the adapter handshake:")
    for v in ["NanoWizard 4.2", "Bruker BioAFM 2.1"]:
        b = BackendSimulator(scan_px=32, max_frames=1)
        ad = JPKAdapter(MockJPKConnection(b, version=v), max_frames=1,
                        dialect="auto")
        list(ad)                                   # connect + handshake
        print(f"  instrument reports {v!r:20s} -> adapter uses "
              f"'{ad.dialect.name}'")

    print("\nForcing a dialect explicitly:")
    b = BackendSimulator(scan_px=32, max_frames=1)
    ad = JPKAdapter(MockJPKConnection(b), max_frames=1, dialect="bruker_bioafm")
    list(ad)
    print(f"  dialect='bruker_bioafm' -> {ad.dialect.name}")


if __name__ == "__main__":
    main()
