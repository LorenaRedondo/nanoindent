"""Generate the contact-model slide deck: comparison figure + editable .pptx.

Builds
  * results/model_comparison_sample.png -- F(delta) for the four models on the
    actual sample (R=20 nm, h=6.2 nm, E=50 MPa, nu=0.5);
  * docs/models_deck.pptx -- an editable PowerPoint deck with the E formulas,
    particularities/constraints, the comparison figure, and an at-a-glance table
    for Hertz, Dimitriadis 2002 (non-bonded), Chadwick 2002 (SIAM), and
    Garcia & Garcia 2018.

Requires the [docs] extra:  pip install "python-pptx>=1.0" matplotlib

Usage
-----
    python examples/model_slides.py [--outdir results] [--pptx docs/models_deck.pptx]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoindent import analytics

# sample (the MSNL-BIO-DC case)
R, E, H, NU = 20.0, 50.0, 6.2, 0.5

# theme (matches the HTML deck)
INK = "EAF1F6"; DIM = "9FB2C0"; ACCENT = "3AA6E0"; WARM = "F08A5D"
BG = "0D1117"; PANEL = "18222D"; GOOD = "5EC6A0"; VIOLET = "B39DDB"


# ==========================================================================
# comparison figure
# ==========================================================================
def comparison_figure(path):
    d = np.linspace(0, 5.0, 200)
    fh = analytics.hertz_force(d, R, E, NU)
    fdf = analytics.dimitriadis_force(d, R, E, NU, H, bonded=False)
    fga = analytics.garcia_bec_force(d, R, E, NU, H)
    fch = analytics.chadwick_force(d, R, E, H)
    d_valid = H ** 2 / R          # delta at which chi = sqrt(R*delta)/h = 1
    ymax = 22000.0

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    # BEC-polynomial validity limit (chi <= 1)
    ax.axvspan(d_valid, 5, color="#9aa5ad", alpha=0.10, lw=0)
    ax.axvline(d_valid, color="#777777", ls=(0, (4, 3)), lw=1.2)
    ax.text(d_valid - 0.08, ymax * 0.5,
            "χ = 1  —  BEC polynomials valid to the left; extrapolated in the "
            "shaded region", rotation=90, fontsize=9, color="#555555",
            va="center", ha="right")
    ax.plot(d, fh, color="#111111", ls=":", lw=2, label="Hertz (half-space)")
    ax.plot(d, fdf, color="#0072B2", lw=2.4,
            label="Dimitriadis 2002 (non-bonded)")
    ax.plot(d, fga, color="#009E73", lw=2.4, label="Garcia & Garcia 2018 (bonded)")
    ax.plot(d, fch, color="#D55E00", ls="--", lw=2.4,
            label="Chadwick 2002 (thin incompr., bonded)")
    ax.set_xlabel("indentation  δ  (nm)")
    ax.set_ylabel("force  F  (pN)")
    ax.set_xlim(0, 5); ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.5, framealpha=0.96, loc="upper left")
    ax.set_title(f"Four models on the same sample:  R = {R:g} nm,  "
                 f"h = {H:g} nm,  E = {E:g} MPa,  ν = {NU:g}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ==========================================================================
# pptx deck
# ==========================================================================
def build_pptx(path, fig_path):
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    W, Hh = prs.slide_width, prs.slide_height

    def rgb(hexs):
        return RGBColor.from_string(hexs)

    def slide():
        s = prs.slides.add_slide(blank)
        bg = s.background.fill
        bg.solid(); bg.fore_color.rgb = rgb(BG)
        return s

    def box(s, l, t, w, h):
        tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        return tb, tf

    def run(p, text, size, color, bold=False, italic=False, font="Calibri"):
        r = p.add_run(); r.text = text
        f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic
        f.name = font; f.color.rgb = rgb(color)
        return r

    def eyebrow(s, text):
        _, tf = box(s, 0.7, 0.5, 12, 0.5)
        p = tf.paragraphs[0]
        run(p, text.upper(), 13, ACCENT, bold=True, font="Consolas")

    def title(s, text, size=34):
        _, tf = box(s, 0.7, 0.95, 12, 1.3)
        p = tf.paragraphs[0]
        run(p, text, size, INK, bold=True)

    def formula(s, l, t, w, label, parts):
        """parts: list of (text, color, italic) rendered as one line."""
        tb, tf = box(s, l, t, w, 1.0)
        tb.fill.solid(); tb.fill.fore_color.rgb = rgb(PANEL)
        tb.line.color.rgb = rgb("26333F"); tb.line.width = Pt(1)
        tf.margin_left = Inches(0.18); tf.margin_top = Inches(0.1)
        tf.margin_bottom = Inches(0.1)
        p0 = tf.paragraphs[0]
        run(p0, label.upper(), 10.5, DIM, bold=True, font="Consolas")
        p = tf.add_paragraph()
        for text, color, ital in parts:
            run(p, text, 21, color, italic=ital, font="Cambria")
        return tb

    def bullets(s, l, t, w, h, header, items):
        _, tf = box(s, l, t, w, h)
        p0 = tf.paragraphs[0]
        run(p0, header.upper(), 11, DIM, bold=True, font="Consolas")
        for text, warn in items:
            p = tf.add_paragraph(); p.space_before = Pt(9)
            run(p, "◆  ", 13, WARM if warn else ACCENT)
            # simple **bold** markup
            for j, seg in enumerate(text.split("**")):
                run(p, seg, 15.5, INK, bold=(j % 2 == 1))

    # ---- 0 · title ------------------------------------------------------
    s = slide()
    eyebrow(s, "Contact mechanics · AFM nanoindentation")
    _, tf = box(s, 0.7, 2.0, 12, 2.2)
    run(tf.paragraphs[0], "Elastic-modulus models for", 44, INK, bold=True)
    p = tf.add_paragraph(); run(p, "thin-film nanoindentation", 44, INK, bold=True)
    _, tf2 = box(s, 0.7, 4.3, 11.5, 1.2)
    run(tf2.paragraphs[0],
        "Four ways to turn a force–indentation curve F(δ) into a Young's "
        "modulus E — and the assumptions that decide which one is valid.",
        18, DIM)
    _, tf3 = box(s, 0.7, 6.4, 12, 0.6)
    run(tf3.paragraphs[0],
        "paraboloid tip   ·   χ = √(Rδ)/h   ·   E★ = E/(1−ν²)",
        13, DIM, font="Consolas")

    # ---- 1 · framework --------------------------------------------------
    s = slide(); eyebrow(s, "The shared picture"); title(s, "One curve, four inversions")
    formula(s, 0.7, 2.2, 5.9, "Reduced (contact) modulus",
            [("E★ = E / (1 − ν²)", INK, True)])
    formula(s, 0.7, 3.5, 5.9, "Bottom-effect variable",
            [("χ = √(R δ) / h", INK, True)])
    bullets(s, 7.0, 2.15, 5.6, 4.5, "Why thickness matters", [
        ("A **rigid substrate** under a thin film makes it feel **stiffer** "
         "than a half-space — the bottom effect.", False),
        ("Ignoring it (bare Hertz) **over-estimates E**, more so as the film "
         "thins (χ grows).", False),
        ("The **substrate bond** matters: a bonded incompressible film is far "
         "stiffer than one **free to slide**.", False),
    ])

    # ---- 2 · Hertz ------------------------------------------------------
    s = slide(); eyebrow(s, "Hertz · 1882 · elastic half-space")
    title(s, "Hertz — the half-space baseline")
    formula(s, 0.7, 2.15, 5.9, "Force",
            [("F = (4/3) E★ √R δ", INK, True), ("3/2", INK, True)])
    formula(s, 0.7, 3.45, 5.9, "Solved for the modulus",
            [("E = 3(1−ν²) F / (4 √R δ", INK, True), ("3/2", INK, True),
             (")", INK, True)])
    bullets(s, 7.0, 2.1, 5.6, 4.7, "Particularities & constraints", [
        ("**Semi-infinite half-space** — no thickness, no substrate. The "
         "reference all others correct.", False),
        ("Linear-elastic, **isotropic, homogeneous**; small strain.", False),
        ("Small contact: **δ ≪ R**, frictionless, **no adhesion**.", False),
        ("**Over-estimates E** for thin films — misses the bottom effect.", True),
    ])

    # ---- 3 · Dimitriadis non-bonded ------------------------------------
    s = slide(); eyebrow(s, "Dimitriadis et al. · Biophys. J. 82, 2798 · 2002 · non-bonded")
    title(s, "Dimitriadis 2002 — thin film, free base", size=30)
    formula(s, 0.7, 2.15, 5.9, "Force — Hertz × BEC polynomial",
            [("F = (4/3) E★ √R δ", INK, True), ("3/2", INK, True),
             (" · Φ", INK, True), ("free", DIM, True), ("(χ)", INK, True)])
    formula(s, 0.7, 3.45, 5.9, "Free-base coefficients (→ ν = ½)",
            [("α₀ = −(0.347 − 0.056 ν) → −0.319,  β₀ = 0.164", INK, True)])
    bullets(s, 7.0, 2.1, 5.6, 4.7, "Particularities & constraints", [
        ("**Finite thickness h** on a substrate the film **does not adhere "
         "to** — free to slide (frictionless base).", False),
        ("BEC quartic in χ = √(Rδ)/h; **reduces to Hertz** as h → ∞.", False),
        ("Valid only for **small χ ≲ 1** (a ≲ h); degrades beyond.", True),
        ("**Softer** than the bonded branch — relieves confinement by sliding.",
         False),
    ])

    # ---- 4 · Chadwick ---------------------------------------------------
    s = slide(); eyebrow(s, "Chadwick · SIAM J. Appl. Math. 62, 1520 · 2002")
    title(s, "Chadwick 2002 — thin incompressible bonded layer", size=28)
    formula(s, 0.7, 2.15, 5.9, "Force — thin-film limit, ν = ½",
            [("F = (2π/3) E R² δ³ / h³", INK, True)])
    formula(s, 0.7, 3.45, 5.9, "Solved for the modulus",
            [("E = 3 F h³ / (2π R² δ³)", INK, True)])
    bullets(s, 7.0, 2.1, 5.6, 4.7, "Particularities & constraints", [
        ("**Opposite limit to Hertz**: valid when contact radius **a ≫ h** "
         "(genuinely thin film).", False),
        ("Strictly **incompressible (ν = ½)** and **bonded** to the substrate.",
         False),
        ("Strong **1/h³ stiffening**; force scales as **δ³**, not δ^{3/2}.",
         False),
        ("**Does not reduce to Hertz** — thin-film regime only.", True),
    ])

    # ---- 5 · Garcia & Garcia -------------------------------------------
    s = slide(); eyebrow(s, "Garcia & Garcia · Biophys. J. 114, 2923 · 2018 · bonded BEC")
    title(s, "Garcia & Garcia 2018 — paraboloid BEC", size=30)
    formula(s, 0.7, 2.15, 5.9, "Force — Hertz × bonded BEC polynomial",
            [("F = (4/3) E★ √R δ", INK, True), ("3/2", INK, True),
             (" · Σ cₖ χ", INK, True), ("k", INK, True)])
    formula(s, 0.7, 3.45, 5.9, "Coefficients",
            [("c = [1, 1.133, 1.497, 1.469, 0.755],  χ = √(Rδ)/h", INK, True)])
    bullets(s, 7.0, 2.1, 5.6, 4.7, "Particularities & constraints", [
        ("**Practical BEC** for cells / films **bonded to a rigid support** — "
         "the workhorse paraboloid correction.", False),
        ("Keeps **E★ = E/(1−ν²)** → works for **arbitrary ν**, not just ½.",
         False),
        ("**Reduces to Hertz** (h→∞) and captures thin-film stiffening — wider "
         "δ/h range than bare Hertz.", False),
        ("Assumes **bonded** base, frictionless contact, small strain.", False),
    ])

    # ---- 6 · comparison figure -----------------------------------------
    s = slide(); eyebrow(s, "Same sample, four models")
    title(s, f"F(δ) for R={R:g} nm, h={H:g} nm, E={E:g} MPa", size=28)
    from PIL import Image
    iw, ih = Image.open(fig_path).size
    disp_w = Inches(8.3); disp_h = Emu(int(disp_w * ih / iw))
    left = Inches(0.7); top = Inches(2.05)
    s.shapes.add_picture(fig_path, left, top, width=disp_w)
    bullets(s, 9.2, 2.15, 3.6, 4.7, "Reading it", [
        ("Same **E and geometry** — the spread is purely the **model "
         "assumption**.", False),
        ("**Dimitriadis-free** sits lowest (softest), **bonded** models "
         "highest.", False),
        ("**Chadwick** (δ³) overtakes at large δ; no Hertz limit.", False),
        ("The true response is **bracketed** by free and bonded.", False),
    ])

    # ---- 7 · at a glance table -----------------------------------------
    s = slide(); eyebrow(s, "At a glance"); title(s, "Which model, when")
    rows = [
        ("Model", "Force law", "Regime", "Substrate", "ν", "Watch out"),
        ("Hertz (1882)", "F ∝ E★√R δ^1.5", "Half-space (h→∞)", "none", "any",
         "Over-estimates E on thin films"),
        ("Dimitriadis 2002 (non-bonded)", "Hertz × Φ_free(χ)", "Thin, χ ≲ 1",
         "free / slip", "general", "Degrades for χ > 1"),
        ("Chadwick 2002 (SIAM)", "F ∝ E R²δ³/h³", "Thin, a ≫ h", "bonded",
         "½ only", "No Hertz limit; thin-film only"),
        ("Garcia & Garcia 2018", "Hertz × Σcₖχ^k", "Thin → thick", "bonded",
         "general", "Assumes bonded hard support"),
    ]
    tshape = s.shapes.add_table(len(rows), 6, Inches(0.7), Inches(2.2),
                                Inches(12.0), Inches(4.4)).table
    widths = [2.7, 2.2, 2.1, 1.7, 1.1, 2.2]
    for j, wv in enumerate(widths):
        tshape.columns[j].width = Inches(wv)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tshape.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(PANEL if i else "223140")
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run(); r.text = val
            r.font.size = Pt(11 if i else 10.5)
            r.font.bold = (i == 0 or j == 0)
            r.font.name = "Consolas" if (i and j in (1,)) else "Calibri"
            r.font.color.rgb = rgb(DIM if i == 0 else INK)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    prs.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--pptx", default="docs/models_deck.pptx")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fig = comparison_figure(os.path.join(args.outdir, "model_comparison_sample.png"))
    print(f"wrote {fig}")
    deck = build_pptx(args.pptx, fig)
    print(f"wrote {deck}")


if __name__ == "__main__":
    main()
