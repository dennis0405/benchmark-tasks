"""Print artwork for the 1-gallon paint can: a made-up paint maker ("Fennwick Paint Co."), not a real
trademark. Rendered with Pillow from free fonts (Liberation: SIL OFL 1.1; DejaVu: Bitstream Vera licence)
at 3 px/mm. Writes textures/can_label.png (the wrap-around body label, 529 x 135 mm) and
textures/lid_sticker.png (the tint-recipe sticker on the lid, 60 x 40 mm).

Run once (with Pillow and the fonts available):  python3 make_textures.py [font_dir]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
FONTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/usr/share/fonts/truetype")
PX = 3                                   # px per mm
CIRC_MM = 529                            # 2 pi x 84.2 mm
LABEL_MM = 135
NAVY = (22, 40, 82)
SWATCH = (58, 104, 150)                  # "Harbor Blue"
CREAM = (246, 243, 234)
INK = (30, 30, 34)
GREY = (112, 112, 118)
RED = (190, 36, 40)
GOLD = (214, 168, 48)


def font(name, size):
    for p in FONTS.rglob(name):
        return ImageFont.truetype(str(p), size)
    raise FileNotFoundError(name)


SERIF_B = lambda s: font("LiberationSerif-Bold.ttf", s)
SERIF_BI = lambda s: font("LiberationSerif-BoldItalic.ttf", s)
SANS_B = lambda s: font("LiberationSans-Bold.ttf", s)
SANS = lambda s: font("LiberationSans-Regular.ttf", s)
NARROW_B = lambda s: font("LiberationSansNarrow-Bold.ttf", s)
MONO = lambda s: font("DejaVuSansMono.ttf", s)


def mm(v):
    return int(round(v * PX))


def panel(d, x0, w):
    """One front panel (the label repeats it twice round the can, as most gallon labels do)."""
    H = mm(LABEL_MM)
    # navy top band with the maker's wordmark
    d.rectangle([x0, 0, x0 + w, mm(30)], fill=NAVY)
    d.text((x0 + w // 2, mm(12)), "Fennwick", font=SERIF_BI(mm(17)), fill=CREAM, anchor="mm")
    d.text((x0 + w // 2, mm(24.5)), "PAINT CO.  ·  EST. 1952", font=SANS_B(mm(4.2)), fill=GOLD, anchor="mm")
    # gold rule
    d.rectangle([x0, mm(30), x0 + w, mm(31.5)], fill=GOLD)
    # product line
    d.text((x0 + w // 2, mm(40)), "INTERIOR LATEX", font=SANS_B(mm(7.5)), fill=NAVY, anchor="mm")
    d.text((x0 + w // 2, mm(48.5)), "Paint + Primer  ·  EGGSHELL", font=SANS(mm(5)), fill=INK, anchor="mm")
    # colour swatch with its name
    sw = (x0 + w // 2 - mm(45), mm(55), x0 + w // 2 + mm(45), mm(88))
    d.rounded_rectangle(sw, radius=mm(3), fill=SWATCH, outline=NAVY, width=mm(0.8))
    d.text((x0 + w // 2, mm(66)), "HARBOR BLUE", font=SANS_B(mm(6.5)), fill=CREAM, anchor="mm")
    d.text((x0 + w // 2, mm(78)), "Colour no. 4417-B  ·  Tinted to order", font=SANS(mm(3.6)), fill=CREAM, anchor="mm")
    # feature bullets
    y = mm(95)
    for line in ("Low odour  ·  Zero VOC colorant", "Washable  ·  Scuff resistant", "Covers up to 400 sq ft / 37 m²"):
        d.text((x0 + w // 2, y), line, font=SANS(mm(3.6)), fill=INK, anchor="mm")
        y += mm(5.5)
    # bottom band: volume
    d.rectangle([x0, mm(114), x0 + w, H], fill=NAVY)
    d.text((x0 + w // 2, mm(124.5)), "1 GALLON  (3.78 L)", font=NARROW_B(mm(9)), fill=CREAM, anchor="mm")


def side_panel(d, x0, w):
    """A narrow side panel: directions, warnings, barcode, lot code."""
    H = mm(LABEL_MM)
    d.rectangle([x0, 0, x0 + w, mm(9)], fill=NAVY)
    d.text((x0 + mm(4), mm(4.5)), "DIRECTIONS", font=SANS_B(mm(4)), fill=CREAM, anchor="lm")
    txt = ("Stir well before use. Apply with brush, roller or",
           "airless sprayer. Recoat in 2 hours. Clean tools",
           "with soap and water. Close lid tightly: set the lid",
           "on squarely and tap all round the rim with a",
           "rubber mallet. Store above 35 °F / 2 °C.",
           "",
           "CAUTION: Keep out of reach of children. Use only",
           "with adequate ventilation. If swallowed, call a",
           "poison control centre. Do not pour down drains.")
    y = mm(14)
    for line in txt:
        d.text((x0 + mm(4), y), line, font=SANS(mm(3.0)), fill=INK, anchor="lm")
        y += mm(4.3)
    # barcode
    rnd = random.Random(7)
    bx, by, bh = x0 + mm(6), mm(70), mm(22)
    x = bx
    while x < bx + mm(48):
        wbar = rnd.choice((1, 1, 2, 3)) * PX // 2 + 1
        d.rectangle([x, by, x + wbar, by + bh], fill=INK)
        x += wbar + rnd.choice((1, 2, 2, 3)) * PX // 2 + 1
    d.text((bx + mm(24), by + bh + mm(4)), "0 84417 20916 3", font=MONO(mm(3.4)), fill=INK, anchor="mm")
    d.text((x0 + mm(4), mm(104)), "LOT 26-311-B  ·  BASE 2", font=MONO(mm(3.2)), fill=GREY, anchor="lm")
    d.rectangle([x0, mm(114), x0 + w, H], fill=NAVY)
    d.text((x0 + w // 2, mm(124.5)), "fennwickpaint.example", font=SANS(mm(3.8)), fill=CREAM, anchor="mm")


def label():
    W, H = mm(CIRC_MM), mm(LABEL_MM)
    img = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(img)
    front_w, side_w = mm(160), mm(CIRC_MM / 2 - 160)
    for k in range(2):
        x0 = k * mm(CIRC_MM / 2)
        panel(d, x0, front_w)
        side_panel(d, x0 + front_w, side_w)
        d.line([x0 + front_w, mm(31.5), x0 + front_w, mm(114)], fill=GREY, width=1)
    return img


def sticker():
    W, H = mm(60), mm(40)
    img = Image.new("RGB", (W, H), (252, 252, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, mm(8)], fill=SWATCH)
    d.text((mm(3), mm(4)), "HARBOR BLUE  4417-B", font=SANS_B(mm(3.6)), fill=CREAM, anchor="lm")
    rows = (("COLORANT", "OZ  48  96"), ("B  Black", "0   12   1"), ("E  Blue", "1   08   0"),
            ("L  Raw Umber", "0   03   1"), ("BASE 2   EGGSHELL   1 GAL", ""))
    y = mm(12)
    for a, b in rows:
        d.text((mm(3), y), a, font=MONO(mm(2.8)), fill=INK, anchor="lm")
        d.text((W - mm(3), y), b, font=MONO(mm(2.8)), fill=INK, anchor="rm")
        y += mm(5)
    d.rectangle([0, H - mm(3), W, H], fill=SWATCH)
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    label().save(OUT / "can_label.png")
    sticker().save(OUT / "lid_sticker.png")
    print("wrote", OUT / "can_label.png", OUT / "lid_sticker.png")
