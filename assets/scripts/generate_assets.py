"""Generate Noesis brand assets (icon, logo, banner x dark/light) as SVG.

Motif: abstracted owl mark. Owl = Athena's bird, classical emblem of
wisdom/insight, which is what "noesis" (Greek: direct intellectual
understanding) names. Rendered as a bold, flat, geometric silhouette (solid
fills, not illustrative/thin-line) so it holds up as a real app icon down to
16-32px, not just a hero illustration.

Run: python3 assets/scripts/generate_assets.py
Writes into ../ (repo assets/ dir) as .svg.
"""
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..")

PALETTES = {
    "dark": dict(
        bg0="#0a0c10", bg1="#14171e",
        accent="#e3a94f", accent2="#f6cd82",
        text="#f2ede2", text_dim="#a89a7e", dot="#4a3d26",
    ),
    "light": dict(
        bg0="#faf8f3", bg1="#f1ece1",
        accent="#b9791f", accent2="#8f5c12",
        text="#1c1a16", text_dim="#6b5f4c", dot="#ddccaa",
    ),
}


def owl_glyph(pal):
    """Owl mark in a local 512x512 design box, centered ~(256, 265)."""
    return (
        f'<path d="M 256 130 L 380 320 Q 256 400 132 320 Z" fill="{pal["accent"]}"/>'
        f'<circle cx="196" cy="230" r="66" fill="{pal["bg0"]}"/>'
        f'<circle cx="316" cy="230" r="66" fill="{pal["bg0"]}"/>'
        f'<rect x="184" y="200" width="24" height="60" rx="6" fill="{pal["accent2"]}"/>'
        f'<rect x="304" y="200" width="24" height="60" rx="6" fill="{pal["accent2"]}"/>'
        f'<path d="M 240 250 L 220 285 L 256 285 Z" fill="{pal["bg0"]}"/>'
    )


def place(glyph, cx, cy, s):
    tx, ty = cx - 256 * s, cy - 256 * s
    return f'<g transform="translate({tx:.1f},{ty:.1f}) scale({s:.4f})">{glyph}</g>'


def svg_header(w, h, extra_defs=""):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs>{extra_defs}</defs>'


def make_icon(variant):
    pal = PALETTES[variant]
    size = 512
    bg_defs = (
        f'<radialGradient id="bg-{variant}" cx="50%" cy="40%" r="75%">'
        f'<stop offset="0%" stop-color="{pal["bg1"]}"/><stop offset="100%" stop-color="{pal["bg0"]}"/>'
        f'</radialGradient>'
    )
    body = (
        f'<rect width="{size}" height="{size}" rx="96" fill="url(#bg-{variant})"/>'
        + place(owl_glyph(pal), 256, 256, 1.0)
    )
    return svg_header(size, size, bg_defs) + body + "</svg>"


def make_logo(variant):
    pal = PALETTES[variant]
    w, h = 760, 220
    body = (
        f'<rect width="{w}" height="{h}" fill="{pal["bg0"]}"/>'
        + place(owl_glyph(pal), 110, 110, 0.42)
        + f'<text x="228" y="130" font-family="Helvetica Neue, Arial, sans-serif" '
        f'font-size="72" font-weight="600" letter-spacing="2" fill="{pal["text"]}">noesis</text>'
    )
    return svg_header(w, h) + body + "</svg>"


def make_banner(variant):
    pal = PALETTES[variant]
    w, h = 1600, 800
    bg_defs = (
        f'<radialGradient id="bbg-{variant}" cx="30%" cy="45%" r="80%">'
        f'<stop offset="0%" stop-color="{pal["bg1"]}"/><stop offset="100%" stop-color="{pal["bg0"]}"/>'
        f'</radialGradient>'
    )
    dots = []
    seed = 7
    for i in range(46):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        x = 60 + (seed % (w - 120))
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        y = 40 + (seed % (h - 80))
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        r = 1 + (seed % 3)
        dots.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{pal["dot"]}" opacity="0.5"/>')

    body = (
        f'<rect width="{w}" height="{h}" fill="url(#bbg-{variant})"/>'
        + "".join(dots)
        + place(owl_glyph(pal), 400, 400, 1.15)
        + f'<text x="760" y="410" font-family="Helvetica Neue, Arial, sans-serif" '
        f'font-size="128" font-weight="600" letter-spacing="3" fill="{pal["text"]}">Noesis</text>'
        + f'<text x="764" y="470" font-family="Helvetica Neue, Arial, sans-serif" '
        f'font-size="34" font-weight="400" letter-spacing="1" fill="{pal["text_dim"]}">Beyond search. Toward understanding.</text>'
    )
    return svg_header(w, h, bg_defs) + body + "</svg>"


def write(name, content):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(content)
    print("wrote", path)


if __name__ == "__main__":
    for variant in ("dark", "light"):
        write(f"noesis-icon-{variant}.svg", make_icon(variant))
        write(f"noesis-logo-{variant}.svg", make_logo(variant))
        write(f"noesis-banner-{variant}.svg", make_banner(variant))
