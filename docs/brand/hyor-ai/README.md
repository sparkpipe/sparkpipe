# HYOR AI logo

The logo is the wordmark **HYOR AI** followed by the **효 seal**. 효 (Hyo, 孝) is the first syllable of Hyoryeong (孝寧), drawn as a network of lines and nodes and stamped after the name the way a seal follows a signature.

The letters are custom-drawn shapes, not a font. Always use these files; never retype the name in a font to stand in for the logo.

## Which file to use

| Folder | File | Use it for |
|---|---|---|
| `logo/` | `hyor-ai-logo-color` | Default. White or light backgrounds. |
| `logo/` | `hyor-ai-logo-reversed` | Dark backgrounds. |
| `logo/` | `hyor-ai-logo-black` | One-color printing, stamps, engraving, embroidery, documents. |
| `logo/` | `hyor-ai-logo-white` | One-color on dark or photo backgrounds. |
| `seal/` | `hyor-seal`, `-1024.png`, `-512.png` | Social media avatars, app icons, square spaces. |
| `seal/` | `apple-touch-icon-180.png` | Website icon on iPhone and iPad home screens. |
| `seal/` | `hyor-seal-black`, `hyor-seal-white` | One-color seal. |
| `favicon/` | `favicon.ico`, `favicon.svg`, PNGs | Browser tab icon. Uses the solid cut of the glyph, which stays clear at 16 px. |

Use `.svg` wherever it is accepted (websites, print shops, design tools); it stays sharp at any size. The `.png` files have transparent backgrounds.

## Colors

| Name | Hex | RGB |
|---|---|---|
| Ink | `#111522` | 17, 21, 34 |
| Seal red | `#D63A26` | 214, 58, 38 |
| White | `#FFFFFF` | 255, 255, 255 |

On sparkpipe.ai, small text in seal red uses the darker `#B8301D` on light pages and `#F0624C` on dark ones, so it stays readable; the seal itself is always `#D63A26`.

## Spacing and size

- **Clear space:** keep empty space on every side of the logo at least half the height of the seal.
- **Minimum size:** full logo 120 px wide on screen, 30 mm wide in print. Seal alone 24 px. Smaller than that, use the favicon.

## Please don't

- Change the colors, or make the HYOR AI letters red.
- Stretch, rotate, outline, or add shadows or effects.
- Move the seal to another position or resize it relative to the letters.
- Put the color logo on a busy photo; use the white version.

## sparkpipe.ai

The site's header carries the logo with no clear space (`site/hyor-ai-logo.svg`, and `site/hyor-ai-logo-reversed.svg` for dark mode), followed by the product name SparkPipe. Below 370 px wide, pages with buttons in the header show the seal alone (`site/hyor-seal.svg`). The footer stamps the seal before the copyright line. The site's icons are:

```html
<link rel="icon" href="/favicon.ico" sizes="32x32">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
```

SparkPipe Chat keeps the SparkPipe bolt (`deployment/hub/open_webui_branding/`).

## Regenerating

`tools/brand_logo.py` holds the letter and glyph geometry and writes every file here and in `site/`. It renders the PNGs with headless Google Chrome; set `CHROME` to its binary when it is not in `/Applications`.

```sh
python3 tools/brand_logo.py
```
