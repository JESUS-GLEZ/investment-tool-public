"""Owned valid-input image controls; no malformed native-input probes."""

from io import BytesIO

from matplotlib.figure import Figure
from PIL import Image, ImageMorph, PdfParser
import pytest


@pytest.mark.parametrize(
    "mode,color", [("RGB", (20, 40, 60)), ("RGBA", (20, 40, 60, 90))]
)
def test_owned_png_roundtrip(mode, color):
    output = BytesIO()
    Image.new(mode, (8, 6), color).save(output, format="PNG")
    output.seek(0)
    with Image.open(output) as restored:
        restored.load()
        assert restored.size == (8, 6)
        assert restored.mode == mode
        assert restored.getpixel((3, 2)) == color


@pytest.mark.parametrize("compression", ["raw", "tiff_lzw"])
def test_owned_tiff_roundtrip(compression):
    output = BytesIO()
    Image.new("RGB", (8, 6), (20, 40, 60)).save(
        output, format="TIFF", compression=compression
    )
    output.seek(0)
    with Image.open(output) as restored:
        restored.load()
        assert restored.size == (8, 6)
        assert restored.getpixel((3, 2)) == (20, 40, 60)


def test_owned_jpeg_draft_and_thumbnail():
    output = BytesIO()
    Image.new("RGB", (64, 48), (30, 60, 90)).save(output, format="JPEG", quality=95)
    output.seek(0)
    with Image.open(output) as restored:
        restored.draft("RGB", (16, 12))
        restored.load()
        restored.thumbnail((16, 12))
        assert restored.size == (16, 12)
        assert all(
            abs(a - b) <= 3 for a, b in zip(restored.getpixel((4, 4)), (30, 60, 90))
        )


def test_owned_multiframe_gif():
    output = BytesIO()
    first = Image.new("RGB", (8, 6), "red")
    second = Image.new("RGB", (8, 6), "blue")
    first.save(output, format="GIF", save_all=True, append_images=[second], duration=10)
    output.seek(0)
    with Image.open(output) as restored:
        assert restored.n_frames == 2
        assert restored.convert("RGB").getpixel((2, 2)) == (255, 0, 0)
        restored.seek(1)
        assert restored.convert("RGB").getpixel((2, 2)) == (0, 0, 255)


def test_owned_palette_tiff():
    image = Image.new("P", (8, 6), 1)
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * (256 * 3 - 6))
    output = BytesIO()
    image.save(output, format="TIFF")
    output.seek(0)
    with Image.open(output) as restored:
        assert restored.convert("RGB").getpixel((2, 2)) == (255, 0, 0)


def test_owned_pdf_append(tmp_path):
    target = tmp_path / "owned.pdf"
    Image.new("RGB", (8, 6), "red").save(target, format="PDF")
    initial_size = target.stat().st_size
    Image.new("RGB", (8, 6), "blue").save(target, format="PDF", append=True)
    assert target.read_bytes().startswith(b"%PDF-")
    assert target.stat().st_size > initial_size
    with PdfParser.PdfParser(filename=str(target)) as parsed:
        assert len(parsed.pages) == 2


def test_owned_nonempty_morphology():
    image = Image.new("L", (5, 5), 0)
    image.putpixel((2, 2), 1)
    _, result = ImageMorph.MorphOp(op_name="dilation8").apply(image)
    assert result.size == image.size
    assert sum(value != 0 for value in result.tobytes()) == 9
    assert set(result.tobytes()) <= {0, 255}


def test_owned_chart_png():
    figure = Figure(figsize=(2, 1), dpi=80)
    axis = figure.subplots()
    axis.plot([0, 1, 2], [1, 0, 1])
    output = BytesIO()
    figure.savefig(output, format="png")
    output.seek(0)
    with Image.open(output) as restored:
        restored.load()
        assert restored.size == (160, 80)
