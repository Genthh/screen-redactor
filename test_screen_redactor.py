"""
Tests for screen_redactor.py

    python3 -m pip install pytest pillow
    python3 -m pytest -v test_screen_redactor.py

Layers
  1. Detection, geometry, CLI           - pure Python, run on any OS.
  2. OCR on a rendered image            - macOS only (Apple Vision).
  3. End-to-end on the real screen      - macOS only, needs Screen Recording
     permission for your terminal. Opens a window showing a FAKE key, starts
     the redactor, takes a real `screencapture` (what Loom / screen share would
     see) and checks the key is unreadable while the rest of the window is not.
     Don't touch the mouse/keyboard for the ~15 s it takes.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import screen_redactor as sr  # noqa: E402

SCRIPT = HERE / "screen_redactor.py"
IS_MAC = sys.platform == "darwin"
mac_only = pytest.mark.skipif(not IS_MAC, reason="needs macOS")


# =========================================================================== #
# 1. Detection
# =========================================================================== #

@pytest.fixture(scope="module")
def det():
    return sr.Detector()


@pytest.mark.parametrize("line", sr.SHOULD_HIT)
def test_detects_secret(det, line):
    assert det.spans(line), f"secret not detected in: {line!r}"


@pytest.mark.parametrize("line", sr.SHOULD_MISS)
def test_ignores_normal_text(det, line):
    assert det.spans(line) == [], f"false positive in: {line!r}"


@pytest.mark.parametrize("line, value", [
    ("OPENAI_API_KEY=sk-proj-AbCdEf1234567890GhIjKlMnOp", "sk-proj-AbCdEf1234567890GhIjKlMnOp"),
    ('"sw-access-key: SWIAQXNHAEZBNLHMWJJUANFAAQ",', "SWIAQXNHAEZBNLHMWJJUANFAAQ"),
    ("Authorization: Bearer abcDEF123456ghiJKL789", "abcDEF123456ghiJKL789"),
    ("aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ("SLACK_BOT_TOKEN=xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij"),
])
def test_covers_value_exactly_not_label(det, line, value):
    """The whole secret is covered and the label next to it stays readable."""
    start = line.index(value)
    spans = det.spans(line)
    assert len(spans) == 1
    s, e, _ = spans[0]
    assert (s, e) == (start, start + len(value))


def test_multiple_secrets_on_one_line(det):
    line = "key=AKIAIOSFODNN7EXAMPLE secret=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    covered = "".join("X" if any(s <= i < e for s, e, _ in det.spans(line)) else c
                      for i, c in enumerate(line))
    assert "AKIA" not in covered and "wJalr" not in covered
    assert covered.startswith("key=") and " secret=" in covered


def test_shopware_mcp_config_from_screenshot(det):
    """The config block shown in the original recording."""
    block = textwrap.dedent("""\
        "mcpServers": {
          "shopware": {
            "command": "npx",
            "args": [
              "-y",
              "mcp-remote@0.8.4",
              "https://shopware-store.ddev.site/api/_mcp",
              "--transport",
              "http-only",
              "--header",
              "sw-access-key: SWIAQXNHAEZBNLHMWJJUANFAAQ",
              "--header",
              "sw-secret-access-key: dGhpc2lzYXZlcnlzZWNyZXRrZXkxMjM0"
            ]""")
    flagged = [ln.strip() for ln in block.splitlines() if det.spans(ln)]
    assert flagged == ['"sw-access-key: SWIAQXNHAEZBNLHMWJJUANFAAQ",',
                       '"sw-secret-access-key: dGhpc2lzYXZlcnlzZWNyZXRrZXkxMjM0"']


def test_custom_pattern():
    d = sr.Detector(extra_patterns=[r"acme_[a-z]{8}"])
    assert d.spans("deploy key acme_abcdefgh")
    assert sr.Detector().spans("deploy key acme_abcdefgh") == []


def test_ignore_pattern():
    d = sr.Detector(ignore_patterns=[r"^AKIAIOSFODNN7EXAMPLE$"])
    assert d.spans("id AKIAIOSFODNN7EXAMPLE") == []


def test_entropy_rule_can_be_disabled():
    line = "key 9f8e7d6c5b4a39281706f5e4d3c2b1a0Zz"
    assert sr.Detector().spans(line)
    assert sr.Detector(entropy=False).spans(line) == []


def test_overlapping_matches_are_merged(det):
    # sk-ant- matches both the anthropic and openai rules; must be one span.
    assert len(det.spans("sk-ant-api03-AbCdEf1234567890GhIjKl")) == 1


def test_detection_is_fast_enough(det):
    """Detection runs on every OCR'd line of every frame."""
    lines = (sr.SHOULD_HIT + sr.SHOULD_MISS) * 100   # ~2,200 lines, a very busy screen
    t = time.perf_counter()
    for ln in lines:
        det.spans(ln)
    assert time.perf_counter() - t < 1.0


# =========================================================================== #
# 1b. Geometry
# =========================================================================== #

def test_utf16_range_ascii():
    assert sr.utf16_range("token: abc", 7, 10) == (7, 3)


def test_utf16_range_after_emoji():
    text = "\U0001F511 token: abcdef123456XYZ"   # emoji is 1 Python char, 2 UTF-16 units
    s = text.index("abcdef")
    loc, length = sr.utf16_range(text, s, len(text))
    assert loc == s + 1 and length == len("abcdef123456XYZ")


def test_to_points_scales_and_pads():
    assert sr.to_points((0.5, 0.25, 0.1, 0.05), (1000, 800), 4) == [496, 196, 108, 48]


def test_overlap():
    a = [0, 0, 10, 10]
    assert sr.overlap(a, [0, 0, 10, 10]) == 1
    assert sr.overlap(a, [20, 20, 5, 5]) == 0
    assert sr.overlap(a, [5, 0, 10, 10]) == pytest.approx(0.5)
    assert sr.overlap(a, [2, 2, 2, 2]) == 1      # small box fully inside


SIZE = (1000, 800)
HIT = (0.5, 0.5, 0.2, 0.03, "openai")


def test_box_appears_for_new_hit():
    boxes, rects = sr.update_boxes([], [HIT], SIZE, now=0.0)
    assert len(rects) == 1


def test_same_hit_refreshes_instead_of_duplicating():
    boxes, _ = sr.update_boxes([], [HIT], SIZE, now=0.0)
    boxes, rects = sr.update_boxes(boxes, [HIT], SIZE, now=0.5)
    assert len(rects) == 1 and boxes[0][4] == 0.5


def test_box_is_held_then_released():
    boxes, _ = sr.update_boxes([], [HIT], SIZE, now=0.0, hold=1.2)
    boxes, rects = sr.update_boxes(boxes, [], SIZE, now=1.0, hold=1.2)
    assert len(rects) == 1, "box should survive a missed frame (OCR flicker / scrolling)"
    boxes, rects = sr.update_boxes(boxes, [], SIZE, now=1.5, hold=1.2)
    assert rects == [], "box should disappear once the secret is gone"


def test_scrolled_secret_covers_old_and_new_position():
    boxes, _ = sr.update_boxes([], [HIT], SIZE, now=0.0)
    moved = (0.5, 0.40, 0.2, 0.03, "openai")
    boxes, rects = sr.update_boxes(boxes, [moved], SIZE, now=0.2)
    assert len(rects) == 2


def test_update_boxes_does_not_mutate_input():
    boxes, _ = sr.update_boxes([], [HIT], SIZE, now=0.0)
    snapshot = [list(b) for b in boxes]
    sr.update_boxes(boxes, [HIT], SIZE, now=0.5)
    assert boxes == snapshot


# =========================================================================== #
# 1c. CLI
# =========================================================================== #

def run_cli(*args, timeout=30):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=timeout)


def test_selftest_passes():
    r = run_cli("--selftest")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all passed" in r.stdout


def test_selftest_output_never_contains_raw_secrets():
    out = run_cli("--selftest").stdout
    for secret in ["SWIAQXNHAEZBNLHMWJJUANFAAQ", "sk-proj-AbCdEf1234567890GhIjKlMnOp",
                   "AKIAIOSFODNN7EXAMPLE", "hunter2hunter2", "s3cr3tPass"]:
        assert secret not in out


def test_help_lists_options():
    out = run_cli("--help").stdout
    for opt in ["--fast", "--hold", "--pattern", "--ignore", "--downscale", "--selftest"]:
        assert opt in out


@pytest.mark.skipif(IS_MAC, reason="checks the non-macOS guard")
def test_refuses_to_run_outside_macos():
    r = run_cli()
    assert r.returncode == 1 and "needs macOS" in r.stdout


# =========================================================================== #
# 2. OCR on a rendered image (macOS)
# =========================================================================== #

MENLO = "/System/Library/Fonts/Menlo.ttc"


@mac_only
def test_ocr_locates_secret_value_in_image(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(MENLO, 32)
    W, H, margin = 1400, 300, 40
    label, value = "sw-access-key: ", "SWIAQXNHAEZBNLHMWJJUANFAAQ"
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 50), label + value, font=font, fill="black")
    draw.text((margin, 180), "The quick brown fox jumps over the lazy dog", font=font, fill="black")
    path = tmp_path / "secret.png"
    img.save(path)

    hits = sr.make_ocr(sr.Detector())(sr.load_cgimage(path))
    assert len(hits) == 1, f"expected one hit, got {hits}"

    nx, ny, nw, nh, _ = hits[0]
    x0, x1 = nx * W, (nx + nw) * W
    top_from_top = (1 - ny - nh) * H          # Vision is bottom-left origin
    value_x0 = margin + font.getlength(label)
    value_x1 = value_x0 + font.getlength(value)

    tol = 12  # px; boxes are padded by --pad on screen anyway
    assert x0 <= value_x0 + tol, "box starts too far right: start of key exposed"
    assert x1 >= value_x1 - tol, "box ends too early: end of key exposed"
    assert x0 >= margin + font.getlength("sw-access") - tol, "box swallowed the label"
    assert 20 < top_from_top < 120, "box is not on the secret's line"


@mac_only
def test_ocr_finds_nothing_on_clean_image(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(MENLO, 32)
    img = Image.new("RGB", (1400, 200), "white")
    ImageDraw.Draw(img).text((40, 60), "https://shopware-store.ddev.site/api/_mcp", font=font, fill="black")
    path = tmp_path / "clean.png"
    img.save(path)
    assert sr.make_ocr(sr.Detector())(sr.load_cgimage(path)) == []


# =========================================================================== #
# 3. End-to-end on the real screen (macOS + Screen Recording permission)
# =========================================================================== #

FAKE_KEY = "sk-proj-Q7mZ4xT9kL2vB8nR5wY1cH6jF3dG0sPa"   # fake, matches the OpenAI rule
CONTROL = "CONTROL LABEL VISIBLE"

WINDOW_HELPER = textwrap.dedent("""
    import sys
    from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
        NSColor, NSFloatingWindowLevel, NSFont, NSMakeRect, NSScreen, NSTextField, NSWindow,
        NSWindowStyleMaskTitled)
    from PyObjCTools import AppHelper
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    f = NSScreen.mainScreen().frame()
    w, h = 1200, 260
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(f.origin.x + (f.size.width - w) / 2, f.origin.y + (f.size.height - h) / 2, w, h),
        NSWindowStyleMaskTitled, NSBackingStoreBuffered, False)
    win.setLevel_(NSFloatingWindowLevel)
    win.setBackgroundColor_(NSColor.whiteColor())
    label = NSTextField.labelWithString_(sys.argv[1])
    label.setFont_(NSFont.fontWithName_size_("Menlo", 30))
    label.setTextColor_(NSColor.blackColor())
    label.setFrame_(NSMakeRect(30, 40, w - 60, h - 80))
    win.contentView().addSubview_(label)
    win.orderFrontRegardless()
    print("ready", flush=True)
    AppHelper.runEventLoop(installInterrupt=True)
""")


def _norm(text):
    return "".join(text.split()).lower()


def _screen_text(tmp_path, name):
    path = tmp_path / f"{name}.png"
    subprocess.run(["screencapture", "-x", "-m", str(path)], check=True, timeout=15)
    return _norm(" ".join(sr.ocr_text(sr.load_cgimage(path))))


def _has_permission():
    try:
        import Quartz
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return False


@mac_only
@pytest.mark.skipif(IS_MAC and not _has_permission(),
                    reason="give your terminal Screen Recording permission")
def test_end_to_end_secret_hidden_in_screen_capture(tmp_path):
    procs = []
    try:
        win = subprocess.Popen([sys.executable, "-c", WINDOW_HELPER,
                                f"{CONTROL}\nOPENAI_API_KEY={FAKE_KEY}"],
                               stdout=subprocess.PIPE, text=True)
        procs.append(win)
        assert win.stdout.readline().strip() == "ready"
        time.sleep(1.0)

        # Baseline: without the redactor the fake key must be readable,
        # otherwise this test would pass for the wrong reason.
        before = _screen_text(tmp_path, "before")
        assert _norm(CONTROL) in before, "test window not visible (another window on top?)"
        assert "q7mz4xt9" in before, "OCR could not read the fake key even without redaction"

        red = subprocess.Popen([sys.executable, str(SCRIPT), "--verbose"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(red)

        start, hidden_after, after = time.monotonic(), None, ""
        while time.monotonic() - start < 15:
            time.sleep(0.5)
            assert red.poll() is None, f"redactor exited: {red.stdout.read()}"
            after = _screen_text(tmp_path, "after")
            if "q7mz4xt9" not in after and "sk-proj" not in after:
                hidden_after = time.monotonic() - start
                break

        assert hidden_after is not None, "fake key still readable in the screen capture after 15 s"
        assert _norm(CONTROL) in after, "redactor covered text that isn't a secret"
        assert "openai_api_key" in after.replace("-", "_"), "the label should stay readable"
        print(f"\nkey hidden {hidden_after:.1f}s after starting the redactor "
              f"(includes startup; screenshots in {tmp_path})")
    finally:
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
