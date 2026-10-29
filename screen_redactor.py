#!/usr/bin/env python3

import argparse
import math
import re
import sys
import threading
import time
from collections import Counter


PREFIX_PATTERNS = {
    "anthropic": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "openai": r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}",
    "stripe": r"\b(?:sk|pk|rk|whsec)_(?:live_|test_)?[A-Za-z0-9]{16,}",
    "github": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "gitlab": r"\bglpat-[A-Za-z0-9_\-]{20,}",
    "slack": r"\bxox[abposr]-[A-Za-z0-9\-]{10,}",
    "aws_key_id": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "google": r"\bAIza[A-Za-z0-9_\-]{30,}",
    "shopware": r"\bSW(?:IA|UA|SC)[A-Za-z0-9]{16,}",
    "huggingface": r"\bhf_[A-Za-z0-9]{30,}",
    "npm": r"\bnpm_[A-Za-z0-9]{30,}",
    "sendgrid": r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}",
    "jwt": r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "url_credentials": r"(?<=://)[^/\s:@]+:[^/\s@]{4,}(?=@)",
}

_KEYWORDS = (
    r"sw-secret-access-key|sw-access-key|x-api-key|"
    r"api[\s_\-]?key|api[\s_\-]?secret|access[\s_\-]?key(?:[\s_\-]?id)?|"
    r"secret(?:[\s_\-]?(?:access[\s_\-]?)?key)?|client[\s_\-]?secret|"
    r"(?:auth|access|refresh|api|bearer)[\s_\-]?token|token|"
    r"passw(?:or)?d|pwd|authorization|private[\s_\-]?key"
)
CONTEXT_RE = re.compile(
    rf"(?i)(?:{_KEYWORDS})[\"'\s]*[:=]\s*[\"']?(?:bearer\s+)?(?P<v>[^\s\"',;`]{{8,}})"
)
BEARER_RE = re.compile(r"(?i)\bbearer\s+(?P<v>[A-Za-z0-9._~+/\-]{12,}=*)")
CANDIDATE_RE = re.compile(r"[A-Za-z0-9_\-+/.]{24,}=*")

CONTEXT_STOPWORDS = {
    "undefined", "required", "optional", "boolean", "function", "nullable",
    "string[]", "your_api_key", "<redacted>", "redacted", "********",
}


def shannon_entropy(s):
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def looks_random(tok):
    tok = tok.strip("=.-_/")
    if len(tok) < 24:
        return False
    has_digit = any(c.isdigit() for c in tok)
    has_alpha = any(c.isalpha() for c in tok)
    if not (has_digit and has_alpha):
        return False
    if tok.count("/") > 2 or tok.count(".") > 2:
        return False
    return shannon_entropy(tok) >= 3.6


class Detector:
    def __init__(self, entropy=True, extra_patterns=(), ignore_patterns=()):
        self.rules = [(name, re.compile(p)) for name, p in PREFIX_PATTERNS.items()]
        self.rules += [(f"custom{i}", re.compile(p)) for i, p in enumerate(extra_patterns)]
        self.ignore = [re.compile(p) for p in ignore_patterns]
        self.entropy = entropy

    def _ignored(self, s):
        return any(r.search(s) for r in self.ignore)

    def spans(self, text):
        found = []
        for name, rx in self.rules:
            for m in rx.finditer(text):
                found.append((m.start(), m.end(), name))
        for name, rx in (("context", CONTEXT_RE), ("bearer", BEARER_RE)):
            for m in rx.finditer(text):
                v = m.group("v")
                if v.lower() in CONTEXT_STOPWORDS or v.lower().startswith(("process.env", "os.environ", "${", "env(")):
                    continue
                found.append((m.start("v"), m.end("v"), name))
        if self.entropy:
            for m in CANDIDATE_RE.finditer(text):
                if looks_random(m.group()):
                    found.append((m.start(), m.end(), "entropy"))

        found = [f for f in found if not self._ignored(text[f[0]:f[1]])]
        found.sort()
        merged = []
        for s, e, r in found:
            if merged and s <= merged[-1][1]:
                ps, pe, pr = merged[-1]
                merged[-1] = (ps, max(pe, e), pr)
            else:
                merged.append((s, e, r))
        return merged


def utf16_range(text, start, end):
    loc = len(text[:start].encode("utf-16-le")) // 2
    length = len(text[start:end].encode("utf-16-le")) // 2
    return loc, length


def to_points(hit, size, pad):
    nx, ny, nw, nh = hit[:4]
    W, H = size
    return [nx * W - pad, ny * H - pad, nw * W + 2 * pad, nh * H + 2 * pad]


def overlap(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    smaller = min(a[2] * a[3], b[2] * b[3]) or 1
    return ix * iy / smaller


def update_boxes(boxes, hits, size, now, pad=4, hold=1.2):
    boxes = [list(b) for b in boxes]
    for hit in hits:
        r = to_points(hit, size, pad) + [now]
        for b in boxes:
            if overlap(b, r) > 0.6:
                b[:] = r
                break
        else:
            boxes.append(r)
    boxes = [b for b in boxes if now - b[4] <= hold]
    return boxes, [tuple(b[:4]) for b in boxes]


def make_ocr(detector, fast=False):
    import Vision

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast
                                 else Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)

    def find(cgimage):
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, {})
        ok, _ = handler.performRequests_error_([request], None)
        if not ok:
            return []
        hits = []
        for obs in request.results() or []:
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            cand = cands[0]
            text = str(cand.string())
            for s, e, rule in detector.spans(text):
                box, _ = cand.boundingBoxForRange_error_(utf16_range(text, s, e), None)
                bb = (box or obs).boundingBox()
                hits.append((bb.origin.x, bb.origin.y, bb.size.width, bb.size.height, rule))
        return hits

    return find


def ocr_text(cgimage):
    import Vision

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, {})
    handler.performRequests_error_([request], None)
    lines = []
    for obs in request.results() or []:
        cands = obs.topCandidates_(1)
        if cands:
            lines.append(str(cands[0].string()))
    return lines


def load_cgimage(path):
    import Quartz
    from Foundation import NSURL

    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None) if src else None


SHOULD_HIT = [
    '"sw-access-key: SWIAQXNHAEZBNLHMWJJUANFAAQ",',
    '"sw-secret-access-key: dGhpc2lzYXZlcnlzZWNyZXRrZXkxMjM0",',
    "Authorization: Bearer abcDEF123456ghiJKL789",
    "OPENAI_API_KEY=sk-proj-AbCdEf1234567890GhIjKlMnOp",
    "ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf1234567890GhIjKl",
    "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789",
    "SLACK_BOT_TOKEN=xoxb-1234567890-abcdefghij",
    "password = hunter2hunter2",
    "postgres://admin:s3cr3tPass@db.example.com:5432/app",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "key 9f8e7d6c5b4a39281706f5e4d3c2b1a0Zz",
]
SHOULD_MISS = [
    "https://shopware-store.ddev.site/api/_mcp",
    "The quick brown fox jumps over the lazy dog",
    '"command": "npx",',
    "getShareableContentExcludingDesktopWindows",
    "apiKey: string;",
    "token: undefined",
    "password: process.env.DB_PASSWORD",
    "Thu Oct 29 2:02 PM",
    "mcp-remote@0.8.4",
]


def selftest():
    d = Detector()
    should_hit, should_miss = SHOULD_HIT, SHOULD_MISS
    ok = True
    for s in should_hit:
        spans = d.spans(s)
        status = "ok  " if spans else "FAIL"
        ok &= bool(spans)
        shown = s
        for a, b, _ in reversed(spans):
            shown = shown[:a] + "\u2588" * (b - a) + shown[b:]
        print(f"[{status}] hit   {shown}")
    for s in should_miss:
        spans = d.spans(s)
        status = "ok  " if not spans else "FAIL"
        ok &= not spans
        print(f"[{status}] miss  {s}")
    print("\nall passed" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


def run(args):
    import objc
    import Quartz
    from AppKit import (
        NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
        NSBezierPath, NSColor, NSCompositingOperationCopy, NSMakeRect, NSRectFillUsingOperation,
        NSScreen, NSScreenSaverWindowLevel, NSView, NSWindow, NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowCollectionBehaviorIgnoresCycle,
        NSWindowCollectionBehaviorStationary, NSWindowSharingReadOnly, NSWindowStyleMaskBorderless,
    )
    from PyObjCTools import AppHelper

    try:
        import ScreenCaptureKit as SCK
        have_sck = hasattr(SCK, "SCScreenshotManager")
    except ImportError:
        SCK, have_sck = None, False

    if not Quartz.CGPreflightScreenCaptureAccess():
        Quartz.CGRequestScreenCaptureAccess()
        print("Screen Recording permission is needed. Enable it for your terminal in\n"
              "System Settings > Privacy & Security > Screen & System Audio Recording,\n"
              "restart the terminal, and run this again.")
        return 1

    def parse_color(hexstr):
        h = hexstr.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 1.0)

    box_color = parse_color(args.color)

    class RedactionView(NSView):
        def initWithFrame_(self, frame):
            self = objc.super(RedactionView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._rects = []
            return self

        def drawRect_(self, dirty):
            NSColor.clearColor().set()
            NSRectFillUsingOperation(dirty, NSCompositingOperationCopy)
            box_color.set()
            for x, y, w, h in self._rects:
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, y, w, h), 3, 3).fill()

    class Overlay:
        def __init__(self, screen):
            frame = screen.frame()
            self.display_id = int(screen.deviceDescription()["NSScreenNumber"])
            self.size = (frame.size.width, frame.size.height)
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
            win.setFrame_display_(frame, False)
            win.setOpaque_(False)
            win.setBackgroundColor_(NSColor.clearColor())
            win.setHasShadow_(False)
            win.setIgnoresMouseEvents_(True)
            win.setLevel_(NSScreenSaverWindowLevel)
            win.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
                | NSWindowCollectionBehaviorIgnoresCycle)
            win.setSharingType_(NSWindowSharingReadOnly)
            win.setReleasedWhenClosed_(False)
            self.view = RedactionView.alloc().initWithFrame_(
                NSMakeRect(0, 0, frame.size.width, frame.size.height))
            win.setContentView_(self.view)
            win.orderFrontRegardless()
            self.window = win
            self.window_id = int(win.windowNumber())
            self.boxes = []
            self.last_count = -1

        def show(self, rects):
            self.view._rects = rects
            self.view.setNeedsDisplay_(True)

    def wait_for(start, timeout=5.0):
        done, out = threading.Event(), []

        def handler(a, b):
            out.extend((a, b))
            done.set()

        start(handler)
        if not done.wait(timeout):
            raise TimeoutError("ScreenCaptureKit did not respond")
        return out

    class SCKCapturer:

        def __init__(self, overlays):
            self.overlays = overlays
            self.filters = {}
            self.refreshed = 0.0

        def refresh(self):
            content, err = wait_for(lambda h: SCK.SCShareableContent
                                    .getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                                        False, True, h))
            if content is None:
                raise RuntimeError(f"cannot list screen content: {err}")
            own_ids = {o.window_id for o in self.overlays}
            own = [w for w in content.windows() if int(w.windowID()) in own_ids]
            if len(own) < len(own_ids):
                return False
            displays = {int(d.displayID()): d for d in content.displays()}
            self.filters = {}
            for o in self.overlays:
                d = displays.get(o.display_id)
                if d is not None:
                    self.filters[o.display_id] = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(d, own)
            self.refreshed = time.monotonic()
            return True

        def capture(self, ov):
            if not self.filters or time.monotonic() - self.refreshed > 10:
                if not self.refresh():
                    return None
            filt = self.filters.get(ov.display_id)
            if filt is None:
                return None
            rect, scale = filt.contentRect(), filt.pointPixelScale() * args.downscale
            cfg = SCK.SCStreamConfiguration.alloc().init()
            cfg.setWidth_(int(rect.size.width * scale))
            cfg.setHeight_(int(rect.size.height * scale))
            cfg.setShowsCursor_(False)
            image, err = wait_for(lambda h: SCK.SCScreenshotManager
                                  .captureImageWithFilter_configuration_completionHandler_(filt, cfg, h))
            return image

    class CGCapturer:

        def capture(self, ov):
            opt = Quartz.kCGWindowImageNominalResolution if args.downscale < 1 else Quartz.kCGWindowImageDefault
            return Quartz.CGWindowListCreateImage(
                Quartz.CGDisplayBounds(ov.display_id),
                Quartz.kCGWindowListOptionOnScreenBelowWindow, ov.window_id, opt)

    detector = Detector(entropy=not args.no_entropy,
                        extra_patterns=args.pattern, ignore_patterns=args.ignore)
    find_secrets = make_ocr(detector, fast=args.fast)

    def worker(overlays, capturer):
        time.sleep(0.5)
        warned = set()
        while True:
            t0 = time.monotonic()
            for ov in overlays:
                with objc.autorelease_pool():
                    try:
                        img = capturer.capture(ov)
                        hits = find_secrets(img) if img is not None else []
                    except Exception as exc:
                        key = type(exc).__name__ + str(exc)[:80]
                        if key not in warned:
                            warned.add(key)
                            print(f"warning: {exc}", file=sys.stderr)
                        continue
                    ov.boxes, rects = update_boxes(ov.boxes, hits, ov.size, time.monotonic(),
                                                   pad=args.pad, hold=args.hold)
                    AppHelper.callAfter(ov.show, rects)
                    if args.verbose and len(rects) != ov.last_count:
                        rules = ", ".join(sorted({h[4] for h in hits})) or "-"
                        print(f"display {ov.display_id}: {len(rects)} redaction(s)  [{rules}]")
                    ov.last_count = len(rects)
            time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    overlays = [Overlay(s) for s in NSScreen.screens()]
    capturer = SCKCapturer(overlays) if have_sck else CGCapturer()
    threading.Thread(target=worker, args=(overlays, capturer), daemon=True).start()

    print(f"Redacting secrets on {len(overlays)} display(s) "
          f"({'ScreenCaptureKit' if have_sck else 'CoreGraphics'}, "
          f"{'fast' if args.fast else 'accurate'} OCR). Start your recording now; Ctrl-C to stop.")
    AppHelper.runEventLoop(installInterrupt=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Black out API keys and tokens on screen while recording (macOS).")
    ap.add_argument("--fast", action="store_true", help="fast OCR: lower latency, weaker on small text")
    ap.add_argument("--interval", type=float, default=0.05, help="minimum seconds between scans (default 0.05)")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds a box stays after its text disappears (default 1.2)")
    ap.add_argument("--pad", type=float, default=4, help="padding around each box in points (default 4)")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="capture at this fraction of native resolution, e.g. 0.5 on 5K displays (default 1.0)")
    ap.add_argument("--color", default="#000000", help="box color as hex (default #000000)")
    ap.add_argument("--no-entropy", action="store_true", help="disable the random-looking-string rule")
    ap.add_argument("--pattern", action="append", default=[], help="extra regex to redact (repeatable)")
    ap.add_argument("--ignore", action="append", default=[], help="regex for strings never to redact (repeatable)")
    ap.add_argument("--verbose", action="store_true", help="print redaction counts and rule names (never the secrets)")
    ap.add_argument("--selftest", action="store_true", help="test the detection rules and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if sys.platform != "darwin":
        print("This script needs macOS (use --selftest elsewhere).")
        return 1
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
