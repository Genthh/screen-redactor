# Screen Redactor

Hides API keys, access keys, tokens and passwords on your Mac's screen while you record with Loom, QuickTime or macOS screen sharing.

It reads the screen with Apple's on-device text recognition, finds anything that looks like a secret, and covers it with a black box. The box sits in a transparent window on top of everything, so the recording shows the box instead of the key. Nothing leaves your Mac and the secrets are never printed.

## Requirements

- macOS 14 (Sonoma) or newer is recommended. Older versions use a fallback capture method.
- Python 3.9 or newer. Python 3.10+ (from Homebrew or python.org) is recommended.

## Install (one time)

Open Terminal and go into this folder. For example, if it's in Downloads:

```bash
cd ~/Downloads/screen-redactor
```

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --only-binary :all: -r requirements.txt
```

### Allow screen recording

1. Open **System Settings → Privacy & Security → Screen & System Audio Recording**.
2. Turn it on for the app you run the script from (Terminal, iTerm or VS Code).
3. Quit that app completely (⌘Q) and open it again.

If macOS asks whether Terminal may access the folder the project is in, click **Allow**.

## Start

```bash
cd ~/Downloads/screen-redactor
source .venv/bin/activate
python3 screen_redactor.py --verbose
```

When you see `Redacting secrets on 1 display(s)… Start your recording now`, it's running. Open something that shows a key and a black box should cover it within about half a second.

Then start your recording. In Loom or your screen share, choose **Full screen / Entire screen**. Window-only recording doesn't include the boxes.

To stop, press **Ctrl-C** in the Terminal window.

## Options

| Option | What it does |
|---|---|
| `--verbose` | Prints when the number of boxes changes (never the secret) |
| `--fast` | Boxes appear quicker; text recognition is slightly weaker on small text |
| `--downscale 0.5` | Uses less CPU, useful on large or 5K displays |
| `--hold 2` | Keeps boxes up longer after text moves (default 1.2 s), useful when scrolling |
| `--pad 6` | Makes the boxes a bit bigger (default 4) |
| `--pattern 'REGEX'` | Also hides your own key format, e.g. `--pattern 'acme_[A-Za-z0-9]{24}'` (repeatable) |
| `--ignore 'REGEX'` | Never covers strings matching this (repeatable) |
| `--no-entropy` | Only hides known key formats and `key: value` lines, not other random-looking strings |
| `--color '#222222'` | Box color |
| `--selftest` | Checks the detection rules and exits |

## What it hides

- Known key formats: OpenAI, Anthropic, GitHub, GitLab, Stripe, Slack, AWS, Google, Shopware (`SWIA…`), Hugging Face, npm, SendGrid, JWTs, private key headers, passwords in URLs.
- Values next to labels such as `api_key`, `token`, `secret`, `password`, `Authorization: Bearer`, `sw-access-key`, `x-api-key`. Only the value is covered and the label stays readable.
- Other long, random-looking strings that mix letters and digits.

## Testing

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python3 -m pytest -v -s test_screen_redactor.py
```

This runs:

- **Detection and geometry tests**, which work on any computer.
- **OCR test**, which draws a key into an image and checks the box lands on it.
- **End-to-end test**, which opens a window with a fake key, starts the redactor, takes a real screenshot and checks the key can't be read. Don't touch the mouse or keyboard for the ~15 seconds it takes. It needs the Screen Recording permission above.

## Good to know

- A secret that suddenly appears can be visible for a fraction of a second before its box lands. `--fast` makes that gap shorter.
- The boxes are visible to you too, not just in the recording.
- If you plug in or remove a display, restart the script.

## Troubleshooting

| Problem | Fix |
|---|---|
| Install fails with `Building wheel for pyobjc-core … error` | Run `pip install --upgrade pip` first, then install again with `--only-binary :all:` as shown above |
| `No matching distribution found` | Your Python is too old for the prebuilt packages. Install Python 3.12 (`brew install python@3.12`), delete `.venv`, and create it again with `python3.12 -m venv .venv` |
| `ModuleNotFoundError: No module named 'objc'` | The virtual environment isn't active. Run `source .venv/bin/activate` |
| Script says Screen Recording permission is needed, or no boxes appear | Enable the permission for your terminal app, then quit and reopen it |
| Something that isn't a secret gets covered | Add `--ignore 'that text'` or use `--no-entropy` |
| The Mac feels slow while it runs | Add `--downscale 0.5` or `--fast` |
