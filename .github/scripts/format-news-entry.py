#!/usr/bin/env python3
"""
Format a bandicoot GitLab release payload into a draft NEWS.md entry.

Usage: format-news-entry.py RELEASES_JSON

Reads the full `/releases` response (array of releases) at RELEASES_JSON
and writes a single entry for the newest release to stdout.

The output is a DRAFT. It is intended to be prepended to NEWS.md by the
upstream-update workflow; a human reviewer still polishes wording and any
upstream-specific edge cases before merging the PR.

Mechanical transforms applied:
  - Capitalize the first letter of each bullet
  - Backtick obvious method/function calls (.foo(), bar()) when not already
    inside backticks
  - Convert common HTML to Markdown: <a href="URL">TXT</a> -> [TXT](URL)
    and <i>TXT</i> -> `TXT`
  - Ensure each bullet ends with sentence punctuation (. ! ?)
  - Wrap bullets at 80 columns with a 4-space continuation indent
  - If the release description has no `*`-bullets (pure prose), each
    non-blank line is promoted to a bullet
"""
from __future__ import annotations

import json
import re
import sys
import textwrap

WIDTH = 80
INITIAL = "  - "
SUBSEQUENT = "    "


def html_to_markdown(text: str) -> str:
    text = re.sub(
        r'<a\s+href="([^"]+)">([^<]+)</a>',
        r'[\2](\1)',
        text,
    )
    text = re.sub(r'<i>([^<]+)</i>', r'`\1`', text)
    return text


def backtick_code(text: str) -> str:
    """Backtick function/method calls not already inside backticks."""
    parts = re.split(r'(`[^`]*`)', text)
    out = []
    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            out.append(part)
            continue
        # .method(...) — leading dot, not preceded by word char or dot
        part = re.sub(
            r'(?<![\w.])(\.[a-zA-Z_]\w*\([^)]*\))',
            r'`\1`',
            part,
        )
        # bare function() — snake_case or lowercase ident, empty args only
        part = re.sub(
            r'(?<![\w.`])\b([a-z_]\w*\(\))',
            r'`\1`',
            part,
        )
        out.append(part)
    return ''.join(out)


def polish(text: str) -> str:
    text = text.strip()
    text = html_to_markdown(text)
    # Capitalize the first alphabetic character (so "(editor's note: ..."
    # becomes "(Editor's note: ...") while leaving pre-capitalized text alone.
    for i, c in enumerate(text):
        if c.isalpha():
            if c.islower():
                text = text[:i] + c.upper() + text[i + 1:]
            break
    text = backtick_code(text)
    if text and text[-1] not in '.!?':
        text += '.'
    return text


def wrap_bullet(text: str) -> str:
    return textwrap.fill(
        text,
        width=WIDTH,
        initial_indent=INITIAL,
        subsequent_indent=SUBSEQUENT,
        break_long_words=False,
        break_on_hyphens=False,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('usage: format-news-entry.py RELEASES_JSON', file=sys.stderr)
        return 2

    with open(argv[1]) as f:
        release = json.load(f)[0]

    tag = release['tag_name'].lstrip('v')
    name = release['name']
    date = release['released_at'][:10]
    desc = release.get('description') or ''
    pkg_version = f'{tag}.1'

    # Every non-blank line in the description becomes a nested bullet.
    # This handles both the `*`-bullet and prose-only styles upstream uses.
    bullets: list[str] = []
    for raw in desc.splitlines():
        m = re.match(r'^\s*\*\s+(.*)$', raw)
        if m:
            bullets.append(m.group(1))
        elif raw.strip():
            bullets.append(raw.strip())

    out: list[str] = []
    out.append(f'# RcppBandicoot {pkg_version}')
    out.append('')
    out.append(f'- Upgraded to {name} ({date})')
    for b in bullets:
        out.append(wrap_bullet(polish(b)))
    out.append('')

    sys.stdout.write('\n'.join(out))
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
