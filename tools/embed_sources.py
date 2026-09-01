#!/usr/bin/env python3
"""Write the tool's sources into the Colab notebook as %%writefile cells.

The notebook used to `git clone` this repository at run time, which tied it to
the repository being reachable: a private repository broke it, and fixing that
meant a token, a Colab secret, and a per-notebook toggle -- three things to get
right before transcribing anything.

Carrying the sources inside the notebook removes all of it. The cost is the
usual one for duplicated code, and CI pays it: `--check` re-embeds and fails if
the result differs, so a change to transcribe.py that never reached the notebook
cannot merge.

    python tools/embed_sources.py            # update the notebook
    python tools/embed_sources.py --check    # verify it is current
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "colab_stt_multilingva.ipynb"
EMBEDDED = ["transcribe.py", "app.py"]


def cell_for(name: str) -> dict:
    body = (ROOT / name).read_text(encoding="utf-8")
    source = f"%%writefile {name}\n{body}"
    return {
        "cell_type": "code",
        "execution_count": None,
        # The marker is how the cell is found again on the next run; matching on
        # content would rewrite whatever happened to look similar.
        "metadata": {"embedded_source": name},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def embed(notebook: dict) -> dict:
    cells = notebook["cells"]
    for name in EMBEDDED:
        at = next(
            (i for i, c in enumerate(cells)
             if c.get("metadata", {}).get("embedded_source") == name),
            None,
        )
        if at is None:
            raise SystemExit(
                f"No cell tagged embedded_source={name!r} in {NOTEBOOK.name}. "
                "Add one (it can be empty) before running this."
            )
        cells[at] = cell_for(name)
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the notebook is out of date")
    args = parser.parse_args()

    current = NOTEBOOK.read_text(encoding="utf-8")
    updated = json.dumps(
        embed(json.loads(current)), ensure_ascii=False, indent=1
    ) + "\n"

    if args.check:
        if current != updated:
            print(
                f"{NOTEBOOK.name} is out of date with {', '.join(EMBEDDED)}.\n"
                "Run: python tools/embed_sources.py",
                file=sys.stderr,
            )
            return 1
        print(f"{NOTEBOOK.name} is current")
        return 0

    NOTEBOOK.write_text(updated, encoding="utf-8")
    print(f"embedded {', '.join(EMBEDDED)} into {NOTEBOOK.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
