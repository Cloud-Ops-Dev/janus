"""Entrypoint for ``python -m janus.admin`` (wrapped by ``bin/janus-admin``)."""

from __future__ import annotations

import sys

from janus.admin.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
