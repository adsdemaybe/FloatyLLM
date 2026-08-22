#!/usr/bin/env python3
# Custom linter: forbid multiline (block) comments in C/C++/CUDA sources.
# Only single-line `//` comments are allowed. Any `/* ... */` block comment is
# flagged, which guarantees no comment spans multiple lines.
# Usage: python3 tools/lint_comments.py [paths...]   (defaults to repo tree)
import sys
import os

EXTS = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".hxx"}


def normalize_line_splices(text):
    # C/C++ line splicing removes backslash-newline before tokenization.
    out = []
    pos = []
    i, n = 0, len(text)
    line, col = 1, 1
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] == "\n":
            i += 2
            line, col = line + 1, 1
            continue
        out.append(c)
        pos.append((line, col))
        if c == "\n":
            line, col = line + 1, 1
        else:
            col += 1
        i += 1
    return "".join(out), pos


def find_block_comments(text):
    # Scan char-by-char, skipping string/char literals and `//` line comments.
    # Return list of (line, col) where a `/*` block comment opens.
    text, pos = normalize_line_splices(text)
    hits = []
    i, n = 0, len(text)
    in_str = False        # inside "..."
    in_char = False       # inside '...'
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif in_char:
            if c == "\\":
                i += 2
                continue
            if c == "'":
                in_char = False
        elif c == "/" and nxt == "/":
            # line comment: skip to end of line
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif c == "/" and nxt == "*":
            hits.append(pos[i])
            i += 2
            continue
        elif c == '"':
            in_str = True
        elif c == "'":
            in_char = True
        i += 1
    return hits


def iter_files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        else:
            for root, _, files in os.walk(p):
                if ".git" in root:
                    continue
                for f in files:
                    if os.path.splitext(f)[1] in EXTS:
                        yield os.path.join(root, f)


def main(argv):
    paths = argv[1:] or ["."]
    violations = 0
    for path in iter_files(paths):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for line, col in find_block_comments(text):
            print(f"{path}:{line}:{col}: block/multiline comment forbidden; use `//`")
            violations += 1
    if violations:
        print(f"\n{violations} block-comment violation(s). Use single-line `//` only.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
