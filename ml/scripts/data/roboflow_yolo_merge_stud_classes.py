from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root")
    a = p.parse_args()

    root = Path(a.dataset_root)
    yml = root / "data.yaml"
    lines = yml.read_text().splitlines()

    i = next(k for k, s in enumerate(lines) if s.strip() == "names:")
    names, j = [], i + 1
    while j < len(lines) and lines[j].lstrip().startswith("- "):
        names.append(lines[j].split("- ", 1)[1].strip())
        j += 1

    merge = {"unsure": "broken", "renewed": "normal"}
    new_names = [n for n in names if n not in ("unsure", "renewed")]
    name2id = {n: k for k, n in enumerate(new_names)}
    old2new = {k: name2id[merge.get(n, n)] for k, n in enumerate(names)}

    label_files = sorted(root.glob("*/labels/*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {root}/*/labels/*.txt")
    for fp in label_files:
        txt = fp.read_text()
        if txt == "":
            continue
        out = []
        for ln in txt.splitlines():
            a0, x, y, w, h = ln.split()
            out.append(f"{old2new[int(a0)]} {x} {y} {w} {h}")
        fp.write_text("\n".join(out) + "\n")

    new_lines = lines[:i] + ["names:"] + [f"- {n}" for n in new_names] + lines[j:]
    new_lines = [f"nc: {len(new_names)}" if s.strip().startswith("nc:") else s for s in new_lines]
    yml.write_text("\n".join(new_lines) + "\n")


if __name__ == "__main__":
    main()

