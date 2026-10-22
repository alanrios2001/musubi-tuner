"""Copy a reproducible, provenance-grouped paired image split before caching.

Usage: python split_paired_dataset.py --original_directory ... --target_directory ...
       --groups_json groups.json --output_directory ... --seed 42 --validation_fraction 0.1

groups.json maps every target filename (including extension) to its original source ID.
All crops, edits, and other derivatives of one source must share that ID. Exact decoded
duplicates are additionally grouped automatically, across both sides of every pair.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def image_files(directory):
    files = sorted(path for path in Path(directory).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    by_stem = {}
    for path in files:
        key = path.stem.casefold()
        if key in by_stem:
            raise ValueError(f"Duplicate image stem: {path}")
        by_stem[key] = path
    return by_stem


def pixel_hash(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        digest = hashlib.sha256(str(image.size).encode())
        digest.update(image.tobytes())
        return digest.hexdigest()


def split_pairs(
    original_directory, target_directory, groups_json, output_directory, seed=42, validation_fraction=0.1, caption_extension=".txt"
):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    output = Path(output_directory).resolve()
    if output.exists():
        raise ValueError(f"Output must not exist (no overwrite): {output}")
    originals = image_files(original_directory)
    targets = image_files(target_directory)
    if not targets or set(originals) != set(targets):
        raise ValueError("Original and target directories must contain the same nonempty set of unique image stems")
    with open(groups_json, encoding="utf-8") as file:
        source_groups = json.load(file)
    filenames = {path.name for path in targets.values()}
    if set(source_groups) != filenames or not all(isinstance(value, str) and value for value in source_groups.values()):
        raise ValueError("groups_json must map every target filename, and only those filenames, to a nonempty source ID")
    if not caption_extension.startswith("."):
        raise ValueError("caption_extension must start with '.'")
    # Validate all inputs and group all identities before creating output.
    keys = sorted(targets)
    parent = {key: key for key in keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        parent[find(a)] = find(b)

    seen = {}
    hashes = {}
    for key in keys:
        target, original = targets[key], originals[key]
        if not target.with_suffix(caption_extension).is_file():
            raise ValueError(f"Missing caption: {target.with_suffix(caption_extension)}")
        identities = [("source", source_groups[target.name])]
        hashes[key] = [pixel_hash(original), pixel_hash(target)]
        identities.extend(("pixels", digest) for digest in hashes[key])
        for identity in identities:
            if identity in seen:
                union(key, seen[identity])
            seen[identity] = key
    grouped = defaultdict(list)
    for key in keys:
        grouped[find(key)].append(key)
    groups = list(grouped.values())
    if len(groups) < 2:
        raise ValueError("At least two independent source groups are required for a leakage-free split")
    random.Random(seed).shuffle(groups)
    wanted = len(keys) * validation_fraction
    running = 0
    choices = []
    for index, group in enumerate(groups[:-1], 1):
        running += len(group)
        choices.append((abs(running - wanted), index))
    cutoff = min(choices)[1]
    validation = {key for group in groups[:cutoff] for key in group}
    manifest = {"seed": seed, "validation_fraction": validation_fraction, "pairs": []}
    for split in ("train", "val"):
        for side in ("original", "target"):
            (output / split / side).mkdir(parents=True, exist_ok=True)
        (output / f"cache_{split}").mkdir()
    for key in keys:
        split = "val" if key in validation else "train"
        target, original = targets[key], originals[key]
        # Normalize control stem to the exact target stem for the repository's pairing rules.
        shutil.copy2(original, output / split / "original" / (target.stem + original.suffix))
        shutil.copy2(target, output / split / "target" / target.name)
        caption = target.with_suffix(caption_extension)
        shutil.copy2(caption, output / split / "target" / caption.name)
        manifest["pairs"].append(
            {
                "target": target.name,
                "original": original.name,
                "source_id": source_groups[target.name],
                "split": split,
                "pixel_hashes": hashes[key],
            }
        )
    manifest["train_pairs"] = len(keys) - len(validation)
    manifest["validation_pairs"] = len(validation)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Training pairs: {manifest['train_pairs']}; validation pairs: {manifest['validation_pairs']}; source groups: {len(groups)}"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_directory", required=True)
    parser.add_argument("--target_directory", required=True)
    parser.add_argument("--groups_json", required=True)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--caption_extension", default=".txt")
    split_pairs(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
