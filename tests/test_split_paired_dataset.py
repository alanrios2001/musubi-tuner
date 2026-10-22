import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


spec = importlib.util.spec_from_file_location(
    "split_paired_dataset", Path(__file__).resolve().parents[1] / "split_paired_dataset.py"
)
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


def inputs(tmp_path):
    original = tmp_path / "original"
    target = tmp_path / "target"
    original.mkdir()
    target.mkdir()
    groups = {}
    for i in range(10):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(original / f"pair_{i}.png")
        Image.new("RGB", (8, 8), (0, i + 20, 0)).save(target / f"pair_{i}.png")
        (target / f"pair_{i}.txt").write_text(f"caption {i}", encoding="utf-8")
        groups[f"pair_{i}.png"] = f"source_{i}"
    # Provenance links derivatives; decoded pixels link re-encoded duplicates transitively.
    groups["pair_1.png"] = groups["pair_0.png"]
    Image.open(original / "pair_1.png").save(original / "pair_2.png")
    group_file = tmp_path / "groups.json"
    group_file.write_text(json.dumps(groups), encoding="utf-8")
    return original, target, group_file


def test_split_groups_pairs_captions_duplicates_reproducibility_and_originals(tmp_path):
    original, target, groups = inputs(tmp_path)
    original_bytes = {path: path.read_bytes() for directory in (original, target) for path in directory.iterdir()}
    a = splitter.split_pairs(original, target, groups, tmp_path / "a")
    b = splitter.split_pairs(original, target, groups, tmp_path / "b")
    assert a == b
    assert a["train_pairs"] + a["validation_pairs"] == 10
    assert a["validation_pairs"] > 0 and a["train_pairs"] > 0
    assignments = {pair["target"]: pair["split"] for pair in a["pairs"]}
    assert len({assignments[f"pair_{i}.png"] for i in range(3)}) == 1
    identities = {"train": set(), "val": set()}
    for pair in a["pairs"]:
        split = pair["split"]
        identities[split].update(pair["pixel_hashes"])
        assert (tmp_path / "a" / split / "original" / pair["target"]).is_file()
        caption = Path(pair["target"]).with_suffix(".txt")
        assert (tmp_path / "a" / split / "target" / caption).read_bytes() == (target / caption).read_bytes()
    assert not identities["train"] & identities["val"]
    # The split also obeys the native datasource's filename/control matching rules.
    from musubi_tuner.dataset.datasources import ImageDirectoryDatasource

    for split in ("train", "val"):
        datasource = ImageDirectoryDatasource(
            str(tmp_path / "a" / split / "target"),
            caption_extension=".txt",
            control_directory=str(tmp_path / "a" / split / "original"),
        )
        assert len(datasource.control_paths) == len(datasource.image_paths)
        for image_path, controls in datasource.control_paths.items():
            assert len(controls) == 1 and Path(controls[0]).stem == Path(image_path).stem
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
    with pytest.raises(ValueError, match="no overwrite"):
        splitter.split_pairs(original, target, groups, tmp_path / "a")


def test_split_requires_complete_provenance_and_captions_before_writing(tmp_path):
    original, target, groups = inputs(tmp_path)
    complete_groups = groups.read_text(encoding="utf-8")
    groups.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="every target"):
        splitter.split_pairs(original, target, groups, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    groups.write_text(complete_groups, encoding="utf-8")
    (target / "pair_0.txt").unlink()
    with pytest.raises(ValueError, match="Missing caption"):
        splitter.split_pairs(original, target, groups, tmp_path / "out")
    assert not (tmp_path / "out").exists()
