from wildfire.metadata import ChipMeta
from wildfire.split import build_group_split


def _item(chip_id: str, kind: str, event: str | None) -> ChipMeta:
    shape = 256 if kind == "af" else 512
    gsd = 375 if kind == "af" else 20
    return ChipMeta(chip_id, kind, shape, shape, gsd, event)


def test_group_split_keeps_same_fire_event_together():
    meta = {
        "af_a": _item("af_a", "af", "fire_1"),
        "bs_a": _item("bs_a", "bs", "fire_1"),
        "af_b": _item("af_b", "af", "fire_2"),
        "bs_b": _item("bs_b", "bs", "fire_3"),
    }
    split = build_group_split(meta, validation_fraction=0.5, seed=42)
    train = set(split["train"])
    validation = set(split["validation"])

    assert not ({"af_a", "bs_a"} & train and {"af_a", "bs_a"} & validation)
    assert train.isdisjoint(validation)
    assert train | validation == set(meta)


def test_group_split_is_deterministic():
    meta = {
        f"af_{index}": _item(f"af_{index}", "af", f"event_{index}")
        for index in range(10)
    }
    first = build_group_split(meta, validation_fraction=0.2, seed=7)
    second = build_group_split(meta, validation_fraction=0.2, seed=7)
    assert first == second


def test_missing_event_id_falls_back_to_individual_chip_group():
    meta = {
        "neg_1": _item("neg_1", "af", None),
        "neg_2": _item("neg_2", "af", None),
        "pos_1": _item("pos_1", "af", "event_1"),
    }
    split = build_group_split(meta, validation_fraction=0.5, seed=3)
    assert set(split["train"]) | set(split["validation"]) == set(meta)
