from wildfire.metadata import ChipMeta
from wildfire.split import build_group_folds, build_group_split


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


def test_group_folds_are_leakage_safe_and_cover_all_chips_once():
    meta = {}
    for index in range(12):
        event = f"event_{index // 2}"
        kind = "af" if index % 3 else "bs"
        meta[f"chip_{index}"] = _item(f"chip_{index}", kind, event)

    manifest = build_group_folds(meta, n_splits=3, seed=42)
    seen_validation: set[str] = set()
    chip_to_fold: dict[str, int] = {}

    for fold in manifest["folds"]:
        train = set(fold["train"])
        validation = set(fold["validation"])
        assert train.isdisjoint(validation)
        assert train | validation == set(meta)
        assert seen_validation.isdisjoint(validation)
        seen_validation |= validation
        for chip_id in validation:
            chip_to_fold[chip_id] = fold["fold"]

    assert seen_validation == set(meta)
    for left, right in (("chip_0", "chip_1"), ("chip_2", "chip_3"), ("chip_4", "chip_5")):
        assert chip_to_fold[left] == chip_to_fold[right]


def test_strict_group_folds_reject_chip_fallbacks():
    meta = {
        "a": _item("a", "af", "event_1"),
        "b": _item("b", "bs", None),
        "c": _item("c", "af", "event_2"),
    }
    try:
        build_group_folds(meta, n_splits=2, seed=1, require_event_groups=True)
    except ValueError as exc:
        assert "strict leakage-safe split requested" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("strict grouping must reject missing event ids")


def test_fold_manifest_reports_leakage_audit():
    meta = {
        "a": _item("a", "af", "event_1"),
        "b": _item("b", "bs", "event_2"),
        "c": _item("c", "af", None),
    }
    manifest = build_group_folds(meta, n_splits=2, seed=1)
    audit = manifest["leakage_audit"]
    assert audit["event_grouped_chips"] == 2
    assert audit["chip_fallbacks"] == 1
    assert audit["strict_event_grouping"] is False
