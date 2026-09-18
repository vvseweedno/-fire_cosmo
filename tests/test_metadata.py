from pathlib import Path

from wildfire.metadata import read_meta_csv


def test_read_test_meta_csv(tmp_path: Path):
    path = tmp_path / "meta.csv"
    path.write_text(
        "chip_id,kind,width,height,gsd,valid_frac,cloud_frac\n"
        "AF_te_000001,af,256,256,375,1.0,nan\n"
        "BS_te_000001,bs,512,512,20,0.99,0.1\n",
        encoding="utf-8",
    )
    meta = read_meta_csv(path)
    assert meta["AF_te_000001"].shape == (256, 256)
    assert meta["BS_te_000001"].kind == "bs"
