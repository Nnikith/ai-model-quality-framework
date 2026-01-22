import pandas as pd

from fakenews.data.ingest_isot import load_isot, add_ids, stratified_split, to_canonical
from fakenews.data.validate import validate_dataframe


def test_isot_ingestion_pipeline(tmp_path):
    true_csv = tmp_path / "True.csv"
    fake_csv = tmp_path / "Fake.csv"

    # Keep it small but enough to allow splitting
    df_true = pd.DataFrame(
        {
            "title": ["Real A", "Real B", "Real C", "Real D"],
            "text": ["real a", "real b", "real c", "real d"],
            "subject": ["politics", "world", "politics", "world"],
            "date": ["January 1, 2020", "January 2, 2020", "January 3, 2020", "January 4, 2020"],
        }
    )
    df_fake = pd.DataFrame(
        {
            "title": ["Fake A", "Fake B", "Fake C", "Fake D"],
            "text": ["fake a", "fake b", "fake c", "fake d"],
            "subject": ["politics", "world", "politics", "world"],
            "date": ["January 5, 2020", "January 6, 2020", "January 7, 2020", "January 8, 2020"],
        }
    )

    df_true.to_csv(true_csv, index=False)
    df_fake.to_csv(fake_csv, index=False)

    df = load_isot(true_csv, fake_csv)
    df = add_ids(df)
    df = stratified_split(df, train_size=0.5, val_size=0.25, test_size=0.25, seed=42)
    df = to_canonical(df)

    result = validate_dataframe(df, dataset_name="unit_test")
    assert result.passed is True

    # Basic invariants
    assert set(df["label"].unique()).issubset({0, 1})
    assert set(df["split"].unique()).issubset({"train", "val", "test"})
    assert df["id"].is_unique
    assert df["source"].nunique() == 1
