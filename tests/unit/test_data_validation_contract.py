import pandas as pd
import pytest
from fakenews.data.validate import validate_dataframe

def make_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a1", "a2", "a3"],
            "text": ["this is real news", "this is fake news", "another real item"],
            "label": [0, 1, 0],
            "source": ["isot", "isot", "isot"],
            "subject": ["politics", "politics", None],
            "date": [None, None, None],
            "split": ["train", "val", "test"],
        }
    )


def test_schema_required_columns_present():
    df = make_valid_df()
    required = {"id", "text", "label", "source", "split"}
    assert required.issubset(df.columns)


def test_label_is_binary_int():
    df = make_valid_df()
    assert set(df["label"].unique()).issubset({0, 1})
    assert pd.api.types.is_integer_dtype(df["label"])


def test_text_not_null_or_empty():
    df = make_valid_df()
    assert df["text"].isna().sum() == 0
    assert (df["text"].str.strip() == "").sum() == 0


def test_split_values_valid():
    df = make_valid_df()
    assert set(df["split"].unique()).issubset({"train", "val", "test"})


def test_id_unique():
    df = make_valid_df()
    assert df["id"].is_unique


def test_rejects_missing_required_column():
    df = make_valid_df().drop(columns=["text"])
    required = {"id", "text", "label", "source", "split"}
    assert not required.issubset(df.columns)


def test_rejects_non_binary_label():
    df = make_valid_df()
    df.loc[0, "label"] = 2
    assert not set(df["label"].unique()).issubset({0, 1})


def test_rejects_empty_text():
    df = make_valid_df()
    df.loc[1, "text"] = "   "
    assert (df["text"].str.strip() == "").sum() > 0

def test_validate_dataframe_passes_on_valid_data():
    df = make_valid_df()
    result = validate_dataframe(df)

    assert result.passed is True
    assert result.errors == []
    assert result.stats["rows_total"] == 3


def test_validate_dataframe_fails_on_missing_columns():
    df = make_valid_df().drop(columns=["text"])
    result = validate_dataframe(df)

    assert result.passed is False
    assert any("Missing required columns" in e for e in result.errors)


def test_validate_dataframe_fails_on_bad_label():
    df = make_valid_df()
    df.loc[0, "label"] = 5

    result = validate_dataframe(df)
    assert result.passed is False
    assert any("binary" in e.lower() for e in result.errors)


def test_validate_dataframe_fails_on_invalid_split():
    df = make_valid_df()
    df.loc[0, "split"] = "training"

    result = validate_dataframe(df)
    assert result.passed is False
    assert any("split" in e.lower() for e in result.errors)