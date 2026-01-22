## Known Limitations

- The v1 baseline (TF-IDF + Logistic Regression) is sensitive to spelling errors and typos.
  Minor misspellings can significantly change predicted probabilities due to token mismatch.
  This is tracked by an xfail robustness test and is expected to improve in v2 (transformer).
