# Dataset

Place the cleaned dataset at:

`data/processed/cleaned_news.csv`

Required columns:

| Column | Description |
|---|---|
| content | Original news text |
| clean_text | Preprocessed text |
| label | 0 = Fake News, 1 = Real News |

Raw ISOT/LIAR/Sinhala source files are not committed to this repository. Add them locally according to their license/usage requirements.

The notebooks validate the required columns before training.
