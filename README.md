# mlb-batting-analysis

By Ichiro Wang and Ajax Mok

## Dataset
- Source: [https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024](https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024)
- Using batting data and ignoring pitching data

### Number of Samples and Features
- Samples: 14051
- Features: 321
    - Numerical: AVG, HR, OPS, wRC+, etc.
    - Categorical: Name, Team.

## Installing packages

```bash
poetry install --no-root
poetry run ipython kernel install --name "mlb-batting-analysis-env" --user
```

## Run a notebook

```bash
poetry run jupyter notebook <notebook_name.ipynb>
```
