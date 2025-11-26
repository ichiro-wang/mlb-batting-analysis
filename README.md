# MLB Batting Analysis

## CMPT 459 Course Project

By Ichiro Wang and Ajax Mok.

## Repository Map

- [data](./data)
  - [raw](./data/raw)
    - [batting_2015_2024.csv](./data/raw/batting_2015_2024.csv)
  - [processed](./data/processed)
    - [batting_2015_2024_reference.csv](./data/processed/batting_2015_2024_reference.csv)
    - [batting_2015_2024_cleaned.csv](./data/processed/batting_2015_2024_cleaned.csv)
- [notebooks](./notebooks)
  - [analysis.ipynb](./notebooks/analysis.ipynb)
- [src](./src)
  - [classification.py](./src/classification.py)
  - [clustering.py](./src/clustering.py)
  - [data_loader.py](./src/data_loader.py)
  - [feature_selection.py](./src/feature_selection.py)
  - [hyperparameter.py](./src/hyperparameter.py)
  - [model_utils.py](./src/model_utils.py)
  - [outlier_detection.py](./src/outlier_detection.py)
  - [preprocessing.py](./src/preprocessing.py)
  - [validation.py](./src/validation.py)
  - [visualization.py](./src/visualization.py)
- [Dockerfile](./Dockerfile)
- [poetry.lock](./poetry.lock)
- [pyproject.toml](./pyproject.toml)
- [requirements.txt](./requirements..txt)
- [README.md](./README.md)

## Data Access Notes

- Raw data file can be found here -> [./data/raw/batting_2015_2024.csv](./data/raw/batting_2015_2024.csv).
- Source: [https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024](https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024)
  - Using batting data and ignoring pitching data.
- Check out the [FanGraphs Library](https://library.fangraphs.com/offense/offensive-statistics-list/) to learn more about the stats provided in the data.

## How To Run Notebook

### Using Docker

```bash
docker build . -t mlb
docker run -p 8888:8888 mlb
```

- Click the link that appears in the terminal. It should look like [http://127.0.0.1:8888/tree?token=...]()

### Using Poetry

- Install poetry [here](https://python-poetry.org/).

```bash
poetry install --no-root
poetry run ipython kernel install --name "mlb-batting-analysis-env" --user
poetry run jupyter notebook notebooks/analysis.ipynb
```

- Click the link that appears in the terminal. It should look like [http://127.0.0.1:8888/tree?token=...]()
- Or, ignore the third command and run locally via VSCode.

### Using Pip

- See [requirements.txt](./requirements..txt)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run locally via VSCode.
