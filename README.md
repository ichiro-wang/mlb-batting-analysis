# MLB Batting Analysis

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
  - [visualization.py](./src/visualization.py)
  - [outlier_detection.py](./src/outlier_detection.py)
  - [feature_selection.py](./src/feature_selection.py)
  - [preprocessing.py](./src/preprocessing.py)
  - [hyperparameter.py](./src/hyperparameter.py)
- [Dockerfile](./Dockerfile)
- [poetry.lock](./poetry.lock)
- [pyproject.toml](./pyproject.toml)
- [requirements.txt](./requirements..txt)
- [README.md](./README.md)

## Data Access Notes

- Source: [https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024](https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024)
  - Using batting data and ignoring pitching data.
  - File can be found here -> [./data/raw/batting_2015_2024.csv](./data/raw/batting_2015_2024.csv).
- Check out the [FanGraphs Library](https://library.fangraphs.com/offense/offensive-statistics-list/) to learn more about the stats provided in the data.

## How To Run Notebook

### Using Docker

```bash
docker build . -t mlb
docker run -p 8888:8888 mlb
```

- Open [http://localhost:8888/tree](http://localhost:8888/tree) on browser to view notebook.

### Using Poetry

- Install poetry [here](https://python-poetry.org/).

#### Install Libraries

```bash
poetry install --no-root
poetry run ipython kernel install --name "mlb-batting-analysis-env" --user
```

#### Run Notebook

```bash
poetry run jupyter notebook notebooks/analysis.ipynb
```

- Open [http://localhost:8888/tree](http://localhost:8888/tree) on browser to view notebook.

### Using Pip

See [requirements.txt](./requirements..txt)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
