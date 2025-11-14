# MLB Batting Analysis

By Ichiro Wang and Ajax Mok.

## Dataset

- Source: [https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024](https://www.kaggle.com/datasets/beckettnewton/mlb-hitting-pitching-data-2015-2024)
- Using batting data and ignoring pitching data.

### Number of Samples and Features

- Samples: 14051
- Features: 321
  - Numerical: AVG, HR, OPS, wRC+, etc.
  - Categorical: Name, Team.

## Running Notebook

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
