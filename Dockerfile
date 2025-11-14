# https://stackoverflow.com/questions/72465421/how-to-use-poetry-with-docker

FROM python:3.14

# Configure Poetry
ENV POETRY_VERSION=2.2.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

# Install dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-root

EXPOSE 8888

# Run the app
COPY . /app
CMD [ "poetry", "run", "jupyter", "notebook", "./notebooks/analysis.ipynb", "--allow-root", "--ip=0.0.0.0", "--no-browser" ]