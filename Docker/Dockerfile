FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
LABEL authors="Gizzatovamir"


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get update && apt install -y  \
    git python3-pip python3-dev \
    nvidia-container-toolkit  \
    cuda-drivers && pip install poetry \
RUN apt-get install make
RUN pip install poetry
COPY poetry.lock pyproject.toml ./
COPY src ./src
COPY Makefile ./src
WORKDIR src
RUN poetry install

ENTRYPOINT ["make", "local"]