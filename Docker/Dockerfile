FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
LABEL authors="Gizzatovamir"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get update  \
    && apt install -y git python3-pip python3-dev python3.8-venv nvidia-container-toolkit cuda-drivers make \
    && pip install poetry

COPY poetry.lock pyproject.toml ./
COPY src ./src
COPY Makefile ./src
WORKDIR src
RUN poetry install --no-root && poetry run pip install git+https://github.com/ros/genpy.git

ENTRYPOINT ["make", "local"]
#ENTRYPOINT ["sleep", "100000000"]