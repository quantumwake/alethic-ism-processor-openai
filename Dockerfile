# Stage 1: Base Image with Miniconda
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

ARG CONDA_ISM_CORE_PATH
ARG CONDA_ISM_DB_PATH
#ARG GITHUB_REPO_URL

# copy the local channel packages (alethic-ism-core, alethic-ism-db)
COPY ${CONDA_ISM_DB_PATH} .
COPY ${CONDA_ISM_CORE_PATH} .

# extract the local channel directories
RUN tar -zxvf $CONDA_ISM_DB_PATH -C /
RUN tar -zxvf $CONDA_ISM_CORE_PATH -C /

# clone the api repo
ADD environment.yaml /app/repo/environment.yaml

# Move to the repository directory
WORKDIR /app/repo

# Force all commands to run in bash
SHELL ["/bin/bash", "--login", "-c"]

# install the conda build package in base
RUN conda install -y conda-build

# reindex local channel
RUN conda index /app/conda/env/local_channel

# Initialize the conda environment
RUN conda env create -f environment.yaml

# Initialize conda in bash config files:
RUN conda init bash

# Activate the environment, and make sure it's activated:
RUN echo "conda activate alethic-ism-processor-openai" > ~/.bashrc

# display information about the current activation
RUN conda info

# Install necessary dependencies for the build process
RUN conda install -y conda-build

#RUN conda clean --all -f -y
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict

# Install Conda packages not done by environment.yaml
##RUN conda install -y redis

# display all packages installed
RUN conda list

COPY entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/entrypoint.sh

# clone the api repo
#RUN git clone --depth 1 ${GITHUB_REPO_URL} repo
ADD . /app/repo

# delete any specific configuration file tied to the current build environment
RUN rm -rf .*

# Set the entrypoint script to be executed
ENTRYPOINT ["entrypoint.sh"]

# Run the pulsar consumer (ism processor for openai)
CMD ["python", "main.py"]

