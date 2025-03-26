# Use the uv base image with Python 3.12 on Debian
FROM ghcr.io/astral-sh/uv:debian

# Set the working directory
WORKDIR /app/repo

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt /app/repo

# Copy the rest of your application code into /app
COPY . /app/repo/

# Create a virtual environment using uv
RUN uv venv

# Activate the venv and install dependencies from PyPI
RUN . .venv/bin/activate && \
    apt-get update && \
    apt-get install -y git cmake build-essential libpq-dev python3-dev && \
    uv pip install --upgrade pip && \
    uv pip install -r requirements.txt

# Copy the entrypoint script and ensure it is executable
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script to be executed on container start
ENTRYPOINT ["entrypoint.sh"]

# Run the pulsar consumer (ism processor for state-router)
CMD ["python", "main.py"]

