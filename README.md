# Alethic Instruction-Based State Machine (OpenAI Processor)

The following processor waits on events from pulsar (but can be extended to use kafka or any pub/sub system)

# Installation via conda
Checkout the ISM core and ISM db repository and build for  
- * modify the environment_local.yml to reflect your environment path for the ISM core and ISM db packages.
- 
- conda env create -f environment_local.yml  
- conda activate alethic-ism-processor-openai

# Installation
- conda install pulsar-client
- conda install pydantic
- conda install python-dotenv
- conda install openai
- conda install tenacity
- conda install pyyaml
- conda install psycopg2

# Remote Alethic Dependencies (if avail otherwise build locally)
- conda install alethic-ism-core
- conda install alethic-ism-db

# Local Dependency (build locally if not using remote channel)
- conda install -c ~/miniconda3/envs/local_channel alethic-ism-core
- conda install -c ~/miniconda3/envs/local_channel alethic-ism-db

