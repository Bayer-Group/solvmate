# Use the official Ubuntu base image
FROM ubuntu:22.04

# Install necessary packages and dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    bzip2 \
    git \
    build-essential \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y wget && apt-get install -y unzip && apt-get update && apt-get install -y libxrender1 libxext6 && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y openjdk-11-jdk && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh 
# Add Conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Create a new conda environment
RUN conda create -n myenv -c conda-forge dgl --yes && \
    conda clean -a


# Activate the environment and install the required packages
# conda install -y -c dglteam/label/th21_cpu dgl && \
RUN /bin/bash -c "source activate myenv && \
    conda install -c dglteam/label/th21_cpu dgl && \
    pip install torchdata && \
    pip install dgllife==0.3.2 && \
    pip install fastapi[standard] && \
    pip install psutil && \
    pip install numpy && \
    pip install pandas && \
    pip install ipython && \
    pip install joblib && \
    pip install pytest && \
    pip install scipy && \
    pip install tornado && \
    pip install tqdm && \
    pip install matplotlib==3.5.2 && \
    pip install seaborn==0.11.2 && \
    pip install xmltodict && \
    pip install scikit-learn && \
    pip install xtbf && \
    pip install rdkit"

COPY . /solvmate

ENV PYTHONPATH="/solvmate/src:${PYTHONPATH}"
ENV PYTHONPATH="/solvmate/:${PYTHONPATH}"

# Set the default command to bash
# CMD ["/bin/bash"]

# bash -c "source activate myenv" && fastapi dev --port 8890 --host 0.0.0.0 /solvmate/sm2/app.py
# source activate myenv
# ENTRYPOINT [ "python", "/solvmate/src/solvmate/app/app.py" ]
# ENTRYPOINT "source activate myenv && fastapi dev --port 8890 --host 0.0.0.0 /solvmate/sm2/app.py"
ENTRYPOINT ["bash", "/solvmate/run_app_docker.sh"]
