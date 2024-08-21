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
    pip install torch==2.2.0 && \
    pip install dgl==2.1.0 && \
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
    pip install rdkit"

# Set the default command to bash
CMD ["/bin/bash"]