FROM ubuntu:22.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && apt-get install -y unzip && apt-get update && apt-get install -y libxrender1 libxext6 && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
    
#RUN export SSL_NO_VERIFY=1 
#RUN conda config --set ssl_verify no
RUN conda config --add channels conda-forge
RUN conda config --add channels bioconda
RUN conda create -y --name solvmate python=3.10
#RUN conda config --add channels conda-forge
RUN activate solvmate 
RUN conda install -y pip
RUN conda install -y -c conda-forge xtb
RUN conda install -y -c bioconda opsin

COPY . /solvmate

# copy only the public domain data over into the docker container
COPY ./data/xtb_features__public_domain.db /solvmate/data/xtb_features.db
COPY ./data/training_data__public_domain.db /solvmate/data/training_data.db
#RUN cd ~/solvmate/ 

#RUN pip install -r /solvmate/requirements.txt 
RUN pip install psutil
RUN pip install numpy
RUN pip install pandas
RUN pip install ipython
RUN pip install joblib
RUN pip install py-spy
RUN pip install pytest
RUN pip install scipy
RUN pip install tornado
RUN pip install tqdm
RUN pip install matplotlib==3.5.2
RUN pip install seaborn==0.11.2
RUN pip install xmltodict
RUN pip install scikit-learn 
RUN pip install rdkit-pypi
RUN pip list

#RUN python /solvmate/scripts/download_models.py
RUN wget "https://github.com/Bayer-Group/solvent-mate/releases/download/v0.1/public_data_models.zip" -O /solvmate/data/archive.zip
RUN unzip /solvmate/data/archive.zip
RUN rm /solvmate/data/archive.zip

ENV PYTHONPATH="/solvmate/src:${PYTHONPATH}"
ENTRYPOINT [ "python", "/solvmate/src/solvmate/app/app.py" ]
#CMD [ "/bin/bash" ]
