# Install XTB
For M1 Macs:
Install using homebrew as described here: https://github.com/grimme-lab/homebrew-qc

Binaries can be downloaded here https://github.com/grimme-lab/xtb/releases/tag/v6.6.0 
or directly installed using conda:
```
conda config --add channels conda-forge
conda search xtb --channel conda-forge
```
or alternatively
```
conda install mamba -c conda-forge
```
Current results were obtained using
xtb version 6.5.1.

# Install OPSIN
Used to convert iupac names -> smiles

As explained here https://anaconda.org/bioconda/opsin
this can be achieved via:
```
conda install -c bioconda opsin
```
Alternatively, jar files can be obtained from
https://github.com/dan2097/opsin/releases.

# Install pip requirements
```
cd path/to/solvmate/
pip install -r requirements.txt
```


# Advanced

## Building Metadata DB for the Nearest Neighbor Search
To rebuild the metadata database, 7-zip is required.
On mac this can be installed with
```
brew install p7zip
```
However, as we (will) make precompiled metadata 
available this should never be necessary from
an end-user perspective.

## Enabling the Ranked Pairs / PageRank algorithms
To enable the ranked pairs or pagerank advanced pair
resolution algorithms the pair_rank module within
`solvmate/src/solvmate/pair_rank` needs to be 
compiled and installed as a c extension on 
the target system. At the moment, this requires a 
working version of GNU make and a compiler on the
target system.
Running
```
cd solvmate/src/solvmate/pair_rank && make
```
should both build, install and test this extension.

TODO: The pagerank algo is actually implemented in
      pure python so should be much simpler to
      distribute. We should therefore cut it out
      of this ugly c extension hack and provide
      it directly within the ranksolv package.
      So that this annoying step is avoided... 



