<!--<img src="/logo.png" width="200">-->
<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/js/hydro_banner.svg" alt="Solvmate 2.0 Logo" width="300"/>

# Solvmate 

A practical web application for the recommendation of organic solvents
based on the paper 
*Solvmate - A hybrid physical/ML approach to solvent recommendation leveraging a rank-based problem framework* 
(Digital Discovery **2024**, https://doi.org/10.1039/D4DD00138A)
(preprint: https://chemrxiv.org/engage/chemrxiv/article-details/662f451f418a5379b0012795).

<img src="/figures/figure_webapp_2.svg" width="300" height="300">

Given a compound as SMILES, and set of solvents as IUPAC names,
the solvents are recommended in increasing solubility.

# Getting Started
## Installation
After cloning the repository,
```
git clone https://github.com/Bayer-Group/solvmate.git
```
change into the main repository where there is a Makefile containing a listing
of all commonly used operations.

### Running with docker
We recommend building a docker container by issuing
```
make docker_build
```
from the main repository directory.

The server can then be run with
```
make docker_run
```
which will start a server at the port 8890.
In case another port is desired, adjust the docker command, e.g.
```
docker run --name solvmate-server -p 80:8890 solvmate:latest
```
would run the server on port 80, instead.

### Accessing the web app
Visit the web app at
```
http://127.0.0.1:8890/main
```

