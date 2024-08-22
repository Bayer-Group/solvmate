# Solvmate 2.0
<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/js/hydro_banner.svg" alt="Solvmate 2.0 Logo" width="600"/>

A practical web application for the recommendation of organic solvents
based on the paper 
*Solvmate - A hybrid physical/ML approach to solvent recommendation leveraging a rank-based problem framework* 
(Digital Discovery **2024**, https://doi.org/10.1039/D4DD00138A)
(preprint: https://chemrxiv.org/engage/chemrxiv/article-details/662f451f418a5379b0012795).

<img src="/figures/figure_webapp_2.svg" width="300" height="300">

Given a compound as SMILES, and set of solvents as IUPAC names,
the solvents are recommended in increasing solubility.

For the original version described in the publication, see v0.1 release under releases.

# Getting Started
## Installation
After cloning the repository,
```
git clone https://github.com/Bayer-Group/solvmate.git
```
change into the main repository where there is a Makefile containing a listing
of all commonly used operations.

### Running with docker
We recommend using docker, because it simplifies installation/running the web app.
Build the container with the command
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

### Usage
After visiting the main url, we are presented with the frontend:
<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/doc/usage_1.png" alt="screenshot" width="600"/>

First we click on "draw API" and are presented with an empty molecule editor canvas:

<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/doc/usage_2.png" alt="screenshot" width="600"/>

We draw lenacapavir as an example molecule:

<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/doc/usage_3.png" alt="screenshot" width="600"/>

After accepting the dialog a sketch of the molecule appears on the header of the main app:

<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/doc/usage_5.png" alt="screenshot" width="600"/>

Next, we can select one of five solvent sets, from the solvents option:

<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/doc/usage_6.png" alt="screenshot" width="600"/>

Solvents can be freely edited by the user in both IUPAC and SMILES formats. Commonly used abbreviations are also supported,
but we advise using IUPAC names that are convenient, while avoiding possible ambiguities caused from abbreviations.
Note that arbitrary solvents are supported.

<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/doc/usage_7.png" alt="screenshot" width="600"/>

Finally, we click on rank to obtain a ranking of the different solvents.

<img src="https://github.com/Bayer-Group/solvmate/blob/main/sm2/doc/usage_8.png" alt="screenshot" width="600"/>

In the above picture, different solvents are ranked according to their solubility. on the vertical axis, more
suitable solvents are shown on the top, less suitable solvents (low predicted solubility) are shown on the bottom.
The horizontal axis gives a prediction of the absolute solubility log S of the compound in units of mol / L.
Absolute solubility calculations have to be seen critical, as they are largely influenced by the crystal lattice energy,
which is difficult to estimate.

### Limitations
Limitations that are currently known:
- Both v1.0 and v2.0 models tend to be biased towards the usually good solvents, e.g. DMSO and
  and DMF often occupy the top rank, even for highly unpolar compounds.
- Very similar solvents are hard to differentiate (e.g. 2-propanol vs. 1-propanol vs. 1-butanol)
- Salts / charged groups are heavily underrepresented in the training data, and typically very large
  errors are observed here.[^A]
- Only works well for drug-like compounds.
Have a look at the underlying datasets to see on what chemical space these models are trained on:
Open Notebook Science Solublity Challenge[^1] 
and *Towards the Prediction of Drug Solubility in Binary Solvent Mixtures at Various Temperatures Using Machine Learning*
by Bao et al.[^2]

> In case any other limitations/bugs are found, please file an issue in the github repository here:
> https://github.com/Bayer-Group/solvmate/issues
> or contact us by email otherwise (see publications for contact information). 

[^1]: https://figshare.com/articles/dataset/Open_Notebook_Science_Challenge_Solubility_Dataset/1514952?file=2217769
[^2]: https://www.researchsquare.com/article/rs-4170106/v1
[^A]: This is really mostly a limitation coming from the limited available data. We would be very happy to hear from (publically available) organic solubility datasets with charged solutes, and
would readily integrate them into training of the models.


