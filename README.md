<img src="/logo.png" width="200">

# Solvmate 

A practical web application for the recommendation of organic solvents
based on the paper
*Solvmate - A hybrid physical/ML approach to solvent recommendation leveraging a rank-based problem framework* (https://chemrxiv.org/engage/chemrxiv/article-details/662f451f418a5379b0012795).

<img src="/figures/figure_webapp_2.svg" width="300" height="300">

Given a compound as SMILES, and set of solvents as IUPAC names,
the solvents are recommended in increasing solubility.



## Usage example:
The following animation shows a simple step-by-step guide (**domain not active (yet)**):

<img src="/figures/usage_animation.gif" width="300">

## Default credentials
The default credentials are user: "user", password: "solvmate"

Have a look in the auth.py module how additional users can be configured.


# References
We thank F. Broda from Leibniz-Institut f. Pflanzenbiochemie for the
development of MolPaintJS, a FOSS molecule editor that we use
for the convenient input of molecules.
