from solvmate import *

"""
A script that generates the SMILES corresponding
to very commonly used solvent names.
Used to display nice chemical nomenclature in the UI.
"""

if __name__ == "__main__":
    smi_to_name = {}
    solvs = """
acetic acid
acetone
acetonitrile
benzene
1-butanol
2-butanol
2-butanone
t-butyl alcohol
carbon tetrachloride
chlorobenzene
chloroform
cyclohexane
1,2-dichloroethane
diethylene glycol
diethyl ether
diglyme
glyme
dimethylformamide
dimethyl sulfoxide
1,4-dioxane
ethanol
ethyl acetate
ethylene glycol
glycerin
hexamethylphosphoramide
hexamethylphosphorous triamide
hexane
methanol
methyl t-butylether
methylene chloride
N-methyl-2-pyrrolidinone
nitromethane
pentane
1-propanol
2-propanol
pyridine
tetrahydrofuran
toluene
triethyl amine
water
o-xylene
m-xylene
p-xylene
1-pentanol
1-hexanol
1-heptanol
1-octanol
2-pentanol
2-hexanol
2-octanol
3-octanol
2-ethylhexanol
4-methylhexan-2-ol
3-heptanol
3-hexanol
pentane
hexane
heptane
octane
nonane
decane
undecane
dodecane
methylcyclohexane
neopentane
isobutane
methyl acetate
2-methoxyethanol
2-ethoxyethanol
iso-butanol
iso-butan-2-ol
neo-pentanol
propylacetate
butylacetate
propylformat
ethylformat
bis-tertbutylether
di-isopropylether
2,2-dimethyl-4-methylpentane
methyl isobutyl ketone
methyl tertbutyl ketone
methyl neopentyl ketone
dibutyl ether
isoamyl alcohol
2-isopropoxyethanol
2-propoxyethanol
2-butoxyethanol
2-methyl pentanol
4-methyl pentan-2-ol
decanol
hexadecane
nonaethylene glycol
octaethylene glycol
heptaethylene glycol
decaethylene glycol
hexaethylene glycol
"""

    for lne in reversed(solvs.strip().split("\n")):
        name = lne.strip()
        print(name)
        smi = name_to_canon(name)
        smi_to_name[smi] = name

    with open(DATA_DIR / "smi_to_name.json", "wt") as fout:
        json.dump(obj=smi_to_name, fp=fout)
