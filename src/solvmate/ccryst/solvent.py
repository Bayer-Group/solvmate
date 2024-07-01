"""
A module that takes care of representing and parsing solvents.
Makes it possible for us to not have to worry about all the
different ways of notation used for the same solvent!

As there is already the nice OPSIN package (MIT licensed),
it is rather trivial for us to recognize all solvents
in a fault-tolerant fashion:
- Parse the IUPAC name using OPSIN to obtain the SMILES.
- Canonicalize the SMILES -> look up the corresponding solvent
  from the table here.
- This way, we can also receive the canonical name of a solvent:
  convert the iupac of the solvent to smiles, then search through
  all known solvents and see whether the smiles matches, then
  return that name as the canonical name for that solvent!
"""
import sys
import numpy as np
from subprocess import PIPE, Popen
from typing import Optional, Tuple

from solvmate import *
from solvmate.ccryst.utils import Silencer

solvents = """
1-Decanol
1-Dodecanol
1-Hexanol
1-hexanol
1-Methyl-2-pyrrolidinon
1-Methyl-2-pyrrolidinone
1-Methylimidazol
1-Methylimidazole
1-Methylpiperazin
1-Methylpiperazine
1-Methylpiperidin
1-Methylpiperidine
1-Methylpyrrol
1-Methylpyrrole
1-Methylpyrrolidin
1-Methylpyrrolidine
1-Nitropropan
1-Nitropropane
1-Nonanol
1-Octanol
1-Pentanol
1-Propanol
1-Propyl acetate
1,1,1-Trichloroethan
1,1,1-Trichloroethane
1,2-Dichlorbenzol
1,2-Dichlorobenzene
1,2-Dichloroethan
1,2-Dichloroethane
1,2-Propandiol
1,2-Propanediol
1,3-Dimethoxybenzene
1,3-Dimethoxybenzol
1,3-Dioxalane
1,3-Dioxolan
1,3-Propandiol
1,3-Propanediol
1,4-Dioxan
1,4-Dioxane
2-(2-Ethoxyethoxy)ethanol
2-(Methoxyethoxy)ethanol
2-Butanol
2-Butanon
2-Butanone
2-Butanone
2-Butoxyethanol
2-Chloroethanol
2-Ethoxyethanol
2-Ethoxyethanol
2-Ethoxyethyl acetate
2-Ethoxyethylacetat
2-Ethoxyethylacetate
2-Ethyl-1-hexanol
2-Heptanon
2-Heptanone
2-Hexanol
2-Hexanon
2-Hexanone
2-Methoxyethanol
2-Methoxyethyl acetate
2-Methoxyethylacetat
2-methyl tetrahydrofuran
2-Methyl-1-butanol
2-Methyl-1-pentanol
2-Methyl-1-propanol
2-Methyl-1-propanol
2-Methyl-1-propanol
2-Methyl-2-butanol
2-Methyl-2-propanol
2-Methyl-THF
2-Methyltetrahydrofuran
2-Octanol
2-Pentanol
2-pentanol
2-pentanon
2-Pentanone
2-Phenoxyethanol
2-Picolin
2-Picoline
2-Propanol
2-Propanol
2,2,2-Trichloroethanol
2,2,2-Trifluoroethanol
2,2,2-Trimethylpentane
2,2,4-Trimethylpentan
2,2,4-trimethylpentane
2,2,4-Trimethylpentane
2,2,4-Trimethylpentane
2,5-Dimethylpyrazin
2,5-Dimethylpyrazine
3-Hexanol
3-Methyl-1-butanol
3-Methyl-1-pentanol
3-Methyl-2-butanol
3-Methyl-2-pentanol
3-Methyl-3-pentanol
3-Methylphenol
3-Pentanol
3-Pentanon
3-Pentanone
3-Picolin
3-Picoline
4-Heptanon
4-Heptanone
4-methylanisol
4-methylanisole
4-Methylpyrimidin
4-Methylpyrimidine
4-Picolin
4-Picoline
Acetic acid
Aceticacid
Aceton
Acetone
acetone
Acetonitril
Acetonitrile
acetonitrile
ACN
AcOEt
Ameisensäure
Anisol
Anisole
anisole
Benzonitril
Benzonitrile
Benzyl alcohol
Benzylalkohol
Bis(2-ethoxyethyl)ether
Bis(2-methoxyethyl)ether
Bromoform
Butanol
Butylamin
Butylamine
Chlorobenzene
Chlorobenzol
Chloroform
Chloroform
Cyclohexan
Cyclohexane
cyclohexane
Cyclohexanol
Cyclohexanon
Cyclohexanone
Cyclopentanol
Decahydronaphtalene
Di-n-butylether
Dichlormethan
Dichloromethan
Dichloromethane
Dichlorométhane
Dichlotmethan
Diethyl ether
Diethylamin
Diethylamine
Diethylcarbonat
Diethylcarbonate
Diethylenglycol
Diethylether
Diethyloxalate
Diisopropylether
Diisopropylether
Dimethoxyethan
Dimethoxyethane
Dimethylacetamid
Dimethylacetamide
Dimethylformamid
Dimethylformamide
Dimethylsulfoxide
Dioxan
Dioxan
Dioxane
dioxane
DIPE
DMF
DMSO
Dodecan
Dodecane
EE
Essigsäure
Ethanol
Ether
Ethoxybenzene 
Phenetole
EthylAcetate
Ethylbutyrat
Ethylbutyrate
Ethylformate
EthylLactate
Ethylmethoxyacetat
Ethylmethoxyacetate
Ethylacetat
EthylAcetate
Ethylene glycol
ethyleneglycol
Ethylenglycol
Ethylformat
Ethyllactat
EtOAc
EtOH
EtOH
Etylenglykol
Fluorbenzol
Fluorobenzene
Formamide
Formic acid
Formic Acid
Formicacid
Heptan
Heptane
Hexadecan
Hexadecane
Hexafluorobenzene
Hexafluorobenzol
Hexafluoroisopropanol
Hexan
HFIPA
HSO4
iPrAc
Isobutylacetat
Isobutylacetate
Isopropanol
isopropyl acetate
Isopropyl acetate
Isopropylacetat
isopropylacetate
m-Xylene
m-xylene
m-xylene
m-xylene
m-Xylol
MEK
MeOH
Mesitylen
Mesitylene
Methanol
methanol
Methoxycyclopentane
Methyl butyrat
Methyl butyrate
Methyl ethyl ketone
Butanone
Methyl propanoate
Methyl Salicylat
Methyl Salicylate
Methyl t-butyl ether
Methyl t-butylether
Methyl-THF
Methylt-butylether
Morpholin
Morpholine
MtBE
MTBE
n-BuOH
n-Butanol
n-Butylacetat
n-Butylacetate
n-Decan
n-Decane
n-Heptan
n-Heptan
n-Heptane
n-heptane
n-heptane
N-Methylmorpholin
N-Methylmorpholine
n-Octan
n-Octane
n-Pentan
N,N-Dimethylformamid
NaHCO3
Nitroethan
Nitroethane
Nitromethan
Nitromethane
o-Xylene
o-Xylol
p-Xylene
p-Xylol
Pentafluoropropyl alcohol
Pentan
Pentane
Perfluoroctan
Perfluorooctane
Petrolether
Phenethyl alcohol
phosphoric acid
Propionic acid
Propionitril
Propionitrile
Propyl acetate
Propyl formate
Propylencarbonat
Propylene Carbonate
Propylformiat
Pyridine
t-Butyl alcohol
t-Butyl alkohol
t-Butylalcohol
tert-Butanol
tert-butyl alcohol
tert-Butylacetate
tert-butylalcohol
tert.Butanol
Tetraethylenglycol
Tetrahydrofuran
Tetrahydrofuran
Tetramethylharnstoff
Tetramethylurea
THF
Toluene
toluene
Toluol
Trichloroethylen
Trichloroethylene
Triethylamine
Veratrole
1,2-Dimethoxybenzene
water
Water
Wasser
H2O
H2SO4
H3PO4
HCl
""".strip().split(
    "\n"
)

import os
import rdkit

KNOWN_ACRONYMS = {
    # "n:ethanol": "ethanol", # why??? TODO: keep this? its for mantis
    "ethyl_acetate": "ethylacetate",
    "Ameisensäureethylester": "formic acid ethyl ester",
    "Ameisensäuremethylester": "formic acid methyl ester",
    "ACN": "acetonitrile",
    "AcOEt": "ethylacetate",
    "AcOH": "aceticacid",
    "EtOAc": "ethylacetate",
    "EE": "diethylether",
    "Et2O": "diethylether",
    "EA": "ethylacetate",
    "DMF": "N,N-dimethylformamide",
    "DMSO": "dimethylsulfoxide",
    "DCM": "dichloromethane",
    "THF": "tetrahydrofuran",
    "wasser": "water",
    "Wasser": "water",
    "EtOH": "ethanol",
    "c2h5oh": "ethanol",
    "c6h6": "benzene",
    "etoh": "ethanol",
    "MeOH": "methanol",
    "meoh": "methanol",
    "PrOH": "n-propanol",
    "iPrOH": "isopropanol",
    "i-PrOH": "isopropanol",
    "i-proh": "isopropanol",
    "DIPE": "diisopropylether",
    "MTBE": "methyl-tert-butylether",
    "H2O": "water",
    "H2SO4": "sulfuric acid",
    "H3PO4": "phosphoric acid",
    "HCl": "hydrochloric acid",
    "MEK": "methylethylketone",
    "NaHCO3": "sodium hydrogen carbonate",
    "Na2CO3": "disodium carbonate",
    "choloroform": "trichloromethane",
    "MtBE": "methyltertbutylether",
    # Typos / inaccuracies:
    "Methyl-THF": "2-methyl-tetrahydrofuran",
    "Dichlotmethan": "dichloromethane",
    "ether": "diethylether",
    "naphtalene": "naphthalene",
    "mehanol": "methanol",
}

KNOWN_ACRONYMS = {k.lower(): v for k, v in KNOWN_ACRONYMS.items()}


def fix_german_halogens(iupac):
    for halogen in [
        "chlor",
        "brom",
        "fluor",
        "iod",
    ]:
        match = re.search(halogen + "[^o]", iupac)
        if match:
            start, end = match.start(), match.end()
            iupac = iupac[0:start] + halogen + "o" + iupac[end - 1 :]
    return iupac


def common_replacements(iupac):
    iupac = iupac.lower()
    if iupac in KNOWN_ACRONYMS:
        return KNOWN_ACRONYMS[iupac]
    iupac = iupac.replace("benzol", "benzene")
    iupac = iupac.replace("säure", "acid")
    iupac = iupac.replace("ameisen", "formic")
    iupac = iupac.replace("essig", "acetic")
    iupac = iupac.replace("alkohol", "alcohol")
    iupac = iupac.replace("harnstoff", "urea")
    iupac = fix_german_halogens(iupac)

    return iupac


import re
import json
from os.path import expanduser
from pathlib import Path
from rdkit import Chem

home = expanduser("~")
_cache_file = Path(home) / ".solvents_cache.txt"
_cache = {}
if _cache_file.exists():
    _cache = json.loads(_cache_file.read_text())


def canonicalize_smiles(smi):
    return Chem.CanonSmiles(smi)


def safe_iupac_to_smiles(iupac: str) -> str:
    try:
        return iupac_to_smiles(iupac)
    except:
        return ""


def iupac_to_smiles(iupac: str) -> str:
    """
    Converts the given IUPAC string into SMILES.
    Takes care of common german naming issues by
    applying several heuristics that should work
    with most german iupac namings.

    Caches computations into a local cache file
    in the users home directory as OPSIN iupac
    nomenclature parsing is quite slow (several
    seconds per entry). This is why the decision
    was made to also cache failed computations!

    >>> iupac_to_smiles('Methanol')
    'CO'
    >>> iupac_to_smiles('Methan-1-ol')
    'CO'
    >>> iupac_to_smiles('methanol')
    'CO'
    >>> iupac_to_smiles('MeOH')
    'CO'
    >>> iupac_to_smiles('chloroform')
    'ClC(Cl)Cl'

    :param iupac: The iupac to convert
    :return: The smiles as a string. Empty string in case of errors.
    """
    global _cache
    global _cache_file
    if iupac in _cache:
        return _cache[iupac]
    if common_replacements(iupac) in _cache:
        return _cache[common_replacements(iupac)]

    smi, err = direct_iupac_to_smiles(iupac)

    if smi:
        smi = canonicalize_smiles(smi)
        _cache[iupac] = smi
        with open(_cache_file, "wt") as fout:
            fout.write(json.dumps(_cache))
        return smi
    else:
        iupac = common_replacements(iupac)

        smi, err = direct_iupac_to_smiles(iupac)

        if err:
            print(
                f"I could not parse the following iupac input: '{iupac}'.",
                file=sys.stderr,
            )
            print(
                "please add a corresponding substitution to KNOWN_ACRONYMS ",
                file=sys.stderr,
            )
            print(
                "or to the known substitution rules in case this is a known solvent. ",
                file=sys.stderr,
            )
            print(
                " details: ",
                file=sys.stderr,
            )
            print(
                err,
                file=sys.stderr,
            )

        if smi:
            smi = canonicalize_smiles(smi)

        _cache[iupac] = smi
        with open(_cache_file, "wt") as fout:
            fout.write(json.dumps(_cache))
        return smi


def direct_iupac_to_smiles(iupac) -> Tuple[str, str]:
    """
    Forces the actual parsing of the IUPAC via
    the OPSIN tool

    >>> direct_iupac_to_smiles("hexan-1-ol")
    ('C(CCCCC)O', '')
    >>> direct_iupac_to_smiles("this-is-invalid!!!")#doctest:+ELLIPSIS
    ('', 'this-is-invalid!!! is unparsable due to ...

    :param iupac: The IUPAC string to parse
    :return: tuple consisting of
        - the resulting SMILES string (empty if failure)
        - the error message (empty if success)
    """
    tmp_fle = random_fle("in")
    with open(tmp_fle, "w") as fout:
        fout.write(iupac)
    p = Popen(
        f"{OPSIN_CMD} " + str(tmp_fle.resolve()), shell=True, stdout=PIPE, stderr=PIPE
    )
    smi, err = p.communicate()
    return smi.decode("utf-8").strip(), err.decode("utf-8").strip()


def old_direct_iupac_to_smiles(iupac) -> str:
    """
    Forces the actual parsing of the IUPAC via
    the OPSIN tool
    :param iupac: The IUPAC string to parse
    :return: The resulting SMILES string.
    """
    tmp_fle = random_fle("in")
    with open(tmp_fle, "w") as fout:
        fout.write(iupac)
    smi = os.popen(f"{OPSIN_CMD} " + str(tmp_fle.resolve())).read()
    return smi


def smiles_to_name(smi: str) -> Optional[str]:
    if not smi:
        return None
    for solvent in solvents:
        if iupac_to_smiles(solvent) == smi:
            return solvent
    return None


def canonical_solvent_name(iupac: str) -> Optional[str]:
    """
    Gives the canonical solvent name so that we have a
    standardized name for every solvent in our datasets.
    >>> canonical_solvent_name("water")
    'water'
    >>> canonical_solvent_name("wasser")
    'water'
    >>> canonical_solvent_name("THF")
    'Tetrahydrofuran'
    >>> canonical_solvent_name("n-hexane")
    'Hexan'
    >>> canonical_solvent_name("n-hexan")
    'Hexan'

    :param iupac:
    :return:
    """
    smi = iupac_to_smiles(iupac)
    canonical_iupac = smiles_to_name(smi)
    return canonical_iupac


def canonical_solvent_name_else_smiles(smi: str) -> str:
    name_attempt = smiles_to_name(smi)
    if name_attempt:
        return name_attempt
    else:
        return smi


def solvent_mixture_iupac_to_smiles(
    solvent_mixture_iupac: str,
):  # -> list[str]:
    """
    Converts a solvent mixture description into a list of the smiles
    of the corresponding chemical components.

    >>> solvent_mixture_iupac_to_smiles('EtOH / Wasser 1:1')
    ['CCO', 'O']
    >>> solvent_mixture_iupac_to_smiles('EtOH/MeOH 1:1')
    ['CCO', 'CO']
    >>> solvent_mixture_iupac_to_smiles('Wasser Methanol 1:1')
    ['CO', 'O']
    """
    # If it is just a iupac name directly, it just refers
    # to a single solvent. So we can redirect to that
    # functionality. If this fails, e.g. because this
    # solvent descriptor is actually a solvent mixture,
    # then iupac_to_smiles will return an emtpy string.
    # This would indicate to us that we need to apply
    # further processing to this solvent (set) descriptor.
    if len(solvent_mixture_iupac.split()) < 3 and (
        " : " not in solvent_mixture_iupac
        and " / " not in solvent_mixture_iupac
        and safe_iupac_to_smiles(solvent_mixture_iupac).strip()
        or "/" not in solvent_mixture_iupac
        and ":" not in solvent_mixture_iupac
    ):
        smi = safe_iupac_to_smiles(solvent_mixture_iupac)
        if smi:
            return [smi]

    # When we arrive here we know that this is most likely
    # a solvent set, though it could also indicate a typo
    # or unparseable single solvent.
    solvent_mixture_iupac = solvent_mixture_iupac.replace(":", " : ").replace(
        "/", " / "
    )
    parts = []
    for part in solvent_mixture_iupac.split():
        smi_part = iupac_to_smiles(part)
        if smi_part:
            parts.append(smi_part)
    return list(sorted(set(parts)))


def iupac_solvent_mixture_to_amounts(solvent_mixture_iupac: str) -> dict:
    """
    Extracts the solvent amounts from the given iupac solvent mixture
    string. Defaults to a uniform mixture in case anything goes wrong
    which should be correct in most cases.

    Here some cases where the solvent mixture specifies the amounts
    correctly:
    >>> iupac_solvent_mixture_to_amounts("MeOH : Water = 10 : 1")
    {'CO': 10.0, 'O': 1.0}
    >>> iupac_solvent_mixture_to_amounts("acetone : cyclohexane = 4 : 2")
    {'CC(C)=O': 4.0, 'C1CCCCC1': 2.0}
    >>> iupac_solvent_mixture_to_amounts("MeOH : DCM : H2O = 10 : 10 : 1")
    {'CO': 10.0, 'ClCCl': 10.0, 'O': 1.0}
    >>> iupac_solvent_mixture_to_amounts("MeOH : DCM : H2O 10 : 100 : 1")
    {'CO': 10.0, 'ClCCl': 100.0, 'O': 1.0}

    Here some cases where the solvent mixture is not correctly specified.
    We see that there are no errors raised, but an empty dict is created
    instead:
    >>> iupac_solvent_mixture_to_amounts("H2O")
    {}
    >>> iupac_solvent_mixture_to_amounts("n-Hexane")
    {}
    >>> iupac_solvent_mixture_to_amounts("MeOH : Water")
    {}
    >>> iupac_solvent_mixture_to_amounts("MeOH : Water = 1 :")
    {}
    """
    org_arg = solvent_mixture_iupac
    solvent_mixture_iupac = solvent_mixture_iupac.replace(":", " ").replace("/", " ")
    solvent_names = []
    solvent_ratios = []
    for part in solvent_mixture_iupac.split():
        try:
            part = float(part)
            solvent_ratios.append(part)
        except ValueError:
            try:
                with Silencer():
                    smi = iupac_to_smiles(part)
                    if smi.strip():
                        solvent_names.append(smi)
            except:
                # we encountered a token that
                # is neither a real number nor
                # a proper solvent name.
                # Hence we skip it ...
                print(
                    f"Could not interpret token {part} in solvent mixture iupac '{org_arg}'"
                )
                continue
    if len(solvent_names) == len(solvent_ratios):
        return dict(zip(solvent_names, solvent_ratios))
    else:
        # This branch will be executed in many cases,
        # e.g. for single solvent mixtures where no
        # one would specify a ratio of 100%
        # Will also be triggered for malformed multi-
        # component mixtures or for unspecified ratios
        # like MeOH/Water without any ratios given.
        # The empty dict will effectively represent
        # a uniform 1:1 which is quite a good handling
        # of this case in general.
        return {}




def canonical_solvent_mixture_name(solvent_set_iupac: str):  # -> list[str]:
    """
    Converts a iupac description of a solvent mixture into a
    the canonical names of the corresponding chemical components
    as a list.

    >>> canonical_solvent_mixture_name("Wasser Methanol 1:1")
    ['MeOH', 'water']

    :param solvent_set_iupac:
    :return:
    """
    smile_parts = solvent_mixture_iupac_to_smiles(solvent_set_iupac)
    iupac_parts = [smiles_to_name(smile_part) for smile_part in smile_parts]
    return list(sorted(iupac_parts))


if __name__ == "__main__":
    print("creating common solvent cache ...")
    for solvent in solvents:
        smi = iupac_to_smiles(solvent)
        success = bool(smi.strip())
        if not success:
            print(solvent, smi.strip(), "success = ", success)
