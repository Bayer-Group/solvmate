import os
import hashlib
from copy import deepcopy
from typing import Optional

import numpy as np
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem import DataStructs, Draw

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

import subprocess
import random

from pathlib import Path
from solvmate import *


def obabel_conversion(input: str, format_in: str, format_out: str):
    """
    Converts from the input format to the given
    output format.

    >>> obabel_conversion(input='''15\\nxtb: 6.5.1 (579679a)\\nC           -2.27018323591655        0.21115627368169       -0.00048220610393\\nC           -1.07815280768851        0.20942936581072       -0.96132168589488\\nC            0.21117010078536        0.26434255797611       -0.21614466382439\\nC            1.15504155307328       -0.67340021246162       -0.27070867755029\\nC            2.42097326686372       -0.67615016331214        0.49098043550222\\nO            3.06968428811897       -1.67829552017005        0.66795981764144\\nO            2.84592469150146        0.49781842130748        0.99691344031191\\nH           -2.23902414729603       -0.66623293656655        0.64169838202005\\nH           -2.25942589324246        1.09976988312784        0.62686561811898\\nH           -3.20089858117776        0.19666330461345       -0.56183930783802\\nH           -1.14413933420006        1.09188629736365       -1.60507309426568\\nH           -1.11021527581142       -0.67971574637669       -1.59223877907991\\nH            0.31790482009810        1.13264779108353        0.42715833531561\\nH            1.01571777343253       -1.56332743632480       -0.86826669705445\\nH            2.26562078145935        1.23340912024738        0.75270108270132''', \
            format_in='xyz',format_out='inchi',) 
    'InChI=1S/C5H8O2/c1-2-3-4-5(6)7/h3-4H,2H2,1H3,(H,6,7)/b4-3+\\n'
    >>> obabel_conversion(input='''15\\nxtb: 6.5.1 (579679a)\\nC           -2.27018323591655        0.21115627368169       -0.00048220610393\\nC           -1.07815280768851        0.20942936581072       -0.96132168589488\\nC            0.21117010078536        0.26434255797611       -0.21614466382439\\nC            1.15504155307328       -0.67340021246162       -0.27070867755029\\nC            2.42097326686372       -0.67615016331214        0.49098043550222\\nO            3.06968428811897       -1.67829552017005        0.66795981764144\\nO            2.84592469150146        0.49781842130748        0.99691344031191\\nH           -2.23902414729603       -0.66623293656655        0.64169838202005\\nH           -2.25942589324246        1.09976988312784        0.62686561811898\\nH           -3.20089858117776        0.19666330461345       -0.56183930783802\\nH           -1.14413933420006        1.09188629736365       -1.60507309426568\\nH           -1.11021527581142       -0.67971574637669       -1.59223877907991\\nH            0.31790482009810        1.13264779108353        0.42715833531561\\nH            1.01571777343253       -1.56332743632480       -0.86826669705445\\nH            2.26562078145935        1.23340912024738        0.75270108270132''', \
            format_in='xyz',format_out='cml',) #doctest:+ELLIPSIS
    '<?xml version="1.0"?>\\n<molecule id="id 6.5" xmlns="http://www.xml-cml.org/schema">\\n <atomArray>\\n  <atom id="a1" elementType="C" hydrogenCount="3" x3="-2.270183" y3="0.211156" z3...


"""
    tmp_file = f"/tmp/__tmp_obabel_in_{random.randint(100000,100000000)}.{format_in}"
    tmp_path = Path(tmp_file)
    tmp_path.write_text(input)
    rslt = subprocess.check_output(["obabel", tmp_file, "-o", format_out]).decode(
        "utf8"
    )
    tmp_path.unlink()
    return rslt


def xyz_to_mol(xyz_str: str) -> "Chem.Mol":
    """
    Converts the given xyz string into a rdkit molecule.
    """
    return Chem.MolFromPDBBlock(
        obabel_conversion(
            input=xyz_str,
            format_in="xyz",
            format_out="pdb",
        )
    )


def organic_subset_atoms():
    return {
        "H",
        "Li",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "K",
        "Ca",
        "As",
        "Se",
        "Br",
        "I",
    }


def is_organic_subset_mol(mol):
    organic_subs = organic_subset_atoms()
    return all(atm.GetSymbol() in organic_subs for atm in mol.GetAtoms())


def neutralize_radicals(mol):
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons() > 0:
            a.SetNumRadicalElectrons(0)
    return mol


def mol_to_image(
    mol,
    width=300,
    height=300,
) -> Image:
    img = random_fle("jpg")
    Draw.MolToImageFile(
        mol, filename=str(img.resolve()), format="JPG", size=(width, height)
    )
    img = Image.open(img)
    return deepcopy(img)


def canonicalize_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def safe_canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol)


sdf_files_cache_location = os.path.expanduser("~/data/sdf_files_cache/")

import sys
import threading
from time import sleep

try:
    import thread
except ImportError:
    import _thread as thread


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print("{0} took too long".format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt
    # raise Exception("interrupt!")


def exit_after(s):
    """
    use as decorator to exit process if
    function takes longer than s seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            result = None
            try:
                result = fn(*args, **kwargs)
            except:
                return result
            finally:
                timer.cancel()
            return result

        return inner

    return outer


# @exit_after(5)
def embed_conformers_cached(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Embeds the given molecule into 3D coordinates.
    Saves a cache of already computed 3D embeddings
    for performance.
    """
    smi = Chem.MolToSmiles(mol)
    sha = hashlib.sha256(smi.encode("utf8")).hexdigest()
    global sdf_files_cache_location

    sdf_file = "{}/{}.sdf".format(sdf_files_cache_location, sha)
    if os.path.exists(sdf_file):
        suppl = Chem.SDMolSupplier(
            sdf_file,
            removeHs=False,
        )
        for mol in suppl:
            return mol
        raise ValueError("unexpected problem with sdf loading!")
    else:
        # TODO: Is this the best way
        # TODO: to embed the molecules in 3D?
        mol = Chem.AddHs(mol)
        ps = rdDistGeom.ETKDGv3()
        ps.enforceChirality = False
        AllChem.EmbedMolecule(mol, enforceChirality=0, maxAttempts=100)
        writer = Chem.SDWriter(sdf_file)
        writer.write(mol, confId=0)
        return mol


def fingerprint_to_numpy(fp):
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def ecfp_fingerprint(
    mol: Chem.Mol,
    radius=4,
    n_bits=2048,
) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )
    return fingerprint_to_numpy(fp)


from rdkit.Chem import rdMolDescriptors


def crystal_propensity_heuristic(mol: Chem.Mol, as_html=True) -> str:
    """
    Employs the simple crystalization propensity heuristic described in
    B.C. Hancock / Journal of Pharmaceutical Sciences 106 (2017) 28-30.

    >>> iup_to_mol = lambda iupac: Chem.MolFromSmiles(opsin_iupac_to_smiles(iupac))
    >>> fun = lambda mol: crystal_propensity_heuristic(mol, as_html=False)
    >>> fun(iup_to_mol("benzene"))
    '1_1'
    >>> fun(iup_to_mol("Acetylsalicylic acid"))
    '1_1'

    Has exactly six rotatable bonds:
    >>> fun(iup_to_mol("1,2,3,4,5,6-hexacyclohexylcyclohexane"))
    '2_2'

    Has exactly seven rotatable bonds:
    >>> fun(iup_to_mol("1,1,2,3,4,5,6-heptacyclohexylcyclohexane"))
    '3_2'

    >>> fun(iup_to_mol("17β-Hydroxyandrost-4-en-3-one"))
    '1_1'
    """
    n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mw = rdMolDescriptors.CalcExactMolWt(mol)

    if n_rot_bonds < 4:
        g1 = "1"
    elif n_rot_bonds < 7:
        g1 = "2"
    elif n_rot_bonds >= 7:
        g1 = "3"
    else:
        assert False

    if mw < 300:
        g2 = "1"
    else:
        g2 = "2"

    if not as_html:
        return "_".join([g1, g2])
    else:
        return (
            """
        <table>
            <tr>
                <td>
                </td>
                <td>
                MW &lt; 300
                </td>
                <td>
                MW &gt; 300
                </td>
            </tr>
            <tr>
                <td>
                0-3 Rot. Bonds 
                </td>
                <td id="cryst_prop_1_1">
                Easy
                </td>
                <td id="cryst_prop_1_2">
                Moderate
                </td>
            </tr>
            <tr>
                <td>
                4-6 Rot. Bonds 
                </td>
                <td id="cryst_prop_2_1">
                easy to moderate
                </td>
                <td id="cryst_prop_2_2">
                moderate to difficult
                </td>
            </tr>
            <tr>
                <td>
                &gt;6  Rot. Bonds 
                </td>
                <td id="cryst_prop_3_1">
                moderate to difficult
                </td>
                <td id="cryst_prop_3_2">
                difficult
                </td>
            </tr>
        </table>
        <style>
        """
            + f"""
            #cryst_prop_{g1}_{g2}"""
            + """{
                font-weight: bold;
                background-color: yellow;
            }
        </style>
        """
        )
