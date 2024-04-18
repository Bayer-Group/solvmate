import hashlib
import random
from typing import Optional
from rdkit import Chem
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent


CM_DATA_DIR = PROJECT_DIR / "data"


def _take_one_file_in_dir(d: Path) -> Optional[Path]:
    if d.exists():
        fles = list(d.iterdir())
        if fles:
            return random.choice(fles)
        else:
            return None
    else:
        return None


def id_to_cosmo_file_path(mol_id: str) -> Optional[Path]:
    return _take_one_file_in_dir(CM_DATA_DIR / mol_id / "COSMO")


def smiles_to_id(smiles: str):
    try:
        smiles_canon = Chem.CanonSmiles(smiles)
    except:
        smiles_canon = ""
    return hashlib.sha1(smiles_canon.encode("utf8")).hexdigest()


def smiles_to_charge(smiles: str):
    if not smiles:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0
    return Chem.rdmolops.GetFormalCharge(mol)
