"""
A module that utilizes XTB to generate solvation
energy fingerprints. These can then be fed into
our machine learning models for crystallization
solvent prediction or for database search and
inspection workflows.
"""
from collections import defaultdict
from joblib import Parallel, delayed
import tqdm
import random
import re
import numpy as np
import pandas as pd
import sqlite3
import shutil
import os
from pathlib import Path

from solvmate import *
from rdkit import Chem
from rdkit.Chem import AllChem,rdDistGeom

import tempfile

# Path to the xtb binary
BIN_XTB = os.getenv("XS__BIN_XTB") or "xtb"

def _patch_bin_xtb_from_deps_dir_if_possible(d:Path):
    global BIN_XTB
    # Fix for windows release
    for xtb_bin_candidate in [
        "xtb.exe", "xtb.out", "xtb"
    ]:
        xtb_bin_candidate = d / xtb_bin_candidate
        if xtb_bin_candidate.exists() and xtb_bin_candidate.is_file():
            BIN_XTB = str(xtb_bin_candidate.resolve())
            info(
                f"found a xtb binary at {xtb_bin_candidate}!"
            )

    for fle in d.iterdir():
        if fle.is_dir():
            # recursively walk the directory tree
            _patch_bin_xtb_from_deps_dir_if_possible(fle)

_patch_bin_xtb_from_deps_dir_if_possible(PROJECT_ROOT / "deps")

import subprocess
import time 
import psutil

try:
    with Silencer() as s:
        xtb_version = subprocess.check_output([f'{BIN_XTB}', '--version'],stderr=subprocess.DEVNULL,)
    if not isinstance(xtb_version,str):
        xtb_version = xtb_version.decode('utf-8')
    xtb_version = re.findall(r"(version \d+.\d+.\d+)",xtb_version)
except:
    print("could not determine xtb version.")
    print("most likely no xtb binary is installed. See: https://xtb-docs.readthedocs.io/en/latest/setup.html")
    raise

if xtb_version:
    xtb_version = xtb_version[0]
    if xtb_version >= "6.5.1":
        print("xtb version:",xtb_version)
    else:
        assert f"detected outdated xtb version: '{xtb_version}'. Please install version >= 6.5.1."\
                "see https://xtb-docs.readthedocs.io/en/latest/setup.html"
else:
    print("could not determine xtb version.")
    print("most likely no xtb binary is installed. See: https://xtb-docs.readthedocs.io/en/latest/setup.html")
    exit(1)


# Try to prioritize memory mapped file system
# to improve speed and reduce strain,
# fallback to potentially memory mapped, or
# non-mem mapped file system otherwise...
TMP_ROOT = Path("/dev/shm/")
if not TMP_ROOT.exists():
    print("Warning: could not find /dev/shm/ mem-mapped io not possible")
    TMP_ROOT = Path("/tmp")
if not TMP_ROOT.exists():
    TMP_ROOT = tempfile.gettempdir()

XTB_TMP_DIR = TMP_ROOT / Path("xtbsolv")

# see https://xtb-docs.readthedocs.io/en/latest/gbsa.html
SOLVENTS_XTB = [
    "Acetone",
    "Acetonitrile",
    "Aniline",
    "Benzaldehyde",
    "Benzene",
    "CH2Cl2",
    "CHCl3",
    "CS2",
    "Dioxane",
    "dmf",
    "Ether",
    "Ethylacetate",
    "Furane",
    "Hexadecane",
    "Hexane",
    "Methanol",
    "Nitromethane",
    "Octanol",
    #"Octanol (wet)", # commented out as the parens cause issues with some shells!
    "Phenol",
    "Toluene",
    "THF",
    "water",
    "dmso",
    ]


class XTBSolv:

    """
    XTBSolvation class. To use, first initialize.

    >>> xs = XTBSolv(db_file="/tmp/_xtb_test.db",)
    >>> xs.setup_db()
    >>> xs.run_xtb_calculations(["CCC","c1ccccc1",])
    >>> df = xs.get_dataframe()
    >>> list(df.columns) # doctest: +NORMALIZE_WHITESPACE
    ['smiles', 'solvent', 'Gsolv', 'Gelec', 'Gsasa',
     'Ghb', 'Gshift', 'TOTAL_ENERGY', 'HOMO_LUMO_GAP', 
     'DIPOLE_MOMENT', 'C6AA', 'C8AA', 'alpha']
    """

    ENERGY_TYPES = [
        "Gsolv", "Gelec", "Gsasa", "Ghb", "Gshift", "TOTAL_ENERGY",
        "HOMO_LUMO_GAP", "DIPOLE_MOMENT",
        "C6AA",
        "C8AA",
        "alpha",
    ]

    ENERGY_TYPES_TO_UNITS = {
        "Gsolv": "kcal/mol", "Gelec": "kcal/mol",
        "Gsasa": "kcal/mol", "Ghb": "kcal/mol",
        "Gshift": "kcal/mol", "TOTAL_ENERGY": "kcal/mol",
        "HOMO_LUMO_GAP": "eV",
        "DIPOLE_MOMENT": "Db",
        "C6AA": "au * bohr^6",
        "C8AA": "au * bohr^8",
        "alpha": "au",
    }

    # These are units we get provided as input by XTB.
    _ENERGY_TYPES_TO_XTB_UNITS = {
        "Gsolv": "Eh", "Gelec": "Eh",
        "Gsasa": "Eh", "Ghb": "Eh",
        "Gshift": "Eh", "TOTAL_ENERGY": "Eh",
        "HOMO_LUMO_GAP": "eV",
        "DIPOLE_MOMENT": "Db",
        "C6AA": "au * bohr^6",
        "C8AA": "au * bohr^8",
        "alpha": "au",
    }

    CONVERSION_FACTORS = {
        # input unit -> output unit -> conversion factor
        "Eh": {"Eh": 1},
        "eV": {"eV": 1},
        "Db": {"Db": 1},
        "Eh": {"kcal/mol":  627.5},
        "au * bohr^6": {"au * bohr^6": 1},
        "au * bohr^8": {"au * bohr^8": 1},
        "au": {"au": 1},
    }


    def __init__(self, db_file, n_confs:int=1,):
        if isinstance(db_file,str):
            db_file = Path(db_file)

        self.n_confs = n_confs

        assert db_file.parent.exists(), f"directory '{db_file.parent}' doesnt exist!"
        self.db_file = db_file

    def get_dataframe(self):
        db = sqlite3.connect(self.db_file,timeout=SQLITE_TIMEOUT_SECONDS,)
        df = pd.read_sql("SELECT * FROM solv_en",con=db)
        db.close()
        return df

    def run_xtb_calculation(
        self, mol:"Chem.Mol",
        solvent:str,
    ) -> "tuple[bool,str]":
        """
        Runs a XTB calculation on the given
        rdkit molecule and solvent. Returns
        the output of the calculation as a
        string. The ephemeral job directory
        is ensured to be deleted at the end
        of the method call.
        """
        assert XTB_TMP_DIR.parent.exists()
        XTB_TMP_DIR.mkdir(exist_ok=True)
        assert XTB_TMP_DIR.exists()
        job_dir = XTB_TMP_DIR / f"job_{random.randint(1,10**9)}"
        assert not job_dir.exists(), "internal logic error"
        cwd_before = os.getcwd()
        try:
            os.mkdir(job_dir)
            os.chdir(job_dir)
            Chem.MolToXYZFile(mol, "input.xyz")

            output_fle = job_dir / "output.out"
            assert not output_fle.exists()
            with Silencer() as s:
                cmd = \
                    f"{BIN_XTB} input.xyz --parallel {XTB_INNER_JOBS} --opt --alpb {solvent} > output.out 2> err.out"

                # normal approach:
                # subprocess.check_output(cmd,shell=True,timeout=XTB_TIMEOUT_SECONDS)
                # ^
                # \__ sadly, this approach does not work because the timeout isnt applied
                #
                # As suggested in https://stackoverflow.com/questions/48763362/python-subprocess-kill-with-timeout
                # we apply the following workaround:
                parent = subprocess.Popen(cmd,shell=True)
                for _ in range(XTB_TIMEOUT_SECONDS): 
                    if parent.poll() is not None:  # process just ended
                        break
                    time.sleep(1)
                else:
                    # the for loop ended without break: timeout
                    parent = psutil.Process(parent.pid)
                    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
                        child.kill()
                    parent.kill()

            if output_fle.exists():
                return True,output_fle.read_text()
            else:
                return False,""
        finally:
            os.chdir(cwd_before)
            # Garbage Collection
            if job_dir.exists():
                shutil.rmtree(job_dir)
        

    def embed_molecule(self, mol:"Chem.Mol")->"Chem.Mol":
        mol = Chem.AddHs(mol)
        ps = rdDistGeom.ETKDGv3()
        ps.enforceChirality = False
        AllChem.EmbedMolecule(
            mol, enforceChirality=0, 
            maxAttempts=100,
            )
        return mol

    def embed_multi(self, mol:"Chem.Mol", n_confs:int,)->"Chem.Mol":
        mol = Chem.AddHs(mol)
        ps = rdDistGeom.ETKDGv3()
        cids = rdDistGeom.EmbedMultipleConfs(mol,n_confs,ps)

        return cids




    @staticmethod
    def parse_xtb_solvation_ens(output:str)->"tuple[bool,dict]":
        """
        Extracts relevant energies from a given XTB output string.
        Success is True only if all specified energy keys could
        be found. 

        No success and only nan values on bogus input:
        >>> XTBSolv.parse_xtb_solvation_ens('test')[0]
        False
        >>> set(XTBSolv.parse_xtb_solvation_ens('test')[1].values())
        {nan}

        GSolv properly parsed
        >>> XTBSolv.parse_xtb_solvation_ens(':: -> dispersion             -0.045996146603 Eh    ::\
         :: -> Gsolv                  -0.056144321652 Eh    ::')[1]['Gsolv']
        -35.23056183663

        Also, units are handled accordingly. For example, the HOMO LUMO gap 
        is specified in electron volts and not in hartrees:
        >>> XTBSolv.parse_xtb_solvation_ens("HOMO-LUMO GAP              14.564164628291 eV   |")[1]['HOMO_LUMO_GAP']
        14.564164628291

        The dipole moment is also extracted:
        >>> s_dpm = 'molecular dipole:   x           y           z       tot (Debye) q only:       -0.015      -0.013       0.132 full:        0.024       0.021      -0.214       0.550'
        >>> XTBSolv.parse_xtb_solvation_ens(s_dpm)[1]['DIPOLE_MOMENT']
        0.55

        >>> inp = "Mol. C6AA /au·bohr⁶  :       2469.682239"
        >>> XTBSolv.parse_xtb_solvation_ens(inp)[1]["C6AA"]
        2469.682239
        >>> inp = ''' \
        ... Mol. C6AA /au·bohr⁶  :       2469.682239 \
        ... Mol. C8AA /au·bohr⁸  :      69773.015226 \
        ... Mol. α(0) /au        :         79.818763 \
        '''
        >>> success,dct = XTBSolv.parse_xtb_solvation_ens(inp)
        >>> success
        False
        >>> dct["C6AA"], dct["C8AA"], dct["alpha"]
        (2469.682239, 69773.015226, 79.818763)


        """
        success = True
        solv_ens = {}
        for key in XTBSolv.ENERGY_TYPES:
            if key == "TOTAL_ENERGY":
                rkey = "TOTAL ENERGY"
            elif key == "HOMO_LUMO_GAP":
                rkey = re.escape("HOMO-LUMO GAP")
            else:
                rkey = key
            in_unit = XTBSolv._ENERGY_TYPES_TO_XTB_UNITS[key]
            out_unit = XTBSolv.ENERGY_TYPES_TO_UNITS[key]
            if key == "DIPOLE_MOMENT":
                rgx = r"molecular dipole:\s+x\s+y\s+z\s+tot \(Debye\)\s+q only:\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+full:\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+(-?\d+\.\d+)"
            elif key == "C6AA":
                rgx = re.escape("Mol. C6AA /au·bohr⁶  :") +r"\s+(-?\d+\.\d+)"
            elif key == "C8AA":
                rgx = re.escape("Mol. C8AA /au·bohr⁸  :") +r"\s+(-?\d+\.\d+)"
            elif key == "alpha":
                rgx = re.escape("Mol. α(0) /au        :") +r"\s+(-?\d+\.\d+)"
            else:
                rgx = rkey+r"\s+(-?\d+\.\d+) "+in_unit
            matches = re.findall(rgx, output)

            if matches:
                CF = XTBSolv.CONVERSION_FACTORS
                assert in_unit in CF.keys(), f"unknown unit in input: {in_unit}"
                conversions_supported_to = CF[in_unit]
                assert out_unit in conversions_supported_to, f"unknown conversion: {in_unit} -> {out_unit}"
                # last value is the relevant energy of final state
                val = float(matches[-1]) * CF[in_unit][out_unit] 
            else:
                success = False
                val = np.NaN
            solv_ens[key] = val

        return success, solv_ens

    def setup_db(self):
        db = sqlite3.connect(self.db_file,timeout=SQLITE_TIMEOUT_SECONDS)
        energy_columns = ",".join([et+" REAL" for et in self.ENERGY_TYPES])
        db.execute(
            f"CREATE table IF NOT EXISTS solv_en (smiles TEXT, solvent TEXT, {energy_columns})"
        )

        db.execute(
            "CREATE table IF NOT EXISTS solv_raw (smiles TEXT, solvent TEXT, output TEXT)"
        )
        db.commit()
        db.close()

    def _run_single_combination(self, smi:str, solvent:str,):
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = self.embed_molecule(mol)
            success,outp = self.run_xtb_calculation(mol,solvent)
        except AssertionError:
            raise
        except KeyboardInterrupt:
            raise
        except:
            success,outp = False,""

        db = sqlite3.connect(self.db_file,timeout=SQLITE_TIMEOUT_SECONDS)
        if success:
            db.execute(
                "INSERT INTO solv_raw(smiles,solvent,output) VALUES (?,?,?)",
                (smi, solvent, outp,),
            )

        success,ens_dct = self.parse_xtb_solvation_ens(outp)
        if success:
            col_vals = [f"'{smi}'",f"'{solvent}'"] + \
                list(map(str,(ens_dct[et] for et in self.ENERGY_TYPES)))
            stmt = ("INSERT INTO solv_en("+
                "smiles, solvent, "+
                ", ".join(self.ENERGY_TYPES)+
                ") VALUES ("+", ".join(col_vals)+");"
            )
            db.execute(
                stmt
            )
        db.commit()
        db.close()

    def run_xtb_calculations(self,
        smiles:"list[str]",
        verbose=False,
        solvents=None,
        ) -> None:
        db = sqlite3.connect(self.db_file,timeout=SQLITE_TIMEOUT_SECONDS)

        smiles = list(set(smiles)) # remove duplicates

        if solvents is None or solvents == "all":
            solvents = SOLVENTS_XTB

        smi_to_solvs = defaultdict(list)
        for smi, solvent in db.execute("SELECT smiles, solvent FROM solv_raw"):
            smi_to_solvs[smi].append(solvent)

        db.close()

        if verbose:
            smiles_iter = tqdm.tqdm(smiles)
        else:
            smiles_iter = smiles


        jobs = (
            (smi,solvent)
            for smi in smiles_iter for solvent in solvents
            for i_conf in range(self.n_confs)
            if solvent not in smi_to_solvs[smi]
        )

        if XTB_OUTER_JOBS > 1:
            Parallel(n_jobs=XTB_OUTER_JOBS)(delayed(self._run_single_combination)(smi,solvent) for smi,solvent in jobs)
        else:
            for smi,solvent in jobs:
                self._run_single_combination(smi,solvent)
                        


    def drop_table_solv_en(self):
        db = sqlite3.connect(self.db_file,timeout=SQLITE_TIMEOUT_SECONDS)
        db.execute("DROP TABLE IF EXISTS solv_en")
        db.commit()
        db.close()

    
    def refresh_from_raw(self):
        """
        Allows to refresh the solv_en table without needing to 
        redo the (potentially expensive) XTB calculations.
        """
        self.drop_table_solv_en()
        self.setup_db()
        self._transfer_raw_to_en()

    def _transfer_raw_to_en(self):
        db = sqlite3.connect(self.db_file,timeout=SQLITE_TIMEOUT_SECONDS)
        loop_counter = 0
        for smi,solvent,outp in db.execute("SELECT smiles, solvent, output FROM solv_raw"):
                # See also method run_xtb_calculations for a similar behavior
                success,ens_dct = self.parse_xtb_solvation_ens(outp)
                if success:
                    col_vals = [f"'{smi}'",f"'{solvent}'"] + \
                        list(map(str,(ens_dct[et] for et in self.ENERGY_TYPES)))
                    stmt = ("INSERT INTO solv_en("+
                        "smiles, solvent, "+
                        ", ".join(self.ENERGY_TYPES)+
                        ") VALUES ("+", ".join(col_vals)+");"
                    )
                    db.execute(
                        stmt
                    )
                loop_counter += 1
        db.commit()
        db.close()






        


                

        