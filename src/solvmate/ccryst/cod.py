from pathlib import Path
import json
from collections import defaultdict
import re
import argparse
import random
import os
import sys


from tqdm import tqdm
from solvmate import DATA_DIR, random_fle

from solvmate.ccryst.utils import in_chunks, run_in_parallel

"""
Module providing convenient access to the 
'Crystallography Open Database' (COD)

In particular, here we scrape both solvents
and SMILES out of the contained .cif-files
and amalgamate this information into a single
CSV file.
"""


cod_root = DATA_DIR / "cod"


class CodHandler:
    def __init__(self, root_path):
        self.root = root_path

    def process_files(self, file_processor, sub_dir=None):
        if sub_dir is None:
            sub_dir = self.root
        for fle in sub_dir.iterdir():
            if fle.is_dir():
                self.process_files(file_processor, fle)
            else:
                file_processor(fle)

    def index_files(self):
        self.index = []
        self.process_files(
            lambda fle: self.index.append(fle), sub_dir=self.root / "cif"
        )
        self.files_by_name = {fle.name: fle for fle in self.index}

    @staticmethod
    def get_default_instance():
        return CodHandler(cod_root)


class SolventMapping:
    def __init__(self, mapping):
        self.mapping = mapping

    def run(self, cifs):
        raise NotImplementedError

    @classmethod
    def get_default_instance(cls):
        return cls(
            {
                "methanol": [
                    r"\bmethanol\b",
                    r"\bmethanole\b",
                    r"\bmeoh\b",
                    r"\bCH3OH\b",
                ],
                "ethanol": [
                    r"\bethanol\b",
                    r"\bethanole\b",
                    r"\betoh\b",
                    r"\bCH3CH2OH\b",
                ],
                "water": [
                    r"\bwater\b",
                ],
                "isopropanol": [
                    r"\bisopropanol\b",
                    r"\biso-propanol\b",
                    "iPrOH",
                    "i-PrOH",
                ],
                "acetone": [
                    r"\baceton\b",
                    r"\bacetone\b",
                ],
                "acetonitrile": [r"\bacetonitril\b", r"\bacetonitrile\b", r"\bACN\b"],
                "ethylacetate": [
                    r"\bethylacetate?\b",
                    r"\bEA\b",
                    r"\bEtOAc\b",
                ],
                "dmso": [
                    r"\bdmso\b",
                    r"\bdimethyl sulfoxide\b",
                ],
                "tetrahydrofuran": [
                    r"\btetrahydrofurane?\b",
                    r"\bTHF\b",
                    r"\boxolane\b",
                ],
                "choloroform": [
                    r"\bchloroforme?\b",
                    r"\bCHCl3\b",
                    r"\btrichloro?methane?\b",
                    r"\btrichloromethane\b",
                ],
                "dichloromethane": [
                    r"\bdichloromethane?\b",
                    r"\bmethylene? chloride?\b",
                    r"\bmethylene?chloride?\b",
                    r"\bDCM\b",
                    r"\bch2cl2\b",
                ],
                "diethylether": [
                    r"\bdiethyl ether\b",
                    r"\bdiethylether\b",
                    r"\bdiethyl-ether\b",
                    r"\bdi-ethyl ether\b",
                    r"\bdi ethyl ether\b",
                    r"\bdi-ethyl-ether\b",
                ],
                "tolouene": [
                    r"\btolouene?\b",
                    r"\btoluol\b",
                    r"\bPhMe\b",
                    r"\bMePh\b",
                ],
                "dimethylformamide": [
                    r"\bdimethylformamide?\b",
                    r"\bDMF\b",
                    r"\bdi - methyl - formamide\b",
                    r"\bdi-methyl-formamide\b",
                    r"\bdi-methylformamide\b",
                    r"\bdimethyl-formamide\b",
                ],
                "hexane": [
                    r"\bhexane?\b",
                ],
                "butanone": [
                    r"\bbutanone?\b",
                    r"\bethylmethylketone?\b",
                    r"\bEMK\b",
                    r"\bMEK\b",
                ],
            }
        )


class GlobalSolventMapping(SolventMapping):
    def __init__(self, mapping):
        super().__init__(mapping)

    def run(self, cifs):
        annots = defaultdict(list)
        for cif in tqdm(cifs):
            try:
                content = cif.read_text()
            except:
                print("Warning: Reading CIF file ", cif, "failed! Continuing")
                continue
            content = content.lower()

            for solvent, solvent_indicators in self.mapping.items():
                for solvent_indicator in solvent_indicators:
                    if re.findall(solvent_indicator, content, re.IGNORECASE):
                        annots[cif.name].append(solvent)
        return dict(annots)


class SpecificSolventMapping(SolventMapping):
    def __init__(self, mapping):
        super().__init__(mapping)

    def run(self, cifs):
        annots = defaultdict(list)
        for cif in tqdm(cifs):
            try:
                content = cif.read_text()
            except:
                print("Warning: Reading CIF file ", cif, "failed! Continuing")
                continue
            content = content.lower()

            # Select only the relevant parts of the cif content.
            # These are:
            #   - _exptl_crystal_recrystallization_method and two lines below
            #   - all lines containing the word "solvate"
            relevant_content_lines = []
            content_lines = content.split("\n")
            for idx, lne in enumerate(content_lines):
                if "_exptl_crystal_recrystallization_method" in lne:
                    relevant_content_lines += [idx, idx + 1, idx + 2]
                if "solvate" in lne:
                    relevant_content_lines += [idx]
            content = "\n".join([content_lines[idx] for idx in relevant_content_lines])
            for solvent, solvent_indicators in self.mapping.items():
                for solvent_indicator in solvent_indicators:
                    if re.findall(solvent_indicator, content, re.IGNORECASE):
                        annots[cif.name].append(solvent)
        return dict(annots)


class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class CifToSmilesConverter:
    def __init__(
        self,
    ):
        self.tmp_file_name = random_fle("smi")

    def run(self, cif):
        devnull = open(os.devnull, "w")

        with RedirectStdStreams(stdout=None, stderr=devnull):
            rslt = os.system(
                "timeout -s9 60 obabel -i cif "
                + str(cif)
                + " -O "
                + str(self.tmp_file_name)
            )  # nosec
            if rslt == 0:
                return self.tmp_file_name.read_text()
            else:
                print("Problem parsing cif:", str(cif))
                return ""


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--map-solvents", action="store_true", dest="map_solvents")
    argp.add_argument("--limit", default=0, type=int)
    argp.add_argument("--n-jobs", default=1, type=int)
    argp.add_argument("--map-smiles", action="store_true", dest="map_smiles")
    args = argp.parse_args()

    print("obtaining index of cif files from cod")
    cod_handler = CodHandler.get_default_instance()
    cod_handler.index_files()
    all_cifs = cod_handler.index

    if args.limit:
        print(f"sampling down to {args.limit} cifs ... ")
        all_cifs = random.sample(all_cifs, args.limit)

    if args.map_solvents:
        print("mapping solvents ... ")
        solv_mapping = SpecificSolventMapping.get_default_instance()

        if args.n_jobs == 1:
            print(f"found {len(all_cifs)} cifs to apply to")
            print("Starting mapping process (should take around 1h)")
            mapping = solv_mapping.run(all_cifs)
        else:
            print(f"mapping solvents in parallel (n_jobs = {args.n_jobs})")
            mapping_chunks = run_in_parallel(
                n_jobs=args.n_jobs,
                callable=lambda cifs: [solv_mapping.run(cifs)],
                inputs=all_cifs,
            )
            mapping = {}
            for mapping_chunk in mapping_chunks:
                mapping.update(mapping_chunk)

        print("writing solvent mapping to file")
        with open(DATA_DIR / Path("default_solvent_mapping.json"), "wt") as fout:
            fout.write(json.dumps(mapping))

    if args.map_smiles:
        with open(DATA_DIR / Path("default_solvent_mapping.json"), "rt") as fin:
            solvent_mapping = json.loads(fin.read())
        smiles_mapping = {}
        print("mapping smiles ...")
        cif_to_smi = CifToSmilesConverter()
        if args.n_jobs == 1:
            for cif in tqdm(all_cifs):
                if cif.name not in solvent_mapping:
                    continue
                smi = cif_to_smi.run(cif)
                smiles_mapping[cif.name] = smi
        else:
            print(f"mapping smiles in parallel (n_jobs = {args.n_jobs})")
            cifs_with_solvent = [cif for cif in all_cifs if cif.name in solvent_mapping]
            smis = run_in_parallel(
                n_jobs=args.n_jobs,
                callable=lambda cifs: list(map(cif_to_smi.run, cifs)),
                inputs=cifs_with_solvent,
            )
            for cif, smi in zip(cifs_with_solvent, smis):
                smiles_mapping[cif.name] = smi

        print("writing smiles mapping to file")
        smiles_mapping_file = Path(BUILD_DIR) / "default_smiles_mapping.json"
        with open(smiles_mapping_file, "wt") as fout:
            fout.write(json.dumps(smiles_mapping))
    print("Done.")


if __name__ == "__main__":
    main()
