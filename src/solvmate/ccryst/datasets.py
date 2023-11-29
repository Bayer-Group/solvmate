import json
import os
import random
import math
from enum import Enum
from pathlib import Path
import pandas as pd
from rdkit import Chem
from solvmate.ccryst.chem_clustering import simple_butina_clustering

from solvmate.ccryst.chem_utils import (
    canonicalize_smiles,
    ecfp_fingerprint,
    opsin_iupac_to_smiles,
)
from solvmate import DATA_DIR
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from solvmate.ccryst.solvent import iupac_solvent_mixture_to_amounts, iupac_to_smiles

from solvmate.ccryst.utils import Silencer, german_to_english_nums, run_in_parallel

RDLogger.DisableLog("rdApp.*")

"""
Module that takes care of loading the data into a common format.
All datasets will be read into the following form:

    comp_no:    A number identifying the compound, optional (missing for CS datasets)
    
    smiles:     The smiles of the compound. Will be canonicalized using rdkit
                so it is also used to identify compounds uniquely.
                
    solvent:    The solvent used in the crystallization, can be either a pure
                solvent or a mixture of solvents. This solvent mixture specification
                can then be further processed using the solvent module, especially
                functions solvent_mixture_iupac_to_smiles and canonical_solvent_mixture_name
                which decipher these solvent mixture specifications.
                
    cryst:      The crystallization outcome. Depending on the dataset, this can
                be either a binary yes/no label or a more detailed quality description
                of the obtained crystals.
                
    bin_cryst:  The crystallization outcome but as a binary result, aka bin_cryst = 0 or 1
    
    
    source:     The source of the datapoint. See DSSource enum.
                Useful for maintaining an overview 
                where single observations originate from / composition of positive
                and negative data points. 
    
    Still TODO:
    [[ num_cryst:  The crystallization outcome but as a numeric regression result. ]]

"""


class DSSource(Enum):
    NOVA = "nova"
    COD = "cod"
    MANTAS = "mantas"


def binarize_outcome(label: str) -> int:
    to_bin = {
        "amorph": 0,
        "überwiegend amorph": 0,
        "amorph mit kristallinen anteilen": 0,
        "teilweise kristallin": 0,
        "kristallin mit amorphen anteilen": 1,
        "überwiegend kristallin": 1,
        "kristallin": 1,
        "kristallin fehlgeordnet": 1,  # ???
        "mesomorph": 0,  # ???
        "No": 0,
        "no": 0,
        "Yes": 1,
        "yes": 1,
        "Yrd": 1,  # < ------ Probably a typo so treat this as yes. happens only once
        "No ": 0,
        "CT": 0,  # < ----------------- TODO: how to handle this? Its crystalline tendencies. I will make it negative now
        # < ----------------- TODO: so that it is in agreement with the original novartis publication.
        "DR": 0,
        "YX": 1,
        "XX": 1,
        "FI": 0,
        "AM": 0,
    }
    if label not in to_bin:
        raise ValueError(
            f"Encountered unknown target '{label}'. I currently don't know how "
            f"to binarize this label. Please add the mapping '{label}': 0 or 1 to "
            f"the function {binarize_outcome.__name__} in module {__file__}"
        )
    return to_bin[label]


def load_mantas_csv(csv=None, verbose=True) -> pd.DataFrame:
    dfm = pd.read_csv(
        os.path.expanduser("~/data/mantas/done_all_sol_precip_clean4_enu8.csv"),
        sep="\t",
    )
    dfm = dfm.rename(
        columns={
            "Solven": "solvent",
            "RP_2": "smiles_reaction",
        }
    )
    dfm["smiles"] = dfm["smiles_reaction"].apply(
        lambda s: s.split(">")[-1]
    )  # reaction to product of reaction
    # get largest product, we do this to avoid getting a byproduct and the "actual product instead"
    dfm["smiles"] = dfm.smiles.apply(
        lambda s: sorted([part for part in s.split(".")], key=len)[-1]
    )
    dfm = dfm[~dfm.solvent.isna()]
    dfm["solvent"] = dfm["solvent"].apply(lambda s: s.replace("|", " : "))
    dfm["cryst"] = "yes"
    dfm["source"] = DSSource.MANTAS.value
    return dfm


def ratio_string_to_tup(ratio_str: str) -> "tuple[int]":
    """
    Converts the given ratio string as is typically used to
    indicate solvent mixture compositions into a tuple of the
    corresponding numbers.

    >>> ratio_string_to_tup("1 : 3")
    (1, 3)
    >>> ratio_string_to_tup("11 : 33")
    (11, 33)
    """
    assert ratio_str.count(":") == 1, "there must be exactly one colon in ratio string!"

    return tuple(map(int, ratio_str.split(":")))


def load_novartis(
    csv=None,
    verbose=True,
):
    """

    - XX:
      crystals >30 μm size in the shortest dimension),
    - YX:
      microcrystals <30 μm size or needle-like structures
    - CT:
      crystalline tendencies indicated by polarized light but no extractable specimen suited for crystallography
    - DR: drop- lets
    - FI: films
    - AM: amorphous

    So basically XX == YX == 1 and CT == DR == FI == AM == 0

    - RS:
      readily soluble (no floating residue after addi- tion of solvent),
    - KS:
      kinetically soluble (no floating residue after vortexing)
    - TS:
      thermally soluble (no floating residue after heating to 40 °C and vortexing)
    - PS:
      partially soluble/insoluble (either a cloudy solution or floating residues after heating and vortexing or the material remains unaffected after addition of solvent, heating and vortexing)


    :param csv:
    :return:
    """

    solvents_to_iupac = {
        "MeOH": "methanol",
        "EtOH": "ethanol",
        "NiMe": "nitromethane",
        "ACT": "acetone",
        "DMF": "N,N-dimethylformamide",
        "iBMK": "isobutylmethylketone",
        "EA": "ethylacetate",
        "tBME": "tertbutylmethylether",
        "MCB": "chlorobenzene",
        "CHCl3": "chloroform",
        "TOL": "methylbenzene",
        "ACN": "acetonitrile",
        "iPrOH": "isopropanol",
        "EMK": "ethylmethylketone",
        "THF": "tetrahydrofurane",
        "DCM": "dichloromethane",
        "DEE": "diethylether",
        "HEX": "hexane",
    }

    if csv is None:
        csv = Path(DATA_DIR) / "novartis_dataset_c7ce00738h6.csv"
    df_wide = pd.read_csv(csv)
    df_wide = df_wide.rename(columns={"SMILES": "smiles"})

    rows = []
    for _, row in df_wide.iterrows():
        smi = row["smiles"]
        for solvent in solvents_to_iupac.keys():
            cryst = row["C-" + solvent]
            rows.append(
                {
                    "smiles": smi,
                    "solvent": solvents_to_iupac[solvent],
                    "cryst": cryst,
                    "comp_no": None,
                }
            )
    df = pd.DataFrame(rows)
    before = len(df)
    df = df[df.smiles.astype(bool)]
    after = len(df)
    if verbose:
        print(f"removed {before-after} rows with no smiles info")

    before = len(df)
    df = df[(~df.cryst.isna()) & df.cryst.astype(bool)]
    after = len(df)
    if verbose:
        print(f"removed {before-after} rows with no cryst info")

    df["smiles"] = df.smiles.apply(canonicalize_smiles)

    df["source"] = DSSource.NOVA.value

    return df


def load_cod(
    smiles_mapping_fle=None,
    solvent_mapping_fle=None,
    verbose=True,
    only_organics=True,
    patch_radicals=True,
    only_relevant_cols=True,
) -> pd.DataFrame:
    if smiles_mapping_fle is None:
        smiles_mapping_fle = DATA_DIR / "default_smiles_mapping.json"
    if solvent_mapping_fle is None:
        solvent_mapping_fle = DATA_DIR / "default_solvent_mapping.json"
    assert (
        smiles_mapping_fle.exists()
    ), "could not find smiles mapping file. please build it!"
    assert (
        solvent_mapping_fle.exists()
    ), "could not find solvent mapping file. please build it!"
    if verbose:
        print("loading smiles mapping file ...")
    smiles_mapping = json.loads(smiles_mapping_fle.read_text())
    if verbose:
        print("loading solvent mapping file ...")
    solvent_mapping = json.loads(solvent_mapping_fle.read_text())
    assert set(smiles_mapping.keys()) == set(
        solvent_mapping.keys()
    ), "keys of solvent and smiles mapping must match"
    cif_ids = list(smiles_mapping.keys())
    rows = []
    for cif_id in cif_ids:
        smi = smiles_mapping[cif_id]
        solv = " / ".join(solvent_mapping[cif_id])
        rows.append(
            {
                "comp_no": cif_id,
                "raw_smiles": smi,
                "solvent": solv,
                "cryst": "yes",
            }
        )

    df = pd.DataFrame(rows)

    before = len(df)
    df = df[df["raw_smiles"].apply(bool)]
    after = len(df)
    if verbose:
        print(f"removed {before-after} entries with missing smiles info")

    df["smiles_unit_cell"] = df.raw_smiles.apply(lambda rs: rs.split()[0].split("."))
    df["smiles"] = df.smiles_unit_cell.apply(lambda cs: sorted(cs, key=len)[-1])

    def smiles_contains_inorganics(smi):
        smil = smi.lower()
        for elt in [
            "ti",
            "v",
            "cr",
            "mn",
            "fe",
            "co",
            "ni",
            "cu",
            "zn",
            "zr",
            "nb",
            "mo",
            "tc",
            "ru",
            "rh",
            "pd",
            "ag",
            "cd",
            "hf",
            "ta",
            "w",
            "re",
            "os",
            "ir",
            "pt",
            "au",
            "hg",
            "rf",
            "db",
            "ce",
            "sm",
            "eu",
            "gd",
            "pb",
        ]:
            if "[" + elt + "]" in smil:
                return True
        return False

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

    def patch_radicals(mol):
        """
        Crystal structures will often miss hydrogens.
        Therefore, many molecules incorrectly contain radical sites.
        This method tries its best to fix those places by setting
        the number of radicals to zero and then adding and removing
        hydrogens so we leave it to rdkit to somehow make sense.
        I eyeballed the results and it seems to does seem right
        in the large majority of the cases!
        """
        for atm in mol.GetAtoms():
            if atm.GetNumRadicalElectrons() > 0:
                atm.SetNumRadicalElectrons(0)
        return Chem.RemoveHs(Chem.AddHs(mol))

    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)

    before = len(df)
    df = df[df["mol"].apply(bool)]
    after = len(df)
    if verbose:
        print(f"removed {before-after} entries where smiles parsing returned no mol")

    if patch_radicals:
        df["mol"] = df.mol.apply(patch_radicals)
        df["smiles"] = df.mol.apply(Chem.MolToSmiles)
    df["is_organic"] = df.mol.apply(is_organic_subset_mol)
    df["smiles_contains_inorganics"] = df.raw_smiles.apply(smiles_contains_inorganics)
    if only_organics:
        before = len(df)
        df = df[df.is_organic & (~df.smiles_contains_inorganics)]
        after = len(df)
        if verbose:
            print(f"removed {before-after} non-organic molecules.")

    df["source"] = DSSource.COD.value

    if only_relevant_cols:
        df = df[["comp_no", "smiles", "solvent", "cryst", "source"]]

    df["smiles"] = df.smiles.apply(canonicalize_smiles)
    df["solv_amounts"] = [
        {} for _ in df.iterrows()
    ]  # TODO: solv ratios currently not known!

    # creates synthetic negative data to balance out this
    # positive-only dataset
    df = balance_cod_with_synthetic_negatives(df, random_state=123)

    return df


def _parse_smiles_in_list(chunk):
    rslt = []
    for lst in chunk:
        rslt.append([Chem.MolFromSmiles(smi) for smi in lst])
    return rslt


def parse_smi(smis):
    return [Chem.MolFromSmiles(smi) for smi in smis]


data_source_to_loader = {
    DSSource.NOVA.value: load_novartis,
    DSSource.COD.value: load_cod,
    DSSource.MANTAS.value: load_mantas_csv,
}


def load_all(
    sources=None,
    skip_sources=None,
    verbose=True,
    do_binarize=True,
    do_solvent_iupac=True,
    n_jobs=6,
) -> pd.DataFrame:
    from solvmate.ccryst.solvent import solvent_mixture_iupac_to_smiles

    if sources is None:
        sources = list(data_source_to_loader.keys())
    if skip_sources:
        for source in skip_sources:
            assert source in sources, "cannot skip unknown source: " + source
        sources = [source for source in sources if source not in skip_sources]

    source_loaders = []
    for source in sources:
        source_loaders.append(data_source_to_loader[source])

    all_data = []
    for source_loader in source_loaders:
        if verbose:
            print(source_loader.__name__)
        all_data.append(source_loader(verbose=verbose))

    df = pd.concat(all_data)
    if do_binarize:
        df["bin_cryst"] = df.cryst.apply(binarize_outcome)
    if do_solvent_iupac:
        df["solvent_mixture_smiles"] = df.solvent.apply(solvent_mixture_iupac_to_smiles)

    if n_jobs > 1:
        df["mol_compound"] = run_in_parallel(
            n_jobs=n_jobs, inputs=df.smiles.tolist(), callable=parse_smi
        )
    else:
        df["mol_compound"] = df.smiles.apply(Chem.MolFromSmiles)
    before = len(df)
    df = df[~df["mol_compound"].isna()]
    after = len(df)
    if verbose:
        print(
            f"removed {before - after} entries where compound smiles couldnt be parsed."
        )
    if n_jobs > 1:
        df["mol_solvent_mixture"] = run_in_parallel(
            n_jobs=n_jobs,
            inputs=df.solvent_mixture_smiles.tolist(),
            callable=_parse_smiles_in_list,
        )
    else:
        df["mol_solvent_mixture"] = df.solvent_mixture_smiles.apply(
            lambda lst: list(map(Chem.MolFromSmiles, lst))
        )
    before = len(df)
    df = df[df["mol_solvent_mixture"].apply(len) > 0]
    after = len(df)
    if verbose:
        print(
            f"removed {before - after} entries where solvent mixture smiles couldnt be parsed."
        )

    df["solvent_label"] = df.solvent_mixture_smiles.apply(str)

    if "solv_amounts" in df.columns:
        df["solv_amounts"] = [
            {} if (itm is None or str(itm).lower() == "nan") else itm
            for itm in df["solv_amounts"].tolist()
        ]
    else:
        df["solv_amounts"] = [{} for _ in df.iterrows()]

    return df


def downsample_sources(df, sources, n_sample: int) -> pd.DataFrame:
    df_downsamp = []
    for source in df.source.unique():
        g = df[df.source == source]
        if source in sources:
            g = g.sample(min(n_sample, len(g)))
        df_downsamp.append(g)
    return pd.concat(df_downsamp)


def add_split_by_col(
    df,
    col="smiles",
    amount_train=0.7,
    amount_test=0.3,
    amount_val=0.0,
    random_seed=None,
) -> None:
    if random_seed:
        random.seed(random_seed)
    split_col = random.choices(
        ["train", "test", "val"],
        weights=[amount_train, amount_test, amount_val],
        k=df[col].nunique(),
    )
    split_dct = {k: v for v, k in zip(split_col, df[col].unique())}
    df["split"] = df[col].map(split_dct)
    return None


def add_cross_fold_by_col(
    df,
    col="smiles",
    n_cv=5,
    random_seed=None,
) -> None:
    if random_seed:
        random.seed(random_seed)
    split_col = random.choices(
        [cv + 1 for cv in range(n_cv)],
        weights=[1 / n_cv for _ in range(n_cv)],
        k=df[col].nunique(),
    )
    split_dct = {k: v for v, k in zip(split_col, df[col].unique())}
    df["cross_fold"] = df[col].map(split_dct)
    return None


class _ClusterInflationError(Exception):
    pass


def _inflate_selected_cluster(
    clusters: "list[int]",
    selected_cluster: int,
    target_size: int,
    random_seed=None,
    leave_out=None,
    requires_perfect_match=True,
):
    """
    A utility function that takes a list of cluster assignments,
    and a cluster to inflate to a desired size. It will then
    proceed by merging more and more of the remaining clusters
    until the target size is reached.
    >>> clusters = [1,1,1,2,2,1,1,2,3,3,1,1,2,2,2,2,3,3,4,5,6,9,8,7,1]
    >>> selected_cluster = 3
    >>> target_size = 8
    >>> _inflate_selected_cluster(clusters,selected_cluster,target_size,random_seed=123)
    [3, 3, 3, 2, 2, 3, 3, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 4, 5, 6, 9, 8, 7, 3]

    >>> clusters = [7,17,17,2,2,17,17,2,3,3,17,1,2,2,2,2,3,3,4,5,6,9,8,7,1]
    >>> selected_cluster = 3
    >>> target_size = 6
    >>> _inflate_selected_cluster(clusters,selected_cluster,target_size,random_seed=129)
    [7, 3, 3, 2, 2, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 3, 4, 5, 6, 9, 8, 7, 1]

    >>> _inflate_selected_cluster(list(range(10)),1,3,random_seed=129)
    [0, 1, 2, 3, 4, 1, 6, 7, 8, 1]

    """
    if random_seed:
        random.seed(random_seed)
    merged_size = sum(cluster == selected_cluster for cluster in clusters)
    while merged_size < target_size:
        cluster_remains = list(
            set([cluster for cluster in clusters if cluster != selected_cluster])
        )

        if leave_out:
            cluster_remains = [
                cluster for cluster in cluster_remains if cluster not in leave_out
            ]

        if not cluster_remains:
            if requires_perfect_match:
                raise _ClusterInflationError
            else:
                return clusters

        cluster_remains = random.choice(cluster_remains)

        clusters = [
            selected_cluster if cluster == cluster_remains else cluster
            for cluster in clusters
        ]
        merged_size = sum(cluster == selected_cluster for cluster in clusters)

    return clusters


def inflate_clusters(
    clusters: "list[int]",
    selected_clusters: "list[int]",
    target_sizes: "list[int]",
    num_retrials=16,
    random_seed=None,
):
    """
    Function thats useful to bootstrap e.g. train/test/val split
    from a given clustering.
    
    The inflation of clusters is deterministic:
    >>> clusters = list(range(10))
    >>> inflate_clusters(clusters,[1,2,3],[2,2,2],random_seed=123,)
    [0, 1, 2, 3, 2, 1, 6, 7, 8, 3]
    >>> inflate_clusters(clusters,[1,2,3],[2,2,2],random_seed=123,)
    [0, 1, 2, 3, 2, 1, 6, 7, 8, 3]

    Setting different seeds gives different results:
    >>> clusters = list(range(10))
    >>> inflate_clusters(clusters,[1,2,3],[2,3,4],random_seed=123,)
    [3, 1, 2, 3, 2, 1, 2, 3, 8, 3]
    >>> inflate_clusters(clusters,[1,2,3],[2,3,4],random_seed=124,)
    [2, 1, 2, 3, 1, 3, 3, 7, 2, 3]

    In the following example we have a random set of clusters and
    we bootstrap a roundabout 50:50 train/test split from it:
    >>> for i in range(5):
    ...     clusters = [1,1,1,1,2,2,2,3,4,5,6,7,8,1,1,1,1,2,3,4,5,]
    ...     clusters = \
                    inflate_clusters( \
                    clusters,[4,5],[math.floor(len(clusters)/2),math.ceil(len(clusters)/2)], random_seed=123+i,)
    ...     sum([cluster == 4 for cluster in clusters]), sum([cluster == 5 for cluster in clusters]),clusters
    (13, 8, [4, 4, 4, 4, 5, 5, 5, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 5, 4, 4, 5])
    (10, 11, [5, 5, 5, 5, 4, 4, 4, 4, 4, 5, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 5])
    (11, 10, [4, 4, 4, 4, 5, 5, 5, 5, 4, 5, 5, 4, 5, 4, 4, 4, 4, 5, 5, 4, 5])
    (10, 11, [4, 4, 4, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 4, 5])
    (11, 10, [5, 5, 5, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 5])
    """

    for _ in range(num_retrials):
        try:
            clusters_candidate = [cluster for cluster in clusters]
            for selected_cluster, target_size in zip(selected_clusters, target_sizes):
                if random_seed is not None:
                    random_seed += 1  # assure that randomness is maintained...
                clusters_candidate = _inflate_selected_cluster(
                    clusters_candidate,
                    selected_cluster,
                    target_size,
                    random_seed=random_seed,
                    requires_perfect_match=False,
                    leave_out=[
                        cluster
                        for cluster in selected_clusters
                        if cluster != selected_cluster
                    ],
                )

            return clusters_candidate
        except _ClusterInflationError:
            continue
    raise _ClusterInflationError(
        "Could not create the desired clustering with given clusters and target sizes"
    )


def clusters_to_split(
    clusters: "list[list[int]]",
    split_ratio: "list[int]",
    eta=0.000001,
    random_seed=None,
):
    """
    Given the clusters in the format cluster-index->instance-indices, this method
    produces a new split with given split ratios.

    For example, given three clusters 1,2,3,4,5,...,10 where 1 contains five instances,
    2 contains 3 instances and 3-10 contain a single instance:
    >>> clusters = [[129,23,111,590,42],[40,27,8],[44],[99],[139],[1],[7],[333],[222],[98]]

    We can use this method to create a (roughly) 50:50 train/test split:
    >>> splits = clusters_to_split(clusters,split_ratio=[0.5,0.5],random_seed=123)
    >>> splits
    [[0, 2, 8, 9], [1, 3, 4, 5, 6, 7]]

    And these represent indeed roughly a 50:50 split:
    >>> [sum(len(clusters[idx]) for idx in split) for split in splits]
    [8, 8]

    >>> for loop in range(10):
    ...     clusters = [[i] for i in range(100)]
    ...     splits = clusters_to_split(clusters,split_ratio=[0.2,0.8],random_seed=123+loop)
    ...     print([len(split) for split in splits])
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]

    """
    if random_seed:
        random.seed(random_seed)
    n_instances = sum(map(len, clusters))
    n_splits = len(split_ratio)
    splits = [[] for _ in range(n_splits)]
    for cluster_idx, cluster in enumerate(clusters):
        while True:
            split = random.randint(0, n_splits - 1)
            if random.random() < split_ratio[split]:
                splits[split].append(cluster_idx)
                split_ratio[split] -= len(cluster) / n_instances
                split_ratio[split] = max(split_ratio[split], eta)
                break
    return splits


def add_split_by_butina_clustering(
    df,
    col="smiles",
    amount_train=0.6,
    amount_val=0.2,
    amount_test=0.2,
    random_seed=None,
    butina_radius=0.4,
    relabel=True,
):
    """
    Creates a train/val/test split on the given dataframe
    by utilizing the butina clustering.

    In normal operation relabel should always be True. If set
    to False, then the original butina clustering labels are
    kept! This is more for debugging/demonstration purposes,
    or really in any case where the raw butina clustering
    labels should be accessed.

    >>> names = ["methylanthracene","ethylanthracene","propylanthracene", \
         "furane", "2-methyl-furane", "3-methyl-furane", "2-ethyl-furane", \
         "glucose", "fructose", "galactose", \
         "bromobenzene", "chlorobenzene", "fluorobenzene", "iodobenzene", \
        ] 
    >>> smiles = [opsin_iupac_to_smiles(name) for name in names]
    >>> df = pd.DataFrame({"smiles": smiles})
    >>> print('\\n'.join([str((row.smiles[0:-1],row.split)) for _,row in add_split_by_butina_clustering(df,random_seed=123).iterrows()])) # doctest:+NORMALIZE_WHITESPACE
    ('CC1=CC=CC2=CC3=CC=CC=C3C=C12', 'train')
    ('C(C)C1=CC=CC2=CC3=CC=CC=C3C=C12', 'train')
    ('C(CC)C1=CC=CC2=CC3=CC=CC=C3C=C12', 'train')
    ('O1C=CC=C1', 'val')
    ('CC=1OC=CC1', 'train')
    ('CC1=COC=C1', 'test')
    ('C(C)C=1OC=CC1', 'train')
    ('O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO', 'train')
    ('OCC(=O)[C@@H](O)[C@H](O)[C@H](O)CO', 'train')
    ('O=C[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO', 'train')
    ('BrC1=CC=CC=C1', 'test')
    ('ClC1=CC=CC=C1', 'val')
    ('FC1=CC=CC=C1', 'test')
    ('IC1=CC=CC=C1', 'val')

    """
    smiles = list(sorted(set(df[col].tolist())))
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    # ecfps =[ecfp_fingerprint(mol) for mol in mols]
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = [rdkit_gen.GetFingerprint(mol) for mol in mols]
    butina_clusters = simple_butina_clustering(
        fingerprints,
        cutoff=butina_radius,
    )

    splits = clusters_to_split(
        butina_clusters,
        split_ratio=[amount_train, amount_val, amount_test],
        random_seed=random_seed,
    )

    smi_to_clu = {}
    for clu_idx, clu in enumerate(butina_clusters):
        for mol_idx in clu:
            smi_to_clu[smiles[mol_idx]] = clu_idx

    clu_to_split = {}
    for split_idx, clusters in enumerate(splits):
        for cluster in clusters:
            clu_to_split[cluster] = split_idx

    if relabel:
        smi_to_split = {smi: clu_to_split[smi_to_clu[smi]] for smi in smiles}
    else:
        smi_to_split = {smi: smi_to_clu[smi] for smi in smiles}

    df["split"] = df[col].map(smi_to_split)

    if relabel:
        df.split = df.split.map(
            {
                0: "train",
                1: "val",
                2: "test",
            }
        )

    return df


def balance_cod_with_synthetic_negatives(
    df,
    k_neg=1,
    random_state=None,
):
    """
    Balances the given dataset but only for the specified sources in
    such a way, that for every positive sample, we also generate
    k_neg negative samples to counterweight. While it is true that
    this process will generate some noise (potentially emitting false
    negative datapoints), it is the only way to combat highly optimistic
    classifiers due to only positive labels within the cod data source.
    Thats why its only applied to cod for now.
    Example:
    >>> df = pd.DataFrame([{"source":"cod","solvent":"dimethylsulfoxid", \
        "solvent_label":"['CS=O(C)']","cryst":1,}, \
        {"source":"cod","solvent":"methanol", \
        "solvent_label": "['CO']","cryst":1,}])
    >>> balance_cod_with_synthetic_negatives(df,k_neg=1,random_state=123,)#doctest:+NORMALIZE_WHITESPACE
    """
    assert "solvent" in df.columns

    if random_state is None:
        random_state = random.randint(1, 10000000)

    df_syn_negs = []

    for _, row in df.iterrows():
        row_dct = row.to_dict()
        solvent = row_dct["solvent"]

        dfso = df[[not solvent.lower() in s.lower() for s in df.solvent.tolist()]]
        dfso = dfso.sample(k_neg, random_state=random_state).copy()
        random_state += 1

        dfso["cryst"] = "no"
        dfso["solvent"] = solvent

        df_syn_negs.append(dfso)

    df_syn_negs = pd.concat(df_syn_negs)

    assert len(df_syn_negs) == k_neg * len(df)

    # Accumulate both synthetic negatives and original positives
    df_out = pd.concat([df_syn_negs, df])

    # Shuffle
    df_out = df_out.sample(frac=1.0, random_state=random_state)

    assert len(df_out) == (1 + k_neg) * len(df)
    return df_out
