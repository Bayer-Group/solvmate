from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import multiprocessing
import tempfile
import contextlib
from scipy.stats import spearmanr


import json
import argparse
from pathlib import Path
import random
import uuid
import pandas as pd
import numpy as np
import os
import sqlite3
import joblib

import datetime

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from typing import TYPE_CHECKING, Iterable

from matplotlib import pyplot as plt

import hashlib


if TYPE_CHECKING:
    from solvmate.ranksolv.recommender import Recommender
    from solvmate.ranksolv.jack_knife_recommender import JackKnifeRecommender

USE_COSMO_RS_FEATURES = True

# The minimum difference that we expect out of pairs in log units.
# Set to 0.25, so 1/4 log(S) units as this should be close
# to the typical experimental accuracy.
MIN_LOG_S_DIFF_THRESHOLD = 0.01  # 0.25 default

# This is per Solvent Job. We allow for a maximum of 20 mins per single solvent job:
XTB_TIMEOUT_SECONDS = 1200
XTB_OUTER_JOBS = min(
    32, multiprocessing.cpu_count() - 1
)  # uses up to a maximum of 32 cpu cores but always leaves one cpu free
XTB_INNER_JOBS = 1  # Used to be 4. but we can get more from outer parallelization

SQLITE_TIMEOUT_SECONDS = 120

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOWE_META_DB = DATA_DIR / "lowe_meta.db"

SOLVENT_SELECTION_DB = DATA_DIR / "solvent_selection.db"

SRC_DIR = PROJECT_ROOT / "src"
DOC_DIR = PROJECT_ROOT / "doc"
FIG_DIR = DOC_DIR / "fig"

DATA_FLE_NOVA = DATA_DIR / "novartis_dataset_c7ce00738h6.csv"
DATA_FLE_OPEN_NOTEBOOK = DATA_DIR / "20150430SolubilitiesSum.xlsx"

# Should perform logging?
DO_LOG = 1

# Is in development?
DEVEL = 1


# The number of jobs used in training classifiers.
# On normal systems (even laptops) eight should
# be a reasonable default.
#
N_JOBS = 8

opsin_cache_file = Path(__file__).parent / ".opsin_cache.json"


def temp_dir():
    if Path("/tmp").exists():
        tmp_dir = Path("/tmp") / "solvmate"
    else:
        tmp_dir = Path(tempfile.gettempdir()) / "solvmate"
    tmp_dir.mkdir(
        exist_ok=True,
    )
    return tmp_dir


def random_fle(
    suf: str,
):
    fle_name = f"{uuid.uuid4().hex}.{suf}"
    return temp_dir() / fle_name


def download_file(
    url: str,
    dest: str,
) -> bool:
    """
    Attempts to download the given file given file first using curl, then
    using wget and finally using pythons urllib.

    A warning is printed in case this download fails.
    """
    import urllib.request

    if os.system(f"wget {url} -O {dest}") and os.system(f"curl {url} > {dest}"):
        try:
            urllib.request.urlretrieve(f"{url}", f"{dest}")
        except:
            warn(f"failed to download file {dest} from url {url}")
            warn(
                f"please download file from url: {url} and move it into location {dest}"
            )


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev_cwd))


def split_in_chunks(
    df: pd.DataFrame,
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    """
    Yields an iterable of chunk_sized chunks of the given
    dataframe.

    Useful to reduce the memory load in e.g. predict calls.

    >>> df = pd.DataFrame({"x": list(range(20)), "y":[itm**2 for itm in range(20)]})
    >>> rslt = list(split_in_chunks(df,8))
    >>> rslt
    [   x   y
    0  0   0
    1  1   1
    2  2   4
    3  3   9
    4  4  16
    5  5  25
    6  6  36
    7  7  49,      x    y
    8    8   64
    9    9   81
    10  10  100
    11  11  121
    12  12  144
    13  13  169
    14  14  196
    15  15  225,      x    y
    16  16  256
    17  17  289
    18  18  324
    19  19  361]
    >>> (pd.concat(rslt) == df).all().all()
    True

    """
    N = len(df)
    start = 0
    while start < N:
        end = start + chunk_size
        yield df.iloc[start:end]
        start = end


OPSIN_CMD = "opsin"


def _patch_opsin_from_deps_dir_if_possible(d: Path):
    global OPSIN_CMD
    # Fix for windows release
    for fle in d.iterdir():
        if fle.is_dir():
            # recursively walk the directory tree
            _patch_opsin_from_deps_dir_if_possible(fle)
        else:
            if fle.suffix == ".jar" and "opsin" in fle.name:
                OPSIN_CMD = f"java -jar {fle.resolve()}"
                print(
                    f"found a .jar file at {fle}. Employing opsin command: `{OPSIN_CMD}`"
                )


_patch_opsin_from_deps_dir_if_possible(PROJECT_ROOT / "deps")


def opsin_iupac_to_smiles(iupac) -> str:
    """
    Forces the actual parsing of the IUPAC via
    the OPSIN tool
    :param iupac: The IUPAC string to parse
    :return: The resulting SMILES string.
    """
    if opsin_cache_file.exists():
        cache = json.loads(opsin_cache_file.read_text())
    else:
        cache = {}
    if iupac in cache:
        return cache[iupac]
    try:
        tmp_dir = temp_dir()
        tmp_fle = f"/{tmp_dir}/__opsin_tmp__{random.randint(1000,10000000)}.in"
        with open(tmp_fle, "w") as fout:
            fout.write(iupac)
        smi = os.popen(f"{OPSIN_CMD} " + tmp_fle).read()
    finally:
        os.remove(tmp_fle)
    cache[iupac] = smi
    with open(opsin_cache_file, "wt") as fout:
        json.dump(obj=cache, fp=fout)
    return smi


def opsin_iupac_to_mol(iupac: str) -> Chem.Mol:
    return Chem.MolFromSmiles(opsin_iupac_to_smiles(iupac))


_DCT_SMI_TO_NAME = None


def smi_mix_to_name(smi: str) -> str:
    return " : ".join([smi_to_name(smi_part) for smi_part in smi.split(".")])


def smi_to_name(smi: str) -> str:
    """
    Attempts to give the corresponding (solvent) IUPAC
    name for the given smiles. Returns the canonical
    smiles if no name was found.
    Therefore useful for displaying results to the
    end user who might not like being represented
    results as SMILES.

    >>> smi_to_name("CCOCC")
    'diethyl ether'
    >>> smi_to_name("CCOC(C)")
    'diethyl ether'
    >>> smi_to_name("CCCCCC")
    'hexane'
    >>> smi_to_name("C1CCCCC1CCN")
    'C1CCCCC1CCN'
    >>> smi_to_name("CC1CCCCC1")
    'methylcyclohexane'
    """
    global _DCT_SMI_TO_NAME
    if _DCT_SMI_TO_NAME is None:
        smi_to_name_fle = DATA_DIR / "smi_to_name.json"
        assert smi_to_name_fle.exists(), "need to build smi_to_name first!"
        _DCT_SMI_TO_NAME = json.loads(smi_to_name_fle.read_text())
    can_smi = Chem.CanonSmiles(smi)
    return _DCT_SMI_TO_NAME.get(can_smi, smi)


def name_to_canon(
    name: str,
) -> str:
    return Chem.CanonSmiles(opsin_iupac_to_smiles(name))


def names_to_canons(*args):
    return list(map(name_to_canon, args))


def warn(msg):
    print("\033[31;1;4m", msg, "\033[0m")


def info(*args, **kwargs):
    print(*args, **kwargs)


def log(*args, **kwargs):
    if DO_LOG:
        print(*args, **kwargs)


def get_training_data(
    include_dont_use_entries=False,
) -> pd.DataFrame:
    con = DATA_DIR / "training_data.db"
    assert (
        con.exists()
    ), "need training data db to build xtb features. run build_training_data.py first!"
    con = sqlite3.connect(con)
    data = pd.read_sql(sql="SELECT * FROM training_data", con=con)
    con.close()

    if not include_dont_use_entries:
        data = data[data["dont_use"] == 0]


    add_cross_fold_by_col(
        data,
        col="solute SMILES",
    )

    assert "conc" in data.columns
    assert "solvent SMILES" in data.columns
    assert "solute SMILES" in data.columns
    return data


def get_xtb_features_data(db_file: Path = None) -> pd.DataFrame:
    if db_file is None:
        db_file = DATA_DIR / "xtb_features.db"

    assert db_file.exists()
    con = sqlite3.connect(db_file)
    data = pd.read_sql(sql="SELECT * FROM solv_en", con=con)
    con.close()

    for col in ["index", "level_0"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    assert "smiles" in data.columns
    assert "solvent" in data.columns
    return data


def get_pairs_data() -> pd.DataFrame:
    con = DATA_DIR / "pairs.db"
    assert con.exists()
    con = sqlite3.connect(con)
    data = pd.read_sql(sql="SELECT * FROM pairs", con=con)
    con.close()

    return data


def get_recommender(fle=None) -> "Recommender":
    from solvmate.ranksolv.recommender import Recommender

    if fle is None:
        fle = DATA_DIR / "recommender.pkl"
    else:
        fle = Path(fle)
    assert fle.exists()
    return Recommender.load(fle.resolve())


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


def ecfp_count_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """

    >>> ecfp_fingerprint(Chem.MolFromSmiles("OCCCCO")).max()
    1
    >>> ecfp_count_fingerprint(Chem.MolFromSmiles("OCCCCO")).max()
    3

    """
    fpgen = AllChem.GetRDKitFPGenerator()
    fp = fpgen.GetCountFingerprint(mol)
    return fingerprint_to_numpy(fp)


def reals_to_placement(xs: list[float]):
    """
    Turns the given list of reals [e.g. concentrations]
    into a corresponding placement ranking [e.g. solvent ranking].
    For example:
    >>> reals_to_placement([0.1, 0.01, 0.5, 0.2,])
    [1, 0, 3, 2]
    """
    xs_sorted = sorted(xs)
    x_to_pos = {x: pos for pos, x in enumerate(xs_sorted)}
    return [x_to_pos[x] for x in xs]


# A quick hack to store the stats so that
# we can easily retrieve all relevant results to e.g.
# compare different models against each other, plot
# figures, create tables et cetera.
#
_DB_STATS = DATA_DIR / "observations.db"


def io_store_stats(
    df: pd.DataFrame,
):
    """
    Persists the given statistic to our
    stats database.
    """
    con = sqlite3.connect(_DB_STATS)

    df.to_sql(
        name="stats",
        con=con,
        if_exists="append",
    )


class Silencer:
    """
    A useful tool for silencing stdout and stderr.
    Usage:
    >>> with Silencer() as s:
    ...         print("kasldjf")

    >>> print("I catched:",s.out.getvalue())
    I catched: kasldjf
    <BLANKLINE>

    Note that nothing was printed and that we can later
    access the stdout via the out field. Similarly,
    stderr will be redirected to the err field.
    """

    def __init__(self):
        self.out = io.StringIO()
        self.err = io.StringIO()

    def __enter__(self):
        self.rs = redirect_stdout(self.out)
        self.re = redirect_stderr(self.err)
        self.rs.__enter__()
        self.re.__enter__()
        return self

    def __exit__(self, exctype, excinst, exctb):
        self.rs.__exit__(exctype, excinst, exctb)
        self.re.__exit__(exctype, excinst, exctb)


def log_function_call(func):
    """

    >>> class C:
    ...     @log_function_call
    ...     def run(self):
    ...         print("hello")
    >>> o = C()
    >>> o.run()
    call   -> run@<doctest solvmate.log_function_call[0]>:2
    hello
    return <- run
    """

    def wrapper(*args, **kwargs):
        if DEVEL:
            print(
                "call   ->",
                func.__name__
                + "@"
                + func.__code__.co_filename
                + ":"
                + str(func.__code__.co_firstlineno),
            )
        rslt = func(*args, **kwargs)
        if DEVEL:
            print("return <-", func.__name__)
        return rslt

    return wrapper


def combine_fold_preds(cols: list[list[str]]):
    """
    combines the recommendations of different predictor folds together.

    For example:
    >>> combine_fold_preds([['OCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'CC#N', 'CCO', 'CC(C)=O'],
    ... ['OCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'CC#N', 'CC(C)=O', 'CCO'],
    ... ['CC#N', 'OCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'CCO', 'CC(C)=O'],
    ... ['OCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'CC#N', 'CCO', 'CC(C)=O'],
    ... ['CC#N', 'OCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'CCO', 'CC(C)=O']])
    ['OCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'CC#N', 'CCO', 'CC(C)=O']
    """
    if not len(cols):
        return []

    ent_idx = {ent: 0 for ent in cols[0]}
    for col in cols:
        for ent in col:
            ent_idx[ent] += list(col).index(ent)
    return [kv[0] for kv in sorted(ent_idx.items(), key=lambda kv: kv[1])]


def remove_close_pairs(df, thresh_log_s, random_state=None):
    """
    We should not evaluate on very close pairs,
    e.g. if the concentrations are almost the same then
    it doesn't make sense to force a ranking in our evaluation.

    This is why the given function removes pairs of entries that
    are too close in concentration by enforcing the specified
    threshold thresh_log_s.
    >>> df = pd.DataFrame({
    ...    "solvent SMILES": ["CO","CCO","O"],
    ...    "conc": [0.01,0.012,10],
    ... })
    >>> remove_close_pairs(df,thresh_log_s=0.5,random_state=123) #doctest:+NORMALIZE_WHITESPACE
    solvent SMILES    conc  log_conc
    1            CCO   0.012 -1.920819
    2              O  10.000  1.000000

    """
    if random_state:
        random.seed(random_state)
    df = df.copy()
    assert "solvent SMILES" in df.columns
    assert "conc" in df.columns

    to_rmvs = []
    df["log_conc"] = df["conc"].apply(np.log10)
    for ia, a in df.iterrows():
        for ib, b in df.iterrows():
            if ia >= ib:
                continue

            ca = a["log_conc"]
            cb = b["log_conc"]
            if abs(ca - cb) < thresh_log_s:
                # we got a hit
                to_rmvs.append(random.choice([ia, ib]))

    for to_rmv in to_rmvs:
        df = df.drop(index=to_rmv)
        return remove_close_pairs(df, thresh_log_s)

    return df


def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    for i in range(1, len(fp_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix


def simple_butina_clustering(
    fps,
    cutoff=0.4,
):
    dist_mat = tanimoto_distance_matrix(fps)
    clusters = Butina.ClusterData(dist_mat, len(fps), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


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


def add_cv_by_butina_clustering(
    df,
    col="smiles",
    n_cvs=5,
    random_seed=None,
    butina_radius=0.4,
):
    """

    >>> df = pd.DataFrame({"smiles": ["CCCPCC","CCCCPCC","COCC","COCCC","c1ccc1C=O","c1c(C)cc1C=O"]})
    >>> add_cv_by_butina_clustering(df,random_seed=123)#doctest:+NORMALIZE_WHITESPACE
            smiles  cv
    0        CCCPCC   0
    1       CCCCPCC   0
    2          COCC   1
    3         COCCC   1
    4     c1ccc1C=O   2
    5  c1c(C)cc1C=O   2
    >>> add_cv_by_butina_clustering(df,random_seed=456,) #doctest:+NORMALIZE_WHITESPACE
            smiles  cv
    0        CCCPCC   2
    1       CCCCPCC   2
    2          COCC   1
    3         COCCC   1
    4     c1ccc1C=O   0
    5  c1c(C)cc1C=O   0
    """
    smiles = list(sorted(set(df[col].tolist())))
    random.seed(random_seed)
    random.shuffle(smiles)
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
        split_ratio=[1 / n_cvs for _ in range(n_cvs)],
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

    smi_to_split = {smi: clu_to_split[smi_to_clu[smi]] for smi in smiles}

    df["cross_fold"] = df[col].map(smi_to_split)

    return df


def obabel_to_smiles(
    fle: Path,
) -> str:
    """
    Converts the given file (in any format that is supported by openbabel)
    into a smiles. Notably, this includes also chemdraw file format,
    such as .cdx and .cdxml formats.
    The distinction is made automatically by openbabel.
    We use this in solvmate to allow a flexible drag-and-drop of different
    chemical files directly into the component.
    """
    tmp_fle = random_fle("smi")
    os.system(f"obabel {fle} -O {tmp_fle}")
    smis = tmp_fle.read_text().split()

    smis_cleaned = []
    for smi in smis:
        try:
            smi = Chem.CanonSmiles(smi)
            if smi:
                smis_cleaned.append(smi)
        except:
            continue
    return ".".join(smis_cleaned)


def apply_delta_s_window(
    df: pd.DataFrame,
    delta_s: float,
    random_state=None,
) -> pd.DataFrame:
    """
    >>> df = pd.DataFrame({"solute SMILES": list("aaa"), "solvent SMILES a": list("abc"), "solvent SMILES b": list("xyz"), "conc": [0.1,0.11,100]})
    >>> apply_delta_s_window(df,1)
    >>> apply_delta_s_window(df,200)
    """
    if random_state:
        random.seed(random_state)
    df = df.copy().sample(
        frac=1.0,
        random_state=random_state,
    )

    to_remove = []
    for solute_smi in df["solute SMILES"].unique():
        g = df[df["solute SMILES"] == solute_smi]

        for ia, a in g.iterrows():
            for ib, b in g.iterrows():
                if ia >= ib:
                    continue
                if ia in to_remove or ib in to_remove:
                    continue
                if abs(a["conc"] - b["conc"]) < delta_s:
                    to_remove.append(random.choice([ia, ib]))

    df = df.drop(to_remove)
    return df


def screen_window_experiment(
    show: bool, df_org, apply_log, recommender: "JackKnifeRecommender"
):

    stats = []
    for delta_log_s in [
        0.01,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ]:
        df = df_org.copy()

        if apply_log:
            df["conc"] = [
                row["conc"] if not np.isnan(row["conc"]) else (row["Solubility"])
                for _, row in df.iterrows()
            ]
            df["conc"] = np.log(df["conc"])
            df["Solubility"] = np.log(df["Solubility"])

        df = apply_delta_s_window(df, delta_log_s)
        for example_idx, solute_smi in enumerate(
            df_org.sort_values(
                [
                    "n_solvents",
                    "solute SMILES",
                ],
                ascending=False,
            )
            .drop_duplicates("solute SMILES")["solute SMILES"]
            .iloc[0:20]
            .tolist()
        ):
            g = df[(df["solute SMILES"] == solute_smi)]

            if len(g) < 2:
                continue  # cannot compute ranking with less than 2 solvents...

            common_solubility_unit = g["Solubility Unit"].value_counts().index[0]
            g = g[g["Solubility Unit"] == common_solubility_unit]
            g = g.drop_duplicates("solvent SMILES")

            # display(g)
            actual = g.sort_values("Solubility", ascending=False)[
                "solvent SMILES"
            ].tolist()

            filter_solvents = list(actual)

            for rc in recommender.rcs:
                rc.featurization = "ab_Gsolv_cat_norm_partit_solv_solu_ecfp"

                rc.top_pairs_ = [
                    (a, b) for a in filter_solvents for b in filter_solvents if a != b
                ]

            rslt = recommender.recommend_smiles_spread([solute_smi])

            rs = rslt[0]

            rs = pd.DataFrame(
                [{"col": col, "idx": idx} for row in rs for idx, col in enumerate(row)]
            )
            rs = rs.groupby("col")["idx"].mean().to_dict()
            spear = -(spearmanr(range(len(actual)), [rs[col] for col in actual])[0])

            pred = [
                kv[0] for kv in sorted([kv for kv in rs.items()], key=lambda kv: kv[1])
            ]

            stats.append({"spearmanr": spear, "delta_log_s": delta_log_s})

            if show:
                display(rs)
                print(spear)

                print(actual, [rs[col] for col in actual])
                rank_compare_plot(solute_smi, actual, pred)
                plt.title(f"R = {spear:.2f}")
                # plt.savefig(f"example_{example_idx}.svg")
                plt.show()
    return pd.DataFrame(stats)
