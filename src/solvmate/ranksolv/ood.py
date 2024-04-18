import argparse
from dataclasses import dataclass
import re
from solvmate import *
from solvmate.ranksolv.jack_knife_recommender import JackKnifeRecommender
from scipy.stats import spearmanr, kendalltau, pearsonr


def parse_ood_data(
    use_blacklist: bool = False,
):
    """
    Parses the out of distribution data within data/ood.txt
    and returns it as a dataframe for convenient handling.
    >>> parse_ood_data()# doctest:+NORMALIZE_WHITESPACE
                                        smiles   solub       solvent_name
    0           CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O   9.000        1,4-dioxane
    1           CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O  88.330            acetone
    2           CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O  64.480         chloroform
    3           CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O  88.650            ethanol
    4           CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O  53.160      ethyl acetate
    ..                                      ...     ...                ...
    120  OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O   0.080       iso-propanol
    121  OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O   2.350           methanol
    122  OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O   7.620           pyridine
    123  OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O   0.006  trichloroethylene
    124  OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O  82.000              water
    <BLANKLINE>
    [125 rows x 3 columns]
    """
    ood_data = (DATA_DIR / "ood.txt").read_text()
    lines = list(ood_data.split("\n")) + [""]
    all_solvent_names = []
    all_solubs = []
    all_smiles = []
    this_smiles, this_solubs, this_solvent_names = None, [], []
    for idx, lne in enumerate(lines):
        if not lne.strip():
            all_solvent_names += this_solvent_names
            all_solubs += this_solubs
            all_smiles += [this_smiles for _ in this_solubs]
            this_smiles, this_solubs, this_solvent_names = None, [], []

        temp = re.findall(r"\((-?[0-9]*.?[0-9]*)°C\)", lne)
        if temp:
            temp = float(temp[0])
            if abs(temp - 20) > 5:
                continue
            solvent_name, lne = lne.split(":")
            if solvent_name.strip() == "water:":
                solvent_name = "water"
            solub = lne.strip().split()[0]
            this_solvent_names.append(solvent_name)
            this_solubs.append(float(solub))
        else:
            this_smiles = lne.strip()
    blacklist = [
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "CC2(C)CCCC(\C)=C2\C=C\C(\C)=C\C=C\C(\C)=C\C=C\C=C(/C)\C=C\C=C(/C)\C=C\C1=C(/C)CCCC1(C)C",
        "ClC5(Cl)[C@]3(Cl)C(\Cl)=C(\Cl)[C@@]5(Cl)[C@H]4[C@H]1C[C@H]([C@@H]2O[C@H]12)[C@@H]34",
    ]

    df = pd.DataFrame(
        {
            "smiles": all_smiles,
            "solub": all_solubs,
            "solvent_name": all_solvent_names,
        }
    )
    if use_blacklist:
        df = df[~df["smiles"].isin(blacklist)]

    df["solute SMILES"] = df["smiles"]
    df["solvent SMILES"] = df["solvent_name"].apply(opsin_iupac_to_smiles)
    df["solvent SMILES"] = df["solvent SMILES"].apply(Chem.CanonSmiles)
    df["conc"] = df["solub"].apply(np.log10)

    df = df[~df["solvent SMILES"].isna()]
    return df


def parse_ood_data_bayer():
    solvent_to_iupac = {
        "n-Heptan": ["n-heptane"],
        "Cyclohexan": ["cyclohexane"],
        "Diisopropylether": ["diisopropylether"],
        "Toluol": ["toluene"],
        "Tetrahydrofuran": ["tetrahydrofurane"],
        "Aceton": ["acetone"],
        "Ethylacetat": ["ethylacetate"],
        "ACN": ["acetonitrile"],
        "2-Propanol": ["2-propanol"],
        "Ethanol": ["ethanol"],
        "EtOH / Wasser 1:1": ["ethanol", "water"],
        "Methanol": ["methanol"],
        "Wasser": ["water"],
        "Dichlormethan": ["dichloromethane"],
    }
    fle = Path(os.path.expanduser("~/data/bayer/ess_solubility_excerpt.xlsx"))
    assert fle.exists()
    df = pd.read_excel(
        fle,
    )
    df["solvent_name"] = df["Lösungsmittel"]
    df["solvent SMILES"] = df["solvent_name"].apply(
        lambda sn: ".".join(
            [opsin_iupac_to_smiles(iup) for iup in solvent_to_iupac[sn.strip()]]
        )
    )
    df["solvent SMILES"] = df["solvent SMILES"].apply(Chem.CanonSmiles)
    df["solute SMILES"] = df["SMILES"]
    df["solub"] = df["Bewertung Löslichkeit"]
    df = df[~df["solub"].isna()]
    df = df[df["solub"].apply(len) != 0]
    df["conc"] = df["solub"].apply(lambda s: s.count("+") - s.count("-"))
    df = df[~df["conc"].isna()]

    fill_up_smiles = list(df["solute SMILES"])
    fill_with = fill_up_smiles[0]
    for i, smi in enumerate(fill_up_smiles):
        assert fill_with
        if not smi or not str(smi).strip() or str(smi).lower() == "nan":
            fill_up_smiles[i] = fill_with
        else:
            fill_with = smi
    df["solute SMILES"] = fill_up_smiles

    return df


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blacklist",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--bayer",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--recf",
        required=False,
    )
    args = parser.parse_args()

    if args.recf:
        fp = args.recf
        fp = Path(fp)
        if not fp.exists():
            fp = DATA_DIR / fp

        assert fp.exists()
    else:
        fp = DATA_DIR / "recommender.pkl"

    print("using recommender from file", fp)
    rec = get_recommender(fle=fp)

    if args.bayer:
        data = parse_ood_data_bayer()
    else:
        data = parse_ood_data(
            use_blacklist=args.blacklist,
        )

    if os.environ.get("_PR_FAST_DEBUG"):
        print("WARNING: fast debug mode enabled. sampling very small subset")
        data = data.sample(len(data) // 10)

    print(eval_on_ood_data(data=data, rec=rec))


@dataclass
class StatsOOD:
    r_spearman: float
    r_kendalltau: float


def eval_on_ood_data(
    data: pd.DataFrame,
    pairs: pd.DataFrame,
    rec: "Recommender",
) -> StatsOOD:

    print("len(data)", len(data))
    preds_all = rec.recommend(
        data["solute SMILES"],
        pairs=pairs,
    )
    preds_df = pd.DataFrame(
        {
            "solute SMILES": solute_smiles,
            "solvent SMILES": solvent_smiles,
            "pred_place": pred_place,
        }
        for (preds, solute_smiles) in zip(
            preds_all,
            data["solute SMILES"],
        )
        for pred_place, solvent_smiles in enumerate(preds)
    )
    data_join = data.merge(
        preds_df,
        how="inner",
        on=[
            "solvent SMILES",
            "solute SMILES",
        ],
    )
    if "solub" in data.columns:
        print(f"{data['solub'].value_counts()=}")
    spears, kts = [], []
    for solute_smiles in data_join["solute SMILES"].unique():
        g = data_join[data_join["solute SMILES"] == solute_smiles]
        spears.append(
            spearmanr(g["pred_place"], reals_to_placement(g["conc"]))[0],
        )
        kts.append(
            kendalltau(g["pred_place"], reals_to_placement(g["conc"]))[0],
        )
    spears = pd.Series(np.array(spears))
    kts = pd.Series(np.array(kts))

    print("*=" * 40)
    print("mean_spear_r_ood", spears.mean(), "+-", spears.std(ddof=1))
    print("mean_kendalltau_ood", kts.mean(), "+-", kts.std(ddof=1))
    return StatsOOD(r_spearman=spears.mean(), r_kendalltau=kts.mean())


def eval_err_estimate_on_ood_data(
    data: pd.DataFrame,
    pairs: pd.DataFrame,
    jkr: "JackKnifeRecommender",
) -> StatsOOD:
    print("len(data)", len(data), "before")
    data = data[
        data["solvent SMILES"].isin(pairs["solvent SMILES a"])
    ]  # <---- I suspect this filter is redundant. TODO: check removal!
    data = data[data["solute SMILES"].isin(pairs["solute SMILES"])]
    print("len(data)", len(data), "after")
    preds_all, preds_err = jkr.recommend_with_err(
        data["solute SMILES"],
        pairs=pairs,
    )

    preds_df = pd.DataFrame(
        {
            "solute SMILES": solute_smiles,
            "solvent SMILES": solvent_smiles,
            "pred_place": pred_place,
            "pred_err": pred_err,
        }
        for (preds, solute_smiles, pred_err) in zip(
            preds_all,
            data["solute SMILES"],
            preds_err,
        )
        for pred_place, solvent_smiles in enumerate(preds)
    )
    data_join = data.merge(
        preds_df,
        how="inner",
        on=[
            "solvent SMILES",
            "solute SMILES",
        ],
    )

    if "solub" in data.columns:
        print(f"{data['solub'].value_counts()=}")

    spears, kts, est_errs = [], [], []
    for solute_smiles in data_join["solute SMILES"].unique():
        g = data_join[data_join["solute SMILES"] == solute_smiles]
        spears.append(
            spearmanr(g["pred_place"], reals_to_placement(g["conc"]))[0],
        )
        kts.append(
            kendalltau(g["pred_place"], reals_to_placement(g["conc"]))[0],
        )
        est_errs.append(g["pred_err"].iloc[0])  # same for all in group

    spears = pd.Series(np.array(spears))
    kts = pd.Series(np.array(kts))

    print("*=" * 40)
    print("GOODNESS of error estimates:")
    print("kendalltau", kendalltau(spears.fillna(spears.mean()), est_errs))
    print("spearmanr", spearmanr(spears.fillna(spears.mean()), est_errs))
    print("pearsonr", pearsonr(spears.fillna(spears.mean()), est_errs))
    plt.plot(spears.fillna(spears.mean()), est_errs, "bo")
    plt.show()
    print("*=" * 40)
    return StatsOOD(r_spearman=spears.mean(), r_kendalltau=kts.mean())


if __name__ == "__main__":
    run()
