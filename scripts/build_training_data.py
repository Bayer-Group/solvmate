from solvmate import *

"""
Script used to generate the training data.
"""


def _canon_else_none(smi: str):
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None


def load_open_notebook_dataset() -> pd.DataFrame:
    data_fle = DATA_FLE_OPEN_NOTEBOOK
    assert data_fle.exists()
    df = pd.read_excel(data_fle)
    df["conc"] = df["concentration (M)"]
    df = df[df["conc"] > 0.0]
    df["conc"] = df["conc"].apply(np.log10)

    df = pd.DataFrame(df)
    df["solvent SMILES"] = df["solvent SMILES"].apply(_canon_else_none)
    df["solute SMILES"] = df["solute SMILES"].apply(_canon_else_none)
    assert df["solvent SMILES"].isna().sum() < len(df) * 0.1
    df["source"] = "open_notebook"

    if os.environ.get("_SM_Y_SCRAMBLING"):
        warn("performing y scrambling")
        conc = df["conc"].tolist()
        random.shuffle(conc)
        df["conc"] = conc

    return df


def load_novartis_dataset():
    dfn = pd.read_csv(DATA_FLE_NOVA)
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
    solvent_to_smiles = {
        solvent: opsin_iupac_to_smiles(solvent_iupac)
        for solvent, solvent_iupac in solvents_to_iupac.items()
    }

    rank_map = {
        "RS": 4,
        "KS": 3,
        "TS": 2,
        "PS": 1,
    }
    nan_counter, non_nan_counter = 0, 0
    df_nova = []
    for _, row in dfn.iterrows():
        smiles = row["SMILES"]
        for solv in solvents_to_iupac:
            conc = row[f"S-{solv}"]
            if str(conc).lower() == "nan":
                nan_counter += 1
                # print("encountered nan")
                continue
            else:
                non_nan_counter += 1
            assert conc in rank_map, f"expected {conc} <= {sorted(rank_map.keys())}"
            conc = rank_map[conc]
            solvent_smiles = solvent_to_smiles[solv]
            solvent_smiles = Chem.CanonSmiles(solvent_smiles)
            smiles = _canon_else_none(smiles)
            df_nova.append(
                {
                    "smiles": smiles,
                    "solute SMILES": smiles,
                    "conc": conc,
                    "solvent SMILES": solvent_smiles,
                    "solvent": solvent_smiles,
                }
            )

    df_nova = pd.DataFrame(df_nova)
    # df_nova = df_nova[
    #    (~df_nova["mol_compound"].isna()) & (~df_nova["mol_solvent_mixture"].isna())
    # ]
    df_nova["source"] = "nova"

    if os.environ.get("_SM_Y_SCRAMBLING"):
        warn("performing y scrambling")
        conc = df_nova["conc"].tolist()
        random.shuffle(conc)
        df_nova["conc"] = conc
    return df_nova


def filter_solvents_by_threshold(td, thresh=100):
    # Currently: only taking those solvents that
    # have more than thresh examples in open_notebook database
    top_solvents = []
    td = td[td.source == "open_notebook"]
    for solv, count in td.value_counts("solvent SMILES").iloc[0:10000].iteritems():
        if count > thresh:
            top_solvents.append(solv)
        else:
            break
    return top_solvents


def filter_relevant_subset(df: pd.DataFrame, N_thresh: int) -> pd.DataFrame:
    """
    Applies the following idea to generate better data:
    If we only have very few measurements, then a datapoint
    really isn't that useful.
    Therefore, we required at least 5 measurements to be
    present for a compound to consider it, aka there need
    to be 5 different solvents measured for a given
    compound to be included in the dataset.
    I saw strong improvements in the ood evaluation (kendalltau doubled!)
    Hence, this seems to be a useful heuristic.
    """

    acceptable = {
        key
        for key, count in df["solute SMILES"].value_counts().to_dict().items()
        if count > N_thresh
    }
    return df[df["solute SMILES"].isin(acceptable)]


if __name__ == "__main__":
    out_file = DATA_DIR / "training_data.db"
    if not out_file.exists():
        con = sqlite3.connect(out_file)
        df = pd.concat(
            [
                load_novartis_dataset(),
                load_open_notebook_dataset(),
            ]
        )
        dont_use = []
        for _, row in df.iterrows():
            blob = str(row).lower().replace(" ", "").replace("\t", "").replace("\n", "")
            if "donotuse" in blob:
                dont_use.append(True)
            else:
                dont_use.append(False)
        df["dont_use"] = dont_use

        print(f"Before nan filter: {len(df)}")
        df = df[(~df["solvent SMILES"].isna()) & (~df["solute SMILES"].isna())]
        print(f"After nan filter: {len(df)}")
        # top_solvents = filter_solvents_by_threshold(df)
        # df = df[df["solvent SMILES"].isin(top_solvents)]

        if os.environ.get("_SM_ONLY_COSMO_CALCULATED_SMILES"):
            warn("FILTERING OUT ONLY COSMO-CALCULATED SMILES:")
            warn("before: " + str(len(df)))
            from cosmonaut import smiles_to_id, id_to_cosmo_file_path

            solu_ids = df["solute SMILES"].apply(smiles_to_id)
            solv_ids = df["solvent SMILES"].apply(smiles_to_id)
            solu_fles = [id_to_cosmo_file_path(mol_id) for mol_id in solu_ids]
            solv_fles = [id_to_cosmo_file_path(mol_id) for mol_id in solv_ids]
            cosmo_filter = [
                bool(solu_fle) and bool(solv_fle)
                for solu_fle, solv_fle in zip(solu_fles, solv_fles)
            ]

            df = df[cosmo_filter]
            warn("after: " + str(len(df)))

        # Used this for comparing with COSMO-RS results.
        df = filter_relevant_subset(
            df,
            N_thresh=3,
        )
        # smiles_filtered = joblib.load("/tmp/smiles_filtered.pkl")
        # df = df[df["solute SMILES"].isin(smiles_filtered)]

        if os.environ.get("_PR_FAST_DEBUG"):
            print("WARNING: fast debug mode enabled. sampling very small subset")
            df = df.sample(len(df) // 10)

        if os.environ.get("_PR_DOWNSAMPLE_BY"):
            fac = float(os.environ.get("_PR_DOWNSAMPLE_BY"))
            print("WARNING: downsampling by factor", fac)

            df = df.sample(int(len(df) / fac))

        df.to_sql("training_data", con)
        con.close()
    else:
        warn(f"file {out_file} exists. skipping step {__file__}")
