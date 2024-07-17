from solvmate import *

"""
A script that builds the pairwise data.
"""


def _sqrt_sample(
    df: pd.DataFrame, N_min: int, hard_downsample=False, hard_downsample_by=None
):
    """
    Applies square root sampling to the given dataframe.
    This is needed to downsample cases where for a single solute
    many solvent measurements have been performed.
    In such cases, aka for a given solute with more than N_min entries,
    downsampling to N_min + sqrt(N - N_min) will be performed, thus
    directly counteracting the quadratic growth for such cases.
    """
    df_out = []
    col_solu = "solute SMILES"

    for solu_smi in df[col_solu].unique():
        g = df[df[col_solu] == solu_smi]
        if len(g) > N_min:
            g = g.sample(int(N_min + (len(g) - N_min) ** 0.5),random_state=123,)

        if hard_downsample:
            assert hard_downsample_by is not None
            g = g.sample(frac=hard_downsample_by,random_state=123,)

        df_out.append(g)

    return pd.concat(df_out)


def run():
    # Given the training_data, aka the labels of the form
    #    smiles, solvent, y=conc
    # as well as the xtb features of the form
    #    smiles, solvent-calc, feature_1, feature_2
    # this script returns the corresponding pairs.
    out_file = DATA_DIR / "pairs.db"
    if not out_file.exists():
        con = sqlite3.connect(out_file)
        training_data = get_training_data()

        # assert {"nova", "open_notebook"} <= set(training_data.source.unique())
        pairs = []
        N_solutes = training_data["solute SMILES"].nunique()
        for smi in training_data["solute SMILES"].unique():
            g = training_data[training_data["solute SMILES"] == smi]

            for ia, pa in g.iterrows():
                for ib, pb in g.iterrows():
                    if (
                        ia == ib
                        or pa["source"] != pa["source"]
                        or abs(pa["conc"] - pb["conc"]) < MIN_LOG_S_DIFF_THRESHOLD
                    ):
                        continue
                    else:
                        pairs.append(
                            {
                                "solute SMILES": pa["solute SMILES"],
                                "solvent SMILES a": pa["solvent SMILES"],
                                "solvent SMILES b": pb["solvent SMILES"],
                                "conc a": pa["conc"],
                                "conc b": pb["conc"],
                                "source": pa["source"],
                                "cross_fold": pa["cross_fold"],
                            }
                        )
                        print("#pairs",len(pairs), ia,"/", N_solutes)
        pairs = pd.DataFrame(pairs)
        pairs["conc diff"] = pairs["conc b"] - pairs["conc a"]

        print("filtering the top solvent pairings ...")
        print("before", len(pairs))
        top_N = 60
        top_N = pairs["solvent SMILES a"].value_counts().iloc[0:top_N].index.tolist()
        pairs = pairs[pairs["solvent SMILES a"].isin(top_N)]
        pairs = pairs[pairs["solvent SMILES b"].isin(top_N)]
        print("after", len(pairs))

        hard_downsample = False
        hard_downsample_by = None
        if os.environ.get("_SM_HARD_DOWN_SAMPLE_BY"):
            hard_downsample_by = float(os.environ.get("_SM_HARD_DOWN_SAMPLE_BY"))
            print("applying hard downsampling factor:", hard_downsample_by)
            assert 0 < hard_downsample_by < 1.0

        print("applying sqrt sampling ...")
        print("before", len(pairs))
        pairs = _sqrt_sample(
            pairs,
            N_min=50,
            hard_downsample=hard_downsample,
            hard_downsample_by=hard_downsample_by,
        )  # 50 = 2 * 5^2
        print("after", len(pairs))

        # assert {"nova", "open_notebook"} <= set(pairs.source.unique())
        pairs.to_sql(
            "pairs",
            con,
            index=False,
        )
        con.close()
    else:
        warn(f"file {out_file} exists. skipping step {__file__}")


if __name__ == "__main__":
    run()
