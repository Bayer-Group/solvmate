from cosmonaut import *
from cosmonaut.cosmors_calc import make_cosmors_features

from solvmate.ccryst.datasets import add_split_by_col  # TODO: copy implementation to here!

from typing import Optional

import random

from sklearn import ensemble
import joblib






class CosmoAbsoluteSolubilityModel:
    def __init__(self, clf=None) -> None:
        if clf is None:
            clf = ensemble.RandomForestRegressor(
                n_jobs=8,
            )
        self.clf = clf

    def fit(self, df: pd.DataFrame):
        df = df[(~df["concentration (M)"].isna()) & (df["concentration (M)"] > 0)]
        df["log_conc"] = df["concentration (M)"].apply(np.log10)

        df["mol_solute"] = df["solute SMILES"].apply(Chem.MolFromSmiles)
        df["mol_solvent"] = df["solvent SMILES"].apply(Chem.MolFromSmiles)

        df = df[~(df["mol_solvent"].isna()) & ~(df["mol_solute"].isna())]

        df["hash_solute"] = df["solute SMILES"].apply(smiles_to_id)
        df["hash_solvent"] = df["solvent SMILES"].apply(smiles_to_id)
        df["cosmo_file_solute"] = df["hash_solute"].apply(id_to_cosmo_file_path)
        df["cosmo_file_solvent"] = df["hash_solvent"].apply(id_to_cosmo_file_path)

        print("removing rows without COSMO result for the solute.")
        print("before:", len(df))
        df = df[~df["cosmo_file_solute"].isna()]
        print("after:", len(df))


        print("removing rows without COSMO result for the solvent.")
        print("before:", len(df))
        df = df[~df["cosmo_file_solvent"].isna()]
        print("after:", len(df))

        joblib.dump(
            value=df["solute SMILES"].tolist(), filename="/tmp/smiles_filtered.pkl"
        )

        df["cosmors_features"] = [
            make_cosmors_features(
                fle_solvent=fle_solvent,
                fle_solute=fle_solute,
                refst="pure_component",
                reduction="sum+concat",
            )
            for fle_solvent, fle_solute in zip(
                df["cosmo_file_solvent"], df["cosmo_file_solute"]
            )
        ]

        joblib.dump(
            value=df,
            filename="/tmp/cosmo_ds.pkl",
        )
        print(
            df["cosmors_features"].isna().sum(),
            "/",
            len(df),
            "cosmors calculations failed",
        )
        df = df[~df["cosmors_features"].isna()]

        add_split_by_col(df, col="solute SMILES")

        df_train, df_test = df[df["split"] == "train"], df[df["split"] == "test"]

        def df_to_XY(dat):
            X = np.vstack(dat["cosmors_features"])
            y = dat["log_conc"]
            return X, y

        X_train, y_train = df_to_XY(df_train)
        X_test, y_test = df_to_XY(df_test)

        clf = self.clf
        clf.fit(X_train, y_train)
        print("score_train,", clf.score(X_train, y_train))
        print("score_test,", clf.score(X_test, y_test))
        print()


if __name__ == "__main__":
    model = CosmoAbsoluteSolubilityModel()
    df = pd.read_excel(CM_DATA_DIR / "20150430SolubilitiesSum.xlsx")
    model.fit(df)
