from solvmate import *
from solvmate import xtb_solv
from sklearn import ensemble
from sklearn import impute


class AbsoluteRecommender:
    def __init__(self, reg=None) -> None:
        if reg is None:
            reg = ensemble.ExtraTreesRegressor(
                n_jobs=8,
                n_estimators=1000,
            )
        self.reg = reg
        self.imputer = impute.SimpleImputer()

    def recommend(
        self,
        smiles: list[str],
        pairs,  # Ignored as not needed
    ) -> list[list[str]]:
        data = pd.DataFrame(
            [
                {
                    "smiles": smi,
                    "solute SMILES": smi,
                    "solvent SMILES": solvent_smiles,
                }
                for smi in smiles
                for solvent_smiles in self.all_solvents
            ]
        )
        # y = data["conc"]
        data = self._featurize(data, fit=False)
        data["preds"] = self.reg.predict(np.vstack(data["X"]))

        recs = []
        for smi in smiles:
            g = data[data["solute SMILES"] == smi]
            recs.append(
                # ascending order because we use real_to_placement,
                # aka the highest concentration comes last!
                g.sort_values("preds", ascending=True)["solvent SMILES"].tolist()
            )

        return recs

    def fit(
        self,
        data: pd.DataFrame,
    ) -> list[list[str]]:
        self.all_solvents = sorted(data["solvent SMILES"].unique().tolist())

        data = self._featurize(
            data,
            fit=True,
        )
        self.reg.fit(np.vstack(data["X"]), data["conc"])

    def _featurize(self, data: pd.DataFrame, fit: bool) -> np.ndarray:
        db_file = DATA_DIR / "xtb_features_predict.db"
        xs = xtb_solv.XTBSolv(db_file=db_file)
        xs.setup_db()

        all_smiles = list(data["solute SMILES"].unique()) + list(
            data["solvent SMILES"].unique()
        )

        print("runnning xtb calculations ...")
        all_smiles = list(set(all_smiles))
        xs.run_xtb_calculations(smiles=all_smiles)
        print("... done runnning xtb calculations")

        xtb_features = get_xtb_features_data(db_file=db_file)
        xtb_features_piv = xtb_features.drop_duplicates(["smiles", "solvent"]).pivot(
            index="smiles",
            columns="solvent",
        )

        xtb_features_piv_solu = xtb_features_piv.copy()
        xtb_features_piv_solv = xtb_features_piv.copy()
        xtb_features_piv_solu.columns = [
            str(col) + "__solu" for col in xtb_features_piv.columns.values
        ]
        xtb_features_piv_solv.columns = [
            str(col) + "__solv" for col in xtb_features_piv.columns.values
        ]
        if fit:
            self.feature_cols = [
                col + suf
                for col in map(str, xtb_features_piv.columns)
                if "smiles" not in col.lower()
                for suf in ["__solu", "__solv"]
            ]
        data = data.merge(
            xtb_features_piv_solu,
            left_on="solute SMILES",
            right_index=True,
        )
        data = data.merge(
            xtb_features_piv_solv,
            left_on="solvent SMILES",
            right_index=True,
        )

        X = data[self.feature_cols].values

        if fit:
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)

        data["X"] = list(X)
        return data
