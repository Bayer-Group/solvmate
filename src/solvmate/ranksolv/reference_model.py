from solvmate import *
from solvmate import xtb_solv
from sklearn import ensemble
from sklearn import impute

from solvmate.ranksolv.featurizer import AbstractFeaturizer, PriorFeaturizer


class AbsoluteRecommender:
    def __init__(self, reg=None, featurizer: AbstractFeaturizer = None) -> None:
        self.reg = reg
        self.imputer = impute.SimpleImputer()
        self.featurizer = featurizer

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
        X = self._featurize(data, fit=False)
        data["preds"] = self.reg.predict(X)

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

        X = self._featurize(data, fit=True)
        self.reg.fit(X, data["conc"])

    def _featurize(self, data: pd.DataFrame, fit: bool) -> np.array:
        if fit:
            self.featurizer.phase = "train"
        else:
            self.featurizer.phase = "predict"

        if "CosmoRSFeaturizer" in str(self.featurizer.__class__.__name__):
            # CosmoRS can only run on solute+solvent!
            X = self.featurizer.run_solute_solvent(compounds=data["solute SMILES"].tolist(),solvents=data["solvent SMILES"].tolist())

        else:
            smiles = (
                data["solute SMILES"].unique().tolist()
                + data["solvent SMILES"].unique().tolist()
            )

            smi_to_x = {
                smi: x for smi, x in zip(smiles, self.featurizer.run_single(smiles))
            }

            X_solu = np.vstack(data["solute SMILES"].map(smi_to_x))
            if len(X_solu.shape) == 1:
                X_solu = X_solu.reshape(-1, 1)
            X_solv = np.vstack(data["solvent SMILES"].map(smi_to_x))
            if len(X_solv.shape) == 1:
                X_solv = X_solv.reshape(-1, 1)

            if isinstance(self.featurizer, PriorFeaturizer):
                # The prior featurizer intends to only encode the solvent side:
                X = X_solv
            else:
                X = np.hstack([X_solu, X_solv])

        return X
