from dataclasses import dataclass
import json
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler


class _KNNRecommenderSolub:
    def __init__(
        self,
    ) -> None:
        self.dist_type = "l2"
        self.lru_cache = []

    def fit(self) -> None:
        df = get_training_data()
        xtb_features = get_xtb_features_data(DATA_DIR / "xtb_features.db")

        df = self._featurize_df(df=df, xtb_features=xtb_features)
        self.df_ = df

    def _featurize_df(
        self, df: pd.DataFrame, xtb_features: pd.DataFrame
    ) -> pd.DataFrame:
        xtb_features_piv = xtb_features.drop_duplicates(["smiles", "solvent"]).pivot(
            index="smiles",
            columns="solvent",
        )
        # Dissolves the multiindex and replaces it with a normal single-index instead.
        # For example, ("a", "b") is turned into '("a", "b")'. That makes it much
        # to operate with normal pandas functions on the dataframe while still retaining
        # all the options to operate in a multiindex way (though then slower!).
        xtb_features_piv.columns = [str(col) for col in xtb_features_piv.columns.values]

        df = df.merge(
            xtb_features_piv,
            left_on="solute SMILES",
            right_on="smiles",
        )

        features_c = sorted(
            [
                col
                for col in xtb_features_piv.columns
                if "(" in col and not "__" in col and "elec" not in col.lower()
            ]
        )
        df["X"] = list(df[features_c].values)
        return df

    def run_df(self, smiles: str) -> pd.DataFrame:
        for k, v in self.lru_cache:
            if k == smiles:
                log("cache hit!")
                return v
        log("cache miss!")

        if not hasattr(self, "df_"):
            self.fit()

        assert hasattr(self, "df_")
        df = self.df_

        xtb_features = get_xtb_features_data(
            db_file=DATA_DIR / "xtb_features_predict.db"
        )
        xtb_features = xtb_features[xtb_features["smiles"] == smiles]
        dfp = pd.DataFrame({"solute SMILES": [smiles]})

        dfp = self._featurize_df(dfp, xtb_features=xtb_features)

        vec = dfp["X"].iloc[0]  # same for all

        vec_diff = np.vstack(df["X"]) - vec
        # TODO: use commented out line above instead of this:
        # vec = vec[0:197]  # hack for now to make it work ...
        # vec_diff = np.random.uniform(-100, 100, (len(df), 197))

        if self.dist_type == "l2":
            df["dist"] = np.linalg.norm(vec_diff, axis=1)
        else:
            assert False, "unknown dist type " + str(self.dist_type)

        df["dist"] /= len(
            vec
        )  # empirically found large distance values just due to high dimensionality

        df["sim"] = 1 / (1 + df["dist"])

        df = df.sort_values("dist")
        self.lru_cache.append((smiles, df))

        if len(self.lru_cache) > 3:
            self.lru_cache = self.lru_cache[-3:]

        return df


@dataclass
class KNNResult:
    similarity: float
    distance: float
    smiles_solute: str
    smiles_solvents: list[str]


_KNN_REC = None
_CURRENT_REC_TYPE = None


class KNNSolubRecommendHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        global _KNN_REC, _CURRENT_REC_TYPE

        try:
            req = json.loads(self.request.body)

            _KNN_REC = _KNNRecommenderSolub()

            rec: _KNNRecommenderSolub = _KNN_REC
            req_smi = req["smiles"]
            req_smi = Chem.CanonSmiles(req_smi)

            start = req["start"]
            end = req["end"]

            dfr = rec.run_df(req_smi)
            dfr["smiles"] = dfr["solute SMILES"]

            if "solvent_label" not in dfr.columns:
                dfr["solvent_label"] = dfr["solvent SMILES"].apply(str)

            dfr = dfr.drop_duplicates(["smiles", "solvent_label"])

            matches = []
            for smi_solute in list(dfr["smiles"].unique())[start:end]:
                g = dfr[dfr["smiles"] == smi_solute]
                g = g.sort_values("conc", ascending=False)
                matches.append(
                    {
                        "similarity": "{0:.3f}".format(g["sim"].iloc[0]),
                        "distance": g["dist"].iloc[0],
                        "smiles_solute": smi_solute,
                        "source": [row["source"] for _, row in g.iterrows()],
                        "conc": [row["conc"] for _, row in g.iterrows()],
                        "smiles_solvents": g["solvent_label"]
                        # .apply(lambda sl: "+".join(sl))
                        .tolist(),
                    }
                )

            resp = json.dumps(
                {
                    "depict_status": "success",
                    "matches": matches,
                }
            )
            self.write(resp)

        except:
            resp = json.dumps({"depict_status": "failure", "depict_data": ""})
            self.write(resp)

            if DEVEL:
                raise
