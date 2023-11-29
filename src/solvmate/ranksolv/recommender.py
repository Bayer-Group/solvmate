from sklearn.inspection import permutation_importance
from solvmate import *
from solvmate.ranksolv.featurizer import AbstractFeaturizer, PriorFeaturizer

try:
    from solvmate.pair_rank import pair_rank
except:
    print("could not import pair rank!")
try:
    from solvmate.pair_rank import page_rank
except:
    print("could not import page rank!")

import joblib


l2 = np.linalg.norm


def norm_mat(
    mat,
    eta=0.00000001,
):
    """
    Normalizes the given matrix mat along its rows,
    e.g.
    >>> mat = np.array([[10,10,10],[1,1,1]])
    >>> norm_mat(mat)
    array([[0.57735027, 0.57735027, 0.57735027],
           [0.57735027, 0.57735027, 0.57735027]])

    """
    l2_rows = np.linalg.norm(mat, axis=1)
    return (mat.T / (l2_rows + eta)).T


class Recommender:
    def __init__(self, reg, featurizer: AbstractFeaturizer) -> None:
        self.featurizer = featurizer
        self.reg = reg
        self.to_absolute_strategy = "mean"  # TODO: make configurable

    @staticmethod
    def load(filename: str):
        return joblib.load(filename)

    def save(self, filename: str):
        joblib.dump(value=self, filename=filename)

    def split_cv(
        self,
        pairs: pd.DataFrame,
        cv_test: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert cv_test in pairs["cross_fold"]
        pairs_train = pairs[pairs["cross_fold"] != cv_test]
        pairs_test = pairs[pairs["cross_fold"] == cv_test]
        return pairs_train, pairs_test

    def fit(
        self,
        pairs: pd.DataFrame,
    ) -> None:
        self._featurize_pairs(pairs)
        X = list(pairs["X"].values)
        y = pairs["conc diff"]
        self.reg.fit(X, y)

    def feature_importance_analysis_sklearn(self, pairs: pd.DataFrame):
        import seaborn as sns

        feat_imps = self.reg.feature_importances_
        feat_col_names = self.featurizer.column_names()
        assert len(feat_imps) == len(feat_col_names)

        df_feat = pd.DataFrame(
            {
                "importance": feat_imps,
                "name": feat_col_names,
            }
        )

        return df_feat

    def feature_importance_permutation_analysis(self, pairs: pd.DataFrame):
        assert len(pairs)
        import seaborn as sns

        X = list(pairs["X"].values)
        y = pairs["conc diff"]
        result = permutation_importance(self.reg, X, y, n_repeats=10, random_state=0)
        feat_imps = result.importances_mean
        feat_col_names = self.featurizer.column_names()
        assert len(feat_imps) == len(feat_col_names)

        df_feat = pd.DataFrame(
            {
                "importance": feat_imps,
                "name": feat_col_names,
            }
        )

        return df_feat

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        all_preds = []

        for chunk in split_in_chunks(
            pairs,
            chunk_size=1024,
        ):
            X = list(chunk["X"].values)
            this_preds = self.reg.predict(X)
            all_preds.append(this_preds)

        all_preds = np.hstack(all_preds)
        return all_preds

    @staticmethod
    def to_top_pairs(
        pairs,
        top_pairs_strategy: str,
    ):
        # I can think of three strategies how to extract the
        # top pairs out of the training data:
        # Right now, I believe the doubles strategy to be the
        # best, because it considers only the most common
        # solvent PAIRS, aka those solvent pairs that we
        # actually have enough data about in our training set.
        #
        # The singleton strategy will just consider the
        # univariate statistics and might e.g. take water
        # and hexane as the top solvents even if their
        # pair (hexane,water) itself might actually be rare.
        #
        # The all strategy will just consider all pairs.
        #
        # It has to be noted that I believe both singleton and
        # all to work best with the mean/median recommenders
        # as the graphs are already complete by construction.
        #
        # For the doubles strategy, I believe or at least hope
        # to see an improvement as the graph is no longer
        # complete. Hence, it is more difficult to reconstruct
        # an absolute ordering from the partial information.
        #
        # That should be exactly the scenario to employ the
        # ranked pairs algorithm.
        assert top_pairs_strategy in ["singleton", "doubles", "all"]

        if top_pairs_strategy == "all":

            top_pairs = list(
                {
                    (row["solvent SMILES a"], row["solvent SMILES b"])
                    for _, row in pairs.iterrows()
                }
            )

        elif top_pairs_strategy == "singleton":
            top_pairs = list(
                {
                    (solv_smi_a, solv_smi_b)
                    for solv_smi_a in pairs["solvent SMILES a"]
                    .value_counts()
                    .index.tolist()[0:10]
                    for solv_smi_b in pairs["solvent SMILES b"]
                    .value_counts()
                    .index.tolist()[0:10]
                    if solv_smi_a != solv_smi_b
                }
            )  # only taking the top 20 solvents for now.
        elif top_pairs_strategy == "doubles":
            top_pairs = (
                pairs[["solvent SMILES a", "solvent SMILES b"]]
                .value_counts()
                .index.tolist()[0:3600]  # 40x40=1600, 1000 worked very well for this
                # TODO              ^____ This should become a hyperparameter
            )
        return top_pairs

    @staticmethod
    def smiles_to_pairs(
        smiles: list[str],
        top_pairs: list[tuple[str, str]],
    ):
        pairs = pd.DataFrame(
            [
                {
                    "smiles": smi,
                    "solute SMILES": smi,
                    "solvent SMILES a": solvent_smiles_a,
                    "solvent SMILES b": solvent_smiles_b,
                }
                for smi in smiles
                for solvent_smiles_a, solvent_smiles_b in top_pairs
                if solvent_smiles_a != solvent_smiles_b
            ]
        )

        return pairs

    def _featurize_pairs(self, pairs: pd.DataFrame):

        if isinstance(self.featurizer, PriorFeaturizer):
            # The prior featurizer intends to only encode the solvent side:
            pairs["X"] = list(
                np.hstack(
                    [
                        self.featurizer.run_single(
                            pairs["solvent SMILES a"].tolist()
                        ).reshape(-1, 1),
                        self.featurizer.run_single(
                            pairs["solvent SMILES b"].tolist()
                        ).reshape(-1, 1),
                    ]
                )
            )
        else:
            pairs["X"] = list(
                self.featurizer.run_pairs(
                    compounds=pairs["solute SMILES"].tolist(),
                    solvents_a=pairs["solvent SMILES a"].tolist(),
                    solvents_b=pairs["solvent SMILES b"].tolist(),
                )
            )

    def recommend_smiles(self, smiles: list[str]):
        assert hasattr(
            self, "top_pairs_"
        ), "need to set top_pairs_ before calling recommend_smiles"
        rec_pairs = Recommender.smiles_to_pairs(smiles, top_pairs=self.top_pairs_)
        self._featurize_pairs(rec_pairs)
        return self.recommend(smiles=smiles, pairs=rec_pairs)

    def recommend(
        self,
        smiles: list[str],
        pairs: pd.DataFrame,
    ) -> list[list[str]]:

        self._featurize_pairs(pairs)

        print("running actual predictions on pairs ...")
        pairs["preds"] = self.predict(pairs)
        print(" ... done running predictions on pairs")

        N_smiles = len(smiles)
        log_every_N = max(100, N_smiles // 10)
        print("running recommendations ...")
        all_recommends = []
        for idx, smi in enumerate(smiles):
            if idx % log_every_N == 0:
                print(f"{idx} / {len(smiles)}")
            pairs_smi = pairs[pairs["solute SMILES"] == smi]

            to_absolute_strategy = self.to_absolute_strategy
            if to_absolute_strategy in ["pair_rank", "page_rank"]:
                g_in = pair_rank.DGraph()
                for i, j, d in zip(
                    pairs_smi["solvent SMILES a"],
                    pairs_smi["solvent SMILES b"],
                    pairs_smi["preds"],
                ):
                    if d >= 0:
                        g_in.add_edge(j, i, d)

                if to_absolute_strategy == "pair_rank":
                    pr = pair_rank.PairRank()
                    g_out = pr.run(g_in)
                    assert not g_out.contains_cycle()

                    recommend = []
                    g_cur = g_out
                    while len(g_cur.nodes()):
                        current_best = g_cur.find_source()[0]
                        recommend.append(current_best)
                        g_cur = g_cur.with_nodes_removed([current_best])

                    all_recommends.append(recommend)
                elif to_absolute_strategy == "page_rank":
                    proba, _ = page_rank.node_rank(
                        g=g_in,
                        d=0.85,
                        eps=0.1,
                        max_steps=300,
                        min_steps=100,
                    )
                    nodes = g_in.nodes()
                    recommend = sorted(
                        [(n, p) for n, p in zip(nodes, proba)], key=lambda np: -np[1]
                    )
                    recommend = [n for n, p in recommend]
                    all_recommends.append(recommend)
                else:
                    assert False
            elif to_absolute_strategy == "mean":

                # assert len(pairs_smi) # <--- TODO: comment this assert in, make sure we have coverage?

                smis = []
                ds = []

                for i in pairs_smi["solvent SMILES a"].unique():
                    g = pairs_smi[pairs_smi["solvent SMILES a"] == i]
                    smis.append(i)
                    ds.append(g["preds"].mean())

                recommend = sorted(
                    [(smi, d) for smi, d in zip(smis, ds)], key=lambda itm: -itm[1]
                )
                recommend = [smi for smi, _ in recommend]

                # assert len(recommend) # <--- TODO: comment this assert in, make sure we have coverage?
                all_recommends.append(recommend)
            else:
                assert False

        print("... done running recommendations using pair rank")

        return all_recommends
