from sklearn.model_selection import GridSearchCV
from solvmate import *
from solvmate.ranksolv import ood
from solvmate.ranksolv.featurizer import (
    CDDDFeaturizer,
    CombinedFeaturizer,
    CountECFPFeaturizer,
    ECFPFeaturizer,
    ECFPSolventOnlyFeaturizer,
    HybridFeaturizer,
    PriorFeaturizer,
    RandFeaturizer,
    XTBFeaturizer,
    CosmoRSFeaturizer,
)
from solvmate.ranksolv.reference_model import AbsoluteRecommender
from solvmate.ranksolv.jack_knife_recommender import JackKnifeRecommender
from solvmate.ranksolv.recommender import Recommender

try:
    from solvmate.pair_rank import pair_rank
except:
    print("could not import pair rank!")
try:
    from solvmate.pair_rank import page_rank
except:
    print("could not import page rank!")

from sklearn import ensemble
import tqdm

from scipy.stats import spearmanr, pearsonr, kendalltau
import tqdm


def apply_solvent_cv(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Implements the split for the following control experiment:
    Each cross fold corresponds to one of the top 5 solvents.
    Solvents that are not part of the top 5 solvents will be
    randomly distributed. Then we check whether we can
    generalize between these cross folds, aka between
    buckets of the top 5 most frequent solvents

    >>> ssa = "solvent SMILES a"
    >>> ssb = "solvent SMILES b"
    >>> solvents = "ABCDEFGHIJKLMN"
    >>> top5 = solvents[:5]
    >>> pairs = pd.DataFrame([{ssa: sa, ssb: sb} for sa in solvents for sb in solvents])
    >>> pairs_cv = apply_solvent_cv(pairs)

    The top5 "cross contamination" has been removed:
    >>> len(pairs), len(pairs_cv)
    (196, 171)

    """
    pairs = pairs.copy()
    top_5 = pairs["solvent SMILES a"].value_counts().iloc[0:5].index.tolist()

    solv_clust = []
    for _, row in pairs.iterrows():
        sa = row["solvent SMILES a"]
        sb = row["solvent SMILES b"]
        if sa in top_5 and sb in top_5:
            # This is "a bridge" between the two solvents,
            # so label it for removal later
            solv_clust.append(-1)
        elif sa in top_5:
            solv_clust.append(top_5.index(sa))
        elif sb in top_5:
            solv_clust.append(top_5.index(sb))
        else:  # both not in top
            solv_clust.append(random.randint(0, len(top_5) - 1))

    pairs["cross_fold"] = solv_clust
    pairs = pairs[pairs["cross_fold"] != -1]
    return pairs


def split_on_solvent_cv(
    pairs: pd.DataFrame, cv_test: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implements the split for the following control experiment:
    Each cross fold corresponds to one of the top 5 solvents.
    Solvents that are not part of the top 5 solvents will be
    randomly distributed. Then we check whether we can
    generalize between these cross folds, aka between
    buckets of the top 5 most frequent solvents

    >>> ssa = "solvent SMILES a"
    >>> ssb = "solvent SMILES b"
    >>> solvents = "ABCDEFGHIJKLMN"
    >>> pairs = pd.DataFrame([{ssa: sa, ssb: sb} for sa in solvents for sb in solvents])
    >>> rslt = split_on_solvent_cv(pairs,1)
    >>> len(rslt[0])
    139
    >>> len(rslt[1])
    57
    >>> rslt = split_on_solvent_cv(pairs,2)
    >>> len(rslt[0])
    157
    >>> len(rslt[1])
    39


    """
    assert 0 <= cv_test <= 5, f"constraint failed: 0 <= {cv_test} <= 5"
    cv_test = cv_test % 5
    pairs = pairs.copy()
    top_N = pairs["solvent SMILES a"].value_counts().index.tolist()
    solv_to_cv = {solv: idx % 5 for idx, solv in enumerate(top_N)}
    pairs["solv_cv_a"] = pairs["solvent SMILES a"].map(solv_to_cv)
    pairs["solv_cv_b"] = pairs["solvent SMILES b"].map(solv_to_cv)
    pairs["solv_cv"] = [
        min(cv_a, cv_b)
        for cv_a, cv_b in zip(pairs["solv_cv_a"].tolist(), pairs["solv_cv_b"].tolist())
    ]

    pairs_test = pairs[pairs["solv_cv"] == cv_test]
    pairs_train = pairs[pairs["solv_cv"] != cv_test]

    solv_test = [s for s, cv in solv_to_cv.items() if cv == cv_test]
    assert len(pairs_test) + len(pairs_train) == len(pairs)

    return pairs_train, pairs_test, solv_test


class RecommenderFactory:
    def __init__(
        self,
        n_estimators_range=None,
        featurizers=None,
        abs_strat_range=None,
        sources=None,
        regs=None,
    ) -> None:
        if n_estimators_range is None:
            n_estimators_range = [100]  # n_estimators for final retrain!

        if featurizers is None:
            featurizers = [
                # Commented out because this would be hell for anyone to
                # actually use because of the long runtimes...
                # It is only included here for reproducibility of the paper:
                #CosmoRSFeaturizer(
                    #phase="train",
                    #pairwise_reduction="concat",
                    #feature_name="cosmors",
                #),
                #CosmoRSFeaturizer(
                    #phase="train",
                    #pairwise_reduction="diff",
                    #feature_name="cosmors",
                #),

                RandFeaturizer(
                    phase="train",
                    pairwise_reduction="concat",
                    feature_name="rand",
                ),
                RandFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="rand",
                ),
                PriorFeaturizer(
                    phase="train",
                    pairwise_reduction="concat",
                    feature_name="prior",
                ),
                PriorFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="prior",
                ),
                ECFPSolventOnlyFeaturizer(
                    phase="train",
                    pairwise_reduction="concat",
                    feature_name="ecfp_solv",
                ),
                ECFPSolventOnlyFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="ecfp_solv",
                ),
                CountECFPFeaturizer(
                    phase="train", pairwise_reduction="concat", feature_name="ecfp_count"
                ),
                CountECFPFeaturizer(
                    phase="train", pairwise_reduction="diff", feature_name="ecfp_count"
                ),
                XTBFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="xtb",
                ),
                XTBFeaturizer(
                    phase="train",
                    pairwise_reduction="concat",
                    feature_name="xtb",
                ),
                ECFPFeaturizer(
                    phase="train", pairwise_reduction="diff", feature_name="ecfp_bit"
                ),
                ECFPFeaturizer(
                    phase="train", pairwise_reduction="concat", feature_name="ecfp_bit"
                ),
                CDDDFeaturizer(feature_name="cddd",phase="train",pairwise_reduction="concat"),
                CDDDFeaturizer(feature_name="cddd",phase="train",pairwise_reduction="diff"),
            ]

        self.featurizers = featurizers

        if abs_strat_range is None:
            abs_strat_range = [
                "absolute",
                "mean",
            ]

        self.abs_strat_range = abs_strat_range

        # The regression models to consider.
        if regs is None:
            regs = [
                ensemble.ExtraTreesRegressor(
                    n_jobs=N_JOBS,
                    n_estimators=n_estimators,
                )
                for n_estimators in n_estimators_range
            ]
            # + [DummyClassifier(strategy="prior", random_state=None, constant=None)] # obsolete
        
        # Wrap a grid search around it
        self.regs = [SimpleGridSearchCV(reg) for reg in regs]
        self.sources = sources

    def train_and_eval_recommenders(
        self,
        perform_cv: bool,
        perform_butina_cv: bool,
        perform_solvent_cv: bool,
        nova_as_ood: bool,
        save_recommender: bool,
        job_name: str,
        only_source: str,
        eval_on_ood: bool,
    ):

        td = get_training_data()
        pairs = get_pairs_data()

        assert not (
            bool(nova_as_ood) and bool(only_source)
        ), "nova_as_ood can only be set without only_source option!"
        if only_source:
            print("only considering source: ", only_source)
            print("train data before:", len(td))
            td = td[td.source == only_source]
            print("train data after:", len(td))

            print("pairs data before:", len(pairs))
            pairs = pairs[pairs.source == only_source]
            print("pairs data after:", len(pairs))

        if nova_as_ood:
            warn("regarding nova as ood dataset!")
            ood_data = td[td.source == "nova"]
            td = td[td.source != "nova"]
        else:
            try:
                ood_data = ood.parse_ood_data_bayer()
            except:
                ood_data = ood.parse_ood_data()

        top_pairs = Recommender.to_top_pairs(pairs, top_pairs_strategy="doubles")

        print("preparing later recommendation pairs...")
        rec_pairs = Recommender.smiles_to_pairs(
            pairs["solute SMILES"].unique().tolist(), top_pairs
        )

        print("preparing later ood pairs...")
        ood_pairs = Recommender.smiles_to_pairs(
            ood_data["solute SMILES"].unique().tolist(), top_pairs
        )

        #if os.environ.get("_SM_ONLY_COSMO_CALCULATED_SMILES"):
        #warn("replacing ood_pairs by normal pairs")
        #ood_data = td
        #ood_pairs = pairs

        if perform_cv:
            add_cross_fold_by_col(pairs, col="solute SMILES",random_seed=123,)

        if perform_butina_cv:
            pairs = pairs[~pairs["solute SMILES"].isna()]
            pairs = pairs[~pairs["solute SMILES"].apply(Chem.MolFromSmiles).isna()]
            add_cv_by_butina_clustering(pairs, col="solute SMILES",random_seed=123,)

        # if perform_solvent_cv:
        # pairs = apply_solvent_cv(pairs)

        if self.sources is not None:
            print(f"only keeping from sources {self.sources}")
            pairs = pairs[pairs["source"].isin(self.sources)]

        for reg in tqdm.tqdm(self.regs):

            for featurizer in self.featurizers:
                stats = []
                for to_abs_strat in self.abs_strat_range:
                    if "DummyClassifier" in str(reg):
                        if to_abs_strat != "absolute":
                            continue  # DummyClassifier sometimes crashes for relative

                    print("-" * 80)
                    print("\n".join(map(str, [reg, featurizer, to_abs_strat])))
                    print("-" * 80)

                    is_absolute = False
                    if to_abs_strat == "absolute":
                        is_absolute = True
                        rc = AbsoluteRecommender(
                            reg=reg,
                            featurizer=featurizer,
                        )
                    else:
                        rc = Recommender(
                            reg=reg,
                            featurizer=featurizer,
                            to_absolute_strategy=to_abs_strat,
                        )

                    if perform_cv:
                        for cv_test in pairs["cross_fold"].unique():

                            if perform_solvent_cv:
                                (
                                    pairs_train,
                                    pairs_test,
                                    solv_test,
                                ) = split_on_solvent_cv(pairs, cv_test)
                            else:
                                pairs_train, pairs_test = (
                                    pairs[pairs["cross_fold"] != cv_test],
                                    pairs[pairs["cross_fold"] == cv_test],
                                )

                            if perform_solvent_cv:
                                td_test = td[td["solvent SMILES"].isin(solv_test)]
                                td_train = td[~td["solvent SMILES"].isin(solv_test)]
                            else:
                                td_test = td[
                                    td["solute SMILES"].isin(
                                        pairs_test["solute SMILES"]
                                    )
                                ]
                                td_train = td[
                                    ~td["solute SMILES"].isin(
                                        pairs_test["solute SMILES"]
                                    )
                                ]
                            assert len(td_test) + len(td_train) == len(td)
                            assert len(td)
                            assert len(td_test)
                            assert len(td_train)

                            if is_absolute:
                                # The absolute model is trained on singletons
                                rc.fit(td_train)
                                if perform_solvent_cv:
                                    for solv in solv_test:
                                        if solv not in rc.all_solvents:
                                            rc.all_solvents.append(solv)
                            else:
                                rc = Recommender(
                                    reg=reg,
                                    featurizer=featurizer,
                                    to_absolute_strategy=to_abs_strat,
                                )
                                # The relative model needs to be trained on pairs
                                print("> fit")
                                rc.fit(pairs_train)
                                print("< fit")

                            rc.featurizer.phase = "predict"
                            # TODO: refactor into method call
                            rc.top_pairs_ = top_pairs
                            print("> rec")
                            smis_test = td_test["solute SMILES"].unique().tolist()
                            preds_all = rc.recommend(
                                smiles=smis_test,
                                pairs=rec_pairs,
                            )
                            print("< rec")
                            preds_df = pd.DataFrame(
                                {
                                    "solute SMILES": solute_smiles,
                                    "solvent SMILES": solvent_smiles,
                                    "pred_place": pred_place,
                                }
                                for (preds, solute_smiles) in zip(
                                    preds_all,
                                    smis_test,
                                )
                                for pred_place, solvent_smiles in enumerate(preds)
                            )
                            data_join = td_test.merge(
                                preds_df,
                                how="inner",
                                on=[
                                    "solvent SMILES",
                                    "solute SMILES",
                                ],
                            )

                            spears, kts = [], []
                            for solute_smiles in data_join["solute SMILES"].unique():
                                g = data_join[
                                    data_join["solute SMILES"] == solute_smiles
                                ]
                                spears.append(
                                    spearmanr(
                                        g["pred_place"],
                                        reals_to_placement(g["conc"]),
                                    )[0],
                                )
                                kts.append(
                                    kendalltau(
                                        g["pred_place"],
                                        reals_to_placement(g["conc"]),
                                    )[0],
                                )

                            if os.environ.get("_SM_DUMP_PREDS"):
                                dj = data_join.copy()
                                dj["reg"] = str(reg)
                                dj["featurizer"] = str(featurizer)
                                dj["feature_name"] = featurizer.feature_name
                                dj["pairwise_reduction"] = featurizer.pairwise_reduction
                                dj["to_abs_strat"] = to_abs_strat
                                dj["created_at"] = datetime.datetime.now()
                                dj_fle = DATA_DIR / "preds_dump.pkl"
                                if dj_fle.exists():
                                    dj = pd.concat([joblib.load(dj_fle), dj])

                                joblib.dump(
                                    value=dj,
                                    filename=dj_fle,
                                )

                            spears = pd.Series(np.array(spears))
                            kts = pd.Series(np.array(kts))

                            mean_spear_r_test = spears.mean()
                            std_spear_r_test = spears.std(ddof=1)
                            mean_kendalltau_test = kts.mean()
                            std_kendalltau_test = kts.std(ddof=1)

                            if eval_on_ood:
                                if not nova_as_ood and featurizer.__class__.__name__ == CosmoRSFeaturizer.__name__:
                                    info("cosmors featurizer+Bayer OOD is not evaluated. ")
                                    ood_stats = ood.StatsOOD(
                                        r_spearman=np.nan, r_kendalltau=np.nan
                                    )
                                else:
                                    ood_stats = ood.eval_on_ood_data(
                                        data=ood_data, pairs=ood_pairs, rec=rc
                                    )
                            else:
                                #info(
                                #"skipping ood eval. To perform it, set _SM_DO_OOD_EVAL environment flag"
                                #)
                                ood_stats = ood.StatsOOD(
                                    r_spearman=np.nan, r_kendalltau=np.nan
                                )

                            stats_dct = {
                                "reg": str(reg.model_) if is_cv_instance(reg,) else str(reg),
                                "featurizer": str(featurizer),
                                "feature_name": featurizer.feature_name,
                                "pairwise_reduction": featurizer.pairwise_reduction,
                                "to_abs_strat": to_abs_strat,
                                "mean_spear_r_test": mean_spear_r_test,
                                "std_spear_r_test": std_spear_r_test,
                                "mean_kendalltau_r_test": mean_kendalltau_test,
                                "std_kendalltau_r_test": std_kendalltau_test,
                                "ood_mean_spear_r": ood_stats.r_spearman,
                                "ood_mean_kendall_tau_r": ood_stats.r_kendalltau,
                                "created_at": datetime.datetime.now(),
                            }
                            print(stats_dct)
                            stats.append(stats_dct)

                            if save_recommender:
                                rc.save(DATA_DIR / f"recommender_{cv_test}.pkl")

                stats = pd.DataFrame(stats)
                stats["job_name"] = job_name
                io_store_stats(stats)

        if False:
            rc_fles = list(DATA_DIR.glob("recommender_*.pkl"))
            jkr = JackKnifeRecommender(rc_fles=rc_fles)

            ood.eval_err_estimate_on_ood_data(
                data=ood_data,
                pairs=ood_pairs,
                jkr=jkr,
            )
