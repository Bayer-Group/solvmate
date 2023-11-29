from solvmate import *
from solvmate.ranksolv import ood
from solvmate.ranksolv.featurizer import XTBFeaturizer, CosmoRSFeaturizer
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


class RecommenderFactory:
    def __init__(
        self,
        n_estimators_range=None,
        featurizations_range=None,
        abs_strat_range=None,
        sources=None,
    ) -> None:
        if n_estimators_range is None:
            n_estimators_range = [
                100,  # TODO: change back n_estimators to be 1000!
            ]

        self.featurizations_range = featurizations_range

        # The regression models to consider.
        regs = [
            ensemble.ExtraTreesRegressor(
                n_jobs=N_JOBS,
                n_estimators=n_estimators,
            )
            for n_estimators in n_estimators_range
        ]
        # + [DummyClassifier(strategy="prior", random_state=None, constant=None)] # obsolete
        self.regs = regs
        self.sources = sources

    def train_and_eval_recommenders(
        self,
        perform_cv: bool,
        save_recommender: bool,
    ):

        td = get_training_data()
        pairs = get_pairs_data()

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

        if os.environ.get("_SM_ONLY_COSMO_CALCULATED_SMILES"):
            warn("replacing ood_pairs by normal pairs")
            ood_data = td
            ood_pairs = pairs

        if perform_cv:
            add_cross_fold_by_col(pairs, col="solute SMILES")

        if self.sources is not None:
            print(f"only keeping from sources {self.sources}")
            pairs = pairs[pairs["source"].isin(self.sources)]

        for reg in tqdm.tqdm(self.regs):

            for featurizer in [
                # CosmoRSFeaturizer(
                #    phase="train",
                #    pairwise_reduction="concat",
                #    feature_name="cosmors",
                # ),
                XTBFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="xtb",
                ),
            ]:
                stats = []
                for to_abs_strat in ["mean"]:
                    if "DummyClassifier" in str(reg):
                        if to_abs_strat != "absolute":
                            continue  # DummyClassifier sometimes crashes for relative

                    print("-" * 80)
                    print("\n".join(map(str, [reg, featurizer, to_abs_strat])))
                    print("-" * 80)

                    is_absolute = False
                    if to_abs_strat == "absolute":
                        is_absolute = True

                        rc = AbsoluteRecommender(reg=reg)
                    else:
                        rc = Recommender(
                            reg=reg,
                            featurizer=featurizer,
                        )

                    if perform_cv:
                        for cv_test in pairs["cross_fold"].unique():
                            pairs_train, pairs_test = (
                                pairs[pairs["cross_fold"] != cv_test],
                                pairs[pairs["cross_fold"] == cv_test],
                            )

                            td_test = td[
                                td["solute SMILES"].isin(pairs_test["solute SMILES"])
                            ]
                            td_train = td[
                                ~td["solute SMILES"].isin(pairs_test["solute SMILES"])
                            ]
                            assert len(td_test) + len(td_train) == len(td)
                            assert len(td)
                            assert len(td_test)
                            assert len(td_train)

                            if is_absolute:
                                # The absolute model is trained on singletons
                                rc.fit(td_train)
                            else:
                                rc = Recommender(
                                    reg=reg,
                                    featurizer=featurizer,
                                )
                                # The relative model needs to be trained on pairs
                                print("> fit")
                                rc.fit(pairs_train)
                                print("< fit")

                            rc.featurizer.phase = "predict"
                            # TODO: refactor into method call
                            rc.top_pairs_ = top_pairs
                            print("> rec")
                            preds_all = rc.recommend(
                                smiles=td_test["solute SMILES"].tolist(),
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
                                    td_test["solute SMILES"],
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
                            spears = pd.Series(np.array(spears))
                            kts = pd.Series(np.array(kts))

                            mean_spear_r_test = spears.mean()
                            std_spear_r_test = spears.std(ddof=1)
                            mean_kendalltau_test = kts.mean()
                            std_kendalltau_test = kts.std(ddof=1)

                            if os.environ.get("_SM_DO_OOD_EVAL"):
                                ood_stats = ood.eval_on_ood_data(
                                    data=ood_data, pairs=ood_pairs, rec=rc
                                )
                            else:
                                info(
                                    "skipping ood eval. To perform it, set _SM_DO_OOD_EVAL environment flag"
                                )
                                ood_stats = ood.StatsOOD(
                                    r_spearman=np.nan, r_kendalltau=np.nan
                                )

                            stats_dct = {
                                "reg": str(reg),
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

                io_store_stats(pd.DataFrame(stats))

        if False:
            rc_fles = list(DATA_DIR.glob("recommender_*.pkl"))
            jkr = JackKnifeRecommender(rc_fles=rc_fles)

            ood.eval_err_estimate_on_ood_data(
                data=ood_data,
                pairs=ood_pairs,
                jkr=jkr,
            )
