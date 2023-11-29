import argparse
from solvmate import *

from solvmate.ranksolv.recommender_factory import RecommenderFactory

"""
Script that trains the recommender.

if --hyperp is specified, a hyerparameter search is performed.
if --finalretrain is specified, only a single model 
retraining is performed.

Exactly one of the two flags must be specified.

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperp",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--finalretrain",
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    assert (
        args.hyperp != args.finalretrain
    ), "specify either hyperp or finalretrain (exactly one)"

    if args.hyperp:
        info("hyperparameter search selected")
        rcf = RecommenderFactory()
        rcf.train_and_eval_recommenders(perform_cv=True, save_recommender=False)

    if args.finalretrain:
        info("final retrain selected")
        rcf = RecommenderFactory(
            abs_strat_range=[
                "mean",
            ],
            featurizations_range=[
                "ab_Gsolv_cat_norm_partit_solv_solu_ecfp",
            ],
        )
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            save_recommender=True,
        )
