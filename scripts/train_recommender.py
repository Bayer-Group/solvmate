import argparse

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from solvmate import *
from solvmate.ranksolv.featurizer import XTBFeaturizer

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
    parser.add_argument(
        "--perform-butina-cv",
        action="store_true",
        required=False,
        dest="perform_butina_cv",
    )
    parser.add_argument(
        "--perform-solvent-cv",
        action="store_true",
        required=False,
        dest="perform_solvent_cv",
    )
    parser.add_argument(
        "--nova-as-ood",
        action="store_true",
        required=False,
        dest="nova_as_ood",
    )
    parser.add_argument(
        "--screen-model-types",
        action="store_true",
        required=False,
        dest="screen_model_types",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        help="name of the job",
        required=False,
        dest="job_name",
    )
    parser.add_argument(
        "--only-source",
        type=str,
        help="filter to only this source",
        required=False,
        dest="only_source",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="reproduce the results of the paper",
        required=False,
        dest="paper",
    )
    parser.add_argument(
        "--to-abs-strat-comparison",
        action="store_true",
        help="compare the different pairwise ranking algorithms",
        required=False,
        dest="to_abs_strat_comparison",
    )
    args = parser.parse_args()

    assert sum(
        [args.hyperp, args.screen_model_types, args.finalretrain, args.paper, args.to_abs_strat_comparison,]
    ), "specify only one of: --hyperp --screen-model-types --finalretrain --paper --to-abs-strat-comparison"


    if args.to_abs_strat_comparison:
        info("to abs strat comparison. Comparing the predictive performance of:") 
        info("mean, ranked_pairs, pair_rank algorithms")
        rcf = RecommenderFactory(
            abs_strat_range=["page_rank","pair_rank","mean",],
            featurizers=[
                XTBFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="xtb",
                ),
            ],
                                 )
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=False,
            perform_solvent_cv=False,
            nova_as_ood=False,
            save_recommender=False,
            job_name="GS9_PAPER_ESI_ABS_STRAT_COMPARISON",
            only_source="open_notebook",
            eval_on_ood=False,
        )
        info("DONE.")

    if args.paper:
        info("reproducing the paper results")
        info("first: cross validation on features, OOD on Bayer")
        rcf = RecommenderFactory()
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=False,
            perform_solvent_cv=False,
            nova_as_ood=False,
            save_recommender=False,
            job_name="GS4_PAPER_FEATURES_BAYER_OOD",
            only_source=None,
            eval_on_ood=True,
        )
        info("DONE.")

        info("second: cross validation on features, OOD on Nova")
        rcf = RecommenderFactory()
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=False,
            perform_solvent_cv=False,
            nova_as_ood=True,
            save_recommender=False,
            job_name="GS4_PAPER_FEATURES_NOVA_OOD",
            only_source=None,
            eval_on_ood=True,
        )
        info("DONE.")

        rcf = RecommenderFactory()
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=True,
            perform_solvent_cv=False,
            nova_as_ood=False,
            save_recommender=False,
            job_name="GS4_PAPER_FEATURES_BUTINA",
            only_source=None,
            eval_on_ood=False,
        )
        info("DONE.")

        info("second: cross validation on features, OOD on Nova")
        rcf = RecommenderFactory()
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=False,
            perform_solvent_cv=True,
            nova_as_ood=True,
            save_recommender=False,
            job_name="GS4_PAPER_FEATURES_SOLVENT_CV",
            only_source=None,
            eval_on_ood=False,
        )
        info("DONE.")
        info("screening model types")
        rcf = RecommenderFactory(
            featurizers=[
                XTBFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="xtb",
                ),
            ],
            regs=[
                Lasso(
                    alpha=1.0,
                    fit_intercept=True,
                    precompute=False,
                    copy_X=True,
                    max_iter=1000,
                    tol=0.0001,
                    warm_start=False,
                    positive=False,
                    random_state=None,
                    selection="cyclic",
                ),
                ExtraTreesRegressor(
                    n_jobs=N_JOBS,
                    n_estimators=100,
                ),
                RandomForestRegressor(
                    n_jobs=N_JOBS,
                    n_estimators=100,
                ),
                MLPRegressor(
                    hidden_layer_sizes=(100,),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size="auto",
                    learning_rate="constant",
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True,
                    random_state=None,
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08,
                    n_iter_no_change=10,
                    max_fun=15000,
                ),
                GradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.1,
                    n_estimators=100,
                    subsample=1.0,
                    criterion="friedman_mse",
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_depth=3,
                    min_impurity_decrease=0.0,
                    init=None,
                    random_state=None,
                    max_features=None,
                    alpha=0.9,
                    verbose=0,
                    max_leaf_nodes=None,
                    warm_start=False,
                    validation_fraction=0.1,
                    n_iter_no_change=None,
                    tol=0.0001,
                    ccp_alpha=0.0,
                ),
            ],
        )
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=False,
            perform_solvent_cv=False,
            nova_as_ood=False,
            eval_on_ood=False,
            save_recommender=False,
            job_name="GS4_PAPER_MODELS",
            only_source=None,
        )



    if args.hyperp:
        info("hyperparameter search selected")
        rcf = RecommenderFactory()
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=args.perform_butina_cv,
            perform_solvent_cv=args.perform_solvent_cv,
            nova_as_ood=args.nova_as_ood,
            save_recommender=False,
            job_name=args.job_name,
            only_source=args.only_source,
        )

    if args.screen_model_types:
        info("screening model types")
        rcf = RecommenderFactory(
            featurizers=[
                XTBFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="xtb",
                ),
            ],
            regs=[
                Lasso(
                    alpha=1.0,
                    fit_intercept=True,
                    precompute=False,
                    copy_X=True,
                    max_iter=1000,
                    tol=0.0001,
                    warm_start=False,
                    positive=False,
                    random_state=None,
                    selection="cyclic",
                ),
                ExtraTreesRegressor(
                    n_jobs=N_JOBS,
                    n_estimators=100,
                ),
                RandomForestRegressor(
                    n_jobs=N_JOBS,
                    n_estimators=100,
                ),
                MLPRegressor(
                    hidden_layer_sizes=(100,),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size="auto",
                    learning_rate="constant",
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True,
                    random_state=None,
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08,
                    n_iter_no_change=10,
                    max_fun=15000,
                ),
                GradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.1,
                    n_estimators=100,
                    subsample=1.0,
                    criterion="friedman_mse",
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_depth=3,
                    min_impurity_decrease=0.0,
                    init=None,
                    random_state=None,
                    max_features=None,
                    alpha=0.9,
                    verbose=0,
                    max_leaf_nodes=None,
                    warm_start=False,
                    validation_fraction=0.1,
                    n_iter_no_change=None,
                    tol=0.0001,
                    ccp_alpha=0.0,
                ),
            ],
        )
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=args.perform_butina_cv,
            perform_solvent_cv=args.perform_solvent_cv,
            nova_as_ood=args.nova_as_ood,
            save_recommender=False,
            job_name=args.job_name,
            only_source=args.only_source,
        )

    if args.finalretrain:
        info("final retrain selected")
        rcf = RecommenderFactory(
            abs_strat_range=[
                "mean",
            ],
            featurizers=[
                XTBFeaturizer(
                    phase="train",
                    pairwise_reduction="diff",
                    feature_name="xtb",
                ),
            ],
        )
        rcf.train_and_eval_recommenders(
            perform_cv=True,
            perform_butina_cv=False,
            perform_solvent_cv=False,
            nova_as_ood=False,
            save_recommender=True,
            job_name=str(args.job_name),
            only_source=args.only_source,
            eval_on_ood=False,
        )
