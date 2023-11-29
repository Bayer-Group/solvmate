from collections import defaultdict
import joblib
from solvmate import *
import seaborn as sns

"""
A script that plots the feature importance analysis of the models,
aka which features contribute most to the pairwise regressors
that the final solvent ranking is generated from.

Only intended for advanced uses, e.g. improving / debugging ML models
"""


def _feature_type_from_name(feature_name: str):
    if feature_name.startswith("solv"):
        return "solv"
    if feature_name.startswith("compound"):
        return "compound"
    raise ValueError("Could not handle fature name '{feature_name}' ")


def _property_group_from_name(feature_name: str):
    """

    >>> _property_group_from_name("compound|('Gelec','water')")
    'Gelec'
    """
    # Use of eval here is unproblematic as it is only called
    # by data scientists in the evaluation phase.
    # not run during production
    return eval(feature_name.split("|")[1])[0] # nosec


def plot_shap_feature_analysis():
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
    )
    from sklearn.model_selection import train_test_split
    from shap import TreeExplainer
    from shap import summary_plot

    for rec_fle in list(DATA_DIR.glob("recommender_*.pkl"))[0:1]:
        rec: Recommender = joblib.load(rec_fle)
        pairs = get_pairs_data()
        pairs = pairs.sample(100)
        rec._featurize_pairs(pairs)
        X_all = np.vstack(pairs["X"].values)
        explainer = TreeExplainer(rec.reg)
        shap_values = np.array(explainer.shap_values(X_all))
        summary_plot(
            shap_values,
            X_all,
            feature_names=rec.featurizer.column_names(),
            max_display=100,
            show=False,
        )

        fle_out = PROJECT_ROOT / "figures" / "shap_feature_importance_analysis.svg"
        plt.savefig(fle_out)
        plt.clf()
        os.system(f"open {fle_out.resolve()}") # nosec


def plot_sklearn_feature_analysis():

    df_feat = []
    for rec_fle in DATA_DIR.glob("recommender_*.pkl"):
        rec = joblib.load(rec_fle)
        df_feat.append(rec.feature_importance_analysis_sklearn(None))

    n_recs = len(df_feat)
    df_feat = pd.concat(df_feat)

    imp_mean_dct = defaultdict(list)
    for imp, name in zip(df_feat["importance"], df_feat["name"]):
        imp_mean_dct[name].append(imp)
    imp_mean_dct = {k: np.array(v).mean() for k, v in imp_mean_dct.items()}

    df_feat["mean_imp"] = df_feat["name"].map(imp_mean_dct)

    df_feat["feat_type"] = df_feat["name"].apply(_feature_type_from_name)

    for feat_type in df_feat["feat_type"].unique():
        g = df_feat[df_feat["feat_type"] == feat_type]

        g = g.sort_values(
            "mean_imp",
            ascending=False,
        ).reset_index(drop=True)
        n_features = 30
        g = g.iloc[0 : n_features * n_recs]
        plt.figure(figsize=(20, 40 / 100 * n_features))
        sns.barplot(
            data=g,
            x="importance",
            y="name",
            order=g["name"].drop_duplicates(),
            palette="light:b_r",
            # order=df_feat["name"],
            # errorbar=("ci", 95),
        )
        fle_out = (
            PROJECT_ROOT / "figures" / f"feature_importance_analysis_{feat_type}.svg"
        )
        plt.savefig(fle_out)
        plt.clf()
        os.system(f"open {fle_out.resolve()}") # nosec


def plot_sklearn_feature_analysis_prop_groups():

    df_feat = []
    for rec_fle in DATA_DIR.glob("recommender_*.pkl"):
        rec = joblib.load(rec_fle)
        df_feat.append(rec.feature_importance_analysis_sklearn(None))

    n_recs = len(df_feat)
    df_feat = pd.concat(df_feat)

    df_feat["prop_group"] = df_feat["name"].apply(_property_group_from_name)

    df_feat["feat_type"] = df_feat["name"].apply(_feature_type_from_name)

    for feat_type in df_feat["feat_type"].unique():
        g = df_feat[df_feat["feat_type"] == feat_type]

        imp_mean_dct = defaultdict(list)
        for imp, name in zip(g["importance"], g["prop_group"]):
            imp_mean_dct[name].append(imp)
        imp_mean_dct = {k: np.array(v).mean() for k, v in imp_mean_dct.items()}

        g["mean_imp"] = g["prop_group"].map(imp_mean_dct)

        g = g.sort_values(
            "mean_imp",
            ascending=False,
        ).reset_index(drop=True)
        # n_features = 30
        # g = g.iloc[0 : n_features * n_recs]
        # n_features = min(30, len(g))
        # plt.figure(figsize=(20, 40 / 100 * n_features))
        sns.barplot(
            data=g,
            x="importance",
            y="prop_group",
            order=g["prop_group"].drop_duplicates(),
            palette="light:b_r",
            # order=df_feat["name"],
            # errorbar=("ci", 95),
        )
        fle_out = (
            PROJECT_ROOT
            / "figures"
            / f"feature_importance_analysis_prop_group_{feat_type}.svg"
        )
        plt.savefig(fle_out)
        plt.clf()
        os.system(f"open {fle_out.resolve()}") # nosec


def plot_permutation_feature_analysis_prop_groups():
    pairs = get_pairs_data()

    pairs = pairs.sample(len(pairs) // 100)  # to speed up calcs
    df_feat = []
    for rec_fle in DATA_DIR.glob("recommender_*.pkl"):
        rec = joblib.load(rec_fle)
        if "X" not in pairs.columns:
            info("featurizing pairs...")
            rec._featurize_pairs(pairs)
            info("...done featurizing pairs")

        info("performing permutation feature importance analysis ...")
        df_feat.append(rec.feature_importance_permutation_analysis(pairs))
        info("... done performing permutation feature importance analysis")

    n_recs = len(df_feat)
    df_feat = pd.concat(df_feat)

    df_feat["prop_group"] = df_feat["name"].apply(_property_group_from_name)

    df_feat["feat_type"] = df_feat["name"].apply(_feature_type_from_name)

    for feat_type in df_feat["feat_type"].unique():
        g = df_feat[df_feat["feat_type"] == feat_type]

        imp_mean_dct = defaultdict(list)
        for imp, name in zip(g["importance"], g["prop_group"]):
            imp_mean_dct[name].append(imp)
        imp_mean_dct = {k: np.array(v).mean() for k, v in imp_mean_dct.items()}

        g["mean_imp"] = g["prop_group"].map(imp_mean_dct)

        g = g.sort_values(
            "mean_imp",
            ascending=False,
        ).reset_index(drop=True)
        # n_features = 30
        # g = g.iloc[0 : n_features * n_recs]
        # n_features = min(30, len(g))
        # plt.figure(figsize=(20, 40 / 100 * n_features))
        sns.barplot(
            data=g,
            x="importance",
            y="prop_group",
            order=g["prop_group"].drop_duplicates(),
            palette="light:b_r",
            # order=df_feat["name"],
            # errorbar=("ci", 95),
        )
        fle_out = (
            PROJECT_ROOT
            / "figures"
            / f"feature_importance_permutation_analysis_prop_group_{feat_type}.svg"
        )
        plt.savefig(fle_out)
        plt.clf()
        os.system(f"open {fle_out.resolve()}") # nosec


def main():
    plot_permutation_feature_analysis_prop_groups()
    plot_sklearn_feature_analysis_prop_groups()
    plot_sklearn_feature_analysis()
    plot_shap_feature_analysis()


if __name__ == "__main__":
    main()
