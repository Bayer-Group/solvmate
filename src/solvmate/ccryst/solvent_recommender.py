import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tqdm

from IPython.display import display

from sklearn.metrics import precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError

from scipy.special import binom


def neg_hyp_geom_accum(N: int, M: int, k: int):
    """
    Computes the expectation value of the number of trials given a
    negative hypergeometric distribution where
    N is number of params overall
    M is number of successfull events
    k is number of successfull draws until win

    returns expected number of trials till k draws
    of M elements from N elements without replacement.

    Average number of dice throws till first 6:
    >>> neg_hyp_geom_accum(N=6,M=1,k=1)
    3.5

    Average number of dice rolls till any face:
    >>> neg_hyp_geom_accum(N=6,M=6,k=1)
    1.0

    Average number of dice rolls accepting anything
    except the six:
    >>> neg_hyp_geom_accum(N=6,M=5,k=2) # doctest:+ELLIPSIS
    2.333...
    """
    hist = {}

    for y in range(1, N + 1):
        hist[y] = (
            binom(M, k - 1)
            * binom(N - M, y - k)
            / binom(N, y - 1)
            * (M - k + 1)
            / (N - y + 1)
        )

    # print(hist)
    acc = 0
    for k, v in hist.items():
        acc += k * v if k * v >= 0 else 0
    return acc


class SolventRecommender:
    """
    Given a list of binary classifiers for each solvent,
    a SolventRecommender will predict the probabilities
    for each solvent so that the client can prioritize
    them acoordingly.
    """

    def __init__(self, clf_dict):
        self.clf_dict = clf_dict
        self.solvents = list(sorted(clf_dict.keys()))
        self.clfs = [clf_dict[solv] for solv in self.solvents]

    def predict(self, X) -> np.ndarray:
        solvents, clfs = self.solvents, self.clfs
        probs = []
        for solv, clf in zip(solvents, clfs):
            p = clf.predict_proba(X)
            probs.append(p[:, 1])
        return np.vstack(probs).T


def evaluate_solv_rec_take_top_n(
    solv_rec,
    df_val,
    top_n: int,
    X_col="X",
    solv_col="solvent_label",
):
    """
    Evaluates the accuracy of a given solvent recommender when taking
    the top_n solvents versus a random selection of solvents.
    The result therefore represents the percentage where one of the
    top_n recommended solvents did indeed correspond to a successful
    crystalization experiment in the past.
    """
    df_val = df_val.copy()  # not mutated

    if X_col is None:
        preds = solv_rec.predict(df_val)
    else:
        preds = solv_rec.predict(df_val[X_col].tolist())
    df_val["preds"] = [col for col in preds]

    correct_mod, correct_null, wrong_mod, wrong_null = 0, 0, 0, 0
    for smiles in df_val["smiles"].unique():
        dfc = df_val[df_val["smiles"] == smiles]

        pos_solvents = dfc[dfc.bin_cryst == 1][solv_col].tolist()
        neg_solvents = dfc[dfc.bin_cryst == 0][solv_col].tolist()

        if not (pos_solvents and neg_solvents):
            continue

        p = dfc["preds"].iloc[0]  # all preds same

        solvents, p = zip(*sorted(zip(solv_rec.solvents, p), key=lambda tup: -tup[1]))
        solvents = list(solvents)
        solvents_top = solvents[0:top_n]
        rand_top = list(solvents)
        random.shuffle(rand_top)
        rand_top = rand_top[0:top_n]
        correct_mod += any(s in pos_solvents for s in solvents_top)
        correct_null += any(s in pos_solvents for s in rand_top)
        wrong_mod += not any(s in pos_solvents for s in solvents_top)
        wrong_null += not any(s in pos_solvents for s in rand_top)

    acc_mod = correct_mod / (correct_mod + wrong_mod)
    acc_null = correct_null / (correct_null + wrong_null)
    return acc_mod, acc_null


def evaluate_solv_rec(solv_rec, df_val, mood="pessimist", X_col="X"):
    # TODO: define a proba threshold below which we abort and reevaluate the metrics with that strategy
    #
    # TODO: If only fails then check that the probability for these solvents should be low!
    # TODO: Conversely, if solvents are in S_1 then we should assign them a high proba.
    #       \
    # .       \___> we can use that to check that the other sources also work out with that recommender
    # .   TODO:  check that positive cases are within upper half of recommendations for these sources.
    # .   TODO:  alternatively:
    # .           for all sources that we cannot treat like ESS,NOVA:
    #                 for all positive compound+Solvent pairs:
    #                       histogram of place of solvent in recommender list. how does that look like?
    #                       ---> but set-wise as we have a set. => easiest fix take only those with a single solvent
    """
    returns trials of the model, trials of the null model

    Evaluates the solvent recommendation by doing the following experiment:


        For C ∈ Compounds:
            𝘚_𝟶 = solvents with no cryst success
            𝘚_1 = solvents with  a cryst success

            If 𝘚_𝟶 ≠ ∅ and 𝘚_1 ≠ ∅:
            Then:
                          |𝘚_1|
               ⍴_ref =  --------------
                        |𝘚_0| + |𝘚_1|

               Ref_Trials = 1 / ⍴_ref

               𝙍 = solvent recommendations
               Trials = 0
               DO:
                   IF PESSIMIST:
                      Trials += 1
                   ELSE:
                      Trials += 1 ONLY IF 𝙧 ∈ 𝘚_𝟶 U S_1
                   𝙧 = next candidate from 𝙍

               WHILE NOT 𝙧 ∈ 𝘚_1.

            Else: // cant compute proba
                ignore sample and continue

    """
    assert mood in [
        "pessimist",
        "optimist",
    ]

    df_val = df_val.copy()  # not mutated

    preds = solv_rec.predict(df_val[X_col].tolist())
    all_trials = []
    df_val["preds"] = [col for col in preds]

    ref_trials = []

    for smiles in df_val["smiles"].unique():
        dfc = df_val[df_val["smiles"] == smiles]

        pos_solvents = dfc[dfc.bin_cryst == 1].solvent_label.tolist()
        neg_solvents = dfc[dfc.bin_cryst == 0].solvent_label.tolist()

        if not (pos_solvents and neg_solvents):
            continue

        # prob_ref = len(pos_solvents) / (len(pos_solvents) + len(neg_solvents))

        p = dfc["preds"].iloc[0]  # all preds same

        solvents, p = zip(*sorted(zip(solv_rec.solvents, p), key=lambda tup: -tup[1]))
        solvents = list(solvents)

        if mood == "pessimist":
            trials = 0
        else:
            trials = 1

        for solv in solvents:
            if mood == "pessimist":
                trials += 1
            if mood == "optimist":
                trials += solv in neg_solvents

            if solv in pos_solvents:
                break

        if trials:  # otherwise didnt even find a solvent match!

            # ref_trials.append( 1 / prob_ref ) # geometric distribution
            N = len(pos_solvents) + len(neg_solvents)
            M = len(pos_solvents)
            k = 1  # first success wins the game!
            ref_trials.append(neg_hyp_geom_accum(N, M, k))

            all_trials.append(trials)

    return all_trials, ref_trials


def train_solvent_recommender(clf_builder, df, solvents, X_col="X"):
    """
    Trains the solvent recommender using the supplied
    clf_builder instance, supplied data df and the list
    of solvents to consider.
    It is assumed that the features are contained within
    the column X_col and that the target is contained within
    the column 'bin_cryst'

    returns solvent recommender, stats dict
    """
    assert (
        "split" in df.columns
    ), "need to have split column = 'train'|'test' to train on df!"
    assert X_col in df.columns, f"need feature column '{X_col}' to train!"
    clf_dict = {}  # solvent -> classifier
    stat_dict = {}  # solvent -> stats

    for solvent in tqdm.tqdm(solvents):
        clf = clf_builder()

        df_solvent = df[df.solvent_label == solvent]

        stat = {}

        df_train = df_solvent[df_solvent.split == "train"]
        df_test = df_solvent[df_solvent.split == "test"]
        df_val = df_solvent[df_solvent.split == "val"]
        X_train = np.vstack(df_train[X_col]).reshape(len(df_train), -1)
        y_train = df_train.bin_cryst.tolist()
        X_test = np.vstack(df_test[X_col]).reshape(len(df_test), -1)
        y_test = df_test.bin_cryst.tolist()
        X_val = np.vstack(df_val[X_col]).reshape(len(df_val), -1)
        y_val = df_val.bin_cryst.tolist()

        FORCE_FIT_BE = True  # only used for forcing proba calib to work
        if FORCE_FIT_BE:
            try:
                clf.fit(X_train, y_train)
            except NotFittedError:
                clf.base_estimator.fit(
                    df_train[X_col].tolist(), df_train.bin_cryst.tolist()
                )
                clf.fit(df_val[X_col].tolist(), df_val.bin_cryst.tolist())
        else:
            clf.fit(df_train[X_col].tolist(), df_train.bin_cryst.tolist())

        clf_dict[solvent] = clf

        stat["train_score"] = clf.score(X_train, y_train)
        stat["test_score"] = clf.score(X_test, y_test)

        stat["train_prfs"] = precision_recall_fscore_support(
            clf.predict(X_train),
            y_train,
            average="binary",
        )
        stat["test_prfs"] = precision_recall_fscore_support(
            clf.predict(X_test),
            y_test,
            average="binary",
        )
        stat["solvent"] = solvent

        prob_true, prob_pred = calibration_curve(
            y_val,
            clf.predict_proba(X_val)[:, 1],
            n_bins=10,
        )

        stat["prob_true"] = prob_true
        stat["prob_pred"] = prob_pred

        maj = sum(y_test) / len(y_test)

        if maj < 0.5:
            maj = 1 - maj

        stat["dummy_test_score"] = maj

        stat_dict[solvent] = stat
    return SolventRecommender(clf_dict), stat_dict


def classifier_stats_report(stats_dict):
    stats_df = pd.DataFrame(stats_dict).T
    stats_df

    display(stats_df)

    sns.barplot(
        data=stats_df,
        x="test_score",
        y="solvent",
    )

    plt.figure(figsize=(10, 10))

    plt.barh(
        [i for i in range(len(stats_df.solvent.tolist()))],
        stats_df.test_score.tolist(),
        color="blue",
        height=0.45,
    )
    plt.barh(
        [i + 0.5 for i in range(len(stats_df.solvent.tolist()))],
        stats_df.dummy_test_score.tolist(),
        color="red",
        height=0.45,
    )
    plt.gca().set_yticks([i + 0.25 for i in range(len(stats_df.solvent.tolist()))])
    plt.gca().set_yticklabels(stats_df.solvent.tolist())
    plt.show()
