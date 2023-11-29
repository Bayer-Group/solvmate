from solvmate import *
from solvmate.ranksolv.recommender import Recommender
import seaborn as sns
from scipy.stats import spearmanr


class PlotDisplay:
    def __init__(
        self,
        data: pd.DataFrame,
        do_iupac_name_conversion=True,
    ) -> None:
        assert "pos_index" in data.columns
        assert "solvent" in data.columns
        self.data = data
        self.do_iupac_name_conversion = do_iupac_name_conversion

    def plot(
        self,
        filter_solvents=None,
    ):

        df = self.data

        if filter_solvents:
            df = df[df["solvent"].isin(filter_solvents)]

        if self.do_iupac_name_conversion:
            df["solvent"] = df["solvent"].apply(smi_to_name)

        solvs, cents = [], []
        for solv in df["solvent"].unique():
            cent = df[df["solvent"] == solv]["pos_index"].median()

            solvs.append(solv)
            cents.append(cent)

        df["solv_cent"] = df["solvent"].map(dict(zip(solvs, cents)))
        df = df.sort_values("solv_cent")

        sns.boxplot(
            data=df,
            y="solvent",
            x="pos_index",
        )


def list_shape(
    lst: list,
    max_depth=100,
    average_over=1000,
):
    """
    Useful for interrogating the shape of lists as we
    are used from numpy arrays but generalized to
    ragged lists as well...

    >>> list_shape([])
    array([], dtype=float64)

    >>> list_shape([0])
    array([1.])

    >>> list_shape([1,2,3])
    array([3.])

    >>> list_shape([[1,2,3]])
    array([1., 3.])

    >>> list_shape([[1,2,3],[4,5,6]])
    array([2., 3.])

    >>> list_shape([[1,2,3],[4,6]]).round(1)
    array([2. , 2.5])

    """
    lst_shapes = []
    for _ in range(average_over):
        lst_shape = _rec_list_shape(lst, max_depth=max_depth)
        lst_shapes.append(lst_shape)
    return np.mean(lst_shapes, axis=0)


def _rec_list_shape(lst: list, max_depth: int):
    assert max_depth
    if not isinstance(lst, list) or len(lst) == 0:
        return []
    else:
        shape = [len(lst)]
        return shape + _rec_list_shape(random.choice(lst), max_depth - 1)


def ragged_transpose_102(lst: list[list[list]]):
    """
    Transpose version that accepts also ragged arrays

    >>> mat_in = [[["a","b"],["c","d","e"]]]
    >>> list_shape(mat_in).round(1)
    array([1. , 2. , 2.5])
    >>> mat_out = ragged_transpose_102(mat_in); mat_out
    [[['a', 'b']], [['c', 'd', 'e']]]
    >>> list_shape(mat_out).round(1)
    array([2. , 1. , 2.5])

    >>> mat_in = [[["abba","babba"],["ca","da","ea"]]]
    >>> mat_out = ragged_transpose_102(mat_in); mat_out
    [[['abba', 'babba']], [['ca', 'da', 'ea']]]
    """
    n_cols = len(lst)
    n_rows = len(lst[0])
    rslt = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for i_row in range(n_rows):
        for i_col in range(n_cols):
            elt = lst[i_col][i_row]
            rslt[i_row][i_col] = elt

    return rslt


class JackKnifeRecommender:
    def __init__(self, rc_fles: list[Path]) -> None:
        self.rcs: list[Recommender] = []
        for rc_fle in rc_fles:
            assert rc_fle.exists()
            self.rcs.append(Recommender.load(rc_fle))

    def recommend_spread(self, smiles: list[str], pairs):
        preds_all_rcs = []
        for rc in self.rcs:
            preds_all_rcs.append(rc.recommend(smiles, pairs))

        try:
            preds_all_rcs = np.array(preds_all_rcs).transpose(1, 0, 2)
        except:
            try:
                preds_all_rcs = np.array(
                    ragged_transpose_102(preds_all_rcs),
                )
            except:
                import pdb

                pdb.set_trace()
        return preds_all_rcs

    def recommend_smiles_spread(self, smiles: list[str]):
        preds_all_rcs = []
        for rc in self.rcs:
            preds_all_rcs.append(rc.recommend_smiles(smiles))

        try:
            preds_all_rcs = np.array(preds_all_rcs).transpose(1, 0, 2)
        except:
            preds_all_rcs = np.array(
                ragged_transpose_102(preds_all_rcs),
            )

        return preds_all_rcs

    def show_plot(
        self,
        preds: np.array,
    ) -> list[PlotDisplay]:

        rslt = []
        for row in preds:
            # A row looks something like this:
            # [["A","B","C","D"],["B","A","C","D"],...]
            # and we are trying to do a violin/box plot of it.
            #
            # This plot will look something like this:
            #    ________
            # . /        \ .........   MeOH
            #   \________/
            #          ______
            # .. .. . /      \ .........   EtOH
            #         \______/
            # .                ______
            # ........ .. ..  /      \ .........   H2O
            # .               \______/
            #
            # Therefore, we have to first split by the hue,
            # and then we identify the spread per instance.
            all_solvents = list(pd.Series(row.reshape(-1)).unique())

            solv_pos_spread = []
            for solvent in all_solvents:
                for col in row:
                    idx = np.where(col == solvent)[0]
                    assert len(idx) == 1
                    solv_pos_spread.append(
                        {
                            "pos_index": idx[0],
                            "solvent": solvent,
                        }
                    )

            solv_pos_spread = pd.DataFrame(solv_pos_spread)

            rslt.append(
                PlotDisplay(
                    data=solv_pos_spread,
                )
            )

        return rslt

    def recommend_with_err(self, smiles: list[str], pairs):
        preds = self.recommend_spread(smiles, pairs)
        assert len(preds) == len(smiles)

        pred_means, stds = [], []
        for smi, row in zip(smiles, preds):

            try:
                all_solvents = list(pd.Series(row.reshape(-1)).unique())
            except:
                all_solvents = list(set(sum(row, [])))

            row_solv_spreads, row_solv_means = [], []
            for solvent in all_solvents:
                inner_solv_spreads = []
                for col in row:
                    col = np.array(col)
                    idx = np.where(col == solvent)[0]
                    assert len(idx) == 1
                    inner_solv_spreads.append(idx)

                assert inner_solv_spreads
                inner_solv_spreads = np.array(inner_solv_spreads)
                row_solv_spreads.append(
                    float(
                        np.median(abs(inner_solv_spreads - np.mean(inner_solv_spreads)))
                        / len(inner_solv_spreads)  # TODO: check
                    )
                )
                row_solv_means.append(np.median(np.array(inner_solv_spreads)))

            if False:
                stds.append(np.array(row_solv_spreads).sum())
            else:
                spears_ij = []
                for i, col_i in enumerate(row):
                    for j, col_j in enumerate(row):
                        if i == j:
                            continue
                        numerals_i = [list(col_i).index(i_val) for i_val in col_i]
                        numerals_j = [list(col_i).index(j_val) for j_val in col_j]
                        spears_ij.append(spearmanr(numerals_i, numerals_j).correlation)
                spears_ij = np.nan_to_num(np.array(spears_ij))
                stds.append(spears_ij.mean())

            solv_by_mean = [(solv, m) for solv, m in zip(all_solvents, row_solv_means)]
            solv_by_mean = sorted(solv_by_mean, key=lambda tup: tup[1])
            pred_means.append([tup[0] for tup in solv_by_mean])

        assert len(pred_means) == len(stds)
        assert len(preds) == len(smiles)
        assert len(pred_means) == len(smiles)
        return pred_means, stds
