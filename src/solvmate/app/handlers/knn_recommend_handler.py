from dataclasses import dataclass
import json
from typing import Callable
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler
from solvmate.ccryst import datasets, chem_utils


def cryst_related_2d_descriptors(mol: Chem.Mol) -> np.ndarray:
    """

    >>> mol = Chem.MolFromSmiles(opsin_iupac_to_smiles("benzaldehyde"))
    >>> mol_similar = Chem.MolFromSmiles(opsin_iupac_to_smiles("benzoic acid"))
    >>> mol_different = Chem.MolFromSmiles(opsin_iupac_to_smiles("octan-1-ol"))
    >>> fun = cryst_related_2d_descriptors
    >>> np.linalg.norm(fun(mol_similar)-fun(mol))
    18.944763386291125
    >>> np.linalg.norm(fun(mol_different)-fun(mol))
    26.637137146716693
    >>> mol_very_similar = Chem.MolFromSmiles(opsin_iupac_to_smiles("2-methylbenzaldehyde"))
    >>> np.linalg.norm(fun(mol_very_similar)-fun(mol))
    14.051835175091744
    """
    HIGH_WEIGHT = 10
    vec = [
        Descriptors.ExactMolWt(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Lipinski.FractionCSP3(mol),
        Lipinski.HeavyAtomCount(mol),
        Lipinski.NHOHCount(mol),
        Lipinski.NOCount(mol),
        Lipinski.NumAliphaticCarbocycles(mol),
        Lipinski.NumAliphaticHeterocycles(mol),
        Lipinski.NumAliphaticRings(mol),
        Lipinski.NumHAcceptors(mol) * HIGH_WEIGHT,
        Lipinski.NumHDonors(mol) * HIGH_WEIGHT,
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
    ]
    return np.array(vec)


class KNNRecommender:
    def run_df(self, smiles: str) -> pd.DataFrame:
        raise NotImplementedError()


class _KNNRecommenderFunctional:
    def __init__(
        self, mol_enc_function: Callable[[Chem.Mol], np.ndarray], sources=None
    ) -> None:
        if DEVEL and sources is None:
            sources = ["nova", "mantas"]
        self.sources = sources
        self.dist_type = "l2"
        self.lru_cache = []
        self.mol_enc_function = mol_enc_function

    def fit(self) -> None:
        df = datasets.load_all(
            sources=self.sources,
        )
        df = df[df["bin_cryst"].astype(bool)]
        df["ecfp"] = df["mol_compound"].apply(self.mol_enc_function)
        self.df_ = df

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

        vec = self.mol_enc_function(Chem.MolFromSmiles(smiles))
        vec_diff = np.vstack(df["ecfp"]) - vec

        if self.dist_type == "l2":
            df["dist"] = np.linalg.norm(vec_diff, axis=1)
        else:
            assert False, "unknown dist type " + str(self.dist_type)

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


class KNNRecommendHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        global _KNN_REC, _CURRENT_REC_TYPE

        try:
            req = json.loads(self.request.body)
            rec_type = req["rec_type"]
            assert rec_type in [
                "ecfp",
                "2DDescriptors",
            ]

            if _KNN_REC is None or _CURRENT_REC_TYPE != rec_type:
                _CURRENT_REC_TYPE = rec_type
                if rec_type == "ecfp":
                    _KNN_REC = _KNNRecommenderFunctional(
                        mol_enc_function=chem_utils.ecfp_fingerprint
                    )
                elif rec_type == "2DDescriptors":
                    _KNN_REC = _KNNRecommenderFunctional(
                        mol_enc_function=cryst_related_2d_descriptors,
                    )
                else:
                    assert False, "unknown rec type"

            rec: KNNRecommender = _KNN_REC
            req_smi = req["smiles"]
            req_smi = Chem.CanonSmiles(req_smi)

            start = req["start"]
            end = req["end"]

            dfr = rec.run_df(req_smi)

            if "solvent_label" not in dfr.columns:
                dfr["solvent_label"] = dfr["solvent_mixture_smiles"].apply(str)

            dfr = dfr.drop_duplicates(["smiles", "solvent_label"])

            matches = []
            for smi_solute in list(dfr["smiles"].unique())[start:end]:
                g = dfr[dfr["smiles"] == smi_solute]

                matches.append(
                    {
                        "similarity": "{0:.3f}".format(g["sim"].iloc[0]),
                        "distance": g["dist"].iloc[0],
                        "smiles_solute": smi_solute,
                        "source": [row["source"] for _, row in g.iterrows()],
                        "smiles_solvents": g["solvent_mixture_smiles"]
                        .apply(lambda sl: "+".join(sl))
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
