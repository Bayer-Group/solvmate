from functools import reduce
from pathlib import Path
import asyncio
import json
import io
import base64
import re
import time
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from rdkit import Chem
from solvmate.app.handlers.base_handler import BaseHandler

from solvmate.ccryst.solvent import solvent_mixture_iupac_to_smiles


def ess_solvents():
    return [
        "n-Heptan",
        "Cyclohexan",
        "Diisopropylether",
        "Toluol",
        "Tetrahydrofuran",
        "Aceton",
        "Ethylacetat",
        "ACN",
        "2-Propanol",
        "Ethanol",
        "EtOH / Wasser 1:1",
        "Methanol",
        "Wasser",
        "Dichlormethan",
        "Isopropanol/Wasser 1/1",
    ]


def default_solvents():
    return """
cyclohexane
octane
heptane
hexadecane
nonane
decane
2,2-dimethyl-4-methylpentane
hexane
methylcyclohexane
water
toluene
benzene
carbon tetrachloride
chlorobenzene
dibutyl ether
di-isopropylether
methyl t-butylether
nitromethane
diethyl ether
decanol
methyl isobutyl ketone
water : Nonaethyleneglycol 1 : 1
Nonaethylene glycol
1-heptanol
methylene chloride
1-octanol
1-hexanol
chloroform
ethanol : water 1 : 1
2-ethylhexanol
4-methyl pentan-2-ol
2-methyl pentanol
ethylene glycol
2-isopropoxyethanol
1-pentanol
1-butanol
2-propoxyethanol
iso-butanol
2-pentanol
acetonitrile
ethyl acetate
butylacetate
1,4-dioxane
t-butyl alcohol
2-butoxyethanol
2-butanol
isoamyl alcohol
2-propanol
1-propanol
methyl acetate
2-ethoxyethanol
ethanol
1,2-dichloroethane
acetone
methanol
2-butanone
tetrahydrofuran
dimethyl sulfoxide
dimethylformamide
""".strip().split(
        "\n"
    )


def _fill_from_iup(df):
    df["smiles"] = df["iupac"].apply(solvent_mixture_iupac_to_smiles)
    df["smiles"] = df["smiles"].apply(lambda lst: ".".join(lst))
    return df


class SolventSelectionStoreHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        global SOLVENT_SELECTION_DB

        if not hasattr(self, "con"):
            self.con = sqlite3.connect(SOLVENT_SELECTION_DB)

        try:
            req = json.loads(self.request.body)

            df_new = pd.DataFrame(req["new_solvent_selections"])
            df_size = reduce(lambda a, b: a * b, df_new.shape)
            if df_size > 1024:
                # size limits for safety
                time.sleep(5)
                return
            df_new = _fill_from_iup(df_new)
            df_new = df_new[~df_new["smiles"].isna()]
            df_new = df_new[df_new["smiles"].apply(len) > 0]

            df_new.to_sql("solvent_selections", self.con, if_exists="replace")

            resp = json.dumps(
                {
                    "query_status": "success",
                }
            )
            self.write(resp)
        except:
            resp = json.dumps(
                {
                    "query_status": "failure",
                }
            )
            self.write(resp)
            if DEVEL:
                raise

    def __del__(
        self,
    ):
        if hasattr(self, "con"):
            print("closing database connection ...")
            self.con.close()


class SolventSelectionFetchHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        global SOLVENT_SELECTION_DB

        if not hasattr(self, "con"):
            self.con = sqlite3.connect(SOLVENT_SELECTION_DB)

            self.con.execute(
                "CREATE table IF NOT EXISTS solvent_selections (smiles TEXT, iupac TEXT, selection TEXT)"
            )
            if not len(pd.read_sql("SELECT * FROM solvent_selections", self.con)):

                df_ess = pd.DataFrame({"iupac": ess_solvents()})

                df_ess = _fill_from_iup(df_ess)

                df_ess["selection"] = "ess"
                df_ess.to_sql("solvent_selections", self.con, if_exists="replace")

                df_def = pd.DataFrame({"iupac": default_solvents()})

                df_def = _fill_from_iup(df_def)

                df_def["selection"] = "default"
                df_def.to_sql("solvent_selections", self.con, if_exists="append")

        try:
            req = json.loads(self.request.body)

            rslt = pd.read_sql(
                sql="SELECT * FROM solvent_selections",
                con=self.con,
            )

            resp = json.dumps(
                {
                    "query_status": "success",
                    "rslt": [
                        {
                            "smiles": row["smiles"],
                            "iupac": row["iupac"],
                            "selection": row["selection"],
                        }
                        for _, row in rslt.iterrows()
                    ],
                }
            )
            self.write(resp)
        except:
            resp = json.dumps({"query_status": "failure", "solvent_selection_list": []})
            self.write(resp)
            if DEVEL:
                raise

    def __del__(
        self,
    ):
        if hasattr(self, "con"):
            print("closing database connection ...")
            self.con.close()
