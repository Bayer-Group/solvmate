from pathlib import Path
import asyncio
import json
import io
import base64
import re
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from rdkit import Chem

from solvmate.app.handlers.base_handler import BaseHandler


def _iou_sim(ta: str, tb: str) -> float:
    """
    Intersection over union similarity for quick and easy approximate
    string similarity deduplication.

    Note: The recipes stem from a public-domain patent reaction dataset

    >>> ta = "0.0072 moles of salicylic acid and 0.029 moles acetic anhydride..."
    >>> tb = "0.0082 holes of salicylic acid and 0.029 moles acetic anhydride..."
    >>> _iou_sim(ta,ta)
    1.0
    >>> _iou_sim(tb,tb)
    1.0
    >>> _iou_sim(ta,tb) # doctest: +ELLIPSIS
    0.72...
    >>> ta = "0.0072 moles of salicylic acid and 0.029 moles acetic anhydride (molar ratio=1:4) were taken in a 20 ml reaction vessel of reaction station. Nano-crystalline sulfated zirconia catalyst (SZ-1), activated at 450° C. in static air with salicylic acid to catalyst weight ratio 10:1, was added and the mixture was heated at 100° C. for 240 minutes. The reaction mixture was filtered to separate the catalyst. The acetyl salicylic acid was crystallized from the reaction mixture. The crude yield of acetyl salicylic acid was 74%. It was further re-crystallized with ethanol-water to obtain pure crystals of acetyl salicylic acid.."
    >>> tb = "0.0072 moles of salicylic acid and 0.029 moles acetic anhydride (molar ratio=1:4) were taken in a 20 ml reaction vessel of reaction station. Nano-crystalline sulfated zirconia catalyst (SZ-1), activated at 450° C. in static air with salicylic acid to catalyst weight ratio 10:1, was added and the mixture was heated at 120° C. for 30 minutes. The reaction mixture was filtered to separate the catalyst. The acetyl salicylic acid was crystallized from the reaction mixture. The crude yield of acetyl salicylic acid was 92-95%. It was further re-crystallized with ethanol-water to obtain pure crystals of acetyl salicylic acid.."
    >>> _iou_sim(ta,tb) # doctest: +ELLIPSIS
    0.91...

    """
    ws_a = set(re.findall("\S+", ta))
    ws_b = set(re.findall("\S+", tb))
    if len(ws_a) + len(ws_b):
        return len(ws_a.intersection(ws_b)) / len(ws_a.union(ws_b))
    else:
        # two empty strings considered to be same
        return 1.0


class MetaDataHandler(BaseHandler):
    def highlight_cryst_sentences(self, text):
        def highl_sent(sent):
            signal_words = ["cryst", "precipi"]
            if any(w in sent for w in signal_words):
                return "<b style='background-color: #cccccc;'>" + sent + "</b>"
            else:
                return sent

        text_out = []
        matches_with_delims = list(re.split("(\\.\s)", text))
        for sentence, delim in zip(
            matches_with_delims[0::2], matches_with_delims[1::2]
        ):
            text_out.append(highl_sent(sentence) + delim)

        return "".join(text_out)

    def drop_near_dups(self, df):

        to_drop = []
        for idx_a, row_a in df.iterrows():
            for idx_b, row_b in df.iterrows():
                ta, tb = row_a["text_recipe"], row_b["text_recipe"]
                if idx_a >= idx_b:
                    continue
                if _iou_sim(ta, tb) > 0.90:
                    if len(ta) > len(tb):
                        to_drop.append(idx_b)
                    else:
                        to_drop.append(idx_a)

        for drop_idx in set(to_drop):
            df = df.drop(drop_idx)
        return df

    @tornado.web.authenticated
    @log_function_call
    def post(self):

        if not hasattr(self, "con"):
            self.con = sqlite3.connect(LOWE_META_DB)

        try:
            req = json.loads(self.request.body)
            req_smi = req["smiles"]

            mol = Chem.MolFromSmiles(req_smi)
            inchi = Chem.MolToInchi(mol)

            rslt = pd.read_sql(
                sql=f"SELECT * FROM recipes WHERE inchi_product='{inchi}'",
                con=self.con,
            )

            rslt = self.drop_near_dups(rslt)

            if not len(rslt):
                info = "-"
            else:
                info = "\n".join(
                    rslt["text_recipe"]
                    .apply(self.highlight_cryst_sentences)
                    .apply(
                        lambda t: f"<p style='background-color: white; padding: 20px; margin-top: 10px;'>{t}</p>"
                    )
                )

            resp = json.dumps(
                {
                    "query_status": "success",
                    "metadata_text": info,
                }
            )
            self.write(resp)
        except:
            resp = json.dumps({"query_status": "failure", "metadata_text": "ERROR"})
            self.write(resp)
            if DEVEL:
                raise

    def __del__(
        self,
    ):
        if hasattr(self, "con"):
            print("closing database connection ...")
            self.con.close()
