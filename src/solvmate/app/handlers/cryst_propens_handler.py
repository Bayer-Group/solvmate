import json
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from rdkit import Chem
from solvmate.app.handlers.base_handler import BaseHandler

from solvmate.ccryst.chem_utils import crystal_propensity_heuristic


class CrystPropensHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        try:
            req = json.loads(self.request.body)
            req_smi = req["smiles"]

            mol = Chem.MolFromSmiles(req_smi)
            html_resp = crystal_propensity_heuristic(
                mol=mol,
                as_html=True,
            )

            resp = json.dumps(
                {
                    "query_status": "success",
                    "html_resp": html_resp,
                }
            )
            self.write(resp)
        except:
            resp = json.dumps({"query_status": "failure", "html_resp": "ERROR"})
            self.write(resp)
            if DEVEL:
                raise
