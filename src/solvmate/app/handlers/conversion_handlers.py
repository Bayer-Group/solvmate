import json
from rdkit import Chem
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler


class MDLMolToSmilesHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        # smi = self.get_argument("smiles")
        try:
            req = json.loads(self.request.body)
            mdl = req["mdl_mol"]
            mol = Chem.MolFromMolBlock(mdl)
            smi = Chem.MolToSmiles(mol)

            resp = json.dumps(
                {
                    "depict_status": "success",
                    "smiles": smi,
                }
            )
            self.write(resp)
        except:
            resp = json.dumps(
                {
                    "depict_status": "failure",
                    "smiles": "",
                }
            )
            self.write(resp)
            if DEVEL:
                raise


class SmilesToMDLMolHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        # smi = self.get_argument("smiles")
        try:
            req = json.loads(self.request.body)
            smi = req["smiles"]
            mol = Chem.MolFromSmiles(smi)
            mdl = Chem.MolToMolBlock(mol)

            resp = json.dumps(
                {
                    "depict_status": "success",
                    "mdl_mol": mdl,
                }
            )
            self.write(resp)
        except:
            resp = json.dumps(
                {
                    "depict_status": "failure",
                    "mdl_mol": "",
                }
            )
            self.write(resp)
            if DEVEL:
                raise
