from pathlib import Path
import json
import io
import base64
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler


IMAGE_DIM = 300


def mol_to_image(mol, width=300, height=300) -> Image:
    rand_fle = str(random_fle("jpg"))
    Draw.MolToImageFile(mol, filename=rand_fle, format="JPG", size=(width, height))
    img = Image.open(rand_fle)
    return deepcopy(img)


def mol_to_svg(mol, width=300, height=300) -> Path:
    fle = random_fle("svg")
    fle.touch()
    Draw.MolToFile(
        mol=mol,
        filename=str(fle),
        size=(width, height),
        imageType="svg",
    )
    return fle


class SmilesInputHandler(BaseHandler):
    @tornado.web.authenticated
    @log_function_call
    def post(self):
        try:
            req = json.loads(self.request.body)
            req_smi = req["smiles"]
            mol = Chem.MolFromSmiles(req_smi)
            img = mol_to_image(mol, width=IMAGE_DIM, height=IMAGE_DIM)

            output = io.BytesIO()
            img.save(output, format="png")
            hex_data = output.getvalue()

            img64 = base64.b64encode(hex_data)

            svg_img = mol_to_svg(mol, width=IMAGE_DIM, height=IMAGE_DIM)
            svg_data = svg_img.read_text()

            resp = json.dumps(
                {
                    "depict_status": "success",
                    "depict_data": img64.decode("utf-8"),
                    "svg_data": svg_data,
                }
            )
            self.write(resp)
        except:
            resp = json.dumps(
                {"depict_status": "failure", "depict_data": "", "svg_data": ""}
            )
            self.write(resp)
            if DEVEL:
                raise
