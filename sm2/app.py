import base64
import io
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from rdkit import Chem
import seaborn as sns
from matplotlib import pyplot as plt

from sm2.model import run_predictions_for_solvents
from solvmate.ccryst.solvent import solvent_mixture_iupac_to_smiles


app = FastAPI()

here = Path(__file__).parent
app.mount("/static", StaticFiles(directory=here / "js"), name="static")

HTML = here / "js" / "app.html"

@app.get("/main/", response_class=HTMLResponse)
async def main_page():
    return HTML.read_text()

@app.post("/rank-by-solubility/")
async def rank_by_solubility(data:dict):
    solute_smiles = data["solute SMILES"]
    solvents = data["solvents"]
    dfo = run_predictions_for_solvents(solute_smiles=solute_smiles,solvents=solvents,)
    dfo = dfo.sort_values("log S",ascending=False,)
    return dfo.to_dict("records")

def _safe_from_smiles(smi):
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None

def _parse_in_order(funs, txt):
    for fun in funs:
        if fun(txt):
            rslt = fun(txt)
            if isinstance(rslt,list):
                return ".".join(rslt)
            else:
                return rslt
    return None

@app.post("/plot-rank-by-solubility/")
async def plot_rank_by_solubility(data:dict):
    solute_smiles = data["solute SMILES"]
    solvents = data["solvents"]

    solvent_smis = [_parse_in_order([solvent_mixture_iupac_to_smiles,_safe_from_smiles], solv) for solv in solvents]
    solvent_smi_to_name = {smi:nme for smi,nme in zip(solvent_smis,solvents)}
    dfo = run_predictions_for_solvents(solute_smiles=solute_smiles,solvents=solvent_smis,)
    dfo = dfo.sort_values("log S",ascending=False,)
    dfo["solvents"] = dfo["solvent SMILES"].map(solvent_smi_to_name)
    
    plt.clf()
    sns.barplot(data=dfo,x="log S",y="solvents")
    buf = io.StringIO()
    plt.tight_layout(w_pad=2,h_pad=2)
    plt.savefig(buf,format="svg")
    plt.clf()
    return {"svg": buf.getvalue(),}
