import base64
import io
from pathlib import Path
import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from rdkit import Chem
import seaborn as sns
from matplotlib import pyplot as plt

from sm2.model import run_predictions_for_solvents
from solvmate.ccryst.solvent import iupac_solvent_mixture_to_amounts, solvent_mixture_iupac_to_smiles


app = FastAPI()

here = Path(__file__).parent
app.mount("/static", StaticFiles(directory=here / "js"), name="static")

HTML = here / "js" / "app.html"

@app.get("/main/", response_class=HTMLResponse)
async def main_page():
    return HTML.read_text()

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


def _extract_temperatures(solvents:str,):
    # Strip any temperature specifications, e.g. '250C, 298K' from 
    # the solvent text. Translate it accordingly
    temps = []
    stripped_solvents = []
    for solv in solvents:
        parts = solv.split()
        stripped_parts = []
        # 0 == 25 C == 298.15 K
        temp = 0
        for part in parts:
            mtch_C = re.match(r"([+-]?([0-9]*[.])?[0-9]+)C",part)
            mtch_K = re.match(r"([+-]?([0-9]*[.])?[0-9]+)K",part)
            if mtch_C:
                temp = float(mtch_C.group(1)) - 25
            elif mtch_K:
                temp = float(mtch_K.group(1)) - 273.15
            else:
                # it's not a temp spec so append
                stripped_parts.append(part)

        temps.append(temp)
        stripped_solvents.append(" ".join(stripped_parts))

    assert len(temps) == len(solvents)
    assert len(temps) == len(stripped_solvents)
    return temps,stripped_solvents
        

@app.post("/plot-rank-by-solubility/")
async def plot_rank_by_solubility(data:dict):
    solute_smiles = data["solute SMILES"]
    solvents = data["solvents"]

    temps,solvents = _extract_temperatures(solvents)

    solvent_amounts = [iupac_solvent_mixture_to_amounts(solv) for solv in solvents]
    solvent_smis = [_parse_in_order([solvent_mixture_iupac_to_smiles,_safe_from_smiles], solv) for solv in solvents]
    # unpack the solvent amounts so that we get instead of a dictionary the
    # solvent amount vector, instead.
    solvent_amounts =  [
        [samnt.get(s,1)/len(list(smix.split("."))) for s in smix.split(".")]
        for samnt,smix in zip(solvent_amounts,solvent_smis)
    ]
    solvent_smis = [smi for smi in solvent_smis if smi]
    solvent_smi_to_name = {smi:nme for smi,nme in zip(solvent_smis,solvents)}
    dfo = run_predictions_for_solvents(solute_smiles=solute_smiles,solvents=solvent_smis,temps=temps,facs=solvent_amounts,)
    dfo = dfo.sort_values("log S",ascending=False,)
    dfo["solvents"] = dfo["solvent SMILES"].map(solvent_smi_to_name)

    dfo["solvents_T"] = dfo["solvents"] + "__" + dfo["temp"]
    
    plt.clf()
    plt.figure(figsize=(10,2+len(dfo)//8))
    sns.barplot(data=dfo,x="log S",y="solvents_T")
    buf = io.StringIO()
    plt.tight_layout(w_pad=2,h_pad=2)
    plt.savefig(buf,format="svg")
    plt.clf()
    return {"svg": buf.getvalue(),}
