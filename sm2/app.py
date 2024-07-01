from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from rdkit import Chem

from sm2.model import run_predictions_for_solvents


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