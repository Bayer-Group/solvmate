from pathlib import Path
import json
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler
from solvmate.app.pool import PROCESS_POOL
from solvmate.ccryst.solvent import solvent_mixture_iupac_to_smiles
from solvmate.ranksolv.jack_knife_recommender import JackKnifeRecommender

import asyncio

JKR = None


def solub_recommend(
    smi: str,
    filter_solvents: list[str],
    allow_unseen_solvents=True,
) -> Path:
    global JKR

    if JKR is None:
        log("loading recommender instance...")
        rcs = list(DATA_DIR.glob("recommender_*.pkl"))

        if not rcs:
            warn(
                "warning: could not find any recommender instances. please train recommenders using `make all_fast` or `make all_slow` if first command fails..."
            )

        JKR = JackKnifeRecommender(rcs)

    if JKR is not None:
        for rc in JKR.rcs:
            rc.featurization = "ab_Gsolv_cat_norm_partit_solv_solu_ecfp"

            if allow_unseen_solvents:
                rc.top_pairs_ = [
                    (a, b) for a in filter_solvents for b in filter_solvents if a != b
                ]

    rslt = JKR.recommend_smiles_spread([smi])

    plot_disps = JKR.show_plot(rslt)

    assert len(plot_disps) == 1
    plot_disp = plot_disps[0]

    if filter_solvents:
        height = 15 / 60 * len(filter_solvents)
    else:
        height = 15

    before = plt.rcParams["svg.fonttype"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.figure(figsize=(10, height))
    plt.tight_layout()
    if allow_unseen_solvents:
        plot_disp.plot(filter_solvents=None)
    else:
        plot_disp.plot(filter_solvents=filter_solvents)

    img = random_fle("svg")
    plt.savefig(img)
    plt.rcParams["svg.fonttype"] = before
    assert img.exists()
    return img


class SolubRecommendHandler(BaseHandler):
    @log_function_call
    @tornado.web.authenticated
    async def post(self):
        try:
            req = json.loads(self.request.body)
            solvents = [
                ".".join(solvent_mixture_iupac_to_smiles(iup))
                for iup in req["solvents"]
            ]
            req_smi = req["smiles"]
            req_smi = Chem.CanonSmiles(req_smi)

            # svg_file = solub_recommend(
            # smi=req_smi,
            # filter_solvents=solvents,
            # )

            loop = asyncio.get_running_loop()
            svg_file = loop.run_in_executor(
                PROCESS_POOL, solub_recommend, req_smi, solvents
            )
            svg_file = await svg_file

            resp = json.dumps(
                {"depict_status": "success", "depict_data": svg_file.read_text()}
            )
            self.write(resp)
        except:
            resp = json.dumps({"depict_status": "failure", "depict_data": ""})
            self.write(resp)

            if DEVEL:
                raise
