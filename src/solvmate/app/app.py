from pathlib import Path
import asyncio
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from solvmate.app.handlers.conversion_handlers import (
    MDLMolToSmilesHandler,
    SmilesToMDLMolHandler,
)
from solvmate.app.handlers.cryst_propens_handler import CrystPropensHandler
from solvmate.app.handlers.iupac_input_handler import IupacInputHandler
from solvmate.app.handlers.knn_recommend_handler import KNNRecommendHandler
from solvmate.app.handlers.knn_solub_recommend_handler import KNNSolubRecommendHandler
from solvmate.app.handlers.login_handler import LoginHandler
from solvmate.app.handlers.main_handler import MainHandler
from solvmate.app.handlers.meta_data_handler import MetaDataHandler
from solvmate.app.handlers.smiles_input_handler import SmilesInputHandler
from solvmate.app.handlers.solub_recommend_handler import SolubRecommendHandler
from solvmate.app.handlers.solvent_selection_handler import (
    SolventSelectionFetchHandler,
    SolventSelectionStoreHandler,
)


static_path_dir = Path(__file__).parent


async def main():
    application = tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/SmilesInputHandler", SmilesInputHandler),
            (r"/IupacInputHandler", IupacInputHandler),
            (r"/SolubRecommendHandler", SolubRecommendHandler),
            (r"/KNNRecommendHandler", KNNRecommendHandler),
            (r"/KNNSolubRecommendHandler", KNNSolubRecommendHandler),
            (r"/MetaDataHandler", MetaDataHandler),
            (r"/CrystPropensHandler", CrystPropensHandler),
            (r"/SolventSelectionStoreHandler", SolventSelectionStoreHandler),
            (r"/SolventSelectionFetchHandler", SolventSelectionFetchHandler),
            (r"/MDLMolToSmilesHandler", MDLMolToSmilesHandler),
            (r"/SmilesToMDLMolHandler", SmilesToMDLMolHandler),
            (r"/login", LoginHandler),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path_dir}),
        ],
        autoreload=True,
        login_url="/login",
        cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
    )

    http_server = tornado.httpserver.HTTPServer(
        application,
        ssl_options={
            "certfile": "cert/cert.pem",
            "keyfile": "cert/key.pem",
        },
    )

    http_server.listen(8890)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
