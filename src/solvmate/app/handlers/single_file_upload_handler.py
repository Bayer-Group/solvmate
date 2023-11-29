from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler
import time


class SingleFileUploadHandler(BaseHandler):
    # see https://techoverflow.net/2015/06/09/upload-multiple-files-to-the-tornado-webserver/
    @log_function_call
    def post(self):
        failure = json.dumps({"status": "failure", "smiles": ""})
        req = json.loads(self.request.body)
        print("************")
        print(req)
        print("************")
        suf = Path(req["filename"]).suffix[1:]
        if not suf:
            self.write(failure)
            return
        fle = random_fle(suf=suf)
        fle.write_text(req["content"])
        smiles = obabel_to_smiles(fle)
        resp = json.dumps(
            {
                "status": "success",
                "smiles": smiles,
            }
        )
        self.write(resp)
