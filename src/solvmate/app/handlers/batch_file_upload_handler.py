from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler
import time


class BatchFileUploadHandler(BaseHandler):
    # see https://techoverflow.net/2015/06/09/upload-multiple-files-to-the-tornado-webserver/
    @log_function_call
    def post(self):
        files = []
        # check whether the request contains files that should get uploaded
        try:
            files = self.request.files["files"]
        except:
            pass

        for xfile in files:
            file = xfile["filename"]
            # the filename should not contain any "evil" special characters
            # basically "evil" characters are all characters that allows you to break out from the upload directory
            index = file.rfind(".")
            filename = (
                file[:index].replace(".", "")
                + str(time.time()).replace(".", "")
                + file[index:]
            )
            filename = filename.replace("/", "")
            # save the file in the upload folder

            print("$$$")
            print(xfile["body"])
            print("$$$")

            # At this point we have collected all the input information.
            # We now generate a job_id for this batch input.
            # This job_id url is to be reported back to the user, e.g. /job/123/
            # So that the user has the possibility to track the progress of
            # the job.

            continue
