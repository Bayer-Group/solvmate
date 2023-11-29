from pathlib import Path
import tornado
import tornado.ioloop
import tornado.web
from solvmate import *
from solvmate.app.handlers.base_handler import BaseHandler


BACKBONE = Path(__file__).parent.parent / "backbone.html"


class MainHandler(BaseHandler):
    @log_function_call
    def read(self):
        with open(BACKBONE) as reader:
            return reader.read()

    @log_function_call
    @tornado.web.authenticated
    def get(self):
        self.write(self.read())
