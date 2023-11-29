import tornado

from solvmate.app.auth import auth_instance


class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        session_id = self.get_secure_cookie("sm_session_id")
        username = self.get_secure_cookie("sm_name")
        if session_id is None:
            session_id = b""
        if username is None:
            username = b""
        session_id, username = str(session_id, "utf-8"), str(username, "utf-8")

        if auth_instance.check_uuid_for_user(username=username, uuid=session_id):
            return username
