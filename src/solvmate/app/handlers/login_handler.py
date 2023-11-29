import random
import time

from solvmate import log_function_call
from solvmate.app.auth import auth_instance
from solvmate.app.handlers.base_handler import BaseHandler


class LoginHandler(BaseHandler):
    @log_function_call
    def get(self):
        self.write(
            '<html><body><form action="/login" method="post">'
            'Name: <input type="text" name="name">'
            'Password: <input type="password" name="pwd">'
            '<input type="submit" value="Sign in">'
            "</form>"
            """
            <script>
            function deleteAllCookies() {
                const cookies = document.cookie.split(";");

                for (let i = 0; i < cookies.length; i++) {
                    const cookie = cookies[i];
                    const eqPos = cookie.indexOf("=");
                    const name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
                    document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
                }
            }
            deleteAllCookies();
            </script>
            """
            "</body></html>"
        )

    @log_function_call
    def post(self):
        # artificially slow down login to avoid brute forcing,
        # and timing attacks ...
        time.sleep(random.randint(1, 5))
        name = self.get_argument("name")
        pwd = self.get_argument("pwd")
        if auth_instance.check_login(username=name, password=pwd):
            session_id = auth_instance.create_uuid_for_user(username=name, password=pwd)
            if session_id:
                self.set_secure_cookie("sm_session_id", session_id)
                self.set_secure_cookie("sm_name", name)
                self.redirect("/")
        else:
            self.redirect("/login")
