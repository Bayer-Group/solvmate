import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid


class AuthHandler:
    """
    A class that handles the authentication of users into
    our system.

    reference: https://nitratine.net/blog/post/how-to-hash-passwords-in-python/

    >>> import random
    >>> ah = AuthHandler(f"/tmp/__auth_handler{random.randint(10000000,100000000)}.tmp.json")
    >>> ah.add_user("tim", "123")
    True
    >>> ah.check_login("tim","321")
    False
    >>> ah.check_login("tim","123")
    True
    >>> ah.check_login("tim","")
    False
    >>> ah.check_login("","tim")
    False
    >>> ah.check_login("","")
    False
    >>> ah.check_login("linda","")
    False
    >>> ah.add_user("linda","")
    False
    >>> ah.add_user("linda","securepassword")
    True
    >>> ah.check_login("tim","securepassword")
    False
    >>> ah.check_login("linda","securepassword")
    True
    >>> ah.check_login("linda","123")
    False
    >>> ah.check_login("tim","123")
    True

    """

    def __init__(self, users_db):
        self.users_db = users_db
        self.users = {}

    def salt_as_str(self, salt):
        salt = base64.b64encode(salt).decode("ascii")
        return salt

    def salt_as_bytes(self, salt):
        salt = base64.b64decode(salt.encode("ascii"))
        return salt

    def add_user(self, username: str, password: str):
        if len(username) < 3 or len(password) < 3:
            return False

        salt = self.salt_as_str(os.urandom(32))
        key = self.hash_password(password, salt)
        self.users[username] = {"salt": salt, "key": key}  # Store the salt and key
        self.save_users_db()

        return True

    def check_login(self, username: str, password: str) -> bool:
        if username not in self.users:
            return False

        salt = self.users[username]["salt"]  # Get the salt
        key = self.users[username]["key"]  # Get the correct key
        login_key = self.hash_password(password, salt)
        return key == login_key

    def hash_password(self, password, salt) -> str:
        salt = self.salt_as_bytes(salt)
        key = base64.b64encode(
            hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        ).decode("utf8")
        return key

    def save_users_db(
        self,
    ):
        with open(self.users_db, "wt") as fle:
            json.dump(obj=self.users, fp=fle)

    def load_users_db(
        self,
    ) -> dict:
        if Path(self.users_db).exists():
            with open(self.users_db, "rt") as fle:
                self.users = json.load(fle)
        else:
            self.users = {}

    def create_uuid_for_user(
        self,
        username: str,
        password: str,
    ) -> str:
        if not self.check_login(username=username, password=password):
            return ""

        user = self.users[username]
        user["uuid"] = str(uuid.uuid4())
        self.save_users_db()
        return user["uuid"]

    def check_uuid_for_user(
        self,
        username: str,
        uuid: str,
    ) -> bool:
        if not uuid or username not in self.users:
            return False
        else:
            user = self.users[username]
            return "uuid" in user and user["uuid"] == uuid


# Instances for testing, replaced by secret script:
auth_instance = AuthHandler(Path(__file__).parent / ".auth.json")
auth_instance.load_users_db()
auth_instance.add_user("user", "solvmate")

# Use the following, e.g. in a private script to create new users:
# auth_instance.add_user("admin", "<insert-password here>")

if __name__ == "__main__":
    if sys.argv[1] == "add":
        if len(sys.argv) == 4:
            user = sys.argv[-2]
            pwd = sys.argv[-1]
            success = auth_instance.add_user(user, pwd)
            if success:
                sys.exit(0)
            else:
                print("failed to add user")
                sys.exit(1)
        else:
            print("usage: python auth.py add <username> <password>")
            exit(1)
