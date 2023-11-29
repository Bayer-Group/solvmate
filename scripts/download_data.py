from solvmate import *


def run():

    md5_nova = "f81fff2127b9d6f3b8034b6c9b0e2dc5"

    if not DATA_FLE_NOVA.exists():
        download_file(
            "https://www.rsc.org/suppdata/c7/ce/c7ce00738h/c7ce00738h6.csv",
            str(DATA_FLE_NOVA),
        )

    assert (
        str(hashlib.md5(DATA_FLE_NOVA.read_bytes(),usedforsecurity=False,).hexdigest()).strip() == md5_nova
    ), f"unexpected hashsum. please check integrity of downloaded nova file at {DATA_FLE_NOVA}"


if __name__ == "__main__":
    run()
