
from solvmate import *
from shutil import unpack_archive

def run():
    zip_file = random_fle("zip")
    download_file(
        "https://github.com/Bayer-Group/solvent-mate/releases/download/v0.1/public_data_models.zip",
        zip_file
    )
    unpack_archive(zip_file,DATA_DIR)



if __name__ == "__main__":
    run()