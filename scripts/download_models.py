
from shutil import unpack_archive

def run():
    zip_file = "/tmp/models.zip"
    download_file(
        "https://github.com/Bayer-Group/solvent-mate/releases/download/v0.1/public_data_models.zip",
        zip_file
    )
    unpack_archive(zip_file,"/solvmate/data")



if __name__ == "__main__":
    run()