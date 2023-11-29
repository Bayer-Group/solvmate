from solvmate import *

"""
A script that downloads and extracts the 
nextmove public-domain patent data by lowe et al..

Only needed in the rare case that somebody would be
interested in the crystallization conditions 
similarity search. As we expect only a very small
minority of users to be interested in this aspect
of the application, this data
---which is occopies 10s of GB on disc---
is only fetched on-demand.

Sadly, this data is only made available in 7-Zip format,
so users / developers interested in exposing the
crystallization conditions search functionality also
have to install 7-Zip binaries and add them on their PATH.
"""


def download_metadata():
    final_dest = (DATA_DIR / "meta_db").absolute()
    final_dest.mkdir(exist_ok=True)
    link_figshare = "https://figshare.com/ndownloader/articles/5104873/versions/1"

    d = temp_dir() / "mdown"
    d.mkdir(exist_ok=True)

    zip_fle = d / "1.zip"
    if not zip_fle.exists():
        info(f"downloading the metadata from {link_figshare}")
        download_file(url=link_figshare, dest=str(zip_fle))
    else:
        info(f"already have file {zip_fle}")

    info(f"file size of original zip: {zip_fle.stat().st_size / 2**20} MB")

    def move_all_xmls_recursive(cur_dir):
        for xml_fle in cur_dir.glob("*.xml"):
            xml_fle.rename(final_dest / xml_fle.name)
        for sub_dir in cur_dir.iterdir():
            if sub_dir.is_dir():
                move_all_xmls_recursive(sub_dir)

    with working_directory(d):
        assert len(list(d.glob("*.zip"))) >= 1
        os.system("unzip 1.zip")
        assert len(list(d.glob("*.7z"))) > 0

        for arch_7z in d.glob("*.7z"):
            out_dir = Path(str(arch_7z).replace(".7z", ""))
            out_dir.mkdir(exist_ok=True)
            os.system(f"7za e {arch_7z} -o{out_dir}") # nosec

        move_all_xmls_recursive(d)


if __name__ == "__main__":
    download_metadata()
