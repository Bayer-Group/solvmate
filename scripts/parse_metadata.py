from solvmate import *
from solvmate.ccryst.metadat import parse_reaction_xml
import tqdm

"""
Parses the metadata and serializes it into the lowe meta 
sqlite database.

This step is needed to make the metadata available for
the crystallization conditions similarity search.
As such it is affecting only a very small percentage
of the intended user base.
"""


def run():
    con = sqlite3.connect(LOWE_META_DB)
    con.execute("DROP TABLE IF EXISTS recipes")

    all_xmls = list((DATA_DIR / "meta_db").glob("*.xml"))
    for xml in tqdm.tqdm(all_xmls):
        xml: Path
        reacts = list(parse_reaction_xml(xml.read_text()))
        reacts = pd.DataFrame([react.to_dict() for react in reacts])
        reacts.to_sql("recipes", con=con, if_exists="append")

    con.close()


if __name__ == "__main__":
    run()
