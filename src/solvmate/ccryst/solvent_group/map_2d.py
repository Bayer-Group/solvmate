
"""
Provides a simple 2D solvent embedding based on the physical properties described within the paper
`Grouping solvents by statistical analysis of solvent property parameters: implication to polymorph screening`
by Chong-Hui, Gu; Hua Li, Rajesh; Gandhia Krishnaswamy, Raghavan
(https://www.sciencedirect.com/science/article/pii/S0378517304004193)
"""
import os
import pathlib

import pandas as pd


def load_phys_solv_table(fle=None,sep="\t") -> pd.DataFrame:
    """
    Loads the physical solvent descriptors table as described in the paper
    `Grouping solvents by statistical analysis of solvent property parameters: implication to polymorph screening`
    by Chong-Hui, Gu; Hua Li, Rajesh; Gandhia Krishnaswamy, Raghavan
    (https://www.sciencedirect.com/science/article/pii/S0378517304004193)

    >>> load_phys_solv_table().values.shape
    (96, 9)
    """
    if fle is None:
        fle = pathlib.Path(os.path.dirname(__file__)) / "map2d.txt"
    text = fle.read_text()
    dct = { }
    first_line = True
    header = None
    for lne in text.split("\n"):
        lne = lne.strip()
        if not lne or lne.startswith("#"): continue

        if first_line:
            header = lne.split()
            for col in header:
                dct[col] = []
            first_line = False
        else:
            vals = lne.split(sep)
            if len(header) != len(vals):
                raise ValueError(f"Dimension mismatch for line `{lne}`")
            for col,val in zip(header,vals):
                try:
                    dct[col].append(float(val.replace("−","-")))
                except:
                    dct[col].append(val)
    return pd.DataFrame(dct)


