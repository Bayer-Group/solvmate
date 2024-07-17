from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from solvmate import *
from solvmate.ccryst.solvent import iupac_to_smiles
from rdkit.Chem import Descriptors

"""
Script used to generate the training data.
"""


def _canon_else_none(smi: str):
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None

def iupac_to_smiles_else_fail(iup:str):
    smi = iupac_to_smiles(iup)
    if not smi:
        raise ValueError(f"could not convert IUPAC to smiles. IUPAC: {iup}")
    return smi


_density_reg = None
def estimate_density(solvent:str):
    """
    Estimates the density of a given sample using a simple lookup table.

    >>> estimate_density(iupac_to_smiles("pentane")) # doctest:+ELLIPSIS
    0.6262
    >>> estimate_density(iupac_to_smiles("octane"))
    0.6837
    >>> estimate_density(iupac_to_smiles("water"))
    0.9982
    """
    global _density_reg

    tab = """
Pentane	; 0.6262
Hexane ;	0.6594
Heptane; 	0.6837
Iso-Octane;	0.6919
Ethyl Ether;	0.7133
Triethylamine;	0.7276
Methyl t-Butyl Ether;	0.7405
Cyclopentane;	0.7454
Cyclohexane;	0.7785
Acetonitrile;	0.7822
Isopropyl Alcohol;	0.7854
Ethyl Alcohol;	0.7892
Acetone;	0.7900
Methanol;	0.7913
Methyl Isobutyl Ketone;	0.8008
Isobutyl Alcohol;	0.8016
n-Propyl Alcohol;	0.8037
Methyl Ethyl Ketone;	0.8049
Methyl n-Propyl Ketone;	0.8082
n-Butyl Alcohol;	0.8097
Isopropyl Myristate;	0.8532
Toluene;	0.8669
Glyme;	0.8691
n-Butyl Acetate;	0.8796
o-Xylene;	0.8802
n-Butyl Chloride;	0.8862
Methyl Isoamyl Ketone;	0.888
Tetrahydrofuran;	0.888
Ethyl Acetate;	0.9006
Dimethyl Acetamide;	0.9415
N,N-Dimethylformamide;	0.9487
2-Methoxyethanol;	0.9646
Pyridine;	0.9832
Water;	0.9982
N-Methylpyrrolidone;	1.0304
1,4-Dioxane;	1.0336
Dimethyl Sulfoxide;	1.1004
Chlorobenzene;	1.1058
Propylene Carbonate;	1.2006
Ethylene Dichloride;	1.253
o-Dichlorobenzene;	1.3058
Dichloromethane;	1.326
1,2,4-Trichlorobenzene;	1.454
Trifluoroacetic Acid;	1.4890
Chloroform;	1.4892
1,1,2-Trichlorotrifluoroethane;	1.564 
"""
    if _density_reg is None:
        reg = KNeighborsRegressor(n_neighbors=1,)
        X,y = [],[]
        for lne in tab.strip().split("\n"):
            lne = lne.strip().split(";")
            mol = lne[0].strip()
            dens = float(lne[1].strip())
            mol = ecfp_count_fingerprint(Chem.MolFromSmiles(iupac_to_smiles_else_fail(mol)))
            y.append(dens)
            X.append(mol)
        reg.fit(np.vstack(X),y)
        _density_reg = reg
    else:
        reg = _density_reg
 
    if isinstance(solvent,str):
        solvent = Chem.MolFromSmiles(solvent)
    return reg.predict(ecfp_count_fingerprint(solvent).reshape(1,-1))[0]


def load_bao_dataset() -> pd.DataFrame:
    """
    
    >>> df = load_bao_dataset()
    >>> df.columns
    Index(['Web of Science Index', 'Drug', 'Solvent_1',
           'Solvent_1_weight_fraction', 'Solvent_1_mol_fraction', 'Solvent_2',
           'Temperature (K)', 'Solubility (mol/mol)', 'DOI', 'Drugs@FDA', 'CAS',
           'solute SMILES', 'Melting_temp (C)', 'Melting_temp (K)', 'Source_1',
           'Source_2', 'solvent SMILES', 'solvent_frac', 'conc', 'source', 'T'],
          dtype='object')
    >>> df.head()
      Web of Science Index                     Drug          Solvent_1  Solvent_1_weight_fraction  Solvent_1_mol_fraction  ... solvent SMILES  solvent_frac      conc source     T
    0                   36  Guanidine hydrochloride  Dimethylformamide                     0.1001                     NaN  ...  CN(C)C=O.CCCO        0.1001  0.924864    bao   5.0
    1                   36  Guanidine hydrochloride  Dimethylformamide                     0.1001                     NaN  ...  CN(C)C=O.CCCO        0.1001  1.026813    bao  10.0
    2                   36  Guanidine hydrochloride  Dimethylformamide                     0.1001                     NaN  ...  CN(C)C=O.CCCO        0.1001  1.132226    bao  15.0
    3                   36  Guanidine hydrochloride  Dimethylformamide                     0.1001                     NaN  ...  CN(C)C=O.CCCO        0.1001  1.210720    bao  20.0
    4                   36  Guanidine hydrochloride  Dimethylformamide                     0.1001                     NaN  ...  CN(C)C=O.CCCO        0.1001  1.302540    bao  25.0
    <BLANKLINE>
    [5 rows x 21 columns]
    """
    data_fle = DATA_DIR / "20240305_Dataset_Raw_Exp.xlsx"
    assert data_fle.exists()
    df = pd.read_excel(data_fle,sheet_name=0,)
    df_compound = pd.read_excel(data_fle,sheet_name=1,)
    df_compound.rename(columns={"SMILES": "solute SMILES",},inplace=True,)
    df = df.merge(df_compound,on="Drug",)
    df["solvent SMILES"] = [
        ".".join([iupac_to_smiles_else_fail(row["Solvent_1"]),iupac_to_smiles_else_fail(row["Solvent_2"])])
        for _,row in df.iterrows()
    ]
    solvent_frac = []
    conc = []
    for _,row in df.iterrows():
        s1 = Chem.MolFromSmiles(row["solvent SMILES"].split(".")[0])
        s2 = Chem.MolFromSmiles(row["solvent SMILES"].split(".")[1])

        mw1 = Descriptors.MolWt(s1)
        mw2 = Descriptors.MolWt(s2)


        sol_mol = row["Solubility (mol/mol)"]

        if str(row["Solvent_1_weight_fraction"]).lower() == "nan":
            # only mol fraction given. Need to calculate weight fraction
            n1 = row["Solvent_1_mol_fraction"]
            # wf = M1 / (M1 + M2) = M1 / ()
            m1 = n1 * mw1
            m2 = (1 - n1) * mw2

            mw_mix = m1 + m2
            solvent_frac.append(m1 / (mw_mix))
        else:
            weight_frac_1 = row["Solvent_1_weight_fraction"]
            solvent_frac.append(weight_frac_1)

            mw_mix = weight_frac_1 * mw1 + (1-weight_frac_1) * mw2

        # mol / mol  / (g / mol) = mol / g = 1/1000 * mol / kg 
        density_mix = estimate_density(s1) * solvent_frac[-1] + estimate_density(s2) *  ( 1 - solvent_frac[-1]) 
        c = 1000 * sol_mol / mw_mix  * density_mix
        conc.append(c)

    # convert the fractions into simple linear weighting factors that are easier to use
    df["mixture_coefficients"] = [[sf, 1-sf] for sf in solvent_frac]
    df["conc"] = conc
    df["source"] = "bao"
    df["T"] = df['Temperature (K)'] - 273.15
    return df


def load_open_notebook_dataset() -> pd.DataFrame:
    """
    
    >>> df = load_open_notebook_dataset()
    >>> df.columns
    Index([                  'Experiment Number (900 series refer to external references)',
                                                                      'sample or citation',
                                                                                     'ref',
                                                                                  'solute',
                                                                           'solute SMILES',
                                                                                 'solvent',
                                                                          'solvent SMILES',
                                                                       'concentration (M)',
                                                                               'wiki page',
                                                                                    'gONS',
                                                                                   'notes',
                                                                              'identifier',
                                                                             'solute type',
                                                            'solubility - solute mass (g)',
                                                                        'solvent mass (g)',
                                                                  'solvent density (g/ml)',
                                      'solute density (g/ml) - from ChemSpider prediction',
                                                                     'solvent volume (ml)',
                                                                      'solute volume (ml)',
                                                                         'total vol. (ml)',
                                                                               'solute MW',
                                                                            'moles solute',
           'calculated concentration (M)- assumes no expansion or contraction upon mixing',
                                                            'calc. conc. (M) from g/100ml',
                                                               'liquid at room temp (y/n)',
                                                              'solute reacts with solvent',
                                                                                    'csid',
                                                                                      True,
                                                                         'solubility g/l ',
                                                          'calculated conc in moles/liter',
                                                                'solubility mole fraction',
                                                                      'solvent MW (g/mol)',
                                                                                    'conc',
                                                                                  'source'],
          dtype='object')

    """
    data_fle = DATA_FLE_OPEN_NOTEBOOK
    assert data_fle.exists()
    df = pd.read_excel(data_fle)
    df["conc"] = df["concentration (M)"]
    df = df[df["conc"] > 0.0]
    df["conc"] = df["conc"].apply(np.log10)

    df = pd.DataFrame(df)
    df["solvent SMILES"] = df["solvent SMILES"].apply(_canon_else_none)
    df["solute SMILES"] = df["solute SMILES"].apply(_canon_else_none)
    assert df["solvent SMILES"].isna().sum() < len(df) * 0.1
    df["source"] = "open_notebook"

    if os.environ.get("_SM_Y_SCRAMBLING"):
        warn("performing y scrambling")
        conc = df["conc"].tolist()
        random.shuffle(conc)
        df["conc"] = conc

    return df


def load_novartis_dataset():
    dfn = pd.read_csv(DATA_FLE_NOVA)
    solvents_to_iupac = {
        "MeOH": "methanol",
        "EtOH": "ethanol",
        "NiMe": "nitromethane",
        "ACT": "acetone",
        "DMF": "N,N-dimethylformamide",
        "iBMK": "isobutylmethylketone",
        "EA": "ethylacetate",
        "tBME": "tertbutylmethylether",
        "MCB": "chlorobenzene",
        "CHCl3": "chloroform",
        "TOL": "methylbenzene",
        "ACN": "acetonitrile",
        "iPrOH": "isopropanol",
        "EMK": "ethylmethylketone",
        "THF": "tetrahydrofurane",
        "DCM": "dichloromethane",
        "DEE": "diethylether",
        "HEX": "hexane",
    }
    solvent_to_smiles = {
        solvent: opsin_iupac_to_smiles(solvent_iupac)
        for solvent, solvent_iupac in solvents_to_iupac.items()
    }

    rank_map = {
        "RS": 4,
        "KS": 3,
        "TS": 2,
        "PS": 1,
    }
    nan_counter, non_nan_counter = 0, 0
    df_nova = []
    for _, row in dfn.iterrows():
        smiles = row["SMILES"]
        for solv in solvents_to_iupac:
            conc = row[f"S-{solv}"]
            if str(conc).lower() == "nan":
                nan_counter += 1
                # print("encountered nan")
                continue
            else:
                non_nan_counter += 1
            assert conc in rank_map, f"expected {conc} <= {sorted(rank_map.keys())}"
            conc = rank_map[conc]
            solvent_smiles = solvent_to_smiles[solv]
            solvent_smiles = Chem.CanonSmiles(solvent_smiles)
            smiles = _canon_else_none(smiles)
            df_nova.append(
                {
                    "smiles": smiles,
                    "solute SMILES": smiles,
                    "conc": conc,
                    "solvent SMILES": solvent_smiles,
                    "solvent": solvent_smiles,
                }
            )

    df_nova = pd.DataFrame(df_nova)
    # df_nova = df_nova[
    #    (~df_nova["mol_compound"].isna()) & (~df_nova["mol_solvent_mixture"].isna())
    # ]
    df_nova["source"] = "nova"

    if os.environ.get("_SM_Y_SCRAMBLING"):
        warn("performing y scrambling")
        conc = df_nova["conc"].tolist()
        random.shuffle(conc)
        df_nova["conc"] = conc
    return df_nova


def filter_solvents_by_threshold(td, thresh=100):
    # Currently: only taking those solvents that
    # have more than thresh examples in open_notebook database
    top_solvents = []
    td = td[td.source == "open_notebook"]
    for solv, count in td.value_counts("solvent SMILES").iloc[0:10000].iteritems():
        if count > thresh:
            top_solvents.append(solv)
        else:
            break
    return top_solvents


def filter_relevant_subset(df: pd.DataFrame, N_thresh: int) -> pd.DataFrame:
    """
    Applies the following idea to generate better data:
    If we only have very few measurements, then a datapoint
    really isn't that useful.
    Therefore, we required at least 5 measurements to be
    present for a compound to consider it, aka there need
    to be 5 different solvents measured for a given
    compound to be included in the dataset.
    I saw strong improvements in the ood evaluation (kendalltau doubled!)
    Hence, this seems to be a useful heuristic.
    """

    acceptable = {
        key
        for key, count in df["solute SMILES"].value_counts().to_dict().items()
        if count > N_thresh
    }
    return df[df["solute SMILES"].isin(acceptable)]


if __name__ == "__main__":
    out_file = DATA_DIR / "training_data.db"
    if not out_file.exists():
        con = sqlite3.connect(out_file)
        df = pd.concat(
            [
                load_novartis_dataset(),
                load_open_notebook_dataset(),
                load_bao_dataset(),
            ]
        )

        print(f"Before nan filter: {len(df)}")
        df = df[(~df["solvent SMILES"].isna()) & (~df["solute SMILES"].isna())]
        print(f"After nan filter: {len(df)}")

        # default solvent mixture is just the solvent with a factor 1.
        # if there are more than one solvents, each of them get's a
        # factor 1, so e.g. H20:MeOH 1:1 = 1*MeOH + 1*H2O
        mcs = []
        for _,row in df.iterrows():
            if "na" in str(row["mixture_coefficients"]).lower():
                parts = row["solvent SMILES"].split(".")
                N = len(parts)
                mcs.append([1/N for _ in parts])
            else:
                mcs.append(row["mixture_coefficients"])
        df["mixture_coefficients"] = mcs

        df["T"] = df["T"].fillna(25)
        dont_use = []
        for _, row in df.iterrows():
            blob = str(row).lower().replace(" ", "").replace("\t", "").replace("\n", "")
            if "donotuse" in blob:
                dont_use.append(True)
            else:
                dont_use.append(False)
        df["dont_use"] = dont_use

        # top_solvents = filter_solvents_by_threshold(df)
        # df = df[df["solvent SMILES"].isin(top_solvents)]

        if os.environ.get("_SM_ONLY_COSMO_CALCULATED_SMILES"):
            warn("FILTERING OUT ONLY COSMO-CALCULATED SMILES:")
            warn("before: " + str(len(df)))
            from cosmonaut import smiles_to_id, id_to_cosmo_file_path

            solu_ids = df["solute SMILES"].apply(smiles_to_id)
            solv_ids = df["solvent SMILES"].apply(smiles_to_id)
            solu_fles = [id_to_cosmo_file_path(mol_id) for mol_id in solu_ids]
            solv_fles = [id_to_cosmo_file_path(mol_id) for mol_id in solv_ids]
            cosmo_filter = [
                bool(solu_fle) and bool(solv_fle)
                for solu_fle, solv_fle in zip(solu_fles, solv_fles)
            ]

            df = df[cosmo_filter]
            warn("after: " + str(len(df)))
        else:
            warn("NOT FILTERING OUT THE COSMO-CALCULATED SMILES ONLY ")

        # Used this for comparing with COSMO-RS results.
        df = filter_relevant_subset(
            df,
            N_thresh=3,
        )
        # smiles_filtered = joblib.load("/tmp/smiles_filtered.pkl")
        # df = df[df["solute SMILES"].isin(smiles_filtered)]

        if os.environ.get("_PR_FAST_DEBUG"):
            print("WARNING: fast debug mode enabled. sampling very small subset")
            df = df.sample(len(df) // 10)

        if os.environ.get("_PR_DOWNSAMPLE_BY"):
            fac = float(os.environ.get("_PR_DOWNSAMPLE_BY"))
            print("WARNING: downsampling by factor", fac)

            df = df.sample(int(len(df) / fac))

        df['mixture_coefficients'] = df['mixture_coefficients'].apply(str)
        df.to_sql("training_data", con)
        con.close()
    else:
        warn(f"file {out_file} exists. skipping step {__file__}")
