from cosmonaut import *
from cosmonaut import cosmo_calc, cosmors_calc


smiles_list = [
    "CNCC",
    "COCC",
]
names_list = [smiles_to_id(smi) for smi in smiles_list]


def test_run_cosmo_calculations():
    cosmo_calc.run_cosmo_calculations(
        smiles_list=smiles_list,
        names_list=names_list,
        charges_list=[0, 0],
        outputs_dir=CM_DATA_DIR,
    )
    assert (
        CM_DATA_DIR / names_list[0]
    ).exists(), "expected an output to be created for first job"
    assert (
        CM_DATA_DIR / names_list[1]
    ).exists(), "expected an output to be created for second job"


def test_run_cosmors_calculations():
    fle_a = list((CM_DATA_DIR / names_list[0] / "COSMO").iterdir())[0]
    fle_b = list((CM_DATA_DIR / names_list[1] / "COSMO").iterdir())[0]
    assert fle_a.exists() and fle_b.exists(), "precond failed"
    cosmors_calc.run_cosmors_calculations(
        fle_solvent=fle_a,
        fle_solute=fle_b,
    )
