from cosmonaut import *
from cosmonaut.cosmo_calc import run_cosmo_calculations


def main():
    data = pd.read_excel(CM_DATA_DIR / "20150430SolubilitiesSum.xlsx")
    data["is_valid"] = ~(data["solute SMILES"].apply(Chem.MolFromSmiles).isna()) & ~(
        data["solvent SMILES"].apply(Chem.MolFromSmiles).isna()
    )
    data = data[data["is_valid"]]
    all_smiles = data["solute SMILES"].tolist() + data["solvent SMILES"].tolist()
    all_smiles = list(set(all_smiles))
    for smi in all_smiles:
        print(smi)
        Chem.CanonSmiles(smi)
    # all_smiles = [Chem.CanonSmiles(smi) for smi in all_smiles if smi]
    all_smiles = list(set(all_smiles))
    print(f"{len(all_smiles)=}")

    smiles_of_interest = []
    for smi in data["solvent SMILES"].unique():
        smiles_of_interest.append(smi)

    for smi in data["solute SMILES"].unique():
        g = data[data["solute SMILES"] == smi]
        if g["solvent SMILES"].nunique() > 5:
            smiles_of_interest.append(smi)

    print(f"{len(smiles_of_interest)=}")

    names_of_interest, charges_of_interest = [], []
    for smi in smiles_of_interest:
        name = smiles_to_id(smi)
        charge = smiles_to_charge(smi)
        names_of_interest.append(name)
        charges_of_interest.append(charge)

    assert len(smiles_of_interest) == len(charges_of_interest)
    assert len(names_of_interest) == len(charges_of_interest)

    run_cosmo_calculations(
        smiles_list=smiles_of_interest,
        names_list=names_of_interest,
        charges_list=charges_of_interest,
        outputs_dir=CM_DATA_DIR,
        n_cores_inner=1,
        n_cores_outer=8,
    )


if __name__ == "__main__":
    main()
