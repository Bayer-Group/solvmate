from rdkit.Chem import rdMolDescriptors

from solvmate.ccryst.utils import safe 


def apply_descriptors(df,
        mol_col="mol_compound",
        descriptors:list[str]=None) -> None:

    if descriptors is None:
        descriptors = [
            descriptor for descriptor in dir(rdMolDescriptors)
            if descriptor.startswith("Calc")
        ]


    for descriptor in descriptors:
        try:
            descriptor_fun = getattr(rdMolDescriptors,descriptor)
            df[descriptor] = df[mol_col].apply(descriptor_fun)
        except:
            continue

    return None





