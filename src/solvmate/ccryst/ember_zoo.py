
"""
Module providing convenient access
to commonly used embeddings.
"""

import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors3D
from solvmate.ccryst import chem_utils

def run_2d_descriptors(mol,descriptors_2d=None,):

    if descriptors_2d is None:
        descriptors_2d = [
            'CalcChi0n',
            'CalcChi0v',
            'CalcChi1n',
            'CalcChi1v',
            'CalcChi2n',
            'CalcChi2v',
            'CalcChi3n',
            'CalcChi3v',
            'CalcChi4n',
            'CalcChi4v',
            'CalcExactMolWt',
            'CalcFractionCSP3',
            'CalcHallKierAlpha',
            'CalcKappa1',
            'CalcKappa2',
            'CalcKappa3',
            'CalcLabuteASA',
            'CalcNumAliphaticCarbocycles',
            'CalcNumAliphaticHeterocycles',
            'CalcNumAliphaticRings',
            'CalcNumAmideBonds',
            'CalcNumAromaticCarbocycles',
            'CalcNumAromaticHeterocycles',
            'CalcNumAromaticRings',
            #'CalcNumAtomStereoCenters',
            'CalcNumAtoms',
            'CalcNumBridgeheadAtoms',
            'CalcNumHBA',
            'CalcNumHBD',
            'CalcNumHeavyAtoms',
            'CalcNumHeteroatoms',
            'CalcNumHeterocycles',
            'CalcNumLipinskiHBA',
            'CalcNumLipinskiHBD',
            'CalcNumRings',
            'CalcNumRotatableBonds',
            'CalcNumSaturatedCarbocycles',
            'CalcNumSaturatedHeterocycles',
            'CalcNumSaturatedRings',
            'CalcNumSpiroAtoms',
            #'CalcNumUnspecifiedAtomStereoCenters',
            'CalcPhi',
            'CalcTPSA',
            ]


    desc_fp = []
    for desc_name in sorted(descriptors_2d):
        desc = getattr(rdMolDescriptors,desc_name)
        desc_fp.append(desc(mol))
    return np.array(desc_fp)


def run_3d_descriptors(mol):
    fp = []
    for attrn in sorted(dir(Descriptors3D)):
        if attrn.startswith("_") or attrn == "rdMolDescriptors": continue
        attr = getattr(Descriptors3D,attrn)
        if mol.GetNumConformers() == 0:
            mol = chem_utils.embed_conformers_cached(mol)
        fp.append(attr(mol))
    return np.array(fp)