

import random
import tqdm
import time
from solvmate import *
from solvmate.ranksolv.featurizer import CDDDFeaturizer, CosmoRSFeaturizer, XTBFeaturizer

def safe_canon_smiles(smi):
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None

if __name__ == "__main__":
    seed = 123
    n_samples = 1
    random.seed(seed)
    compounds = ["".join(random.choices("CCCCCCON",k=size,)) for size in range(3,10) for n_samples in range(3)]
    
    compounds = [safe_canon_smiles(comp) for comp in compounds]
    compounds = [comp for comp in compounds if comp]

    feats = [XTBFeaturizer(
                phase="train",
                pairwise_reduction="diff",
                feature_name="xtb",
                skip_calculations=False,
            ),
            CDDDFeaturizer(feature_name="cddd",phase="train",pairwise_reduction="diff"),
            CosmoRSFeaturizer(
                phase="train",
                pairwise_reduction="diff",
                feature_name="cosmors",
                skip_calculations=False,
            ),
    ]
    runtime_stats = []
    for comp in tqdm.tqdm(compounds):
        for feat in feats:
            start = time.time()
            if hasattr(feat,"run_solute_solvent"):
                feat.run_solute_solvent(compounds=[comp],solvents=["CO",])
            else:
                feat.run_single([comp,"CO",])
            end = time.time()
            runtime_stats.append({
                "feature_name": feat.feature_name,
                "n_atoms": Chem.MolFromSmiles(comp).GetNumAtoms(),
                "runtime / seconds": end - start,
                })

    runtime_stats = pd.DataFrame(runtime_stats)
    import seaborn as sns
    sns.lineplot(data=runtime_stats,x="n_atoms",y="runtime / seconds",hue="feature_name",)
    plt.savefig( DATA_DIR / "runtime_stats.svg")
    plt.clf()
            
            
    
