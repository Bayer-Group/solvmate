import seaborn as sns
from smal.all import *
from model import run_predictions_for_solvents
from scipy.stats import spearmanr
if __name__ == "__main__":
    here = Path(__file__).parent
    df = pd.read_csv(here / "data" / "training_data_singleton.csv")
    df = df[df["source"] == "nova"]
    df["exp log S"] = df["conc"]
    spears = []
    solutes = list(df["solute SMILES"].unique())
    for solu in solutes:
        g_solu = df[df["solute SMILES"] == solu]
        preds = run_predictions_for_solvents(solu, g_solu["solvent SMILES"].tolist(),)
        g_solu = g_solu.merge(preds,on=["solvent SMILES",])
        spears.append(
            spearmanr(g_solu["exp log S"], g_solu["log S"])[0]
        )

    stats = pd.DataFrame({"spearman": spears, "solute": solutes,})
    plt.clf()
    sns.boxplot(data=stats,y="spearman",)
    here = Path(__file__).parent
    plt.savefig(here / "results" / "spearmanr_on_nova_boxplot.svg")
    plt.clf()
    sns.violinplot(data=stats,y="spearman",)
    here = Path(__file__).parent
    plt.savefig(here / "results" / "spearmanr_on_nova_violinplot.svg")
    plt.clf()

    print("mean spears:",stats["spearman"].mean())

    stats = stats.sort_values("spearman",ascending=False,)
    print("top 10:")
    for _,row in stats.iloc[0:10].iterrows():
        print( 
            row["solute"],
            row["spearman"]
        )

    stats = stats.sort_values("spearman",ascending=True,)
    print("bottom 10:")
    for _,row in stats.iloc[0:10].iterrows():
        print( 
            row["solute"],
            row["spearman"]
        )