from solvmate import *
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

def _post_process_svg(fle_out:Path):
    """
    post processes the given svg to make the figure
    easier to digest visually by highlighting the
    pairwise reduction parts
    """
    svg = fle_out.read_text()

    for pat,repl in [
        ("__diff","""<tspan dx="5" fill="gray"> (-) </tspan>"""),
        ("__concat","""<tspan dx="5" fill="gray"> (+) </tspan>"""),
        ]:
        while pat in svg:
            svg = svg.replace(pat,repl)
    fle_out.write_text(svg)

if __name__ == "__main__":
    con = sqlite3.connect(DATA_DIR / "observations.db")
    df = pd.read_sql("SELECT * FROM stats",con=con)
    # These are the cols in the dataframe:
    # print(df.columns)
    # Index(['index', 'reg', 'featurizer', 'feature_name', 'pairwise_reduction',
    # 'to_abs_strat', 'mean_spear_r_test', 'std_spear_r_test',
    # 'mean_kendalltau_r_test', 'std_kendalltau_r_test', 'ood_mean_spear_r',
    # 'ood_mean_kendall_tau_r', 'created_at', 'job_name'],
    # dtype='object')


    # these here I have manually computed by following the instructions
    # from this notebook here:
    # https://github.com/fhvermei/SolProp_ML/blob/main/sample_files/example_with_csv_inputs.ipynb

    df_vermeire = pd.DataFrame({
        "job_name":["GS_PAPER_FEATURES_BAYER_OOD","GS_PAPER_FEATURES_NOVA_OOD","GS_PAPER_FEATURES",],
        "ood_mean_spear_r": [0.612,0.554,None,],
        "mean_spear_r_test": [0.645,0.645,0.645,]
    })
    df_vermeire["reg"] = ""
    df_vermeire["feature_name"] = "vermeire"
    df_vermeire["to_abs_strat"] = "N/A"
    df_vermeire["pairwise_reduction"] = ""

    df = pd.concat([df,df_vermeire,])

    df["feature_name__pairwise_reduction"] = df["feature_name"]+"__"+df["pairwise_reduction"]
    for job_name in ['GS_PAPER_FEATURES_BAYER_OOD', 'GS_PAPER_FEATURES_NOVA_OOD',
       'GS_PAPER_FEATURES_BUTINA', 'GS_PAPER_FEATURES_SOLVENT_CV',
       'GS_PAPER_MODELS', 'GS_PAPER_FEATURES',]:
        sns.set_theme(style='darkgrid',palette="Paired")
        dfs = df[df.job_name == job_name] 

        # remove the ecfp__solv and hybrid features because they don't add any new info:
        dfs = dfs[dfs.feature_name.apply(lambda fn: (not "solv" in fn) and (not "hybrid" in fn))]

        dfs = dfs.sort_values(["feature_name__pairwise_reduction","to_abs_strat",])

        if job_name == "GS_PAPER_FEATURES":
            dfs = df[df.job_name == "GS_PAPER_FEATURES_BAYER_OOD"] # doesn't matter because test eval stays
            dfs = dfs[dfs.feature_name.apply(lambda fn: (not "solv" in fn) and (not "hybrid" in fn))]
            dfs = dfs.sort_values(["feature_name__pairwise_reduction","to_abs_strat",])
            sns.barplot(
                data=dfs,
                y="feature_name__pairwise_reduction",
                x="mean_spear_r_test",
                hue="to_abs_strat",    
            )
        elif "models" in job_name.lower(): 
            sns.barplot(
                data=dfs,
                y="reg",
                x="mean_spear_r_test",
                hue="to_abs_strat",    
            )
        else:
            sns.barplot(
                data=dfs,
                y="feature_name__pairwise_reduction",
                x="ood_mean_spear_r" if "ood" in job_name.lower() else "mean_spear_r_test",
                hue="to_abs_strat",    
            )
        
        plt.tight_layout()
        plt.title(job_name)
        fle_out = DATA_DIR / f"paper_{job_name}.svg"
        plt.savefig(fle_out)
        plt.clf()
        _post_process_svg(fle_out)
        os.system(f"open {fle_out}")


