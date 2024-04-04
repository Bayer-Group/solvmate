from solvmate import *
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def _boldify(svg:str)->str:
    return '''<tspan font-weight="bold">'''+svg+'''</tspan>'''


def _post_process_svg(fle_out:Path):
    """
    post processes the given svg to make the figure
    easier to digest visually by highlighting the
    pairwise reduction parts
    """
    svg = fle_out.read_text()

    for pat,repl in [
        ("ood_mean_spear_r", "Spearman's ρ"),
        ("mean_spear_r_test", "Spearman's ρ"),
        (
            "GS3_PAPER_FEATURES_BAYER_OOD", _boldify("ood eval bayer")
        ),
        (
            "GS3_PAPER_FEATURES_NOVA_OOD", _boldify("ood eval novartis")
        ),
        (
            "GS3_PAPER_FEATURES_SOLVENT_CV", _boldify("solvent splits")
        ),
        (
            "GS3_PAPER_MODELS", _boldify("regressor models")
        ),
        (
            "GS3_PAPER_FEATURES_BUTINA", _boldify("butina clustering")
        ),
        (
            "GS3_PAPER_FEATURES", _boldify("random splits")
        ),
        ("feature_name", "feature name"),
        ]:
        while pat in svg:
            svg = svg.replace(pat,repl)
    fle_out.write_text(svg)

if __name__ == "__main__":
    con = sqlite3.connect(DATA_DIR / "observations_paper.db")
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
        "job_name":["GS3_PAPER_FEATURES_BAYER_OOD","GS3_PAPER_FEATURES_NOVA_OOD","GS3_PAPER_FEATURES",],
        "ood_mean_spear_r": [0.612,0.554,None,],
        "mean_spear_r_test": [0.645,0.645,0.645,]
    })
    df_vermeire["reg"] = ""
    df_vermeire["feature_name"] = "vermeire"
    df_vermeire["to_abs_strat"] = "absolute"
    df_vermeire["pairwise_reduction"] = ""

    df = pd.concat([df,df_vermeire,])

    #df["feature_name__pairwise_reduction"] = df["feature_name"]+"__"+df["pairwise_reduction"]
    df["to_abs_strat"] = [
        "absolute" if row["to_abs_strat"] == "absolute"
        else 
        "relative_diff" if row["pairwise_reduction"] == "diff"
        else "relative_concat" if row["pairwise_reduction"] == "concat"
        else "N/A"
        for _,row in df.iterrows()
    ]

    job_names = [
            'GS3_PAPER_MODELS',
            'GS3_PAPER_FEATURES',
            'GS3_PAPER_FEATURES_BUTINA',
            'GS3_PAPER_FEATURES_SOLVENT_CV',
            'GS3_PAPER_FEATURES_NOVA_OOD',
            'GS3_PAPER_FEATURES_BAYER_OOD', 
         ]
    short_name = {

            'GS3_PAPER_MODELS':"model",
            'GS3_PAPER_FEATURES':"rand",
            'GS3_PAPER_FEATURES_BUTINA':"butina",
            'GS3_PAPER_FEATURES_SOLVENT_CV':"solvent",
            'GS3_PAPER_FEATURES_NOVA_OOD':"ood(novartis)",
            'GS3_PAPER_FEATURES_BAYER_OOD':"ood(bayer)", 
    }

    one_latex = None

    for job_name in job_names:
        sns.set_theme(style='darkgrid',)#palette="Paired")
        dfs = df[df.job_name == job_name] 

        # remove the ecfp__solv and hybrid features because they don't add any new info:
        dfs = dfs[dfs.feature_name.apply(lambda fn: (not "solv" in fn) and (not "hybrid" in fn))]

        dfs = dfs.sort_values(["feature_name","to_abs_strat",])

        colors = ["#797979f7", "#3485fdf7", "#4ab563f7",]
        # Set your custom color palette
        sns.set_palette(sns.color_palette(colors))
        if job_name == "GS3_PAPER_FEATURES":
            dfs = df[df.job_name == "GS3_PAPER_FEATURES_BAYER_OOD"] # doesn't matter because test eval stays
            dfs = dfs[dfs.feature_name.apply(lambda fn: (not "solv" in fn) and (not "hybrid" in fn))]
            dfs = dfs.sort_values(["feature_name","to_abs_strat",])
            sns.barplot(
                data=dfs,
                y="feature_name",
                x="mean_spear_r_test",
                hue="to_abs_strat",    
            )
        elif "models" in job_name.lower(): 
            dfs["reg"] = dfs["reg"].apply(lambda reg: reg.split("(")[0])
            colors = [colors[0],colors[-1]]
            # Set your custom color palette
            sns.set_palette(sns.color_palette(colors))
            sns.barplot(
                data=dfs,
                y="reg",
                x="mean_spear_r_test",
                hue="to_abs_strat",    
            )
        else:
            sns.barplot(
                data=dfs,
                y="feature_name",
                x="ood_mean_spear_r" if "ood" in job_name.lower() else "mean_spear_r_test",
                hue="to_abs_strat",    
            )

        latex = dfs.copy()
        spear_col = "ood_mean_spear_r" if "ood" in job_name.lower() else "mean_spear_r_test"
        latex = latex[[
            "feature_name",
            "to_abs_strat",
            spear_col,
        ]]

        latex["g_id"] = latex["feature_name"] + "__" + latex["to_abs_strat"]
        latex_g = []
        for g_id in sorted(latex["g_id"].unique()):
            g = latex[latex["g_id"] == g_id]
            latex_g.append({
                "g_id": g_id,
                #"feature_name": g["feature_name"].iloc[0],
                #"to_abs_strat": g["to_abs_strat"].iloc[0],
                f"rho mean": g[spear_col].mean(),
                f"rho std": g[spear_col].std(),
            })

        latex_g = pd.DataFrame(latex_g)
        for col,n_digits in [("rho mean",3), ("rho std",2),]:
            latex_g[col] = latex_g[col].round(n_digits)

        latex_g[f"rho ({short_name[job_name]})"] = latex_g["rho mean"].apply(str) + "+-" + latex_g["rho std"].apply(str)
        latex_g = latex_g.drop(columns=["rho mean", "rho std",])

        if "models" not in job_name.lower():
            if one_latex is None:
                one_latex = latex_g
            else:
                one_latex = one_latex.merge(latex_g,on="g_id",)

        #latex_h = latex.groupby(["feature_name","to_abs_strat",])
        #latex = latex_h.agg(["mean","std",])
        #import pdb; pdb.set_trace()
        #latex = latex.rename(columns={spear_col: "Spearman's rho"})
        #latex_g = latex_g.to_latex(index=False,)
        #Path(DATA_DIR / f"tabular__{job_name}.tex").write_text(latex_g)
        #latex = latex.replace(spear_col, )
        
        if False:
            plt.tight_layout()
            plt.title(job_name)
            fle_out = DATA_DIR / f"paper_{job_name}.svg"
            plt.savefig(fle_out)
            plt.clf()
            _post_process_svg(fle_out)
            os.system(f"open {fle_out}")


    one_latex["feature"] = one_latex["g_id"].apply(lambda g_id: g_id.split("__")[0])
    one_latex["relation"] = one_latex["g_id"].apply(lambda g_id: g_id.split("__")[1])
    one_latex = one_latex.drop(columns=["g_id",])
    cols_first = ["feature","relation",]
    one_latex = one_latex[cols_first + [col for col in one_latex.columns if col not in cols_first]]
    one_latex = one_latex.to_latex(index=False,)
        

    def remove_duplicate_line_starters(txt:str):
        lines = txt.split("\n")
        out_lines = txt.split("\n")
        for i in range(len(lines)-1):
            this_line = lines[i]
            next_line = lines[i+1]
            if this_line.split("&")[0] == next_line.split("&")[0]:
                print("!!!")
                out_lines[i+1] = " & ".join(["\t~"]+(next_line.split("&")[1:]))
        return "\n".join(out_lines)

    one_latex = remove_duplicate_line_starters(one_latex)
    one_latex = one_latex.replace("+-","$\\pm$")

    print("%",80*"=")
    print(one_latex)
    print("%",80*"=")


        


