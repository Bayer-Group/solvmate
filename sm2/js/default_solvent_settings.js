const _ess_solvents = `
n-Heptan
Cyclohexan
Diisopropylether
Toluol
Tetrahydrofuran
Aceton
Ethylacetat
ACN
2-Propanol
Ethanol
EtOH / Wasser 1:1
Methanol
Wasser
Dichlormethan
Isopropanol/Wasser 1/1
`;

const _all_solvents = `
cyclohexane
octane
heptane
hexadecane
nonane
decane
2,2-dimethyl-4-methylpentane
hexane
methylcyclohexane
water
toluene
benzene
carbon tetrachloride
chlorobenzene
dibutyl ether
di-isopropylether
methyl t-butylether
nitromethane
diethyl ether
decanol
methyl isobutyl ketone
water : Nonaethyleneglycol 1 : 1
Nonaethylene glycol
1-heptanol
methylene chloride
1-octanol
1-hexanol
chloroform
ethanol : water 1 : 1
2-ethylhexanol
4-methyl pentan-2-ol
2-methyl pentanol
ethylene glycol
2-isopropoxyethanol
1-pentanol
1-butanol
2-propoxyethanol
iso-butanol
2-pentanol
acetonitrile
ethyl acetate
butylacetate
1,4-dioxane
t-butyl alcohol
2-butoxyethanol
2-butanol
isoamyl alcohol
2-propanol
1-propanol
methyl acetate
2-ethoxyethanol
ethanol
1,2-dichloroethane
acetone
methanol
2-butanone
tetrahydrofuran
dimethyl sulfoxide
dimethylformamide
`;

const DEFAULT_SOLVENT_SETTINGS = [
    _ess_solvents.trim().split("\n").join("|"),
    _all_solvents.trim().split("\n").join("|"),
    "",
    "",
    ""
];