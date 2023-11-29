from sklearn.impute import SimpleImputer
from solvmate import *
from solvmate import xtb_solv


class AbstractFeaturizer:
    def __init__(
        self,
        phase: str,
        pairwise_reduction: str,
        feature_name: str,
    ) -> None:
        assert phase in ["train", "predict"]
        assert pairwise_reduction in ["diff", "concat"]

        self.phase = phase
        self.pairwise_reduction = pairwise_reduction
        self.feature_name = feature_name
        self.imputer = SimpleImputer()

    def __str__(self):
        return f"{self.__class__}: feature_name={self.feature_name} pairwise_reduction={self.pairwise_reduction}"

    def run_single(self, smiles: list[str]) -> np.array:
        raise NotImplemented

    # TODO: extend this interface to accept mixtures of solvents!
    def run_pairs(
        self, compounds: list[str], solvents_a: list[str], solvents_b: list[str]
    ) -> np.array:
        X_by_type = {}
        for typ, smis_index in [
            ("compound", compounds),
            ("solv_a", solvents_a),
            ("solv_b", solvents_b),
        ]:
            smis_set = list(set(smis_index))
            X = self.run_single(smis_set)
            X = {smi: x for smi, x in zip(smis_set, X)}
            X = np.vstack([X[smi] for smi in smis_index])
            X_by_type[typ] = X

        if self.pairwise_reduction == "diff":
            X_solv = X_by_type["solv_b"] - X_by_type["solv_a"]
            return np.hstack([X_by_type["compound"], X_solv])
        if self.pairwise_reduction == "concat":
            return np.hstack(
                [X_by_type["compound"], X_by_type["solv_a"], X_by_type["solv_b"]]
            )
        else:
            assert False, "unknown pairwise_reduction: " + self.pairwise_reduction

    def column_names(self) -> list[str]:
        if self.pairwise_reduction == "diff":
            return [
                "|".join([col_type, col])
                for col_type in ["compound", "solv"]
                for col in self.column_names_single()
            ]
        if self.pairwise_reduction == "concat":
            return [
                "|".join([col_type, col])
                for col_type in ["compound", "solv_a", "solv_b"]
                for col in self.column_names_single()
            ]

    def column_names_single(self) -> list[str]:
        raise NotImplemented()


class XTBFeaturizer(AbstractFeaturizer):
    """
    Provides a convenient interface to XTB features that can be readily fed
    into machine learning models. Imputations, train vs. predict db differences
    as well as pivoting of the xtb features is already taken care of.

    Here is an example on how to run a simple xtb featurization:
    >>> xf = XTBFeaturizer(phase="train", pairwise_reduction="diff", feature_name="test", db_storage_dict={"train":"/tmp/_featurizer_test_130u80.db","predict":"/tmp/_featurizer_test_130u80.db"})
    >>> xf.run_single(["C1C(O)CC1"]).shape
    (1, 253)

    A simple sanity check: 1-propanol should be more similar to 2-propanol than hexadecane:
    >>> xf = XTBFeaturizer(phase="train", pairwise_reduction="diff", feature_name="test", db_storage_dict={"train":"/tmp/_featurizer_test_130u80.db","predict":"/tmp/_featurizer_test_130u80.db"})
    >>> rslt = xf.run_single(["C(O)CC","CC(O)C","CCCCCCCCCCCCCCCC"])
    >>> rslt.shape
    (3, 253)
    >>> ab = np.linalg.norm(rslt[1]-rslt[0])
    >>> ac = np.linalg.norm(rslt[2]-rslt[0])

    >>> assert ab < ac, f"{ab} < {ac}"


    """

    def __init__(
        self,
        phase: str,
        pairwise_reduction: str,
        feature_name: str,
        skip_calculations=False,
        db_storage_dict=None,
    ) -> None:

        self._db_storage_dict = db_storage_dict
        self.skip_calculations = skip_calculations

        # The names of the feature columns. Useful
        # for feature importance analysis.
        self.column_names_single_ = None

        super().__init__(phase, pairwise_reduction, feature_name)

    def _db_storage_for_phase(self, phase):
        if hasattr(self, "_db_storage_dict") and self._db_storage_dict is not None:
            return self._db_storage_dict[phase]

        if phase == "train":
            return DATA_DIR / "xtb_features.db"
        elif phase == "predict":
            return DATA_DIR / "xtb_features_predict.db"
        else:
            assert False, f"unknown phase {phase}"

    def run_single(self, smiles: list[str]) -> np.array:
        db_file = self._db_storage_for_phase(self.phase)
        xs = xtb_solv.XTBSolv(db_file=db_file)
        xs.setup_db()
        if self.skip_calculations:
            print("skipping XTB calculations!")
        else:
            xs.run_xtb_calculations(smiles=smiles)
        df_xtb = xs.get_dataframe()
        vec_xtb = self._xtb_features_to_vector(df_xtb, smiles)
        return vec_xtb

    def _xtb_features_to_vector(
        self,
        df_xtb: pd.DataFrame,
        smiles: list[str],
    ) -> np.array:
        xtb_features_piv = df_xtb.drop_duplicates(["smiles", "solvent"]).pivot(
            index="smiles",
            columns="solvent",
        )
        # Dissolves the multiindex and replaces it with a normal single-index instead.
        # For example, ("a", "b") is turned into '("a", "b")'. That makes it much easier
        # to operate with normal pandas functions on the dataframe while still retaining
        # all the options to operate in a multiindex way (though then slower!).
        xtb_features_piv.columns = [str(col) for col in xtb_features_piv.columns.values]
        xtb_features_piv.reset_index(inplace=True)

        smis_contained = set(xtb_features_piv["smiles"].unique())

        # Fills up the rows where the xtb feature generation failed with rows that only
        # contain a smiles and nothing else. Later in this function, we will employ
        # the simple imputer to fill up these values.
        smis_need_to_add = list(set(smiles).difference(smis_contained))
        if smis_need_to_add:
            xtb_features_piv = pd.concat(
                [
                    xtb_features_piv,
                    pd.DataFrame({"smiles": smis_need_to_add}),
                ]
            )

        xtb_features_piv.set_index("smiles", inplace=True)
        xtb_features_piv = xtb_features_piv.reindex(smiles)
        xtb_features_piv.reset_index(inplace=True)

        feat_cols = [col for col in xtb_features_piv.columns if "(" in col]
        self.column_names_single_ = feat_cols

        feat_vals = xtb_features_piv[feat_cols].values
        if self.phase == "train":
            feat_vals = self.imputer.fit_transform(feat_vals)
        elif self.phase == "predict":
            feat_vals = self.imputer.transform(feat_vals)
        else:
            assert False

        return feat_vals

    def column_names_single(self) -> list[str]:
        assert (
            self.column_names_single_ is not None
        ), "need to call run_single at least once before!"
        return self.column_names_single_


try:
    from cosmonaut import *
    from cosmonaut.cosmo_calc import run_cosmo_calculations
    from cosmonaut.cosmors_calc import make_cosmors_features
except ModuleNotFoundError:
    info("could not load cosmonaut package")


class CosmoRSFeaturizer(AbstractFeaturizer):
    def __init__(
        self,
        phase: str,
        pairwise_reduction: str,
        feature_name: str,
        skip_calculations=False,
    ) -> None:
        super().__init__(phase, pairwise_reduction, feature_name)
        self.skip_calculautions = skip_calculations

    def run_pairs(
        self, compounds: list[str], solvents_a: list[str], solvents_b: list[str]
    ) -> np.array:
        smiles_set = list(set(compounds + solvents_a + solvents_b))

        names, charges = [], []
        for smi in smiles_set:
            try:
                name = smiles_to_id(smi)
            except:
                import pdb

                pdb.set_trace()
            charge = smiles_to_charge(smi)
            names.append(name)
            charges.append(charge)

        assert len(smiles_set) == len(charges)
        assert len(smiles_set) == len(names)

        if self.skip_calculautions:
            print("skipping COSMO calculations!")
        else:
            print("running COSMO calculations...")
            run_cosmo_calculations(
                smiles_list=smiles_set,
                names_list=names,
                charges_list=charges,
                outputs_dir=CM_DATA_DIR,
                n_cores_inner=1,
                n_cores_outer=8,
            )
            print("... finished running COSMO calculations.")

        cosmo_fle_solvents_a = [
            id_to_cosmo_file_path(smiles_to_id(smi)) for smi in solvents_a
        ]
        cosmo_fle_solvents_b = [
            id_to_cosmo_file_path(smiles_to_id(smi)) for smi in solvents_b
        ]
        cosmo_fle_solutes = [
            id_to_cosmo_file_path(smiles_to_id(smi)) for smi in compounds
        ]

        print("running COSMO-RS calculations ...")
        X_a = [
            np.array(
                make_cosmors_features(
                    fle_solvent=fle_solvent,
                    fle_solute=fle_solute,
                    refst="pure_component",
                    reduction="id",
                )
            )
            for fle_solvent, fle_solute in zip(cosmo_fle_solvents_a, cosmo_fle_solutes)
        ]

        X_b = [
            np.array(
                make_cosmors_features(
                    fle_solvent=fle_solvent,
                    fle_solute=fle_solute,
                    refst="pure_component",
                    reduction="id",
                )
            )
            for fle_solvent, fle_solute in zip(cosmo_fle_solvents_b, cosmo_fle_solutes)
        ]

        shape = list(
            {
                tuple(row.shape)
                for mat in [X_a, X_b]
                for row in mat
                if row is not None and len(row.shape)
            }
        )
        assert len(shape) == 1, f"expected shapes to be unique but found: {shape}"
        success_shape = shape[0]
        X_a = [
            np.full(success_shape, np.nan) if row is None or not len(row.shape) else row
            for row in X_a
        ]
        X_a = np.vstack(X_a)
        X_b = [
            np.full(success_shape, np.nan) if row is None or not len(row.shape) else row
            for row in X_b
        ]
        X_b = np.vstack(X_b)
        print("... finished running COSMO-RS calculations.")

        if self.phase == "train":
            X_a = self.imputer.fit_transform(X_a)
            X_b = self.imputer.fit_transform(X_b)
        elif self.phase == "predict":
            X_a = self.imputer.transform(X_a)
            X_b = self.imputer.transform(X_b)
        else:
            assert False, f"unknown phase: {self.phase}"

        if self.pairwise_reduction == "diff":
            return X_b - X_a
        elif self.pairwise_reduction == "concat":
            return np.hstack(
                [
                    X_a,
                    X_b,
                ]
            )
        else:
            assert False
