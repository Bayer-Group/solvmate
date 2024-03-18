from collections import OrderedDict
import random
import numpy as np
import joblib
from pathlib import Path

try:
    from opencosmorspy import Parameterization, COSMORS
except:
    print("could not import opencosmorspy!")


from dataclasses import dataclass

from cosmonaut import *


@dataclass
class COSMO_RS_Result:

    feature_vec: np.ndarray

    @staticmethod
    def _example_ordered_dict():
        array = np.array
        return OrderedDict(
            [
                ("x", array([[0.5, 0.5]])),
                ("T", array([298.15])),
                ("tot", OrderedDict([("lng", array([[4.38288174, -0.57636597]]))])),
                (
                    "enth",
                    OrderedDict(
                        [
                            ("lng", array([[4.4968558, -0.33857875]])),
                            ("aim_A_int", array([[7222.3606538, -10021.49849661]])),
                            ("aim_A_hb", array([[-469.18557283, -12509.73110977]])),
                            ("aim_tau", array([[10.54586416, 81.69193686]])),
                            ("aim_A_mf", array([[7691.54622663, 2488.23261315]])),
                            ("aim_E_mf", array([[7691.54622663, 2488.23261315]])),
                            ("aim_E_hb", array([[-1172.96393207, -31274.32777441]])),
                            ("aim_sigma_mf_abs", array([[0.05534355, 0.01463925]])),
                            ("aim_E_int", array([[6518.58229456, -28786.09516126]])),
                            ("aim_sigma_mf", array([[-0.00264621, -0.00283544]])),
                            ("pm_A_int", array([[8273.15386082, -11072.28604867]])),
                            ("pm_A_hb", array([[639.70349027, -13618.61821409]])),
                            ("pm_tau", array([[8.01794529, 84.21985923]])),
                            ("pm_A_mf", array([[7633.45037056, 2546.33216542]])),
                            ("pm_E_mf", array([[7633.45037056, 2546.33216542]])),
                            ("pm_E_hb", array([[1599.25872566, -34046.54553524]])),
                            ("pm_sigma_mf_abs", array([[0.05504271, 0.0149401]])),
                            ("pm_E_int", array([[9232.70909622, -31500.21336981]])),
                            ("pm_sigma_mf", array([[-0.00307302, -0.00240868]])),
                        ]
                    ),
                ),
                ("comb", OrderedDict([("lng", array([[-0.11397406, -0.23778722]]))])),
            ]
        )

    @staticmethod
    def from_ordered_dict(od: OrderedDict) -> "COSMO_RS_Result":
        """

        >>> od = COSMO_RS_Result._example_ordered_dict()
        >>> COSMO_RS_Result.from_ordered_dict(od)#doctest:+NORMALIZE_WHITESPACE
        COSMO_RS_Result(feature_vec=array([[-1.13974060e-01, -2.37787220e-01],
           [-4.69185573e+02, -1.25097311e+04],
           [ 7.22236065e+03, -1.00214985e+04],
           [ 7.69154623e+03,  2.48823261e+03],
           [-1.17296393e+03, -3.12743278e+04],
           [ 6.51858229e+03, -2.87860952e+04],
           [ 7.69154623e+03,  2.48823261e+03],
           [-2.64621000e-03, -2.83544000e-03],
           [ 5.53435500e-02,  1.46392500e-02],
           [ 1.05458642e+01,  8.16919369e+01],
           [ 4.49685580e+00, -3.38578750e-01],
           [ 6.39703490e+02, -1.36186182e+04],
           [ 8.27315386e+03, -1.10722860e+04],
           [ 7.63345037e+03,  2.54633217e+03],
           [ 1.59925873e+03, -3.40465455e+04],
           [ 9.23270910e+03, -3.15002134e+04],
           [ 7.63345037e+03,  2.54633217e+03],
           [-3.07302000e-03, -2.40868000e-03],
           [ 5.50427100e-02,  1.49401000e-02],
           [ 8.01794529e+00,  8.42198592e+01],
           [ 4.38288174e+00, -5.76365970e-01]]))

        """
        rslt = []
        keys = sorted(od.keys())
        for key in keys:
            val = od[key]
            if isinstance(val, OrderedDict):
                inner_od = val
                inner_keys = sorted(inner_od.keys())

                for key in inner_keys:
                    val = inner_od[key]
                    assert val.shape in [
                        (1,),
                        (1, 2),
                    ]
                    if val.shape == (1, 2):
                        rslt.append(val)
        rslt = np.vstack(rslt)
        assert rslt.shape == (21, 2)
        return COSMO_RS_Result(feature_vec=rslt)


_COSMO_RS_MEMO_FLE = CM_DATA_DIR / ".cosmo_rs_memo.pkl"
if _COSMO_RS_MEMO_FLE.exists():
    _COSMO_RS_MEMO = joblib.load(_COSMO_RS_MEMO_FLE)
else:
    _COSMO_RS_MEMO = {}


def make_cosmors_features(
    fle_solvent: Path, fle_solute: Path, refst: str, reduction: str, T=298.15
):
    assert reduction in [
        "id",
        "sum",
        "diff",
        "sum+concat",
    ]
    xs = []
    for x_solvent in [
        0.90, 0.99, 0.999,
    ]:
        x_solute = 1.0 - x_solvent
        x = run_cosmors_calculations(
            fle_solvent=fle_solvent,
            fle_solute=fle_solute,
            x_solvent=x_solvent,
            x_solute=x_solute,
            refst=refst,
            T=T,
        )
        if x is None:
            # if even any of the single calculations is None, then
            # the overall result is None.
            return None
        else:
            x = x.feature_vec

        if reduction == "id":
            x = x.reshape(-1)
        elif reduction == "sum":
            x = x[:, 0] + x[:, 1]
        elif reduction == "diff":
            x = x[:, 0] - x[:, 1]
        elif reduction == "sum+concat":
            x_flat = x.reshape(-1)
            x_sum = x[:, 0] + x[:, 1]
            x = np.hstack([x_flat, x_sum])
        else:
            assert False, f"unknown reduction {reduction}"

        xs.append(x)

    return np.hstack(xs)


def run_cosmors_calculations(
    fle_solvent: Path,
    fle_solute: Path,
    x_solvent: float,
    x_solute: float,
    refst: str,
    T=298.15,
) -> COSMO_RS_Result:
    assert refst in [
        "cosmo",
        "pure_component",
        "reference_mixture",
    ]
    assert x_solvent >= x_solute
    key = "__".join(
        [
            str(fle_solvent.resolve()),
            str(fle_solute.resolve()),
            "{:.6f}".format(x_solvent),
            "{:.6f}".format(x_solute),
            refst,
        ]
    )
    if key in _COSMO_RS_MEMO:
        return _COSMO_RS_MEMO[key]
    else:
        try:
            assert fle_solvent.exists(), f"could not find file {fle_solvent}"
            assert fle_solute.exists(), f"could not find file {fle_solute}"
            crs = COSMORS(par="default_orca")
            crs.par.calculate_contact_statistics_molecule_properties = True

            crs.add_molecule([fle_solvent])
            crs.add_molecule([fle_solute])

            x = np.array(
                [
                    x_solvent,
                    x_solute,
                ]
            )
            crs.add_job(x, T, refst=refst)

            results = crs.calculate()
            results = COSMO_RS_Result.from_ordered_dict(results)
            _COSMO_RS_MEMO[key] = results
            if random.random() < 0.01:
                joblib.dump(value=_COSMO_RS_MEMO, filename=_COSMO_RS_MEMO_FLE)
            return results

        except Exception as e:
            if "Unknown elements" in str(e):
                # Known issue, if element type cannot be handled then we can't
                # handle that, so return None instead.
                # TODO: find out which elements are causing this issue, and
                # TODO: ask tmg how we could possibly add those additional
                # TODO: element types
                _COSMO_RS_MEMO[key] = None
                return None
            else:
                print(e)
                return None
