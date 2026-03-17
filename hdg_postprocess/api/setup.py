from pathlib import Path

import numpy as np
import scipy.io


_DEFAULT_CX_ALPHA = np.array(
    [-1.87744894e01, 4.51800000e-01, -3.58100000e-02, 8.00400000e-03, -6.83700000e-04]
)


def load_reference_element(path):
    """Load a SOLEDGE-HDG reference-element MAT file into the dictionary format used internally."""
    ref_path = Path(path)
    ref_elem = scipy.io.loadmat(ref_path)
    if "refEl" in ref_elem:
        name = "refEl"
    elif "referenceelement" in ref_elem:
        name = "referenceelement"
    else:
        raise KeyError(f"Unsupported reference-element keys in {ref_path}: {sorted(ref_elem.keys())}")

    return {
        "IPcoordinates": ref_elem[name][0, 0][0],
        "IPweights": ref_elem[name][0, 0][1][:, 0],
        "N": ref_elem[name][0, 0][2],
        "Nxi": ref_elem[name][0, 0][3],
        "Neta": ref_elem[name][0, 0][4],
        "IPcoordinates1d": ref_elem[name][0, 0][5],
        "IPweights1d": ref_elem[name][0, 0][6],
        "N1d": ref_elem[name][0, 0][7],
        "N1dxi": ref_elem[name][0, 0][8],
        "faceNodes": ref_elem[name][0, 0][9] - 1,
        "innerNodes": ref_elem[name][0, 0][10],
        "faceNodes1d": ref_elem[name][0, 0][11] - 1,
        "NodesCoord": ref_elem[name][0, 0][12],
        "NodesCoord1d": ref_elem[name][0, 0][13],
        "degree": ref_elem[name][0, 0][14],
    }


def make_neutral_diffusion_parameters():
    """Return the default neutral-diffusion parameter bundle used in the demos and tests."""
    return {
        "const": False,
        "dnn_soft": True,
        "dnn_max": 2e8,
        "dnn_min": 3.0,
        "dnn_w": 0.01,
        "dnn_width": 10,
        "ti_soft": True,
        "ti_min": 1e-6,
        "ti_w": 0.01,
        "ti_width": 10,
    }


def make_turbulence_parameters(dk_min=1e-6, dk_max=1e2):
    """Return the default turbulence-diffusion parameter bundle used in the demos and tests."""
    return {
        "dk_min": dk_min,
        "dk_max": dk_max,
        "dk_min_adim": 0.0,
        "dk_max_adim": 0.0,
    }


def make_atomic_parameters(radiation_model="none", data_dir=None):
    """Return an atomic parameter bundle from an explicit local data directory."""
    if data_dir is None:
        raise ValueError("Please provide atomic data_dir explicitly when constructing atomic parameters.")
    data_dir = Path(data_dir)

    atomic = {
        "iz": {
            "database": "AMJUEL 2.1.5JH",
            "alpha": np.load(data_dir / "alpha_iz.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
        "cx": {
            "database": "OpenADAS expanded",
            "alpha": _DEFAULT_CX_ALPHA.copy(),
            "te_min": 0.1,
            "te_max": 2e4,
        },
        "rec": {
            "database": "AMJUEL 2.1.8JH",
            "alpha": np.load(data_dir / "alpha_rec_2.1.8JH.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
        "Eiz": {
            "database": "AMJUEL 2.1.5JH",
            "alpha": np.load(data_dir / "alpha_energy_iz.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
        "Erec": {
            "database": "AMJUEL 2.1.8JH",
            "alpha": np.load(data_dir / "alpha_energy_rec.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
    }
    if radiation_model == "nitrogen_cooling":
        cooling = np.load(data_dir / "LZ_Nitrogen_adas_fit_te_2e-1_4e3.npy")
        cooling[0] -= np.log(1.60217662e-19)
        atomic["cooling_factor"] = {
            "database": "ADAS",
            "alpha": cooling,
            "te_min": 0.1,
            "te_max": 3e3,
        }
    elif radiation_model == "tungsten_cooling":
        cooling = np.load(data_dir / "LZ_Tungsten_adas_fit_te_2e0_4e4.npy")
        cooling[0] -= np.log(1.60217662e-19)
        atomic["cooling_factor"] = {
            "database": "ADAS",
            "alpha": cooling,
            "te_min": 2.0,
            "te_max": 4e4,
        }
    elif radiation_model != "none":
        raise ValueError(f"Unsupported radiation model {radiation_model!r}")
    return atomic


def configure_solution_setup(
    solution,
    *,
    reference_element=None,
    radiation_model=None,
    atomic_data_dir=None,
    neutral_diffusion=False,
    turbulence=False,
):
    """Apply the common reference-element and additional-parameter setup used by tutorials and demos."""
    if reference_element is not None:
        if isinstance(reference_element, (str, Path)):
            solution.mesh.metadata.reference_element = load_reference_element(reference_element)
        else:
            solution.mesh.metadata.reference_element = reference_element

    if radiation_model is not None:
        solution.additional_parameters.set_atomic(
            make_atomic_parameters(radiation_model=radiation_model, data_dir=atomic_data_dir)
        )

    if neutral_diffusion:
        params = make_neutral_diffusion_parameters() if neutral_diffusion is True else dict(neutral_diffusion)
        solution.additional_parameters.set_neutral_diffusion(params, solution.parameters["adimensionalization"])

    if turbulence:
        params = make_turbulence_parameters() if turbulence is True else dict(turbulence)
        solution.additional_parameters.set_turbulence(params, solution.parameters["adimensionalization"])

    return solution
