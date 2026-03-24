import numpy as np

from hdg_postprocess.core.solution import surfaces as surface_ops


_ELECTRON_CHARGE = 1.602176634e-19
_ATOMIC_MASS_UNIT = 1.66053906660e-27


def bohm_profile(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
    ion_mass_amu=2.0,
    ion_charge=1.0,
):
    profiles = surface_transport_inputs(
        solution,
        rho,
        rho_inner=rho_inner,
        rho_edge=rho_edge,
        method=method,
        width=width,
    )
    profiles["chi_bohm"] = bohm_component(
        profiles,
        ion_mass_amu=ion_mass_amu,
        ion_charge=ion_charge,
    )
    return profiles


def gyrobohm_profile(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
    ion_mass_amu=2.0,
    ion_charge=1.0,
):
    profiles = surface_transport_inputs(
        solution,
        rho,
        rho_inner=rho_inner,
        rho_edge=rho_edge,
        method=method,
        width=width,
    )
    profiles["chi_gyrobohm"] = gyrobohm_component(
        profiles,
        ion_mass_amu=ion_mass_amu,
        ion_charge=ion_charge,
    )
    return profiles


def mixed_bohm_gyrobohm(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
    ion_mass_amu=2.0,
    ion_charge=1.0,
):
    profiles = surface_transport_inputs(
        solution,
        rho,
        rho_inner=rho_inner,
        rho_edge=rho_edge,
        method=method,
        width=width,
    )
    chi_bohm = bohm_component(profiles, ion_mass_amu=ion_mass_amu, ion_charge=ion_charge)
    chi_gyrobohm = gyrobohm_component(profiles, ion_mass_amu=ion_mass_amu, ion_charge=ion_charge)
    chi_i, chi_e, diffusion = mixed_transport_from_components(chi_bohm, chi_gyrobohm)
    profiles.update(
        {
            "chi_bohm": chi_bohm,
            "chi_gyrobohm": chi_gyrobohm,
            "chi_i": chi_i,
            "chi_e": chi_e,
            "diffusion": diffusion,
        }
    )
    return profiles


def surface_transport_inputs(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
):
    rho_values = np.asarray(rho, dtype=float)
    if rho_values.ndim != 1 or rho_values.size < 2:
        raise ValueError("rho must be a one-dimensional array with at least two points")

    te = np.asarray(surface_ops.te_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    density = np.asarray(surface_ops.average_on_surfaces(solution, "n", rho_values, method=method, width=width), dtype=float)
    q_value = np.asarray(surface_ops.q_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    btor = np.asarray(surface_ops.average_on_surfaces(solution, "btor", rho_values, method=method, width=width), dtype=float)
    minor_radius = np.asarray(surface_ops.minor_radius_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    edge_minor_radius = float(surface_ops.minor_radius_on_surfaces(solution, rho_edge, method=method, width=width))
    delta_te = float(
        surface_ops.delta_te_on_surfaces(
            solution,
            rho_inner=rho_inner,
            rho_outer=rho_edge,
            method=method,
            width=width,
        )
    )

    pressure_like = density * te
    dte_dr = np.gradient(te, minor_radius, edge_order=2)
    dpe_dr = np.gradient(pressure_like, minor_radius, edge_order=2)

    return {
        "rho": rho_values,
        "te": te,
        "density": density,
        "q": q_value,
        "btor": btor,
        "minor_radius": minor_radius,
        "edge_minor_radius": edge_minor_radius,
        "delta_te": delta_te,
        "pressure_like": pressure_like,
        "dte_dr": dte_dr,
        "dpe_dr": dpe_dr,
    }


def bohm_component(profiles, *, ion_mass_amu=2.0, ion_charge=1.0):
    cs = _sound_speed(profiles["te"], ion_mass_amu)
    rho_s = _gyro_radius(cs, profiles["btor"], ion_mass_amu, ion_charge)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_over_pe_dpe = profiles["edge_minor_radius"] * profiles["dpe_dr"] / profiles["pressure_like"]
        return rho_s * cs * profiles["q"]**2 * a_over_pe_dpe * profiles["delta_te"]


def gyrobohm_component(profiles, *, ion_mass_amu=2.0, ion_charge=1.0):
    cs = _sound_speed(profiles["te"], ion_mass_amu)
    rho_s = _gyro_radius(cs, profiles["btor"], ion_mass_amu, ion_charge)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_over_te_dte = profiles["edge_minor_radius"] * profiles["dte_dr"] / profiles["te"]
        return rho_s**2 * cs / profiles["edge_minor_radius"] * a_over_te_dte


def mixed_transport_from_components(chi_bohm, chi_gyrobohm):
    chi_i = 1.6e-4 * chi_bohm + 1.75e-2 * chi_gyrobohm
    chi_e = 8.0e-5 * chi_bohm + 3.5e-2 * chi_gyrobohm
    with np.errstate(divide="ignore", invalid="ignore"):
        diffusion = chi_i * chi_e / (chi_i + chi_e)
    return chi_i, chi_e, diffusion


def _sound_speed(te_ev, ion_mass_amu):
    ion_mass = ion_mass_amu * _ATOMIC_MASS_UNIT
    return np.sqrt(np.maximum(te_ev, 0.0) * _ELECTRON_CHARGE / ion_mass)


def _gyro_radius(sound_speed, btor, ion_mass_amu, ion_charge):
    ion_mass = ion_mass_amu * _ATOMIC_MASS_UNIT
    omega_ci = ion_charge * _ELECTRON_CHARGE * np.abs(btor) / ion_mass
    with np.errstate(divide="ignore", invalid="ignore"):
        return sound_speed / omega_ci
