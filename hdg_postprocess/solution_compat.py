def _resolve_nested_attr(obj, path):
    value = obj
    for part in path:
        value = getattr(value, part)
    return value


def _make_generated_property(root_attr, path, doc, setter_attr=None):
    def getter(self):
        base = self if not root_attr else getattr(self, root_attr)
        return _resolve_nested_attr(base, path)

    getter_parts = tuple(part for part in (root_attr.strip("_"),) + tuple(path) if part)
    getter.__name__ = f"get_{'_'.join(getter_parts)}"

    if setter_attr is None:
        return property(getter, doc=doc)

    def setter(self, value):
        setattr(self, setter_attr, value)

    setter.__name__ = f"set_{'_'.join(getter_parts)}"
    return property(getter, setter, doc=doc)


_STATE_PROPERTY_DOCS = {
    "combined_to_full": ("", ("_combined_to_full",), "Flag which tells if the solution has been combined to full"),
    "combined_boundary": ("", ("_combined_boundary",), "Flag which tells if the solution has been combined on a boundary of mesh"),
    "combined_simple_solution": ("", ("_combined_simple_solution",), "Flag which tells if the solution has been combined to simple one on full mesh"),
    "full_phys_initialized": ("", ("_full_phys_initialized",), "Flag which tells if physical values has been initialized (full)"),
    "simple_phys_initialized": ("", ("_simple_phys_initialized",), "Flag which tells if physical values has been initialized (simple)"),
    "e": ("", ("_e",), "elemental_charge"),
    "cons_idx": ("", ("_cons_idx",), "dictionary with keys are the cons variables, values are the indexes o the corresponding equation"),
    "phys_idx": ("", ("_phys_idx",), "dictionary with keys are the phys variables, values are the indexes o the corresponding equation"),
    "ion_energy_sheath_loss_total": (
        "_summary",
        ("boundary", "ion_energy_sheath_loss_total"),
        "Total ion energy loss in sheath on a full solution mesh using conservative values as inputs",
    ),
    "electron_energy_sheath_loss_total": (
        "_summary",
        ("boundary", "electron_energy_sheath_loss_total"),
        "Total electron energy loss in sheath on a full solution mesh using conservative values as inputs",
    ),
}

_SOURCE_TOTAL_DOCS = {
    "ion_gain_iz_total": "Total ion energy sink due to ionization on a full solution mesh using conservative values as inputs",
    "ion_sink_rec_total": "Total ion energy sink due to recombination on a full solution mesh using conservative values as inputs",
    "ion_sink_cx_total": "Total ion energy sink due to charge exchange on a full solution mesh using conservative values as inputs",
    "electron_sink_iz_total": "Total electron energy sink due to ionization on a full solution mesh using conservative values as inputs",
    "electron_sink_rec_total": "Total electron energy sink due to recombination on a full solution mesh using conservative values as inputs",
    "electron_gain_rec_total": "Total electron energy source due to recombination on a full solution mesh using conservative values as inputs",
    "external_heating_total": "Total external heating source on a full solution mesh using conservative values as inputs",
    "external_heating_e_total": "Total external heating source on electrons on a full solution mesh using conservative values as inputs",
    "external_heating_i_total": "Total external heating source on ions on a full solution mesh using conservative values as inputs",
    "ohmic_source_total": "Total ohmic heating source on a full solution mesh using conservative values as inputs",
}


def attach_compat_properties(cls):
    for name, (root_attr, path, doc) in _STATE_PROPERTY_DOCS.items():
        setattr(cls, name, _make_generated_property(root_attr, path, doc))

    for name, doc in _SOURCE_TOTAL_DOCS.items():
        setattr(cls, name, _make_generated_property("_summary", ("sources", name), doc))
