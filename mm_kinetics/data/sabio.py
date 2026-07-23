from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pint

UNIT_REGISTRY = pint.UnitRegistry()


@dataclass(frozen=True)
class SabioIngestionResult:
    accepted: pd.DataFrame
    excluded: pd.DataFrame
    provenance: dict[str, str]


def _normalize_header(name: object) -> str:
    return re.sub("[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


ALIASES: dict[str, tuple[str, ...]] = {
    "entry_id": ("entry_id", "sabiork_id", "sabio_rk_id", "id"),
    "enzyme_id": ("enzyme_id", "uniprot_id", "uniprotid", "enzyme"),
    "enzyme_family": ("enzyme_family", "ec_number", "ecnumber"),
    "substrate_id": (
        "substrate_id",
        "substrate",
        "parameter_associated_species",
        "associated_species",
    ),
    "km": ("km", "km_value"),
    "km_unit": ("km_unit", "parameter_unit", "unit"),
    "kcat": ("kcat", "kcat_value", "turnover_number"),
    "kcat_unit": ("kcat_unit", "turnover_number_unit"),
    "parameter_type": ("parameter_type", "parametertype", "kinetic_constant_type"),
    "parameter_value": ("parameter_value", "parameter_start_value", "start_value"),
    "parameter_unit": ("parameter_unit", "parameter_value_unit", "unit"),
    "temperature": ("temperature", "temperature_value"),
    "temperature_unit": ("temperature_unit",),
    "ph": ("ph", "p_h"),
    "organism": ("organism", "organism_name"),
    "study_id": ("study_id", "pubmed_id", "pubmedid", "publication"),
    "kinetic_law": ("kinetic_law", "rate_equation", "kinetic_law_type"),
}


def _find_column(columns: set[str], field: str) -> str | None:
    return next((candidate for candidate in ALIASES[field] if candidate in columns), None)


def _quantity(value: object, unit: object, target: str) -> float:
    if pd.isna(value) or pd.isna(unit) or str(unit).strip() == "":
        return np.nan
    return float((float(value) * UNIT_REGISTRY(str(unit))).to(target).magnitude)


def ingest_sabio_export(
    path: str | Path,
    *,
    retrieval_date: str | None,
    query: str | None,
    source_version: str = "export",
) -> SabioIngestionResult:
    export_path = Path(path)
    if not export_path.exists():
        raise FileNotFoundError(export_path)
    separator = "\t" if export_path.suffix.lower() in {".tsv", ".tab"} else ","
    raw = pd.read_csv(export_path, sep=separator)
    raw = raw.rename(columns={column: _normalize_header(column) for column in raw.columns})
    columns = set(raw.columns)
    mapping = {field: _find_column(columns, field) for field in ALIASES}
    required_identity = ["enzyme_id", "substrate_id"]
    missing = [field for field in required_identity if mapping[field] is None]
    if missing:
        raise ValueError(f"SABIO-RK export lacks required identity columns: {missing}")
    long_form = all(
        mapping[field] is not None
        for field in ("parameter_type", "parameter_value", "parameter_unit")
    )
    if mapping["km"] is None and mapping["kcat"] is None and (not long_form):
        raise ValueError(
            "SABIO-RK export has neither wide Km/kcat columns nor long-form parameter fields"
        )
    retrieved = retrieval_date or datetime.now(UTC).date().isoformat()
    accepted: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for source_index, row in raw.iterrows():
        reason: list[str] = []
        kinetic_law_value = row[mapping["kinetic_law"]] if mapping["kinetic_law"] else None
        kinetic_law = "" if pd.isna(kinetic_law_value) else str(kinetic_law_value).strip()
        if (
            kinetic_law
            and kinetic_law.lower() != "nan"
            and ("michaelis" not in kinetic_law.lower())
        ):
            reason.append("incompatible kinetic law")
        enzyme_id = row[mapping["enzyme_id"]]
        substrate_id = row[mapping["substrate_id"]]
        if pd.isna(enzyme_id) or pd.isna(substrate_id):
            reason.append("missing enzyme or substrate identity")
        km_molar = np.nan
        kcat_per_s = np.nan
        try:
            if long_form:
                parameter_type = str(row[mapping["parameter_type"]]).strip().lower()
                parameter_value = row[mapping["parameter_value"]]
                parameter_unit = row[mapping["parameter_unit"]]
                if parameter_type in {"km", "k_m", "michaelis constant", "michaelis_constant"}:
                    km_molar = _quantity(parameter_value, parameter_unit, "mole/liter")
                elif parameter_type in {"kcat", "k_cat", "turnover number", "turnover_number"}:
                    kcat_per_s = _quantity(parameter_value, parameter_unit, "1/second")
                else:
                    reason.append(f"unsupported parameter type: {parameter_type or 'missing'}")
            elif mapping["km"] is not None:
                km_unit_column = mapping["km_unit"]
                km_molar = _quantity(
                    row[mapping["km"]],
                    row[km_unit_column] if km_unit_column else None,
                    "mole/liter",
                )
            if not long_form and mapping["kcat"] is not None:
                kcat_unit_column = mapping["kcat_unit"]
                kcat_per_s = _quantity(
                    row[mapping["kcat"]],
                    row[kcat_unit_column] if kcat_unit_column else None,
                    "1/second",
                )
        except (ValueError, pint.errors.PintError) as error:
            reason.append(f"unit conversion failed: {error}")
        if not np.isfinite(km_molar) and (not np.isfinite(kcat_per_s)):
            reason.append("no usable Km or kcat value")
        if np.isfinite(km_molar) and km_molar <= 0:
            reason.append("Km is nonpositive")
        if np.isfinite(kcat_per_s) and kcat_per_s <= 0:
            reason.append("kcat is nonpositive")
        record: dict[str, object] = {
            "source_row": int(source_index),
            "assay_id": str(row[mapping["entry_id"]])
            if mapping["entry_id"]
            else f"SABIO-{source_index:08d}",
            "enzyme_id": str(enzyme_id),
            "substrate_id": str(substrate_id),
            "enzyme_family": str(row[mapping["enzyme_family"]])
            if mapping["enzyme_family"]
            else None,
            "study_id": str(row[mapping["study_id"]]) if mapping["study_id"] else None,
            "organism": str(row[mapping["organism"]]) if mapping["organism"] else None,
            "temperature_kelvin": np.nan,
            "pH": pd.to_numeric(row[mapping["ph"]], errors="coerce") if mapping["ph"] else np.nan,
            "km_molar": km_molar,
            "kcat_per_s": kcat_per_s,
            "kinetic_law": kinetic_law or None,
            "source": "SABIO-RK",
            "source_version": source_version,
            "retrieval_date": retrieved,
            "source_query": query,
            "source_file": export_path.name,
        }
        if mapping["temperature"]:
            try:
                unit = row[mapping["temperature_unit"]] if mapping["temperature_unit"] else "kelvin"
                record["temperature_kelvin"] = _quantity(
                    row[mapping["temperature"]], unit, "kelvin"
                )
            except (ValueError, pint.errors.PintError):
                reason.append("temperature conversion failed")
        if reason:
            record["exclusion_reason"] = "; ".join(dict.fromkeys(reason))
            excluded.append(record)
        else:
            record["log10_km"] = np.log10(km_molar) if np.isfinite(km_molar) else np.nan
            record["log10_kcat"] = np.log10(kcat_per_s) if np.isfinite(kcat_per_s) else np.nan
            record["log10_efficiency"] = (
                np.log10(kcat_per_s / km_molar)
                if np.isfinite(km_molar) and np.isfinite(kcat_per_s)
                else np.nan
            )
            accepted.append(record)
    provenance = {
        "source": "SABIO-RK",
        "source_file": str(export_path.resolve()),
        "retrieval_date": retrieved,
        "query": query or "unspecified",
        "source_version": source_version,
    }
    return SabioIngestionResult(pd.DataFrame(accepted), pd.DataFrame(excluded), provenance)
