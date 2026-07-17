import numpy as np

from mm_kinetics.synthetic import assay_level_truth, generate_synthetic_dataset


def test_synthetic_dataset_has_known_truth_and_assay_structure():
    raw = generate_synthetic_dataset(
        n_enzymes=4, n_substrates=3, assays_per_pair=2, n_concentrations=6, random_seed=2
    )
    assert len(raw) == 4 * 3 * 2 * 6
    assert raw.assay_id.nunique() == 4 * 3 * 2
    assert (raw.velocity_molar_per_s >= 0).all()
    assert np.isfinite(raw[["true_km_molar", "true_kcat_per_s"]]).all().all()
    assays = assay_level_truth(raw)
    assert len(assays) == raw.assay_id.nunique()
    np.testing.assert_allclose(assays.log10_km, np.log10(assays.km_molar))
    assert "substrate_molar" not in assays
