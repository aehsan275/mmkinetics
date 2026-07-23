from mm_kinetics.data.splitting import grouped_train_validation_test_split
from mm_kinetics.synthetic import assay_level_truth, generate_synthetic_dataset


def _assays():
    raw = generate_synthetic_dataset(
        n_enzymes=12, n_substrates=2, n_concentrations=5, random_seed=10
    )
    return assay_level_truth(raw)


def test_group_split_has_no_enzyme_leakage():
    splits = grouped_train_validation_test_split(
        _assays(), group_column="enzyme_id", validation_size=0.2, test_size=0.2, random_seed=1
    )
    splits.assert_disjoint()
    assert not set(splits.train.enzyme_id) & set(splits.test.enzyme_id)
