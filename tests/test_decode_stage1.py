import numpy as np
import pandas as pd

from data.data_process import data_process


def legacy_mix_cells(dp, x, y, cell_type_list):
    fracs = dp.mixup_fraction(len(cell_type_list))
    samp_fracs = np.multiply(fracs, dp.sample_size)
    samp_fracs = list(map(round, samp_fracs))
    fracs = np.divide(samp_fracs, sum(samp_fracs))

    fracs_complete = [0] * len(cell_type_list)
    for i, act in enumerate(cell_type_list):
        idx = cell_type_list.index(act)
        fracs_complete[idx] = fracs[i]

    artificial_samples = []
    for i, ct in enumerate(cell_type_list):
        cells_sub = x.loc[y[dp.random_type] == ct]
        if cells_sub.shape[0] > 0 and samp_fracs[i] <= len(cells_sub):
            cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
            cells_sub = cells_sub.iloc[cells_fraction, :]
            artificial_samples.append(cells_sub)
        else:
            return None

    df_samp = pd.concat(artificial_samples, axis=0)
    df_samp = df_samp.sum(axis=0)
    return df_samp, fracs_complete


def build_inputs():
    x = pd.DataFrame(
        [
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 2.0, 1.0],
            [3.0, 1.0, 2.0],
        ]
    )
    y = pd.DataFrame({"CellType": ["A", "A", "B", "B", "C", "C"]})
    return x, y


def test_mix_cells_matches_legacy_for_fixed_seed():
    dp = data_process(["A", "B", "C"], tissue_name="test", sample_size=4)
    x, y = build_inputs()
    celltype_pools = dp._build_celltype_pools(x, y)

    np.random.seed(2026)
    legacy_sample, legacy_label = legacy_mix_cells(dp, x, y, dp.type_list)

    np.random.seed(2026)
    optimized_sample, optimized_label = dp.mix_cells(
        cell_type_list=dp.type_list,
        celltype_pools=celltype_pools,
    )

    pd.testing.assert_series_equal(optimized_sample, legacy_sample)
    assert optimized_label == legacy_label


def test_mix_cells_returns_none_when_a_fraction_exceeds_available_cells():
    dp = data_process(["A", "B"], tissue_name="test", sample_size=3)
    x = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]])
    y = pd.DataFrame({"CellType": ["A", "B"]})
    celltype_pools = dp._build_celltype_pools(x, y)

    original_mixup_fraction = dp.mixup_fraction
    dp.mixup_fraction = lambda cell_num: np.array([0.99, 0.01])
    try:
        assert dp.mix_cells(cell_type_list=dp.type_list, celltype_pools=celltype_pools) is None
    finally:
        dp.mixup_fraction = original_mixup_fraction


def test_mix_cells_matches_legacy_on_rounded_zero_fraction_case():
    dp = data_process(["A", "B", "C"], tissue_name="test", sample_size=3)
    x, y = build_inputs()
    celltype_pools = dp._build_celltype_pools(x, y)

    original_mixup_fraction = dp.mixup_fraction
    dp.mixup_fraction = lambda cell_num: np.array([0.8, 0.19, 0.01])
    try:
        np.random.seed(7)
        legacy_sample, legacy_label = legacy_mix_cells(dp, x, y, dp.type_list)

        np.random.seed(7)
        optimized_sample, optimized_label = dp.mix_cells(
            cell_type_list=dp.type_list,
            celltype_pools=celltype_pools,
        )
    finally:
        dp.mixup_fraction = original_mixup_fraction

    pd.testing.assert_series_equal(optimized_sample, legacy_sample)
    assert optimized_label == legacy_label
