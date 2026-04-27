from __future__ import annotations


def test_dangerous_upgrade_detection() -> None:
    from bike_router.services.surface_ai import is_dangerous_upgrade

    assert is_dangerous_upgrade("paved_rough", "paved_good") is True
    assert is_dangerous_upgrade("unpaved_hard", "paved_good") is True
    assert is_dangerous_upgrade("unpaved_soft", "paved_good") is True
    assert is_dangerous_upgrade("rough_or_unpaved", "paved_good") is True
    assert is_dangerous_upgrade("paved_good", "paved_good") is False
    assert is_dangerous_upgrade("unpaved_soft", "paved_rough") is False


def test_dangerous_upgrade_metrics_raw_effective_and_severity() -> None:
    from bike_router.services.surface_ai import dangerous_upgrade_metrics

    metrics = dangerous_upgrade_metrics(
        ["paved_rough", "unpaved_soft", "paved_good"],
        ["paved_good", "paved_good", "paved_good"],
        ["unknown", "paved_good", "paved_good"],
        [10.0, 20.0, 30.0],
    )

    assert metrics["dangerous_upgrade_count_raw"] == 2
    assert metrics["dangerous_upgrade_count_effective"] == 1
    assert metrics["dangerous_upgrade_rate_raw_among_non_good"] == 1.0
    assert metrics["dangerous_upgrade_rate_effective_among_non_good"] == 0.5
    assert metrics["dangerous_upgrade_length_m_effective"] == 20.0
    assert metrics["dangerous_upgrade_weighted_score_raw"] == 4.0
    assert metrics["dangerous_upgrade_weighted_score_effective"] == 3.0
