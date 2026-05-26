import pytest

from simpoly.poly_arena.simulation import protocol
from simpoly.poly_arena.simulation.stages import NPT

COMMON_KWARGS = dict(units="metal", time_step=0.0005)


class TestTgCoolingProtocol:
    @pytest.fixture()
    def p(self) -> protocol.LAMMPSProtocol:
        return protocol.build_tg_cooling_protocol(
            temp_start_k=500,
            temp_end_k=200,
            temp_step_k=50,
            **COMMON_KWARGS,
        )

    def test_only_npt_stages(self, p: protocol.LAMMPSProtocol) -> None:
        assert all(isinstance(s, NPT) for s in p.stages)

    def test_temperature_schedule(self, p: protocol.LAMMPSProtocol) -> None:
        names = [s.name for s in p.stages]
        assert names == [
            "cool_000_500K",
            "cool_001_450K",
            "cool_002_400K",
            "cool_003_350K",
            "cool_004_300K",
            "cool_005_250K",
            "cool_006_200K",
        ]

    def test_non_aligned_endpoint(self) -> None:
        p = protocol.build_tg_cooling_protocol(
            temp_start_k=533, temp_end_k=213, temp_step_k=20, **COMMON_KWARGS
        )
        names = [s.name for s in p.stages]
        assert names[0] == "cool_000_533K"
        assert names[-1] == f"cool_{len(names) - 1:03d}_213K"


def test_21step_then_cooling_concatenation() -> None:
    eq = protocol.build_21steps_protocol(temp_final_k=533, **COMMON_KWARGS)
    cool = protocol.build_tg_cooling_protocol(
        temp_start_k=533, temp_end_k=213, temp_step_k=20, **COMMON_KWARGS
    )
    combined = protocol.build_21step_then_cooling_protocol(
        temp_start_k=533, temp_end_k=213, temp_step_k=20, **COMMON_KWARGS
    )
    assert len(combined.stages) == len(eq.stages) + len(cool.stages)
    names = [s.name for s in combined.stages]
    assert len(set(names)) == len(names)
    assert names[len(eq.stages)] == "cool_000_533K"
