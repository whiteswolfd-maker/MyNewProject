"""§15.4、§15.6、§15.7：球心硬约束、正值输出、前沿次序、速度可为负；
理想气体法向激波满足三条 RH 且违反熵条件的状态不被接受；
接触面同压同速但不同密度/内能不被错误惩罚。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from spherical_blast_pinn.losses.terms import contact_loss, shock_loss
from spherical_blast_pinn.models.air_net import AirFields
from spherical_blast_pinn.models.front_net import FrontNet
from spherical_blast_pinn.models.product_net import ProductNet
from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.moving_frame import FrontData
from spherical_blast_pinn.physics.similitude import SimilitudeScales

import torch


R0 = 0.05
SCALES = SimilitudeScales.for_radius(R0, time_scale=2.0, detonation_velocity=6930.0)
GAMMA = 1.4
AMBIENT = mat.ambient_air()


# --------------------------------------------------------------------------
# §15.4 硬结构与输出范围
# --------------------------------------------------------------------------


def test_centre_velocity_is_exactly_zero_by_construction():
    """``u_p = U_* xi z_u`` 在 ``xi = 0`` 精确为零，不需要球心损失。"""
    net = ProductNet(scales=SCALES, seed=0)
    xi = torch.zeros((1, 1), dtype=torch.float64)
    zero_s = torch.zeros((1, 1), dtype=torch.float64)
    fields = net(zero_s, xi)
    assert float(fields.velocity) == 0.0


def test_product_density_and_energy_are_strictly_positive():
    net = ProductNet(scales=SCALES, seed=0)
    s = torch.linspace(0.0, 5.0, 7, dtype=torch.float64).reshape(-1, 1)
    xi = torch.linspace(1e-4, 0.999, 7, dtype=torch.float64).reshape(-1, 1)
    fields = net(s, xi)
    assert torch.all(fields.density > 0.0)
    assert torch.all(fields.specific_energy > 0.0)


def test_velocity_is_not_clipped_and_can_be_negative():
    """速度允许为负：做一个负速度的场，检查它真的能取到负值。

    这里直接构造一个输出层权重使 ``z_u < 0``，而不是希望随机初始化给出负值。
    """
    net = ProductNet(scales=SCALES, seed=0)
    with torch.no_grad():
        net.mlp.network[-1].weight.zero_()
        net.mlp.network[-1].bias.copy_(torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64))
    s = torch.zeros((1, 1), dtype=torch.float64)
    xi = torch.tensor([[0.5]], dtype=torch.float64)
    fields = net(s, xi)
    assert float(fields.velocity) < 0.0


def test_front_ordering_and_anchoring_hold_by_construction():
    net = FrontNet(scales=SCALES, initial_contact_hat=1.08, initial_gap_hat=0.02, seed=0)
    s = torch.linspace(0.0, 5.0, 9, dtype=torch.float64).reshape(-1, 1)
    positions = net.positions(s)
    assert torch.all(positions.contact_hat >= 1.0), "R_c 必须不小于 R_0"
    assert torch.all(positions.gap_hat > 0.0), "空气层厚度必须为正"
    assert torch.all(positions.shock_hat > positions.contact_hat)


def test_front_positions_match_the_reference_at_the_handover_time():
    net = FrontNet(scales=SCALES, initial_contact_hat=1.08, initial_gap_hat=0.02, seed=0)
    positions = net.positions(torch.zeros((1, 1), dtype=torch.float64))
    assert float(positions.contact_hat) == pytest.approx(1.08, rel=1e-9)
    assert float(positions.gap_hat) == pytest.approx(0.02, rel=1e-9)


def test_unanchored_construction_can_drop_below_R0():
    """关掉锚定就该能掉到 1 以下——否则"锚定"这件事没有被真正验证。"""
    net = FrontNet(
        scales=SCALES,
        initial_contact_hat=1.08,
        initial_gap_hat=0.02,
        anchor_contact_at_charge_surface=False,
        seed=0,
    )
    with torch.no_grad():
        net.mlp.network[-1].weight.zero_()
        net.mlp.network[-1].bias.copy_(torch.tensor([-5.0, 0.5413], dtype=torch.float64))
    positions = net.positions(torch.zeros((1, 1), dtype=torch.float64))
    assert float(positions.contact_hat) < 1.0


# --------------------------------------------------------------------------
# §15.6 主激波的完整 RH
# --------------------------------------------------------------------------


@dataclass
class _StubBatch:
    s_contact: torch.Tensor
    s_shock: torch.Tensor
    scales: SimilitudeScales
    charge_radius: float


class _StubFront:
    """一组固定前沿：``R_s = Rs0 + D t``，接触面同速。"""

    def __init__(self, shock_speed: float, contact_speed: float, contact: float, shock: float):
        self.shock_speed = float(shock_speed)
        self.contact_speed = float(contact_speed)
        self.contact = float(contact)
        self.shock = float(shock)

    def data(self, s, *, charge_radius=None, create_graph=True):
        shape = s.shape
        return FrontData(
            contact_radius=torch.full(shape, self.contact, dtype=s.dtype),
            shock_radius=torch.full(shape, self.shock, dtype=s.dtype),
            contact_speed=torch.full(shape, self.contact_speed, dtype=s.dtype),
            shock_speed=torch.full(shape, self.shock_speed, dtype=s.dtype),
        )


class _StubAir:
    """给出一个规定的波后状态。"""

    def __init__(self, density, velocity, energy):
        self.density = float(density)
        self.velocity = float(velocity)
        self.energy = float(energy)

    def fields(self, s, eta):
        shape = s.shape
        density = torch.full(shape, self.density, dtype=s.dtype)
        energy = torch.full(shape, self.energy, dtype=s.dtype)
        return AirFields(
            density=density,
            velocity=torch.full(shape, self.velocity, dtype=s.dtype),
            specific_energy=energy,
            pressure=(GAMMA - 1.0) * density * energy,
        )


class _StubModel:
    def __init__(self, front, air, undisturbed):
        self._front = front
        self._air = air
        self.undisturbed = undisturbed
        self.gamma_air = GAMMA

    def front_data(self, s, *, charge_radius=None, create_graph=True):
        return self._front.data(s, charge_radius=charge_radius)

    def air_fields(self, s, eta):
        return self._air.fields(s, eta)


def _ideal_gas_jump(shock_mach: float):
    """理想气体法向激波的正解。"""
    m = float(shock_mach)
    rho0, p0 = AMBIENT.density, AMBIENT.pressure
    density_ratio = ((GAMMA + 1.0) * m * m) / ((GAMMA - 1.0) * m * m + 2.0)
    pressure_ratio = (2.0 * GAMMA * m * m - (GAMMA - 1.0)) / (GAMMA + 1.0)
    c0 = math.sqrt(GAMMA * p0 / rho0)
    density = rho0 * density_ratio
    pressure = p0 * pressure_ratio
    velocity = 2.0 * c0 * (m * m - 1.0) / ((GAMMA + 1.0) * m)
    energy = pressure / ((GAMMA - 1.0) * density)
    return density, velocity, energy, m * c0


@pytest.mark.parametrize("mach", [2.0, 5.0, 20.0])
def test_an_exact_ideal_gas_shock_satisfies_all_three_rh_conditions(mach):
    density, velocity, energy, speed = _ideal_gas_jump(mach)
    model = _StubModel(
        _StubFront(shock_speed=speed, contact_speed=velocity, contact=R0, shock=1.5 * R0),
        _StubAir(density, velocity, energy),
        AMBIENT,
    )
    batch = _StubBatch(
        s_contact=torch.zeros((4, 1), dtype=torch.float64),
        s_shock=torch.zeros((4, 1), dtype=torch.float64),
        scales=SCALES,
        charge_radius=R0,
    )
    breakdown = shock_loss(batch, model)
    assert float(breakdown.total) < 1.0e-20, breakdown.parts
    assert float(breakdown.monitors.get("entropy_violation_rate", 0.0)) == 0.0


def test_a_rarefaction_is_not_accepted_as_a_shock():
    """把波后状态换成一个稀疏波的状态，RH 残差必须显著非零。"""
    density, velocity, energy, speed = _ideal_gas_jump(5.0)
    model = _StubModel(
        _StubFront(shock_speed=speed, contact_speed=velocity, contact=R0, shock=1.5 * R0),
        # 密度低于波前：这是稀疏，不是激波。
        _StubAir(0.4 * AMBIENT.density, -velocity, energy),
        AMBIENT,
    )
    batch = _StubBatch(
        s_contact=torch.zeros((4, 1), dtype=torch.float64),
        s_shock=torch.zeros((4, 1), dtype=torch.float64),
        scales=SCALES,
        charge_radius=R0,
    )
    breakdown = shock_loss(batch, model)
    assert float(breakdown.total) > 1.0e-4


# --------------------------------------------------------------------------
# §15.7 接触面
# --------------------------------------------------------------------------


class _StubFields:
    def __init__(self, density, velocity, energy, pressure):
        self.density = density
        self.velocity = velocity
        self.specific_energy = energy
        self.pressure = pressure


class _ContactModel:
    """接触面两侧的一对状态。

    ``front_speed`` 是界面速度，``field_velocity`` 是两侧流体速度。两者分开
    是必要的：把它们写成同一个数，"速度不匹配"这一项就永远检验不到——它会
    恒等于零，而一个恒等于零的检验比没有检验更糟。
    """

    def __init__(self, front_speed, field_velocity=None, product_pressure=5.0e8, air_pressure=5.0e8):
        self._front = float(front_speed)
        self._field = float(front_speed if field_velocity is None else field_velocity)
        self._product_pressure = float(product_pressure)
        self._air_pressure = float(air_pressure)

    def front_data(self, s, *, charge_radius=None, create_graph=True):
        shape = s.shape
        return FrontData(
            contact_radius=torch.full(shape, R0, dtype=s.dtype),
            shock_radius=torch.full(shape, 1.5 * R0, dtype=s.dtype),
            contact_speed=torch.full(shape, self._front, dtype=s.dtype),
            shock_speed=torch.full(shape, 2.0 * self._front, dtype=s.dtype),
        )

    def product_fields(self, s, xi):
        shape = s.shape
        density = torch.full(shape, 1630.0, dtype=s.dtype)
        energy = torch.full(shape, 3.0e6, dtype=s.dtype)
        return _StubFields(
            density=density,
            velocity=torch.full(shape, self._field, dtype=s.dtype),
            energy=energy,
            pressure=torch.full(shape, self._product_pressure, dtype=s.dtype),
        )

    def air_fields(self, s, eta):
        shape = s.shape
        density = torch.full(shape, 12.0, dtype=s.dtype)
        energy = torch.full(shape, 2.0e6, dtype=s.dtype)
        return _StubFields(
            density=density,
            velocity=torch.full(shape, self._field, dtype=s.dtype),
            energy=energy,
            pressure=torch.full(shape, self._air_pressure, dtype=s.dtype),
        )


def test_contact_is_blind_to_a_density_and_energy_jump():
    """接触面上密度与比内能**不要求**相等，所以它们不同不该被惩罚。

    如果实现里偷偷加了密度匹配或内能匹配，这一项会立刻变大。
    """
    speed = 4200.0
    batch = _StubBatch(
        s_contact=torch.zeros((16, 1), dtype=torch.float64),
        s_shock=torch.zeros((16, 1), dtype=torch.float64),
        scales=SCALES,
        charge_radius=R0,
    )
    model = _ContactModel(speed)
    breakdown = contact_loss(batch, model, contact_pressure=5.0e8)
    assert float(breakdown.total) == pytest.approx(0.0, abs=1e-24)


def _batch(n: int = 16) -> _StubBatch:
    return _StubBatch(
        s_contact=torch.zeros((n, 1), dtype=torch.float64),
        s_shock=torch.zeros((n, 1), dtype=torch.float64),
        scales=SCALES,
        charge_radius=R0,
    )


def test_contact_penalises_a_pressure_mismatch():
    """两侧速度都等于界面速度，只有压力不同：应当只由压力项贡献。"""
    model = _ContactModel(4200.0, product_pressure=5.0e8, air_pressure=7.5e8)
    breakdown = contact_loss(_batch(), model, contact_pressure=5.0e8)
    assert float(breakdown.parts["contact_speed_product"]) == pytest.approx(0.0, abs=1e-24)
    assert float(breakdown.parts["contact_speed_air"]) == pytest.approx(0.0, abs=1e-24)
    assert float(breakdown.parts["contact_pressure"]) > 1.0e-3


def test_contact_penalises_a_velocity_mismatch():
    """压力相同，只有两侧速度与界面速度不同：应当只由速度项贡献。"""
    model = _ContactModel(9000.0, field_velocity=4200.0)
    breakdown = contact_loss(_batch(), model, contact_pressure=5.0e8)
    assert float(breakdown.parts["contact_pressure"]) == pytest.approx(0.0, abs=1e-24)
    expected = ((4200.0 - 9000.0) / SCALES.reference_velocity) ** 2
    assert float(breakdown.parts["contact_speed_product"]) == pytest.approx(expected, rel=1e-12)
    assert float(breakdown.parts["contact_speed_air"]) == pytest.approx(expected, rel=1e-12)
