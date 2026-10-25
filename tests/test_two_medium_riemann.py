import math

import numpy as np
import pytest

from spherical_blast_pinn.physics import material as mat
from spherical_blast_pinn.physics.eos import (
    jwl_pressure_numpy,
    jwl_pressure_scalar,
    jwl_sound_speed_squared_numpy,
    jwl_sound_speed_squared_scalar,
)
from spherical_blast_pinn.physics.two_medium_riemann import (
    RiemannError,
    solve_ideal_gas_riemann,
    solve_jwl_air_riemann,
)


def _solution(**kwargs):
    params = mat.product_jwl()
    e_p = mat.PRODUCT_VOLUME_ENERGY / params.rho0
    return solve_jwl_air_riemann(
        product_eos=params,
        product_density=params.rho0,
        product_specific_energy=e_p,
        air_density=mat.AIR_DENSITY,
        air_pressure=mat.AIR_PRESSURE,
        air_gamma=mat.AIR_GAMMA,
        **kwargs,
    )


def test_default_riemann_wave_pattern_and_star_state():
    sol = _solution()
    assert sol.wave_pattern == "product_rarefaction-contact-air_shock"
    assert sol.air_initial_pressure < sol.p_star < sol.product_initial_pressure
    assert sol.u_star > 0.0
    assert sol.shock_speed > sol.u_star > 0.0
    assert sol.product_rho_star > 0.0
    assert sol.product_e_star > 0.0
    assert sol.product_c_star > 0.0
    assert sol.air_rho_star > sol.air_initial_density
    assert sol.air_e_star > sol.air_initial_specific_energy
    assert sol.air_c_star > 0.0
    # The product initial pressure is not the star pressure, and the air shock
    # speed is not the detonation velocity.
    assert sol.p_star != sol.product_initial_pressure
    assert sol.shock_speed != mat.DETONATION_VELOCITY_REFERENCE
    assert sol.checks["contact_pressure_residual"] < 1.0e-8
    assert sol.checks["contact_velocity_residual"] < 1.0e-10
    assert sol.checks["air_mass_rh_residual"] < 1.0e-10
    assert sol.checks["air_momentum_rh_residual"] < 1.0e-10
    assert sol.checks["air_energy_rh_residual"] < 1.0e-10


def test_product_release_relation_uses_actual_jwl_and_invariant():
    sol = _solution()
    p_eos = jwl_pressure_scalar(
        sol.product_rho_star, sol.product_e_star, mat.product_jwl()
    )
    assert abs(p_eos - sol.p_star) / sol.p_star < 1.0e-8
    c2 = jwl_sound_speed_squared_scalar(
        sol.product_rho_star, sol.product_e_star, mat.product_jwl()
    )
    assert c2 > 0.0
    assert math.isclose(math.sqrt(c2), sol.product_c_star, rel_tol=1.0e-12)

    # Independently recompute the left-going Riemann invariant from the table.
    V_star = sol.isentrope.state_at_pressure(sol.p_star)["V"]
    table = sol.isentrope
    mask = table.V <= V_star
    V = np.concatenate([table.V[mask], [V_star]])
    c = np.concatenate([table.c[mask], [sol.product_c_star]])
    integrand = c / V
    invariant = float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(V)))
    assert abs(invariant - sol.u_star) / sol.u_star < 1.0e-4


def test_air_shock_satisfies_all_three_rankine_hugoniot_relations():
    sol = _solution()
    rho0 = sol.air_initial_density
    p0 = sol.air_initial_pressure
    e0 = sol.air_initial_specific_energy
    D = sol.shock_speed
    u = sol.u_star
    rho1 = sol.air_rho_star
    e1 = sol.air_e_star
    p1 = sol.p_star
    mass = rho0 * D - rho1 * (D - u)
    momentum = (p1 - p0) - rho0 * D * u
    energy = (e1 - e0) - 0.5 * (p1 + p0) * (1.0 / rho0 - 1.0 / rho1)
    assert abs(mass) / (rho0 * D) < 1.0e-10
    assert abs(momentum) / (p1 - p0) < 1.0e-10
    assert abs(energy) / max(abs(e1), 1.0) < 1.0e-10
    assert D - u > 0.0
    assert rho1 > rho0


def test_rarefaction_head_tail_and_fan_states():
    sol = _solution()
    c0 = sol.product_initial_sound_speed
    assert math.isclose(sol.rarefaction_head_speed, -c0, rel_tol=1.0e-14)
    assert math.isclose(
        sol.rarefaction_tail_speed,
        sol.u_star - sol.product_c_star,
        rel_tol=1.0e-14,
    )
    head = sol.isentrope.state_at_xi(sol.rarefaction_head_speed)
    tail = sol.isentrope.state_at_xi(sol.rarefaction_tail_speed)
    assert math.isclose(head["rho"], sol.product_initial_density, rel_tol=1.0e-12)
    assert head["u"] == 0.0
    assert math.isclose(head["p"], sol.product_initial_pressure, rel_tol=1.0e-10)
    assert math.isclose(tail["rho"], sol.product_rho_star, rel_tol=1.0e-9)
    assert math.isclose(tail["u"], sol.u_star, rel_tol=1.0e-7)
    assert tail["p"] > 0.0
    assert np.all(np.diff(sol.isentrope.xi) > 0.0)
    assert np.all(np.diff(sol.isentrope.p) < 0.0)


def test_integration_step_sensitivity_is_small():
    coarse = _solution(integration_step_factor=0.005)
    fine = _solution(integration_step_factor=0.00125)
    assert abs(coarse.p_star - fine.p_star) / fine.p_star < 1.0e-8
    assert abs(coarse.u_star - fine.u_star) / fine.u_star < 1.0e-8
    assert abs(coarse.shock_speed - fine.shock_speed) / fine.shock_speed < 1.0e-10


def test_standard_ideal_gas_sod_and_equal_state():
    sod = solve_ideal_gas_riemann(
        left_density=1.0,
        left_velocity=0.0,
        left_pressure=1.0,
        left_gamma=1.4,
        right_density=0.125,
        right_velocity=0.0,
        right_pressure=0.1,
        right_gamma=1.4,
    )
    assert math.isclose(sod.p_star, 0.30313017805054077, rel_tol=1.0e-8)
    assert math.isclose(sod.u_star, 0.9274526200489095, rel_tol=1.0e-8)
    assert sod.left_wave == "rarefaction"
    assert sod.right_wave == "shock"
    assert sod.checks["contact_velocity_residual"] < 1.0e-10

    equal = solve_ideal_gas_riemann(
        left_density=1.0,
        left_velocity=0.0,
        left_pressure=1.0,
        left_gamma=1.4,
        right_density=1.0,
        right_velocity=0.0,
        right_pressure=1.0,
        right_gamma=1.4,
    )
    assert equal.p_star == 1.0
    assert equal.u_star == 0.0
    assert equal.left_wave == equal.right_wave == "none"
    assert equal.left_rho_star == equal.right_rho_star == 1.0


def test_scalar_jwl_twins_match_the_numpy_eos():
    params = mat.product_jwl()
    e_p = mat.PRODUCT_VOLUME_ENERGY / params.rho0
    for rho, e in (
        (params.rho0, e_p),
        (94.6749796251097, 800549.0112611477),
        (500.0, 1.0e6),
    ):
        p_np = float(jwl_pressure_numpy(np.array([rho]), np.array([e]), params)[0])
        p_scalar = jwl_pressure_scalar(rho, e, params)
        assert p_scalar == pytest.approx(p_np, rel=1.0e-13)
        c2_np = float(
            jwl_sound_speed_squared_numpy(np.array([rho]), np.array([e]), params)[0]
        )
        c2_scalar = jwl_sound_speed_squared_scalar(rho, e, params)
        assert c2_scalar == pytest.approx(c2_np, rel=1.0e-12)


def test_invalid_riemann_inputs_fail_explicitly():
    params = mat.product_jwl()
    e_p = mat.PRODUCT_VOLUME_ENERGY / params.rho0
    with pytest.raises(RiemannError):
        solve_jwl_air_riemann(
            product_eos=params,
            product_density=params.rho0,
            product_specific_energy=e_p,
            air_density=mat.AIR_DENSITY,
            air_pressure=mat.AIR_PRESSURE * 2.0e5,  # p_air > p_product
            air_gamma=mat.AIR_GAMMA,
        )
    with pytest.raises(RiemannError):
        _solution(integration_step_factor=0.0)
    with pytest.raises(RiemannError):
        _solution(root_relative_tolerance=0.0)
    with pytest.raises(RiemannError):
        solve_ideal_gas_riemann(
            left_density=1.0,
            left_velocity=0.0,
            left_pressure=1.0,
            left_gamma=1.0,
            right_density=1.0,
            right_velocity=0.0,
            right_pressure=1.0,
            right_gamma=1.4,
        )
