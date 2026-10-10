"""真实数据整链验证（提示词 §16.1 第 3 步）。

覆盖"原始读取 → 导出 → 重读 → 采样 → 观测损失"这条闭环，用的数据是**按当前
提取器**重新提取的 50 mm（含 §9 的逐帧几何重构、逐字段有效性、球对称份额）。

数据不在本机时整个模块跳过——不是失败，是"没跑"，这两者在报表上必须能分开。
`REAL_CASE_DIR` 指向的目录由

.. code-block:: bash

    python -m spherical_blast_pinn.cli.extract_profiles \
      --config configs/base.yaml --case tnt_50mm \
      --deck <deck>/input.k --d3plot-dir <d3plot dir> \
      --out F:/2/extracted_geom/50mm_c --charge-radius 0.05 \
      --states 0 20 100 300 500 778

生成（早期/中期/末期的少量代表状态，§16.1 第 2 步）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import torch

REAL_CASE_DIR = Path("F:/2/extracted_geom/50mm_c")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiments" / "geom_smoke.yaml"
RADIUS = 0.05
TIME_SCALE = 2.0

pytestmark = pytest.mark.skipif(
    not (REAL_CASE_DIR / "raw_line.npz").exists()
    or not (REAL_CASE_DIR / "material.npz").exists(),
    reason=(
        f"the re-extracted case is not on this machine ({REAL_CASE_DIR}); this is a "
        "not-run, not a failure. See the module docstring for the extraction command"
    ),
)


@pytest.fixture(scope="module")
def real_case():
    from spherical_blast_pinn.data.extracted_case import load_extracted_case

    return load_extracted_case(REAL_CASE_DIR, case_id="tnt_50mm")


@pytest.fixture(scope="module")
def reference():
    """参考 B 初场。``allow_unaccepted=True`` 是**诊断**用法（§5）。

    正式入口必须让它保持 ``False``——这里只是要一个能喂进采样器的场，好把链路
    本身跑通，**不**主张这个场已经验收。
    """
    from spherical_blast_pinn.cli._common import build_reference
    from spherical_blast_pinn.config import load_config

    return build_reference(load_config(CONFIG_PATH), allow_unaccepted=True)


def test_the_reextracted_case_satisfies_the_data_contract(real_case):
    """§15：这批数据在契约上必须真的就绪，而不是"看起来像就绪"。"""
    from spherical_blast_pinn.data.contract import evaluate_data_contract

    verdict = evaluate_data_contract([real_case])
    assert verdict.ready, verdict.reasons

    detail = verdict.detail["cases"][0]
    assert detail["symmetry_factor"] == 8.0
    assert detail["detonation_velocity_m_s"] > 0.0
    assert "volume fraction" in detail["material_masks_from"].lower()
    assert detail["field_validity_fields"] == [
        "density", "pressure", "specific_energy", "velocity"
    ]
    eligibility = detail["front_supervision_eligibility"]
    assert set(eligibility) == {"contact", "shock"}
    assert detail["n_frames_supervisable"] > 0


def test_the_reextracted_case_carries_per_field_validity(real_case):
    """逐字段有效性必须真的带着，而不是退化成一张联合掩码。

    **这份数据上两者的交集恰好相等**，这里不作"必须更宽"的断言：那是一条关于
    某一批数据的经验性质，不是关于机制的。要让"更宽"成立，得有一帧在某些字段上
    缺失、在另一些上完好——六帧代表状态里没有出现，而为了让测试通过去伪造一帧
    比不测更糟。机制本身由 ``tests/test_data_contract_fields.py`` 用构造出来的
    导出直接验证（缺字段时必须退回 ``None`` 而不是伪装成"都有效"）。
    """
    validity = real_case.field_validity
    assert validity is not None
    assert sorted(validity) == ["density", "pressure", "specific_energy", "velocity"]
    joint = real_case.valid
    for name, mask in validity.items():
        assert mask.shape == joint.shape
        assert mask.dtype == bool
    per_field = np.stack(
        [validity["density"], validity["velocity"], validity["specific_energy"]],
        axis=0,
    ).all(axis=0)
    # 三项联合起来必须**至少**覆盖联合掩码：逐字段只会放宽，不会收紧
    assert np.all(per_field >= joint)


def test_the_observation_closure_runs_on_real_data(real_case, reference):
    """读取 → 采样 → 观测算子 → 损失，在真实数据上真的走一遍。"""
    from spherical_blast_pinn.losses.objective import contact_pressure_scale, total_loss
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet
    from spherical_blast_pinn.physics.similitude import SimilitudeScales
    from spherical_blast_pinn.sampling.sampler import RadiusSampler, SamplingConfig

    scales = SimilitudeScales.for_radius(
        RADIUS, time_scale=TIME_SCALE, detonation_velocity=real_case.detonation_velocity
    )
    view = reference.scale_to(RADIUS)
    sampler = RadiusSampler(
        case=real_case, scales=scales, reference=view,
        config=SamplingConfig(
            n_pde_product=16, n_pde_air=16, n_ic_product=8, n_ic_air=8,
            n_contact=4, n_shock=4, n_data_product=16, n_data_air=16, n_data_front=4,
        ),
        seed=0,
    )
    # 完整窗口：三组监督都必须真的有数据进来
    sampler.set_time_window(1.0)
    batch = sampler.batch()

    for name in ("s_data_product", "s_data_air", "rho_data_product", "lo_data_product",
                 "hi_data_product", "coverage_data_product"):
        value = getattr(batch, name)
        assert value is not None and value.numel() > 0, (
            f"{name} 是空的：真实数据没有进入监督"
        )
    # 支持区是**一段**，不是退化成一个点
    assert float((batch.hi_data_product - batch.lo_data_product).min()) > 0.0

    # 阶段 1 的窄窗口（§14.2）：这份导出只有 6 个代表状态，其中确认的激波都在
    # 后期，所以早期窗口里**空气组本来就是空的**。那不是缺陷，是这批数据的实际
    # 形状；要紧的是空组不会把整条链路弄坏，也不会伪装成"这一项误差为零"。
    narrow = RadiusSampler(
        case=real_case, scales=scales, reference=view,
        config=SamplingConfig(
            n_pde_product=16, n_pde_air=16, n_ic_product=8, n_ic_air=8,
            n_contact=4, n_shock=4, n_data_product=16, n_data_air=16, n_data_front=4,
        ),
        seed=0,
    )
    narrow.set_time_window(0.25)
    early = narrow.batch()
    assert early.s_data_product.numel() > 0, "早期窗口里应当仍有产物监督"
    assert early.s_data_air.numel() == 0, (
        "这批数据的早期窗口里不该有空气监督；出现了说明空气侧的门槛被放松了"
    )

    model = CoupledModel(
        product=ProductNet(scales=scales, width=8, depth=1, seed=0),
        air=AirNet(scales=scales, width=8, depth=1, seed=1),
        front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.1, seed=2),
        scales=scales,
    )
    pressure_scale = contact_pressure_scale(
        view, charge_radius=RADIUS, detonation_velocity=real_case.detonation_velocity
    )
    breakdown = total_loss([batch], model, contact_pressure=pressure_scale, pde_weight=0.1)
    assert torch.isfinite(breakdown.total)
    breakdown.total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_a_real_data_checkpoint_resumes_reproducibly(real_case, reference, tmp_path):
    """检查点保存 → 恢复：恢复出来的必须与连续跑是**同一个**模型（§15）。"""
    from spherical_blast_pinn.data import compatibility as compat
    from spherical_blast_pinn.models.air_net import AirNet
    from spherical_blast_pinn.models.coupled_model import CoupledModel
    from spherical_blast_pinn.models.front_net import FrontNet
    from spherical_blast_pinn.models.product_net import ProductNet
    from spherical_blast_pinn.physics.similitude import SimilitudeScales
    from spherical_blast_pinn.sampling.sampler import SamplingConfig
    from spherical_blast_pinn.training.trainer import Trainer

    scales = SimilitudeScales.for_radius(
        RADIUS, time_scale=TIME_SCALE, detonation_velocity=real_case.detonation_velocity
    )
    sampling = SamplingConfig(
        n_pde_product=16, n_pde_air=16, n_ic_product=8, n_ic_air=8,
        n_contact=4, n_shock=4, n_data_product=16, n_data_air=16, n_data_front=4,
    )

    class _Config:
        def section(self, name):
            return {}

        def get(self, key, default=None):
            return default

        def path_of(self, key, default=None):
            return default

        def to_dict(self):
            return {}

    def make(output: Path) -> Trainer:
        model = CoupledModel(
            product=ProductNet(scales=scales, width=8, depth=1, seed=0),
            air=AirNet(scales=scales, width=8, depth=1, seed=1),
            front=FrontNet(scales=scales, width=8, depth=1, initial_contact_hat=1.1, seed=2),
            scales=scales,
        )
        return Trainer(
            config=_Config(), cases=[real_case], reference=reference, model=model,
            scales_by_radius={RADIUS: scales}, time_scale=TIME_SCALE,
            sampling=sampling, seed=3, output_dir=output,
            log_every=1000, checkpoint_every=1000, check_every=1000,
            compatibility={real_case.case_id: compat.CaseCompatibility(
                real_case.case_id, status=compat.INCOMPATIBLE
            )},
            allow_unverified_data=True,
        )

    steps = 2
    continuous = make(tmp_path / "a")
    continuous.run(max_updates=steps, verbose=False)

    interrupted = make(tmp_path / "b")
    interrupted.run(max_updates=steps, verbose=False)
    checkpoint = interrupted.output_dir / "last.pt"
    assert checkpoint.exists()

    # 从一个**全新的** trainer 恢复，参数必须与连续跑一致。
    resumed = make(tmp_path / "c")
    resumed.load(checkpoint)
    assert resumed.step == steps
    for name, value in continuous.model.state_dict().items():
        assert torch.equal(
            value, resumed.model.state_dict()[name]
        ), f"{name} 在恢复后与连续跑不一致"


def test_the_reference_gate_still_refuses_on_real_data():
    """研究闸门在这批新数据上**依然**是关着的。

    数据契约就绪不等于物理依据核实。这条测试的作用是：任何一天有人把
    ``assert_reference_accepted`` 放宽到能过，它会立刻失败。
    """
    from spherical_blast_pinn.cli._common import ReferenceNotAccepted, build_reference
    from spherical_blast_pinn.config import load_config

    with pytest.raises(ReferenceNotAccepted) as error:
        build_reference(load_config(CONFIG_PATH))
    message = str(error.value)
    assert "NOT ACCEPTED" in message
    assert "physics" in message.lower() or "物理" in message
