"""命令行之间共享的装配逻辑。

一次运行要按固定的顺序把东西建起来，而这个顺序本身就是规范的一部分：

1. 参考场先算，因为初值只有这一个来源；
2. ``hat T_*`` 由**训练**半径的有效时间范围冻结，之后不许再变；
3. 尺度集合由半径与 ``hat T_*`` 得到；
4. 模型由尺度与参考初值得到（前沿网络的输出层偏置取自参考初值）。

任何一步跳过或调换，都会让某个下游模块悄悄用上不一致的数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import Config, load_config
from ..data import compatibility as compat
from ..data.extracted_case import DetectorSettings, ExtractedCase, FieldMapping, load_extracted_case
from ..models.coupled_model import CoupledModel, build_model
from ..physics.detonation import DETONATION_MODEL_VERSION, DetonationConfig
from ..physics.precursor import PrecursorConfig, ReferenceField, solve_reference_field
from ..physics import material as mat
from ..physics.similitude import (
    SimilitudeScales,
    freeze_time_scale,
)
from ..sampling.sampler import SamplingConfig


#: 参考场缓存格式版本。**任何会改变这个场怎么算出来的东西**都必须 +1：
#: 坐标语义、查询语义、产物场构造、网格交接方式、前置演化格式。
#:
#: 为什么不能只靠"求解参数"做键：上面那些改动一个参数都不动——网格数、CFL、
#: 增长比、爆速全是原样——产出的场却不同。少了这一版，一次修好能量记账之后，
#: 缓存里那份旧场会被原样复用，报告里连一个字都不会变。这个坑踩过两次
#: （产物场的压力功、交接的守恒搬移）。
REFERENCE_FIELD_VERSION = 12

#: 参考场初值的**物理方案**标记。方案 A 时代写下的缓存里没有这个键，
#: 所以它们在签名比对时一律不命中——不是靠文件名或时间戳猜的。
REFERENCE_CONSTRUCTION = "scheme_b_products_only_detonation"

#: 只有这个状态才允许进入正式初值入口与训练入口。
ACCEPTED_STATUS = "ACCEPTED"


class ReferenceNotAccepted(RuntimeError):
    """参考场尚未通过 §5 验收，不得用于正式初值或训练。

    故意**不**继承 ``SetupError``：它不是"配置没配好"，而是一次真实的物理验收
    失败，必须能被单独捕获与单独特判，不能被"把配置改一改"消掉。
    """


def _field_content_hash(reference: Any) -> str:
    """场**内容**标识。委托给 :func:`physics.precursor.field_content_hash`。

    实现在 ``physics`` 侧，因为**指标在 ``solve_reference_field`` 里算出来的时候
    就要绑定它**——绑定不能只在 CLI 这一层做，否则"重新保存"这条路会绕过它。
    """
    from ..physics.precursor import field_content_hash

    return field_content_hash(reference)


def assert_reference_accepted(reference: ReferenceField) -> None:
    """§5：未通过验收的场不得被正式入口接受。

    这不是一句元数据注释——注释不会阻止任何事。加载与训练入口调用本函数，
    状态不是 ``ACCEPTED`` 就抛错。诊断命令要生成试算产物，必须显式传
    ``allow_unaccepted_reference=True``，并在产物里保留未验收的标记。

    **§7.3：不能只读 ``acceptance_status`` 字符串。** 字符串是上一次判定写下的
    结果，而判定所依据的**规则**（容差、验收项、物理/数据要求）可能已经改了。
    所以这里拿保存下来的 ``acceptance_metrics`` 与 ``acceptance_evidence``，按
    **当前**的 ``assess_acceptance`` 重新判定一次。一个只带
    ``{"acceptance_status": "ACCEPTED"}``、没有指标与证据的元数据，重新判定会
    因为"缺少证据 / 物理未核实 / 数据未就绪"而失败——这正是要堵的漏洞。

    **§7.2：证据必须属于当前场。** 重新判断保存的指标，还不等于检查了当前场。
    若元数据里带了场内容哈希，而当前数组算出来的哈希对不上，说明场在验收之后被
    改过，证据失效——直接拒绝，而不是拿旧指标继续判。

    **验收记录必须绑定到它被计算时的那份场内容**（Trusty URIs 的"内容寻址"
    原则：内容变了就是另一个对象，不能继承原对象的身份与证据）。``field_content_hash``
    只在**保存时**算，防的是"保存后场被改"；``acceptance_field_hash`` 在
    **验收判定时**算，防的是"重新保存把旧指标挂到了新场上"——两条都要查，缺一
    条就有一个方向漏掉。
    """
    metadata = getattr(reference, "metadata", {}) or {}
    from ..physics.precursor import assess_acceptance

    metrics = dict(metadata.get("acceptance_metrics") or {})
    evidence = dict(metadata.get("acceptance_evidence") or {})
    physics = metadata.get("physics_basis_status")
    # **这一层不评估数据契约**：参考场的验收是"这份场算得对不对"，与"数据有没有
    # 按新契约重提"是两件事，各自阻塞各自的入口（§7.1 要求三类状态分开记）。
    # 以前把数据的 ready 传进来，结果是参考场永远不可能 ACCEPTED——一个不相关的
    # 条件把数值/物理的判定也一起钉死了。
    verdict = assess_acceptance(
        metrics,
        convergence_evidence=evidence,
        physics_basis=physics,
    )
    if verdict["acceptance_status"] != ACCEPTED_STATUS:
        # 先报**真正的失败原因**（指标超限 / 缺证据 / 物理未核实 / 数据未就绪）。
        # 内容标识检查是**附加**的一道门，不替代这一条——否则所有失败的场都会
        # 得到一句"缺少哈希"，把真正的阻塞项盖住。
        raise ReferenceNotAccepted(
            f"参考场按当前规则重判为 {verdict['acceptance_status']!r}，"
            f"不是 {ACCEPTED_STATUS!r}；原因：{verdict.get('acceptance_reason', '未记录')}"
        )

    # 到这里数值 + 物理 + 数据三态都过了。**再加一道内容绑定检查**：证据必须
    # 确实属于当前这份场。缺任何一个标识都拒绝——"缺证据默认通过"是不能接受的。
    #
    # 查两个标识，它们防的是两个不同的方向：
    #   ``metrics_field_hash``    指标是不是在这份场上算的（防"旧指标挂新数据"）
    #   ``acceptance_field_hash`` 验收记录是不是发给这份场的（防"重新保存改身份"）
    current = _field_content_hash(reference)
    for key, label in (
        ("metrics_field_hash", "指标绑定的场标识"),
        ("acceptance_field_hash", "验收记录绑定的场标识"),
    ):
        stored = metadata.get(key)
        if stored is None:
            raise ReferenceNotAccepted(
                f"缺少{label}（{key}）：无法确认验收证据属于当前场。"
                "缺证据不得默认为通过——重新求解并重新验收。"
            )
        if stored != current:
            raise ReferenceNotAccepted(
                f"{label}与当前场不一致：这份验收证据属于另一份场"
                "（内容变了就是另一个对象，不能继承原对象的证据）。"
                "重新求解并重新验收，不得复用旧证据。"
            )


def _attach_cache_provenance(reference: ReferenceField, cache: Path, signature: Mapping[str, Any]) -> None:
    """把这份场**来自哪个缓存文件、签名是什么**记在元数据里。

    ``record_acceptance`` 要凭这两个字段把判定写回**正式缓存**。少了它们，那个
    函数会静默什么都不做——内存里的理由更新了、磁盘上的缓存还是旧的"缺少证据"，
    而两条路径看起来都"成功返回"。
    """
    reference.metadata["_cache_path"] = str(cache)
    reference.metadata["_cache_signature"] = dict(signature)


def record_acceptance(reference: ReferenceField, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """把**完整**的 §5 判定写回参考场的元数据，并重新落盘缓存。

    只有跑完了加密、时间步与尺度三项核查的调用方（``cli.reference_field``）才有
    资格调用它。缺证据时 ``assess_acceptance`` 会直接记未验收，所以这里不需要
    再判断一次"够不够"。

    **物理与数据状态要原样带下去**：``assess_acceptance`` 的 ``physics_basis`` 与
    ``data_contract`` 默认是"未核实 / 未就绪"。若这里不把已有状态传进去，一次
    "数值验收写回"会把本来已经 ``已核实 / ready`` 的状态覆盖回默认值——状态链路
    在这里被悄悄切断。所以先从元数据里读出已有状态，再传。
    """
    from ..physics.precursor import assess_acceptance

    metadata = reference.metadata or {}
    metrics = dict(metadata.get("acceptance_metrics") or {})
    # **指标必须是在这份场上算出来的。** 这是本函数唯一能防止"旧指标被重新保存
    # 挂到新数据上"的地方：指标在 ``solve_reference_field`` 里算出来时就绑定了
    # 当时的场内容（``metrics_field_hash``）。若当前场已经变了，写回就等于给新数据
    # 发一张旧证明——必须拒绝，而不是重新盖一个当前的哈希。
    current = _field_content_hash(reference)
    metrics_hash = metadata.get("metrics_field_hash")
    if metrics_hash is None:
        raise ReferenceNotAccepted(
            "验收指标没有绑定到场内容（缺 metrics_field_hash）：无法确认这些指标"
            "是在当前这份场上算的。拒绝写回。"
        )
    if metrics_hash != current:
        raise ReferenceNotAccepted(
            "验收指标是在另一份场上算出来的：当前场的内容标识与指标绑定的不一致。"
            "内容变了就是另一个对象，旧指标不能作为新数据的合格证明——重新求解"
            "并重新验收。"
        )

    verdict = assess_acceptance(
        metrics,
        convergence_evidence=evidence,
        physics_basis=metadata.get("physics_basis_status"),
        # 数据契约不在这一层判（见 assert_reference_accepted 的说明）。
    )
    reference.metadata.update(verdict)
    reference.metadata["acceptance_evidence"] = dict(evidence)
    # 把验收记录绑定到**被验收的那份场内容**（Trusty URIs 原则）：保存操作只负责
    # 存储，不重新赋予"验收通过"的含义。这里记下指标与证据对应的是哪一份场，
    # 场之后被改（重新保存得到新标识），旧验收记录就不再用，加载时会被拒。
    #
    # 用的是上面已经核对过的 ``current``——不是重新算一遍。重新算会把"当前场的
    # 哈希"写进去，正是上一版留下漏洞的写法。
    reference.metadata["acceptance_field_hash"] = current

    cache_dir = reference.metadata.get("_cache_path")
    if cache_dir:
        signature = reference.metadata.get("_cache_signature")
        if signature is not None:
            _save_reference(reference, Path(cache_dir), signature)
    return verdict


class SetupError(RuntimeError):
    """运行的前置条件不满足。"""


def configure_stdout() -> None:
    """把标准输出切成 UTF-8。

    报告里有中文和 ``µ``、``ρ`` 一类字符，而 Windows 控制台默认是 GBK；不切换
    的话，报告写文件没问题，但打印到终端会直接抛 ``UnicodeEncodeError``，把一次
    已经算完的运行变成一次崩溃。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive
                pass


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="配置文件路径（YAML）")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="覆盖配置项，形如 --set training.max_updates=10（可重复）",
    )
    parser.add_argument("--verbose", action="store_true")


def load_config_from_args(args) -> Config:
    return load_config(args.config, args.overrides)


# --------------------------------------------------------------------------
# 参考场
# --------------------------------------------------------------------------


#: 参考半径，方案的尺度基准（§4）。相对时间 epsilon 以它为准。
REFERENCE_CHARGE_RADIUS = 0.05

#: 参考交接延迟 epsilon_ref，s。
REFERENCE_START_DELAY = 1.0e-6


def reference_detonation_velocity(config: Config) -> float:
    """卡片爆速 ``D``，m/s。

    **不接受"没有"**：没有它就退回方案 A 的时间轴（只减 ``epsilon``），而两条
    时间轴都只有秒的量纲。所以在返回之前就把口径定下来，调用方不需要各自判断。
    """
    section = config.section("reference_field")
    material_section = config.section("materials") if hasattr(config, "section") else {}
    return float(
        section.get("detonation_velocity")
        or (material_section or {}).get("detonation_velocity_reference")
        or mat.DETONATION_VELOCITY_REFERENCE
    )


def reference_detonation_config(config: Config) -> DetonationConfig:
    section = config.section("reference_field")
    return DetonationConfig(cells=int(section.get("detonation_cells", 2000)))


def handover_relative_time(config: Config, radius: float) -> float:
    """相对 ``T_b`` 的交接时刻 ``epsilon(R_0) = k epsilon_ref``，s。

    **这不是方案 A 的那个 1e-6 秒绝对时间。** 方案 B 里 PINN 的初值在
    ``T_h = T_b + epsilon``，而参考场的采样时间是相对 ``T_b`` 的，所以"初值那一
    刻"对应的采样时间是 ``epsilon``。两个数都是 1e-6 量级，混用不会报错，只会
    让整条曲线挪到另一个时刻。
    """
    return REFERENCE_START_DELAY * (float(radius) / REFERENCE_CHARGE_RADIUS)


def reference_config(config: Config) -> PrecursorConfig:
    section = config.section("reference_field")
    return PrecursorConfig(
        cells_inside=int(section.get("cells_inside", 2000)),
        outer_radius_factor=float(section.get("outer_radius_factor", 40.0)),
        growth=float(section.get("growth", 1.005)),
        cfl=float(section.get("cfl", 0.2)),
        q1=float(section.get("q1", 1.5)),
        q2=float(section.get("q2", 0.06)),
    )


def build_reference(
    config: Config, *, verbose: bool = False, allow_unaccepted: bool = False
) -> ReferenceField:
    """求解参考场，或在缓存可用时直接读取。

    缓存键里带着求解参数与**构造来源**，所以改了网格、增长比或初值物理方案都不会
    被旧缓存冒充。

    ``allow_unaccepted`` 只给诊断命令用。正式训练入口必须让它保持 ``False``——
    §5 要求"未验收试算场不得被正式入口接受"，而这句话只有在有代码执行它的时候
    才成立。
    """
    section = config.section("reference_field")
    radius = float(section.get("charge_radius_m", 0.05))
    settings = reference_config(config)
    factor = float(section.get("handover_margin", 1.2))
    count = int(section.get("sample_count", 41))
    detonation_cells = int(section.get("detonation_cells", 2000))
    detonation_velocity = reference_detonation_velocity(config)

    cache_dir = config.path_of("training.output_dir", default=Path("outputs/run"))
    cache = Path(cache_dir) / "reference_field.npz"
    # ``field_version`` 是缓存格式的版本，不是求解参数的版本。第 2 版起参考场带
    # **逐帧的当前单元中心**（第 1 版把初始单元中心当成了空间坐标，查询出来的
    # 状态属于另一个位置）。旧缓存必须失效并重建——它里面的坐标本身就是错的，
    # 参数一样并不能让它变对。
    # `construction` 是**初值物理方案**的标记，与格式版本分开记。缺了它，
    # 方案 A 时代写下的缓存会因为其余参数相同而命中——它的时间轴、初值场、
    # 甚至 `initial_condition` 字段都属于 A，但文件本身看不出这一点。
    #
    # 签名里**不**放验收状态：验收是**场自己的属性**（写在 `metadata` 里随缓存
    # 一起走），由 `assert_reference_accepted` 在每次加载后重新判。把它写进签名
    # 等于让"当初是不是验收通过"决定"现在能不能复用"，而这两件事会脱钩。
    signature = {
        "field_version": REFERENCE_FIELD_VERSION,
        # 产出这个场的**物理模型与格式**的版本。它不随任何求解参数变化，所以
        # 少了它，一次"参数没变、格式变了"的修复不会让缓存失效。
        "detonation_model_version": DETONATION_MODEL_VERSION,
        "construction": REFERENCE_CONSTRUCTION,
        "detonation_cells": detonation_cells,
        "charge_radius": radius,
        "cells_inside": settings.cells_inside,
        "growth": settings.growth,
        "outer_radius_factor": settings.outer_radius_factor,
        "cfl": settings.cfl,
        "q1": settings.q1,
        "q2": settings.q2,
        "handover_margin": factor,
        "detonation_velocity": detonation_velocity,
        "sample_count": count,
    }
    if cache.exists():
        # ``with`` 是必须的，不是风格问题：``np.load`` 返回的 ``NpzFile`` **持有
        # 文件句柄**，不关就一直开着。而这条路径后面会走到 ``_save_reference`` →
        # ``os.replace(tmp, cache)``——在 Windows 上替换一个仍被打开的文件会
        # ``PermissionError: [WinError 5] 拒绝访问``，也就是**读到缓存之后再也存不
        # 回去**。Linux 上不报错，所以这个坑只在 Windows 上露头。
        with np.load(cache, allow_pickle=True) as payload:
            stored = json.loads(str(payload["signature"].reshape(-1)[0]))
            if stored.get("construction") != REFERENCE_CONSTRUCTION:
                if verbose:
                    print(
                        "[reference] 缓存不是方案 B 构造的 "
                        f"(construction={stored.get('construction')!r})；拒绝复用并重解"
                    )
            elif stored == signature:
                if verbose:
                    print(f"[reference] cache hit: {cache}")
                cached = _reference_from_npz(payload)
                _attach_cache_provenance(cached, cache, signature)
                if not allow_unaccepted:
                    assert_reference_accepted(cached)
                return cached
            if verbose:
                print(f"[reference] cache signature differs; re-solving")

    if verbose:
        print(
            f"[reference] solving 1D spherical Lagrangian precursor "
            f"(R0={radius} m, {settings.cells_inside} cells inside)"
        )
    # 参考场的时间跨度由**交接时刻**决定（§3）：PINN 的初值就是 T_h 那一刻的两侧
    # 场，所以前置求解必须至少覆盖到 T_h。旧配置写的是 4 * epsilon_ref（方案 A 的
    # 数），在 50 mm 上只到 4 us，而 T_h = 8.215 us——查询会在覆盖范围外直接失败。
    # 采样窗口是**相对 T_b** 的时间：0 是波前到达装药表面的时刻，交接时刻
    # T_h = T_b + epsilon 落在相对时间 epsilon 处。窗口取 [0, margin*epsilon]，
    # 于是既覆盖 epsilon 段，又在它之后留出 margin-1 的余量。
    # 这里**不能**再用绝对物理时间：T_h = 8.215 us 会被当成相对时间，跑到
    # T_b + 8.215 us，是实际物理时长的两倍多。
    epsilon = handover_relative_time(config, radius)
    margin = float(section.get("handover_margin", 1.2))
    sample_times = np.linspace(0.0, margin * epsilon, count)
    reference = solve_reference_field(
        charge_radius=radius,
        config=settings,
        # 爆速是必填的：漏传会直接失败，而不是悄悄退回 A 的时间轴。
        detonation_velocity=detonation_velocity,
        detonation_config=reference_detonation_config(config),
        sample_times=list(sample_times),
        check_outer_boundary=float(section.get("check_outer_boundary", 0.5)),
    )
    if not allow_unaccepted:
        # 新建的场同样要过闸门：构造本身没通过 §5 验收时，**第一次**求解也不能
        # 悄悄返回一个"看起来算完了"的场。
        assert_reference_accepted(reference)
    _attach_cache_provenance(reference, cache, signature)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _save_reference(reference, cache, signature)
    return reference


def _save_reference(reference: ReferenceField, path: Path, signature: dict[str, Any]) -> None:
    # 把场**内容**哈希写进元数据（§7.2）：验收证据必须能核对到当前这份场。
    # 加载时 ``assert_reference_accepted`` 会重算并比对——场被改过就拒绝。
    reference.metadata["field_content_hash"] = _field_content_hash(reference)
    # **原子写**（§7.3）：先写同目录下的临时文件，再 ``os.replace`` 覆盖目标。
    # 直接 ``np.savez_compressed(path, ...)`` 会在写坏（磁盘满、进程被杀）时留下
    # 半份文件；下一次加载读到一个截断的 npz，报错位置离真正的错误十万八千里。
    # ``os.replace`` 在同一文件系统内是原子的，目标要么是旧文件、要么是完整的新
    # 文件，不存在中间态。
    #
    # **临时文件名必须以 ``.npz`` 结尾**：``np.savez_compressed`` 对字符串/Path 会
    # 在文件名不以此结尾时**自动补** ``.npz``（NumPy 文档明确）。若 tmp 取
    # ``formal.npz.tmp``，NumPy 实际写的是 ``formal.npz.tmp.npz``，随后
    # ``os.replace("formal.npz.tmp", ...)`` 直接 FileNotFoundError。所以这里用
    # ``<原名>.tmp.npz``，NumPy 看到 ``.npz`` 结尾就不会再补。
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        times=reference.times,
        cell_radii=reference.cell_radii,
        density=reference.density,
        velocity=reference.velocity,
        specific_energy=reference.specific_energy,
        material=reference.material,
        contact_radius=reference.contact_radius,
        shock_radius=reference.shock_radius,
        reference_radius=np.array([reference.reference_radius]),
        signature=np.array([json.dumps(signature, sort_keys=True)]),
        metadata=np.array([json.dumps(reference.metadata, sort_keys=True, default=str)]),
    )
    os.replace(tmp, path)


def _reference_from_npz(payload) -> ReferenceField:
    if "cell_radii" not in payload.files:
        raise SetupError(
            "this reference-field cache predates the per-frame Lagrangian coordinates "
            "(it has a single 'radii' array). Delete it and re-solve: its spatial "
            "coordinates were the *initial* cell centres, so every query returned the "
            "state of a different location."
        )
    return ReferenceField(
        times=payload["times"],
        cell_radii=payload["cell_radii"],
        density=payload["density"],
        velocity=payload["velocity"],
        specific_energy=payload["specific_energy"],
        material=payload["material"],
        contact_radius=payload["contact_radius"],
        shock_radius=payload["shock_radius"],
        reference_radius=float(payload["reference_radius"][0]),
        metadata=json.loads(str(payload["metadata"].reshape(-1)[0])),
    )


# --------------------------------------------------------------------------
# 算例
# --------------------------------------------------------------------------


def case_specs(config: Config, *, split: str | None = None) -> list[dict[str, Any]]:
    section = config.section("data")
    specs = list(section.get("cases", []) or [])
    if split is not None:
        specs = [s for s in specs if str(s.get("split")) == split]
    return specs


def load_cases(
    config: Config,
    *,
    split: str | None = None,
    strict: bool | None = None,
) -> list[ExtractedCase]:
    """读取算例；缺数据时的严格程度取决于它在这次运行里扮演什么角色。

    * ``split="train"``（以及显式 ``strict=True``）—— 缺一个训练算例是**致命**的。
      少一个半径意味着"三个半径等权联合更新"这句话不再成立，而那是最不该被
      静默降级的东西。
    * 其它情况（缺一个验证或盲测算例）—— 只是**那一次对照跑不了**。报告缺失，
      继续处理其余的。

    ``strict=None`` 时按 ``split == "train"`` 推断。
    """
    if strict is None:
        strict = split == "train"
    section = config.section("data")
    mapping = FieldMapping.from_mapping(section.get("field_map"))
    settings = DetectorSettings(
        **{
            key: value
            for key, value in (section.get("detector") or {}).items()
            if key in DetectorSettings.__dataclass_fields__
        }
    )
    cases: list[ExtractedCase] = []
    missing: list[str] = []
    for specification in case_specs(config, split=split):
        directory = Path(str(specification["directory"]))
        if not (directory / "raw_line.npz").exists():
            missing.append(f"{specification.get('case_id')} -> {directory}")
            continue
        case = load_extracted_case(
            directory,
            case_id=specification.get("case_id"),
            field_mapping=mapping,
            settings=settings,
            gamma_air=float(config.get("air.gamma", 1.4)),
        )
        case.source["split"] = specification.get("split")
        expected = float(specification.get("charge_radius_m", case.charge_radius))
        if abs(expected - case.charge_radius) > 1.0e-9:
            raise SetupError(
                f"case {case.case_id!r}: configuration says R_0 = {expected} m but the "
                f"data declares {case.charge_radius} m"
            )
        cases.append(case)
    if missing and strict:
        raise SetupError(
            "these configured cases have no readable dataset:\n  "
            + "\n  ".join(missing)
            + "\nMissing data is reported, never silently substituted."
        )
    if not cases:
        raise SetupError(f"no case matched split={split!r}")
    if missing:
        for case in cases:
            case.source.setdefault("notes", [])
        setattr(cases[0], "_missing_cases", list(missing))
    return cases


# --------------------------------------------------------------------------
# 数据兼容性报告
# --------------------------------------------------------------------------


def compatibility_report_path(config: Config) -> Path | None:
    """兼容性报告的位置。

    优先用 ``data.compatibility_report``；没设时退回 ``output_dir/dyna_audit/``
    下由 ``audit_dyna`` 写出的那一份。两处都没有就是"没有证据"，而不是"通过"。
    """
    section = config.section("data")
    declared = section.get("compatibility_report")
    if declared:
        return Path(str(declared))
    audit_dir = config.path_of("training.output_dir", default=Path("outputs/run")) / "dyna_audit"
    candidate = audit_dir / compat.REPORT_FILENAME
    return candidate if candidate.exists() else None


def resolve_case_compatibility(
    config: Config, cases: Sequence[ExtractedCase]
) -> tuple[dict[str, compat.CaseCompatibility], Path | None]:
    """把每个训练算例的兼容性状态凑齐。

    报告里没有的算例不是"默认通过"，而是 ``unknown``；它自己记录的分区来源仍然
    会被读一遍——那一项与报告无关，是算例数据本身的性质。
    """
    path = compatibility_report_path(config)
    stored = compat.load_report(path) if path is not None else {}
    resolved: dict[str, compat.CaseCompatibility] = {}
    for case in cases:
        entry = stored.get(case.case_id)
        partition, evidence = compat.material_partition_verdict(case)
        if entry is None:
            missing = compat.compatibility_from_findings(
                case.case_id,
                (),
                source=str(path) if path is not None else "no compatibility report found",
                material_partition=partition,
                material_partition_evidence=evidence,
            )
            missing.reasons.insert(
                0,
                "没有读到这个算例的初始化审查结论（"
                + (f"{path} 里没有它的条目" if path is not None
                   else "没有找到 data_compatibility.json；先跑 audit_dyna")
                + "）",
            )
            resolved[case.case_id] = missing
            continue
        # 报告里的分区状态若比算例自身的更乐观，以算例自身的为准：报告是快照，
        # 数据是当下的事实。
        if partition == compat.INCOMPATIBLE and entry.checks.get("material_partition") != compat.INCOMPATIBLE:
            entry.checks["material_partition"] = partition
            entry.reasons.append("material_partition：算例记录显示分区不是来自体积分数")
            entry.status = compat.INCOMPATIBLE
        entry.evidence.setdefault("material_partition", evidence)
        resolved[case.case_id] = entry
    return resolved, path


# --------------------------------------------------------------------------
# 运行上下文
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    """一次运行需要的全部对象。"""

    config: Config
    reference: ReferenceField
    training_cases: list[ExtractedCase]
    time_scale: float
    time_scale_provenance: dict[str, Any]
    scales: dict[float, SimilitudeScales]
    model: CoupledModel | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    notes: list[str] = field(default_factory=list)
    compatibility: dict[str, compat.CaseCompatibility] = field(default_factory=dict)
    compatibility_report: str | None = None
    #: 数据契约判定（§15）。训练入口按它阻塞；predict/compare 只记录不阻塞。
    data_contract: Any = None
    #: 验证隔离判定（§16）。同样只在训练入口阻塞。
    validation_isolation: Any = None

    @property
    def training_radii(self) -> list[float]:
        return [float(c.charge_radius) for c in self.training_cases]

    def scales_for(self, charge_radius: float) -> SimilitudeScales:
        return self.scales[float(charge_radius)]

    def describe(self) -> dict[str, Any]:
        return {
            "training_cases": [c.describe() for c in self.training_cases],
            "training_radii": self.training_radii,
            "time_scale": self.time_scale,
            "time_scale_provenance": self.time_scale_provenance,
            "scales": {str(k): v.as_dict() for k, v in self.scales.items()},
            "sampling": {
                name: getattr(self.sampling, name)
                for name in SamplingConfig.__dataclass_fields__
            },
            "data_compatibility": {
                "report": self.compatibility_report,
                "cases": {k: v.as_dict() for k, v in self.compatibility.items()},
            },
            "data_contract": (
                None if self.data_contract is None else self.data_contract.as_dict()
            ),
            "validation_isolation": (
                None if self.validation_isolation is None
                else self.validation_isolation.as_dict()
            ),
            "notes": self.notes,
        }


def build_context(
    config: Config,
    *,
    verbose: bool = False,
    with_model: bool = True,
    seed: int | None = None,
) -> RunContext:
    """按固定顺序装配一次运行。"""
    reference = build_reference(config, verbose=verbose)
    training = load_cases(config, split="train")

    # ``hat T_*`` 只用训练半径。把验证或盲测半径放进来会改变整个训练的时间输入
    # 分布，那是一种没有写进任何地方的泄漏。
    valid_times = {
        float(case.charge_radius): float(case.last_valid_pinn_time()) for case in training
    }
    time_scale, provenance = freeze_time_scale(valid_times)
    scales = {
        float(case.charge_radius): SimilitudeScales.for_radius(
            case.charge_radius,
            time_scale=time_scale,
            # 方案 B 的时间原点 T_h = R_0/D + epsilon 要卡片爆速。从算例自身取，
            # 不给模块默认值：省略它等于悄悄退回 A 的时间轴。
            detonation_velocity=case.detonation_velocity,
        )
        for case in training
    }

    compatibility, report_path = resolve_case_compatibility(config, training)
    # §15 数据契约。**只在这里求值、不在这里抛错**：predict/compare 也需要
    # build_context，而它们读的是模型与参考场，数据没就绪不该阻止它们。阻塞放在
    # 训练入口（cli/train.py），那里才真正消费监督标签。
    from ..data.contract import evaluate_data_contract

    contract = evaluate_data_contract(training)

    # §16 验证隔离：500 mm 不得参与训练、初值标定、归一化极值、早停或选模。
    # 与数据契约一样，**只在这里求值、不在这里抛错**（predict/compare 也要走
    # build_context，它们不训练、不该被这条拦住）；阻塞在训练入口。
    from ..data.isolation import evaluate_validation_isolation

    reference_radius = config.get("reference_field.charge_radius_m", None)
    isolation = evaluate_validation_isolation(
        config,
        training,
        reference_radius_m=None if reference_radius is None else float(reference_radius),
    )
    context = RunContext(
        config=config,
        reference=reference,
        training_cases=training,
        data_contract=contract,
        validation_isolation=isolation,
        time_scale=time_scale,
        time_scale_provenance=provenance,
        scales=scales,
        sampling=SamplingConfig.from_config(config),
        compatibility=compatibility,
        compatibility_report=str(report_path) if report_path is not None else None,
    )
    if with_model:
        first = training[0]
        view = reference.scale_to(first.charge_radius)
        contact, shock = view.contact_and_shock_at(scales[first.charge_radius].start_time())
        context.model = build_model(
            config,
            scales=scales[first.charge_radius],
            initial_contact_hat=float(np.asarray(contact).reshape(-1)[0] / first.charge_radius),
            initial_gap_hat=float(
                (np.asarray(shock).reshape(-1)[0] - np.asarray(contact).reshape(-1)[0])
                / first.charge_radius
            ),
            seed=seed,
        )
    return context


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "detach"):
        return float(value.detach())
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=jsonable), encoding="utf-8")
    return path


__all__ = [
    "RunContext",
    "SetupError",
    "add_common_arguments",
    "build_context",
    "build_reference",
    "configure_stdout",
    "case_specs",
    "jsonable",
    "load_cases",
    "load_config_from_args",
    "reference_config",
    "write_json",
]
