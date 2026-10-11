"""compare 命令：命令行这一层必须真的跑起来（提示词 §1、§7）。

复核给的必现崩溃：

.. code-block:: python

    with np.load(args.prediction, allow_pickle=True) as handle:
        data = {key: handle[key] for key in handle.files}
    prediction = {key: data[key] for key in data.files}      # data 已经是字典

``dict`` 没有 ``files`` 属性，所以 ``compare`` 在**任何**输入下都抛
``AttributeError``。已有的单元测试只覆盖 ``compare_fields()``，正好绕开了
命令行这一层——所以崩溃一直没被发现。

这里的测试因此必须写到"命令真的产出了报告文件"为止，不能只调内部函数。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spherical_blast_pinn.cli import compare as compare_module
from spherical_blast_pinn.data.extracted_case import load_extracted_case
from spherical_blast_pinn.physics import material as mat
from test_data_contract_fields import _write_export

CASE_ID = "tnt_synth"
CONTACT = 0.05
SHOCK = 0.10


def _fronts(n_t: int) -> tuple[np.ndarray, np.ndarray]:
    """与算例的 ``material`` 表**逐帧一致**的前沿：``R_c = 0.05(1 + 0.05 i)``。

    用常数前沿会让一小段单元落在"参考说产物、前沿说空气"的位置上，凭空造出
    ``region_mismatch``。这里要测的是插值与计数，不是错配，所以两边对齐。
    """
    index = np.arange(n_t, dtype=np.float64)
    contact = 0.05 * (1.0 + 0.05 * index)
    return contact, contact * 2.0


def _region(radii: np.ndarray, contact: np.ndarray, shock: np.ndarray) -> np.ndarray:
    return np.where(
        radii[None, :] < contact[:, None], 0,
        np.where(radii[None, :] < shock[:, None], 1, 2),
    ).astype(np.int64)


def _prediction(case, **overrides) -> dict:
    """与算例**逐字一致**的一份预测，用来验证"自洽时误差为零"。"""
    n_t = case.times.size
    contact, shock = _fronts(n_t)
    radii = np.asarray(case.radius, dtype=np.float64)
    data = {
        "start_time": float(case.start_time),
        "pinn_time": np.asarray(case.pinn_times, dtype=np.float64),
        "physical_time": np.asarray(case.times, dtype=np.float64),
        "radius": radii,
        "charge_radius": float(case.charge_radius),
        "region": _region(radii, contact, shock),
        "contact_radius": contact,
        "shock_radius": shock,
        "density": np.asarray(case.density, dtype=np.float64),
        "velocity": np.asarray(case.velocity, dtype=np.float64),
        "specific_energy": np.asarray(case.specific_energy, dtype=np.float64),
        "pressure": np.asarray(case.pressure, dtype=np.float64),
        "scale_product_density": 1630.0,
        "scale_air_density": 1.225,
        "scale_velocity": mat.REFERENCE_VELOCITY,
        "scale_specific_energy": mat.PRODUCT_SPECIFIC_ENERGY,
        "ambient_pressure": mat.AIR_PRESSURE,
        "scaled_time_coverage": 3.0e-4,
    }
    data.update(overrides)
    return data


def _write_prediction(path: Path, prediction: dict) -> Path:
    np.savez(path, **prediction)
    return path


def _write_config(directory: Path, case_directory: Path) -> Path:
    base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(
        "\n".join([
            f"inherits: {base.as_posix()}",
            "",
            "data:",
            "  cases:",
            f"    - case_id: {CASE_ID}",
            "      charge_radius_m: 0.05",
            f"      directory: {case_directory.as_posix()}",
            "      split: train",
        ]) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def environment(tmp_path):
    case_directory = _write_export(tmp_path / "case")
    case = load_extracted_case(case_directory, case_id=CASE_ID)
    config = _write_config(tmp_path / "config", case_directory)
    return case, config, tmp_path


# --------------------------------------------------------------------------
# §1 命令行必须真的跑完并写出报告
# --------------------------------------------------------------------------


def test_the_compare_command_runs_end_to_end_and_writes_a_report(environment):
    """**复核点名的必现崩溃。** 走真实 ``main(argv)``，检查报告文件真的产生。

    这条测试不替换任何内部实现：预测文件是真实写出的 npz，算例是真实读进来的
    导出目录，配置是真实加载的 yaml，比对与写出都是被调用的那一份代码。
    """
    case, config, tmp_path = environment
    prediction_path = _write_prediction(
        tmp_path / "predict.npz", _prediction(case)
    )
    output = tmp_path / "out"

    code = compare_module.main([
        "--config", str(config),
        "--prediction", str(prediction_path),
        "--case", CASE_ID,
        "--output", str(output),
    ])
    assert code == 0

    report_path = output / f"compare_{CASE_ID}.json"
    assert report_path.exists(), "报告文件没有产生"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["case_id"] == CASE_ID
    assert report["checks"]["charge_radius_match"] is True
    assert report["checks"]["time_origin_match"] is True
    assert report["fields"]["available"] is True
    # 预测与参考逐字一致 ⇒ 每个可比场量的相对 L2 都是零。
    for medium in ("product", "air"):
        for name in ("density", "velocity", "specific_energy"):
            key = f"{medium}_{name}_relative_l2"
            if key in report["fields"]:
                assert report["fields"][key] == pytest.approx(0.0, abs=1e-12), key


def test_the_report_says_which_reasons_dropped_points(environment):
    """§7.6/§7.7：报告里要能把"跨间断不能插""支撑点非有限""域外"分开看。"""
    case, config, tmp_path = environment
    prediction = _prediction(case)
    # 把一部分**被判分的**场量改成 NaN：这些点必须被**计数**，不能静默消失。
    # （压力不在 ``compare_fields`` 的被判分场量里，所以这里改的是密度；帧也要
    # 挑一个**可监督**的——第 0 帧在参考里不可用，改它等于什么都没测。）
    density = np.array(prediction["density"], dtype=np.float64)
    density[2, :5] = np.nan
    prediction["density"] = density
    prediction_path = _write_prediction(tmp_path / "predict.nan.npz", prediction)
    output = tmp_path / "out_nan"

    assert compare_module.main([
        "--config", str(config), "--prediction", str(prediction_path),
        "--case", CASE_ID, "--output", str(output),
    ]) == 0
    report = json.loads(
        (output / f"compare_{CASE_ID}.json").read_text(encoding="utf-8")
    )
    assert report["fields"]["sample_reason_counts"]["prediction_non_finite_at_support"] > 0
    assert report["fields"]["sample_reason_meaning"]["prediction_non_finite_at_support"]


# --------------------------------------------------------------------------
# §1.2 缺字段 / 错形状 / 单帧：明确拒绝，且**不写出报告**
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "drop, needle",
    [
        ("density", "density"),
        ("region", "region"),
        ("scale_velocity", "scale_velocity"),
        ("contact_radius", "contact_radius"),
    ],
)
def test_a_missing_required_field_is_refused_with_its_name(environment, drop, needle):
    case, config, tmp_path = environment
    prediction = _prediction(case)
    prediction.pop(drop)
    path = tmp_path / f"missing_{drop}.npz"
    with pytest.raises(compare_module.PredictionError, match=needle):
        compare_module.load_prediction(_write_prediction(path, prediction))


def test_a_field_with_the_wrong_frame_count_is_refused(environment):
    case, config, tmp_path = environment
    prediction = _prediction(case)
    prediction["velocity"] = np.ones((1, np.asarray(prediction["radius"]).size))
    with pytest.raises(compare_module.PredictionError, match="velocity"):
        compare_module.validate_prediction(prediction)


def test_a_descending_radial_axis_is_refused(environment):
    case, config, tmp_path = environment
    prediction = _prediction(case)
    prediction["radius"] = np.asarray(prediction["radius"])[::-1].copy()
    with pytest.raises(compare_module.PredictionError, match="strictly increasing"):
        compare_module.validate_prediction(prediction)


def test_a_single_frame_prediction_is_refused_not_extrapolated(environment):
    """单帧预测没有定义"两帧之间"的值。入口明确拒绝，不给一个常数。"""
    case, config, tmp_path = environment
    prediction = _prediction(case)
    for key in ("pinn_time", "physical_time", "density", "velocity",
                "specific_energy", "pressure", "region"):
        array = np.asarray(prediction[key])
        prediction[key] = array[:1]
    prediction["contact_radius"] = np.asarray(prediction["contact_radius"])[:1]
    prediction["shock_radius"] = np.asarray(prediction["shock_radius"])[:1]
    with pytest.raises(compare_module.PredictionError, match="pinn_time"):
        compare_module.validate_prediction(prediction)


def test_a_time_axis_that_does_not_match_the_start_time_is_refused(environment):
    """§1.3：``physical_time`` 与 ``pinn_time + start_time`` 对不上，说明这份
    文件混用了两套时间约定——两条轴各自都单调递增，各自都"看起来没问题"。"""
    case, config, tmp_path = environment
    prediction = _prediction(case)
    prediction["start_time"] = float(prediction["start_time"]) + 1.0e-4
    with pytest.raises(compare_module.PredictionError, match="start_time"):
        compare_module.validate_prediction(prediction)


def test_a_missing_prediction_file_is_refused(tmp_path):
    with pytest.raises(compare_module.PredictionError, match="does not exist"):
        compare_module.load_prediction(tmp_path / "nope.npz")


def test_a_charge_radius_mismatch_stops_the_comparison(environment):
    """§1.3：半径语义对不上就不能继续输出"看起来有效"的对照结论。"""
    case, config, tmp_path = environment
    prediction = _prediction(case)
    prediction["charge_radius"] = 0.08
    path = _write_prediction(tmp_path / "wrong_radius.npz", prediction)
    with pytest.raises(compare_module.SetupError, match="装药半径"):
        compare_module.main([
            "--config", str(config), "--prediction", str(path),
            "--case", CASE_ID, "--output", str(tmp_path / "out_r"),
        ])


# --------------------------------------------------------------------------
# §7 插值与区域：规则一致、原因可查
# --------------------------------------------------------------------------


def _smooth_prediction() -> dict:
    """一个**光滑同材料**的解析场：内部点插值必须与解析值逐点相符。"""
    radii = np.linspace(0.01, 0.10, 41)
    times = np.linspace(0.0, 4.0, 9)
    density = np.empty((times.size, radii.size))
    for i, t in enumerate(times):
        density[i] = 1000.0 + 10.0 * t + 2.0 * radii
    return {
        "pinn_time": times,
        "physical_time": times,
        "start_time": 0.0,
        "radius": radii,
        "density": density,
        "velocity": np.zeros_like(density),
        "specific_energy": np.zeros_like(density),
        "pressure": np.zeros_like(density),
        "region": np.zeros(density.shape, dtype=np.int64),
        "contact_radius": np.full(times.size, 0.005),
        "shock_radius": np.full(times.size, 0.50),
    }


def test_a_smooth_same_material_field_is_interpolated_exactly():
    """双线性插值对双线性函数是精确的：``1000 + 10 t + 2 r`` 必须被逐点复现。"""
    prediction = _smooth_prediction()
    radii = np.array([0.0325, 0.055, 0.0775])
    for t in (0.5, 1.75, 3.25):
        values, why = compare_module.sample_field(prediction, "density", radii, t)
        expected = 1000.0 + 10.0 * t + 2.0 * radii
        assert list(why) == [compare_module.SAMPLE_OK] * 3
        np.testing.assert_allclose(values, expected, rtol=1e-12)


def test_the_last_frame_uses_the_last_frame_s_fronts():
    """**区域按查询时刻判定。** 查询最后一帧时必须用末帧的前沿。

    修前 ``_region_at`` 取时间区间的**左端**那一帧，于是查询末帧用的是倒数第二帧
    的前沿——前沿移动越快，错得越多，而报告上看不出任何异常。
    """
    times = np.array([0.0, 1.0, 2.0])
    prediction = {
        "pinn_time": times,
        "physical_time": times,
        "start_time": 0.0,
        "radius": np.array([0.05, 0.15, 0.25]),
        "density": np.ones((3, 3)),
        "contact_radius": np.array([0.05, 0.05, 0.30]),
        "shock_radius": np.array([0.20, 0.20, 0.40]),
    }
    # 末帧的前沿把 0.15 划进产物区；倒数第二帧把它划进空气区。
    assert compare_module._region_at(prediction, np.array([0.15]), 2.0)[0] == 0
    assert compare_module._region_at(prediction, np.array([0.15]), 1.0)[0] == 1
    # 两帧之间用**插值后**的前沿：t = 1.5 时接触面在 0.175，所以 0.15 仍是产物。
    assert compare_module._region_at(prediction, np.array([0.15]), 1.5)[0] == 0


def test_a_radius_outside_the_prediction_grid_is_unavailable_not_labelled():
    """域外的半径不能贴上"产物/空气/环境"的标签——那会让它被记成"材料错配"
    或"模型非有限"，而它其实根本不在预测范围内。"""
    prediction = _smooth_prediction()
    radii = np.array([0.001, 0.5])
    region = compare_module._region_at(prediction, radii, 1.0)
    assert list(region) == [-1, -1]
    values, why = compare_module.sample_field(prediction, "density", radii, 1.0)
    assert np.all(~np.isfinite(values))
    assert list(why) == [compare_module.SAMPLE_RADIUS_OUTSIDE] * 2


def test_a_query_time_outside_the_prediction_coverage_says_so():
    prediction = _smooth_prediction()
    values, why = compare_module.sample_field(
        prediction, "density", np.array([0.05]), 99.0
    )
    assert not np.isfinite(values[0])
    assert why[0] == compare_module.SAMPLE_TIME_OUTSIDE


def test_an_unusable_front_record_makes_the_region_unavailable():
    prediction = _smooth_prediction()
    prediction["contact_radius"] = np.full(9, np.nan)
    assert np.all(compare_module._region_at(prediction, np.array([0.05]), 1.0) == -1)


def test_the_reason_counts_and_the_mutually_exclusive_classes_close(environment):
    """§7.7：每个候选点恰好落进一个互斥类别，计数闭合。"""
    case, config, tmp_path = environment
    prediction = _prediction(case)
    # 制造一批"预测在支撑点上是 NaN"的点。
    density = np.array(prediction["density"], dtype=np.float64)
    density[:, 40:80] = np.nan
    prediction["density"] = density
    report = compare_module.compare_fields(prediction, case)

    for key, entry in report["counts"].items():
        total = sum(
            entry[name] for name in (
                "reference_missing", "outside_domain", "region_mismatch",
                "buffer_zone", "interpolation_unavailable", "model_non_finite",
                "unclassified", "compared",
            )
        )
        assert total == entry["candidates"], (key, entry)

    assert report["sample_reason_counts"]["prediction_non_finite_at_support"] > 0
    assert "density" in report["unavailable_fields"][0] or any(
        "density" in name for name in report["unavailable_fields"]
    )


def test_a_field_with_no_comparable_point_is_unavailable_not_zero(environment):
    case, config, tmp_path = environment
    prediction = _prediction(case)
    # 预测在这个半径范围外时，产物区的密度没有可比点。
    prediction["radius"] = np.asarray(prediction["radius"]) + 0.4
    prediction["contact_radius"] = np.full(case.times.size, 0.45)
    prediction["shock_radius"] = np.full(case.times.size, 0.50)
    prediction["region"] = np.where(
        np.asarray(prediction["radius"])[None, :] < 0.45, 0,
        np.where(np.asarray(prediction["radius"])[None, :] < 0.50, 1, 2),
    ).astype(np.int64)
    report = compare_module.compare_fields(prediction, case)
    assert report["product_density_available"] is False
    assert not np.isfinite(report["product_density_relative_l2"])
    assert "product_density" in report["unavailable_fields"]
    # **顶层与字段不能互相矛盾**：一个可比点都没有时，顶层必须也说"不可评价"，
    # 而不是留着 `available: True` 让结论看起来只是"某些字段空缺"。
    assert report["comparable_points"] == 0
    assert report["available"] is False
    assert report["reason"]


def test_the_coverage_of_the_prediction_is_reported_separately(environment):
    """时间域外的参考帧**不属于候选范围**，要单独报告覆盖范围与排除数量。"""
    case, config, tmp_path = environment
    prediction = _prediction(case)
    # 只保留前半段：后面的参考帧落在预测覆盖之外。
    keep = np.arange(case.times.size)[: max(2, case.times.size // 3)]
    for name in ("pinn_time", "physical_time", "density", "velocity",
                 "specific_energy", "pressure", "region"):
        prediction[name] = np.asarray(prediction[name])[keep]
    for name in ("contact_radius", "shock_radius"):
        prediction[name] = np.asarray(prediction[name])[keep]
    report = compare_module.compare_fields(prediction, case)
    assert report["reference_frames_outside_prediction_coverage"] > 0
    assert report["reference_frames_used"] > 0
    assert report["prediction_time_coverage_s"][0] <= report["prediction_time_coverage_s"][1]
    assert report["coverage_note"]


def test_an_ambient_pressure_of_101325_instead_of_101320_is_refused(environment):
    """**本项目的环境压是 101320 Pa，不是标准大气压 101325 Pa。**

    两者差 5 Pa，而超压、峰值超压与冲量全都相对环境压定义。混用不会报错，只会
    让每条曲线的基线整体挪 5 Pa——所以这里按约定拒绝，而不是"差不多就行"。
    """
    case, config, tmp_path = environment
    prediction = _prediction(case)
    prediction["ambient_pressure"] = 101325.0
    with pytest.raises(compare_module.PredictionError, match="101320"):
        compare_module.validate_prediction(prediction)


def test_a_missing_ambient_pressure_is_refused_not_defaulted(environment):
    """它是被消费的场量，不能用仓库里的常数顶替预测自己的环境态。"""
    case, config, tmp_path = environment
    prediction = _prediction(case)
    prediction.pop("ambient_pressure")
    with pytest.raises(compare_module.PredictionError, match="ambient_pressure"):
        compare_module.load_prediction(
            _write_prediction(tmp_path / "no_ambient.npz", prediction)
        )
