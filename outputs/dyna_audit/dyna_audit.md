# DYNA 真实物理初始化审查

本表由 `python -m spherical_blast_pinn.cli.audit_dyna` 生成。每一项都带证据来源；
没有证据的项标为“缺证据”，不编造结论。

## tnt_50mm

卡片：`F:\tnt\tnt_files\dp0\SYS\MECH\input.k`

| 审查项 | 结论 | 证据文件及位置 | 与方案 A 是否一致 | 仍缺什么信息 |
|---|---|---|---|---|
| 1. 整球瞬时初始化还是有限速度爆轰 | 有限速度爆轰（programmed burn）。卡片有 4 张 *MAT_HIGH_EXPLOSIVE_BURN 与一张 *INITIAL_DETONATION，起爆点 (0.0, 0.0, 0.0)，lt = 0.0 | F:\tnt\tnt_files\dp0\SYS\MECH\input.k: *MAT_HIGH_EXPLOSIVE_BURN / *INITIAL_DETONATION | 不一致 | 起爆点是否恰好是球心，需要几何坐标核对（见第 7 项） |
| 5. 单位制 | mm_s_t；判据是炸药密度而不是文件名：卡片 ro = 1.63e-09，按该单位制换算（×1e+12）得 1630 kg/m^3，与规范的 1630 kg/m^3 一致 | F:\tnt\tnt_files\dp0\SYS\MECH\input.k: *MAT_HIGH_EXPLOSIVE_BURN 的 ro = 1.63e-09 | 一致 |  |
| 3a. JWL 系数与规范 §4 是否一致 | 一致（已按识别出的单位制换算到 SI） | F:\tnt\tnt_files\dp0\SYS\MECH\input.k: *EOS_JWL 第 2 号 | 一致 |  |
| 3b/c/d. E0、V0 与能量参考零点 | E0 = 6000（按参考体积的能量密度）；V0 = 1；换算到 SI 后 E0 = 6e+09 J/m^3 | F:\tnt\tnt_files\dp0\SYS\MECH\input.k: *EOS_JWL 第 2 行 | 一致 | 能量参考零点需要 *DATABASE_EXTENT_BINARY 的 engflg 与运行时能量统计共同确认；仅凭卡片不能断定 |
| 6. 终止时间与输出间隔 | endtim = 0.001（时间单位）；d3plot 间隔 = 1e-06；→ 约 1000 个输出帧 | F:\tnt\tnt_files\dp0\SYS\MECH\input.k | 一致 | 外边界条件需要 *BOUNDARY_* 卡片（见第 8 项） |
| 7. 计算域形状与对称面 | 三个方向跨度相同（包围盒跨度 [5000.0, 5000.0, 5000.0]，最大半径 8660.25），与整球或八分之一球相符；但楔形体的包围盒也是正方体，所以对称面仍需核对 *BOUNDARY_SPC_SET | F:\tnt\tnt_files\dp0\SYS\MECH\input.k: *NODE 包围盒 ((-5000.0, 0.0), (0.0, 5000.0), (0.0, 5000.0)) | 缺证据 | 对称面靠 *BOUNDARY_SPC_SET 定义，其自由度组合需要逐条核对；质量与能量统计是否需要几何倍数取决于此 |
| 8. 外边界条件与对称面 | 3 组 *BOUNDARY_SPC_SET，约束自由度（x/y/z）= ['1/0/0', '0/1/0', '0/0/1']；1 组 *BOUNDARY_NON_REFLECTING。三组各约束一个方向的平动，与三个对称面一致；配合非反射外边界，计算域是整球的八分之一，质量与能量统计需要乘以 8 | F:\tnt\tnt_files\dp0\SYS\MECH\input.k: *BOUNDARY_SPC_SET / *BOUNDARY_NON_REFLECTING / *SET_NODE_LIST | 缺证据 | 非反射边界在强爆炸下的实际吸收效果需要运行时数据核对；仅凭卡片不能断定晚期有无反射污染 |
| 2. 时间零点代表什么 | 时间零点不是整球同时转化。逐帧的强压缩前沿是移动的：t = 9.862e-07 s 时在 r = 0.006002 m（P = 1.03e+10 Pa），t = 6.948e-06 s 时在 r = 0.04802 m。线性拟合斜率 7029 m/s，与卡片爆速 6930 m/s 的相对偏差 1.435% | F:\2\extracted\raw_line.npz: 逐帧取 P > 1e+09 Pa 的最外侧半径，对 t 作线性拟合（7 帧，截距 -8.585e-04 m） | 不一致 | 卡片没有输出反应进度字段（*DATABASE_EXTENT_BINARY 未开启 ialemat 一类的材料状态输出），反应速率本身无法逐帧确认，只能由前沿位置间接判断 |

<details><summary>卡片明细</summary>

```json
{
  "title": null,
  "termination_time": 0.001,
  "d3plot_interval": 1e-06,
  "initial_detonation": [
    0.0,
    0.0,
    0.0
  ],
  "initial_detonation_time": 0.0,
  "control_ale": {
    "field_0": -1.0,
    "field_1": 1.0,
    "field_2": 1.0,
    "field_3": -1.0,
    "field_4": 0.0,
    "field_5": 0.0,
    "field_6": 0.0,
    "field_7": 0.0
  },
  "n_parts": 7,
  "part_names": {
    "1": "air",
    "2": "tnt",
    "3": "1n1",
    "4": "tnt1",
    "5": "air1",
    "6": "air11",
    "7": "tnt11"
  },
  "detonation_velocity_raw_card_units": 6930000.0,
  "explosive_density_raw": 1.63e-09,
  "unit_system_inferred": "mm_s_t",
  "explosive_density_kg_m3": 1629.9999999999998,
  "density_to_si_factor": 999999999999.9999,
  "jwl_card_raw": {
    "A": 371200.0,
    "B": 3747.1,
    "R1": 4.15,
    "R2": 0.95,
    "omeg": 0.3,
    "E0": 6000.0,
    "V0": 1.0
  },
  "e0_volume_energy_raw": 6000.0,
  "e0_volume_energy_si": 6000000000.0,
  "v0": 1.0,
  "specific_energy_j_kg": 3680981.5950920247,
  "geometry": {
    "n_nodes": 1250370,
    "n_solid_elements": 1220736,
    "bounding_box": [
      [
        -5000.0,
        0.0
      ],
      [
        0.0,
        5000.0
      ],
      [
        0.0,
        5000.0
      ]
    ],
    "radius_min": 0.0,
    "radius_max": 8660.254037844386,
    "total_volume": 125000000000.00977,
    "counts_by_part": {
      "1": 335808,
      "2": 30528,
      "3": 127200,
      "4": 31800,
      "5": 349800,
      "6": 316800,
      "7": 28800
    },
    "radius_by_part": {
      "1": [
        49.99804582672533,
        8660.254037844386
      ],
      "2": [
        25.0,
        50.00148932244871
      ],
      "3": [
        0.0,
        43.30127018922193
      ],
      "4": [
        25.0,
        50.00007488454647
      ],
      "5": [
        49.99859538039408,
        8660.254037844386
      ],
      "6": [
        49.99799868656757,
        8660.254037844386
      ],
      "7": [
        25.0,
        50.001158033904126
      ]
    },
    "volume_by_part": {
      "1": 41666644852.77331,
      "2": 16605.3899926844,
      "3": 15625.00000000485,
      "4": 16605.807454745052,
      "5": 41666644852.53536,
      "6": 41666644852.796585,
      "7": 16605.345151449506
    },
    "element_format": "two_line"
  },
  "n_nodes": 1250370,
  "n_elements": 1220736,
  "detonation_velocity_raw": 6930000.0,
  "detonation_velocity_m_s": 6930.0,
  "warnings": []
}
```
</details>

## tnt_80mm

卡片：`F:\2\runs\100mm_full\input.k`

| 审查项 | 结论 | 证据文件及位置 | 与方案 A 是否一致 | 仍缺什么信息 |
|---|---|---|---|---|
| 1. 整球瞬时初始化还是有限速度爆轰 | 有限速度爆轰（programmed burn）。卡片有 1 张 *MAT_HIGH_EXPLOSIVE_BURN 与一张 *INITIAL_DETONATION，起爆点 (0.0, 0.0, 0.0)，lt = 0.0 | F:\2\runs\100mm_full\input.k: *MAT_HIGH_EXPLOSIVE_BURN / *INITIAL_DETONATION | 不一致 | 起爆点是否恰好是球心，需要几何坐标核对（见第 7 项） |
| 5. 单位制 | si；判据是炸药密度而不是文件名：卡片 ro = 1630，按该单位制换算（×1）得 1630 kg/m^3，与规范的 1630 kg/m^3 一致 | F:\2\runs\100mm_full\input.k: *MAT_HIGH_EXPLOSIVE_BURN 的 ro = 1630 | 一致 |  |
| 3a. JWL 系数与规范 §4 是否一致 | 一致（已按识别出的单位制换算到 SI） | F:\2\runs\100mm_full\input.k: *EOS_JWL 第 1 号 | 一致 |  |
| 3b/c/d. E0、V0 与能量参考零点 | E0 = 6e+09（按参考体积的能量密度）；V0 = 1；换算到 SI 后 E0 = 6e+09 J/m^3 | F:\2\runs\100mm_full\input.k: *EOS_JWL 第 2 行 | 一致 | 能量参考零点需要 *DATABASE_EXTENT_BINARY 的 engflg 与运行时能量统计共同确认；仅凭卡片不能断定 |
| 6. 终止时间与输出间隔 | endtim = 0.002（时间单位）；d3plot 间隔 = 1e-06；→ 约 2000 个输出帧 | F:\2\runs\100mm_full\input.k | 一致 | 外边界条件需要 *BOUNDARY_* 卡片（见第 8 项） |
| 7. 计算域形状与对称面 | 非等向计算域（包围盒跨度 [2.46014, 0.854708, 0.148478]，最大半径 2.5），既不是整球也不是规则八分之一球 | F:\2\runs\100mm_full\input.k: *NODE 包围盒 ((0.00187938524, 2.46201938), (0.000342020143, 0.855050358), (0.0, 0.148477937)) | 缺证据 | 对称面靠 *BOUNDARY_SPC_SET 定义，其自由度组合需要逐条核对；质量与能量统计是否需要几何倍数取决于此 |
| 8. 外边界条件与对称面 | 1 组 *BOUNDARY_SPC_SET，约束自由度（x/y/z）= ['1/1/1']；1 组 *BOUNDARY_NON_REFLECTING。这些组同时约束三个方向，是固定壁面而不是对称面；计算域不是靠对称面剪出来的，几何倍数取决于网格实际张角，需另行确认 | F:\2\runs\100mm_full\input.k: *BOUNDARY_SPC_SET / *BOUNDARY_NON_REFLECTING / *SET_NODE_LIST | 缺证据 | 非反射边界在强爆炸下的实际吸收效果需要运行时数据核对；仅凭卡片不能断定晚期有无反射污染 |
| 2. 时间零点代表什么 | 时间零点不是整球同时转化。逐帧的强压缩前沿是移动的：t = 1.984e-06 s 时在 r = 0.01701 m（P = 1.255e+09 Pa），t = 9.967e-06 s 时在 r = 0.1351 m。线性拟合斜率 1.471e+04 m/s，与卡片爆速 6930 m/s 的相对偏差 112.272% | C:\Users\work\Documents\GitHub\2\extracted_80mm\raw_line.npz: 逐帧取 P > 1e+09 Pa 的最外侧半径，对 t 作线性拟合（5 帧，截距 -9.172e-03 m） | 缺证据 | 卡片没有输出反应进度字段（*DATABASE_EXTENT_BINARY 未开启 ialemat 一类的材料状态输出），反应速率本身无法逐帧确认，只能由前沿位置间接判断 |

<details><summary>卡片明细</summary>

```json
{
  "title": "Spherical TNT R0=100 mm air blast  |  SI units (m, kg, s, Pa)",
  "termination_time": 0.002,
  "d3plot_interval": 1e-06,
  "initial_detonation": [
    0.0,
    0.0,
    0.0
  ],
  "initial_detonation_time": 0.0,
  "control_ale": {
    "field_0": -1.0,
    "field_1": 1.0,
    "field_2": 1.0,
    "field_3": -1.0,
    "field_4": 0.0,
    "field_5": 0.0,
    "field_6": 0.0,
    "field_7": 0.0
  },
  "n_parts": 2,
  "part_names": {
    "2": "air_mesh",
    "1": "tnt_charge"
  },
  "detonation_velocity_raw_card_units": 6930.0,
  "explosive_density_raw": 1630.0,
  "unit_system_inferred": "si",
  "explosive_density_kg_m3": 1630.0,
  "density_to_si_factor": 1.0,
  "jwl_card_raw": {
    "A": 371200000000.0,
    "B": 3747100000.0,
    "R1": 4.15,
    "R2": 0.95,
    "omeg": 0.3,
    "E0": 6000000000.0,
    "V0": 1.0
  },
  "e0_volume_energy_raw": 6000000000.0,
  "e0_volume_energy_si": 6000000000.0,
  "v0": 1.0,
  "specific_energy_j_kg": 3680981.5950920247,
  "geometry": {
    "n_nodes": 4833,
    "n_solid_elements": 2144,
    "bounding_box": [
      [
        0.00187938524,
        2.46201938
      ],
      [
        0.000342020143,
        0.855050358
      ],
      [
        0.0,
        0.148477937
      ]
    ],
    "radius_min": 0.0019999999973872537,
    "radius_max": 2.5000000042390154,
    "total_volume": 0.0408808495407769,
    "counts_by_part": {
      "1": 392,
      "2": 1752
    },
    "radius_by_part": {
      "1": [
        0.0019999999973872537,
        0.10000000002564487
      ],
      "2": [
        0.09999999995925218,
        2.5000000042390154
      ]
    },
    "volume_by_part": {
      "1": 2.616353437195946e-06,
      "2": 0.040878233187339706
    },
    "element_format": "one_line"
  },
  "n_nodes": 4833,
  "n_elements": 2144,
  "detonation_velocity_raw": 6930.0,
  "detonation_velocity_m_s": 6930.0,
  "warnings": []
}
```
</details>
