# 数据质量审查

本表由 `python -m spherical_blast_pinn.cli.audit_data` 生成。

| 算例 | R_0 (m) | 帧数 | 可监督帧 | t_PINN 终止 (s) | 接触面有效率 | 主激波有效率 | 最薄空气层单元数 |
|---|---|---|---|---|---|---|---|
| tnt_50mm | 0.050 | 779 | 14 | 4.7945e-05 | 0.02 | 1.00 | 5 |
| tnt_80mm | 0.080 | 483 | 26 | 1.2827e-04 | 0.05 | 0.20 | 17 |
| tnt_100mm | 0.100 | 30 | 3 | 1.7785e-04 | 0.10 | 0.87 | 97 |

## 字段来源与已知限制

### tnt_50mm

- 目录：`F:\2\extracted`
- 字段映射：`{'time': 't', 'radius': 'x', 'density': 'rho', 'velocity': 'u', 'pressure': 'P'}`
- 比内能来源：reconstructed from (rho, p) via EOS; not a measured field
- 材料分区来源：density-steepest-descent contact detector (heuristic)
- 识别参数：`{'shock_pressure_ratio': 0.001, 'contact_smoothing_cells': 5, 'contact_density_ratio_floor': 1.02, 'minimum_cells_per_medium': 3}`
- 有效时间范围：物理 0.0000e+00 – 1.0001e-03 s；启动时刻 1.0000e-06 s

### tnt_80mm

- 目录：`C:\Users\work\Documents\GitHub\2\extracted_80mm`
- 字段映射：`{'time': 't', 'radius': 'x', 'density': 'rho', 'velocity': 'u', 'pressure': 'P'}`
- 比内能来源：reconstructed from (rho, p) via EOS; not a measured field
- 材料分区来源：density-steepest-descent contact detector (heuristic)
- 识别参数：`{'shock_pressure_ratio': 0.001, 'contact_smoothing_cells': 5, 'contact_density_ratio_floor': 1.02, 'minimum_cells_per_medium': 3}`
- 有效时间范围：物理 0.0000e+00 – 1.0023e-03 s；启动时刻 1.6000e-06 s

### tnt_100mm

- 目录：`C:\Users\work\Documents\GitHub\2\extracted_100mm`
- 字段映射：`{'time': 't', 'radius': 'x', 'density': 'rho', 'velocity': 'u', 'pressure': 'P'}`
- 比内能来源：reconstructed from (rho, p) via EOS; not a measured field
- 材料分区来源：density-steepest-descent contact detector (heuristic)
- 识别参数：`{'shock_pressure_ratio': 0.001, 'contact_smoothing_cells': 5, 'contact_density_ratio_floor': 1.02, 'minimum_cells_per_medium': 3}`
- 有效时间范围：物理 0.0000e+00 – 1.3677e-03 s；启动时刻 2.0000e-06 s
