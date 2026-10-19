# 历史归档：方案 A 运行结果

本目录保存的是**旧方案 A**（`scheme A: whole charge converts at t = 0`）的历史
运行产物，原样迁移、字节未改。它们**不是**当前方案 B 的结果，**不构成当前验收
结论**，也不得被引用为"训练/预测已通过"的证据。

## 为什么判定为方案 A

以下判据都来自文件内容本身，不靠目录名或版本号推测：

- 运行清单/训练结果：`metadata.reference_field.initial_condition` 明确写着
  `scheme A: whole charge converts at t=0 (uniform, at rest, e = E0/rho0)`，
  `metadata.model.scales.start_time = 1e-06`（50 mm 只含交接延迟 `epsilon`，
  没有爆轰渡越时间 `T_b = R_0/D`）。
- 参考场报告：`start_time_s = 1e-06`。
- 预测结果：`start_time_s = 1.6e-06`（80 mm 的 `epsilon`；方案 B 的
  `T_h = R_0/D + epsilon` 应为 `13.144012e-06`），且其 `checkpoint` 指向方案 A
  的运行目录。
- 对比结果：`checks.time_origin_s = 1.6e-06`、`time_origin_rule` 为
  `t_PINN = t_phys - epsilon(R_0)`（只减 `epsilon`），同样是方案 A。
- `checks.json`：没有独立的方案字段，但其数组与同目录运行清单的
  `metadata.checks` **逐项完全相同**（已用结构化 JSON 比较核实），因此属于
  同一次方案 A 运行。

当前方案 B 的 50 mm 时间原点是 `T_h ≈ 8.215007 µs`，与上面这些记录相差一个
`T_b ≈ 7.215 µs`。

## 迁移

- 原路径 → 本目录下的同名相对路径（例如 `outputs/final/checks.json` →
  `outputs/_archive_scheme_a/final/checks.json`）。
- 迁移前后 SHA256 见 [MIGRATION_SHA256.txt](MIGRATION_SHA256.txt)；两者逐位一致。
- 其中 `outputs/final/run_manifest.json`、`outputs/final/train_result.json`、
  `outputs/full/reference/reference_field_report.json` 在后续提交 `faea39c`
  中已被删除；这里按基线 `2a74161` 的**原始 blob** 恢复到归档（Git blob 哈希
  与基线一致），没有修改其中任何字段。
- 本迁移**不**改写历史 JSON 里的方案、数值或路径。

## 使用约束

- 不要把这些文件当作当前方案 B 的训练/预测/验收结果。
- 若需要当前方案 B 的结论，必须用当前配置重新运行、重新产出。
