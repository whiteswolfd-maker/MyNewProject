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
- 迁移前后 SHA256 见 [MIGRATION_SHA256.txt](MIGRATION_SHA256.txt)。清单口径与
  核验方法见下节；23 份数据的 Git blob 字节与基线 `2a74161` 中的原文件完全相同。
- 其中 `outputs/final/run_manifest.json`、`outputs/final/train_result.json`、
  `outputs/full/reference/reference_field_report.json` 在后续提交 `faea39c`
  中已被删除；这里按基线 `2a74161` 的**原始 blob** 恢复到归档（Git blob 哈希
  与基线一致），没有修改其中任何字段。
- 本迁移**不**改写历史 JSON 里的方案、数值或路径。

## 哈希口径与核验

`MIGRATION_SHA256.txt` 的统一口径是：

> 对**指定提交中 Git blob 的原始内容字节**计算 SHA256。

它不是 Git 对象 ID；不对工作区文本文件直接计算；不做换行归一化、文本解码再
编码或空白清理。旧清单混用了工作区换行形式：在 `core.autocrlf=true` 的检出上，
23 条里有 20 条是对 CRLF 工作区字节算的，另外 3 条是对 blob 字节算的——归档
数据本身没有损坏（23 份 Git blob 与基线 `2a74161` 完全一致），只是口径不一致。
本清单已全部统一为 blob 字节口径。

只读核验（在仓库根目录执行）。默认核对 `HEAD`；可用 `COMMIT=origin/main`
核对远端实际提交。缺文件、路径重复、数量不是 23、blob 字节不同或 SHA256 不符，
都会以非零状态退出：

```bash
COMMIT="${COMMIT:-HEAD}" python - <<'PY'
import hashlib, os, subprocess, sys
from pathlib import Path

BASE = "2a74161"
COMMIT = os.environ["COMMIT"]
LIST = Path("outputs/_archive_scheme_a/MIGRATION_SHA256.txt")
DOCS = {"outputs/_archive_scheme_a/README.md",
        "outputs/_archive_scheme_a/MIGRATION_SHA256.txt"}

def run(*args):
    return subprocess.run(["git", *args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

def blob(rev, path):
    # 直接对 git cat-file 返回的 **bytes** 计算，不经文本管道。
    return subprocess.check_output(["git", "cat-file", "blob", f"{rev}:{path}"])

def exists(rev, path):
    return run("cat-file", "-e", f"{rev}:{path}").returncode == 0

rows, errors = [], []
for n, line in enumerate(LIST.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip() or line.startswith("#"):
        continue
    try:
        head, archive = line.split("  ->  ")
        digest, original = head.split("  ", 1)
    except ValueError:
        errors.append(f"line {n}: malformed record")
        continue
    rows.append((digest, original, archive.strip()))

if len(rows) != 23:
    errors.append(f"expected 23 records, found {len(rows)}")
for label, values in (("original", [r[1] for r in rows]),
                      ("archive", [r[2] for r in rows])):
    if len(set(values)) != len(values):
        errors.append(f"duplicate {label} path in list")

listed = {r[2] for r in rows}
actual = {p for p in subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", COMMIT,
     "outputs/_archive_scheme_a"]).decode().splitlines()
    if p not in DOCS}
if listed != actual:
    errors.append(f"archive set mismatch: missing={sorted(actual - listed)} "
                  f"extra={sorted(listed - actual)}")

for digest, original, archive in rows:
    if not exists(BASE, original):
        errors.append(f"{original}: missing in baseline {BASE}"); continue
    if not exists(COMMIT, archive):
        errors.append(f"{archive}: missing in {COMMIT}"); continue
    a, b = blob(BASE, original), blob(COMMIT, archive)
    if a != b:
        errors.append(f"{original} vs {archive}: blob bytes differ"); continue
    if hashlib.sha256(a).hexdigest() != digest:
        errors.append(f"{archive}: sha256 mismatch")

if errors:
    print("FAILED")
    for error in errors:
        print(" -", error)
    sys.exit(1)
print(f"OK: {len(rows)} records; blob bytes identical; sha256 matches")
PY
```

## 使用约束

- 不要把这些文件当作当前方案 B 的训练/预测/验收结果。
- 若需要当前方案 B 的结论，必须用当前配置重新运行、重新产出。
