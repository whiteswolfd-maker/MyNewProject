# contact_air 空气资格与 R_b 诊断（2026-11-09）

本轮目标：把“原始单元内能重建失败 → 整壳所有字段作废”的空气污染判断收窄到
空气字段实际贡献单元；为 contact_air 的 `R_c ~ R_b` 域单独生成空气监督资格；
查清 `R_b` 候选由什么阈值决定。**没有改网络结构、损失公式、交替训练、学习率、
epsilon、前置时钟、EOS 或物理模型。**

---

## 1. 实际根目录、运行环境与数据目录

### 1.1 根目录

- 本轮**实际编辑目录**（用户当前要改的代码）：
  `E:\2-codex-coupled-handoff-pinn\2-codex-coupled-handoff-pinn\2-codex-coupled-handoff-pinn1`
- 该目录**不是 git 检出**（无 `.git`）。
- `C:\Users\26715\Desktop\deskbook\deskbook` 是用户之后用于上传的 deskbook 目录，
  不是本轮编辑目录；本轮**没有向它复制或覆盖任何文件**，也没有提交或推送。
  用户改完 E 盘代码后再自行复制到 deskbook 上传。

### 1.2 Python 环境

本机没有 `py` 启动器；`python` 是 Simcenter 自带的 3.6，不能用。实际使用：

```text
E:\Anaconda3\envs\pytorch\python.exe
Python 3.12.12
numpy 1.26.4
scipy 1.16.3
torch 2.9.1+cu130
lasso-python 2.0.3
PyYAML 6.0.3
pytest 9.1.1
```

说明：

- 必须 `export KMP_DUPLICATE_LIB_OK=TRUE`，否则导入 torch 时
  `libiomp5md.dll` 重复加载会终止进程。
- 本机原先没有 `lasso-python`。本轮安装了 `lasso-python==2.0.3`（仓库
  `VALIDATED_LASSO_VERSION`），安装过程把该 conda 环境的 numpy 从 2.3.5
  降到 1.26.4；安装后 torch/scipy 导入与全量测试均正常。
- `E:\Anaconda3\python.exe`（base）有 pytest 但没有 torch，不能用于本仓库。
- `C:\Users\26715\Desktop\deskbook_sync\venv_clean` 存在，但没有 lasso，
  本轮未使用。

### 1.3 四个数据目录

`input.k` 在四个 `MECH` 目录内均存在；没有去父目录或别处补路径。

| 工况 | d3plot 族 | 状态数（LASSO 探针 + 边界复核） | `input.k` | 由单元体积/体积分数反推的装药半径 | 相对偏差 |
|---|---:|---:|---|---:|---:|
| 50 mm `E:/tnt/tnt_files/dp0/SYS/MECH` | 780 个文件（含 master） | **779** | 有 | 49.9979 mm | 4.2e-5 |
| 80 mm `G:/500mmtnt/80mmtnt_files/dp0/SYS-3/MECH` | 783 | **782** | 有 | 79.9943 mm | 7.1e-5 |
| 100 mm `G:/500mmtnt/100mmtnt_files/dp0/SYS-3/MECH` | 783 | **782** | 有 | 99.9929 mm | 7.1e-5 |
| 500 mm `D:/500mmtnt/500mmtnt_files/dp0/SYS-3/MECH` | 783 | **782** | 有 | 499.9610 mm | 7.8e-5 |

材料卡核对（四份 deck 完全一致，50 mm 的 B 为 3747.1，其余为 3747）：

```text
*MAT_HIGH_EXPLOSIVE_BURN: ro=1.63E-09, D=6930000, pcj=21000
*EOS_JWL: A=371200, B=3747/3747.1, R1=4.15, R2=0.95, omega=0.3, E0=6000, V0=1
*INITIAL_DETONATION: (0,0,0), lt=0
unit system: mm_s_t
```

装药半径不是按目录名认定的：上表最后一列由 `_charge_radius_from_volume`
独立反推，和目录名对应工况的偏差都在 8e-5 以内。500 mm 仍只作为 `role: test`，
本轮没有用它的内部场或接触面标签做训练或阈值调优。

---

## 2. 迁移检查

| 要求 | 现状 |
|---|---|
| contact_air 独立入口 | 已有：`python -m spherical_blast_pinn.cli.contact_air`，实现与 CLI 在 `src/spherical_blast_pinn/contact_air/` |
| `material_phase` | 已有：由 ALE 体积分数固定为 0/1/2，写入 `material.npz`，读取器与 contact_air 都使用 |
| `training_location_valid` | 已有：材料/旧撤标/污染/前沿保护带的位置资格，不含联合有限性 |
| 逐字段有效性 | 已有 `field_valid_*`；本轮新增 contact_air 专用的 `air_field_valid_*` |
| 接触面、空气两个网络 + 两个持久 Adam 交替更新 | 已有：`contact_air/trainer.py`，先 AirNet 后 ContactNet，反向传播后检查非目标网络无梯度且参数未移动 |

结论：新电脑代码没有落后于这些迁移项。本轮没有重复实现网络/训练/损失；
只修正数据侧污染作用范围和空气资格，并补上 R_b 诊断。

---

## 3. 本轮修改文件与新旧行为差异

### 3.1 修改文件

| 文件 | 改动 |
|---|---|
| `src/spherical_blast_pinn/cli/extract_profiles.py` | 按介质贡献体积统计污染；新增 `air_training_location_valid`、`air_field_valid_*`、`air_contaminated_shell_fraction`、`product_contaminated_shell_fraction`、占用/污染体积、按相位的无效单元计数、接触带剔除原因计数 |
| `src/spherical_blast_pinn/data/extracted_case.py` | 读取并校验上述新数组；暴露 `air_training_location_valid`、`air_field_validity`、`shock_has_supervision` 等 |
| `src/spherical_blast_pinn/contact_air/data.py` | `build_air_observations` 使用 `material_phase` 与 `air_training_location_valid`；不再用旧 `training_location_valid`；叠加覆盖率与逐字段掩码；记录逐帧剔除原因和受扰/环境计数；`CaseArrays` 增加空气专用字段 |
| `src/spherical_blast_pinn/contact_air/rb_diagnostic.py`（新） | R_b 逐字段候选、连续外缘、binding field、动态样本、背景噪声、阈值敏感性 |
| `scripts/extract_contact_air_profile.py`（新） | 从 `configs/contact_air/base.yaml` 的 `data_sources` 读取真实 deck/d3plot 路径并提取 |
| `scripts/diagnose_contact_air_rb.py`（新） | 对真实帧输出候选附近逐点 CSV、逐层计数、背景与敏感性报告 |
| `configs/contact_air/base.yaml` | 增加 `data_sources`（正斜杠 Windows 路径） |
| `configs/contact_air/diagnostic_background_thresholds.yaml`（新） | **仅诊断**：把阈值放到实测背景之上，用于验证链路；不进入生产配置 |
| `README.md`, `AGENTS.md` | 更新 50 mm 数据源为 `E:/tnt/...`；README 增加 2026-11-09 空气资格/R_b 说明 |
| `tests/test_contact_air_regressions.py` | 新增按介质污染、无 confirmed shock 仍保留空气资格、`material_phase` 归因测试；更新 profile 夹具 |
| `tests/test_contact_air_rb_diagnostic.py`（新） | 连续外缘、速度控制、密度阻断、全环境帧、壳层计数标签 |
| `tests/test_contact_air_extraction_bridge.py` | profile 夹具补新数组 |

### 3.2 新旧行为差异

**旧行为（仍然保留给旧消费者）：**

```text
任一原始单元内能重建失败（state_valid=False）
  -> 全介质 contaminated_shell_fraction > 0
  -> 整个球壳的所有字段作废
没有 confirmed shock
  -> 全帧空气 material 撤标、valid=False
```

**新行为（contact_air 专用，不改变旧 `valid` / `training_location_valid`）：**

```text
空气污染 = 失效空气单元的贡献体积 / 空气占据体积
  -> 只有真正参与空气平均的单元能否决空气字段
  -> 产物单元没有空气权重，不能否决空气
  -> 混合单元不进入任何纯介质平均
rho/u：直接测量，空气位置可用且有限即可
p/e ：还要求没有缺失的空气贡献（air_missing_contribution=False）
空气资格：material_phase==1 + 非空壳层 + 接触混合带保护带之外
  -> 不要求 confirmed shock
  -> 不因未确认的候选激波删除外侧可靠空气
激波检测状态：单独记录，不参与空气资格
```

没有静默丢弃无效贡献：空气缺失贡献会写进 `air_missing_contribution`，并否决
p/e；rho/u 仍可用，因为它们的测量不依赖产物/空气的 EOS 反解。

缓存格式因此升到 **version 6**：version 5 的 air observations 是用旧联合掩码
选取的，不能静默转换；`load_case` / `load_prediction_case` 会拒绝 version 5，
必须重新 `build-cache`。

---

## 4. 真实 50 mm 帧的逐层筛选计数与剔除原因

提取命令只读真实 d3plot，状态编号和时间来自本次运行：

| 帧 | 状态号 | 时间 (s) | 空气相壳层 | 空气位置可用 | 接触带剔除 | 空气缺失贡献 | 空壳层/零空气体积 | 旧全介质污染>0 的空气壳层 | 新空气污染>0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0 | 2084 | 2082 | 2 | 0 | 0 | 0 | 0 |
| 1 | 19 | 1.89862058e-5 | 1657 | 1582 | 75 | 0 | 0 | **209** | **0** |
| 2 | 20 | 1.99870592e-5 | 1620 | 1559 | 61 | 0 | 0 | **172** | **0** |
| 3 | 22 | 2.19677877e-5 | 1570 | 1513 | 57 | 0 | 0 | **122** | **0** |

逐帧无效单元按相位（本次真实提取）：

| 状态 | 无效产物单元 | 无效空气单元 | 无效混合单元 |
|---:|---:|---:|---:|
| 0 | 218328 | 0 | 0 |
| 19 | 0 | 0 | 6975 |
| 20 | 0 | 0 | 7221 |
| 22 | 0 | 0 | 7097 |

关于“哪些原始单元、权重多少”：

- profile **没有逐单元 ID**（`raw_cell_ids_available=False`），所以不能列出
  “第 j 个原始单元”的清单。提取器按壳层写出：
  `air_occupied_volume_m3`（空气占据体积）、`air_contaminated_volume_m3`
  （其中失效空气贡献）、`product_occupied_volume_m3`、
  `product_contaminated_volume_m3`、`mixed_occupied_volume_m3`。
- 空气平均的权重是 `alpha_air * 重叠体积`；rho/p 体积加权，u/e 质量加权。
  因此空气污染比例的分母就是实际参与空气平均的体积，而不是整个壳层体积。
- 被拒壳层的三类原因在本轮 50 mm 上是：接触混合带保护带 2/75/61/57 个壳层；
  空气缺失贡献 0；空壳层或零空气体积 0。覆盖率未剔除（全部 ≥0.5）。

结论：

1. **参与空气平均的单元与导致旧污染的单元不是同一组。** 纯产物
   (`alpha>=0.95`)、纯空气 (`alpha<=0.05`) 和混合单元在体积分数掩码上互斥；
   产物单元没有空气权重，混合单元不进入任何纯介质平均。
2. 状态 0 的污染来自产物侧（尚未释放化学能），旧口径污染了 416 个产物壳层，
   但空气污染为 0。
3. 状态 19/20/22 的无效单元全部是混合单元；旧全介质口径把 209/172/122 个空气
   壳层整层作废，新空气口径为 0。
4. 空气侧的剔除原因只有接触混合带保护带（2/75/61/57 个壳层）；没有空气缺失
   贡献、没有空壳层/零空气体积。覆盖率在本次 50 mm 抽取中全部 ≥0.5，未剔除。

---

## 5. 受扰区与环境区分别可用的 rho/u/p 点数

使用**诊断阈值**（见第 6 节；生产阈值下 R_b 不存在，因此无法形成 R_c~R_b 域）
通过真实 `build_air_observations` 得到。所有计数都是**球壳样本数**，不是原始
ALE 单元数；profile 没有逐单元 ID。

| 状态 | 域内可用壳层 | 受扰（三场同时非环境） | 环境（三场同时通过） | rho 可用 | u 可用 | p 可用 |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 640 | 622 | 18 | 640 | 640 | 640 |
| 20 | 624 | 622 | 2 | 624 | 624 | 624 |
| 22 | 587 | 584 | 3 | 587 | 587 | 587 |
| 合计 | 1851 | 1828 | 23 | 1851 | 1851 | 1851 |

- 状态 0 没有 `R_b`，不进入 `R_c ~ R_b` 观测；其 2082 个空气位置可用壳层只作为
  静态帧检查。
- 状态 20 **没有 confirmed shock**，仍选中 624 个壳层：
  `frames_with_eligible_air_without_confirmed_shock = 1`。这是本轮“激波状态不
  自动否决整帧空气”的直接验证。
- 受扰区与环境区分开统计。环境点没有掩盖受扰区：受扰区 1828 个壳层，环境区
  只有 23 个（占选中样本 1.24%）。
  `disturbed_air_absent_while_ambient_present = False`；如果受扰区完全无数据而
  只剩环境点，这个标志会变成 True，不会被“大量环境点”掩盖。
- 逐字段可用性与三场同时可用一致（本次 50 mm 没有出现 p/e 缺失贡献）。
  如果出现缺失贡献，rho/u 仍会保留，p/e 会被 `air_field_valid_*` 单独否决。

旧路径对照：同一批帧上旧 `training_location_valid` 对空气壳层全为 False，
旧路径可用的空气训练点数为 **0**；新口径在诊断阈值下得到 1851 个壳层样本。
这不是“阈值一放宽就自动正确”，见下一节。

---

## 6. R_b 候选的控制阈值与稳定性

### 6.1 生产阈值：R_b 不存在

`configs/contact_air/base.yaml`：

```text
pressure_pa: 0.10132 Pa      (1e-6 relative)
density_kg_m3: 1.0e-9        (~8.2e-10 relative)
velocity_m_s: 1.0e-6
min_ambient_cells: 2
```

在已确认的未扰动区域（`r >= pressure candidate * 1.05`）实测数值背景：

| 状态 | 压力最大相对偏差 | 密度最大相对偏差 | 速度最大绝对偏差 |
|---:|---:|---:|---:|
| 0 | 8.37e-8 | **7.51e-3** | 0 |
| 19 | 2.58e-7 | **2.01e-3** | 9.48e-4 m/s |
| 20 | 3.33e-7 | **1.93e-3** | 9.56e-4 m/s |
| 22 | 6.00e-7 | **1.84e-3** | 9.85e-4 m/s |

生产密度阈值比密度背景低约 6–7 个数量级，速度阈值比速度背景低约 3 个数量级。
因此：

- 压力有连续环境段（候选约 0.05032 / 0.28962 / 0.29519 / 0.30251 m）；
- 密度没有连续环境段；
- 速度没有连续环境段；
- **三场同时通过的 R_b 候选不存在。**

`build-cache` 在 `t0 = 1.953133291397989e-5 s` 处硬失败：

```text
** case 'tnt_50mm': t0=1.953133291397989e-05 s is outside every continuous
   R_b segment; the initial condition cannot be built without gap filling or
   extrapolation **
```

这是当前**剩余阻塞**。不是空气污染或激波检测造成的，也不是靠删点/填常数能
解决的。生产配置的阈值本轮**没有**改动。

### 6.2 背景阈值：仍会把静态帧误判

把密度放到 `1.225e-4`（1e-4 相对）、速度放到 `1e-5 m/s`（仍在密度/速度背景
附近）后：

- 状态 0 的静态帧在 1.3659 m 处出现候选，且被识别为外缘 → **静态帧检查失败**。
- 这说明 1e-4 仍然低于远场密度背景，不能作为“未扰动”的判据。

### 6.3 1% 背景阈值：静态帧通过，但候选不稳定

把密度放到 `1.225e-2`（1e-2 相对，高于 7.5e-3 背景）、速度 `2e-3 m/s`：

| 状态 | 三场候选半径 | binding field | 候选到最后一个受扰壳层 | 原始单元厚度 | 受扰样本连续？ | 受扰段数 / 内部环境壳层 |
|---:|---:|---|---:|---:|---|---|
| 0 | 无 | - | - | - | 否（全环境） | 0 / 0 |
| 19 | 0.28962 m | 压力 | 7.87e-4 m | 0.0329 m | **否** | 2 段 / 17 个 |
| 20 | 0.29519 m | 压力 | 8.03e-4 m | 0.0329 m | **否** | 2 段 / 1 个 |
| 22 | 0.30251 m | 压力 | 8.22e-4 m | 0.0329 m | **否** | 3 段 / 2 个 |

- 候选在**一个原始单元厚度**内贴近受扰区外缘（gap ≈ 0.0008 m << 0.0329 m），
  所以“候选是不是扰动外缘”在原生网格分辨率上可以说是；但按壳层网格看不是
  （壳层宽度远小于原始单元）。
- 受扰壳层样本**不连续**：中间夹了 1–17 个环境壳层。`classify_separation_frames`
  会因此把帧标成 `unresolvable`，不会把“能定位候选”当成“已分离”。
- 125 组阈值组合（压力/速度/密度各 1e-6…1e-2）中只有 **75 组**给出候选；
  候选半径范围 **0.208–1.366 m**，跨约 6.5 倍。候选对阈值敏感，不是稳定读数。
- 因此不能把 1% 阈值当成已验证的物理 `R_b`。它只用于验证代码路径和输出诊断。

### 6.4 静态帧检查

- 生产阈值：三场候选不存在（但密度/速度没有连续段，检查不是“选择性”的证据）。
- 1e-4 阈值：静态帧被误判为有外缘 → 失败。
- 1e-2 阈值：静态帧没有候选、受扰壳层为 0 → 通过。
- 三种情况的逐点证据在 `rb_*/rb_diagnostic.json` 和
  `rb_*/near_candidate_points.csv`。

---

## 7. 针对性回归测试与实际结果

实际执行：

```text
E:\Anaconda3\envs\pytorch\python.exe -m pytest -q -rs
560 passed, 2 skipped, 1 warning
```

（本轮最终全量执行；此前一次为 559 passed，新增“环境点掩盖受扰区”标志测试后为 560。）

两个 skip 都是 `tests/test_dyna_supervision.py` 的“本地当前 50 mm 提取/handoff
不存在”，属于“没跑”不是“失败”。

本轮新增/更新的重点测试：

- `test_air_contamination_is_contribution_weighted_not_product_contamination`
  —— 产物侧无效单元不能否决空气壳层；新空气污染为 0，旧全介质污染可 >0。
- `test_air_eligibility_survives_a_frame_without_a_confirmed_shock`
  —— 旧 `material` 仍撤标，新 `air_training_location_valid` 保留。
- `test_air_observations_use_phase_material_not_the_legacy_material_label`
  —— 防止 `build_air_observations` 误用旧 `material` 导致空气点只剩几十个。
- `test_air_observations_keep_rho_and_u_when_pressure_is_invalid`
  —— 逐字段有效性确实控制取数。
- `test_contact_air_rb_diagnostic.py`
  —— 连续外缘、速度控制候选、密度阻断候选、全环境帧不是爆炸外缘、壳层样本数
  不能冒充原始单元数。

本轮未启动长时间训练；未删除原始数据；未提交或推送。

---

## 8. 可直接在新电脑执行的命令

以下命令假设当前目录是仓库根目录，使用本机实际解释器：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PY="E:/Anaconda3/envs/pytorch/python.exe"
```

### 8.1 提取 50 mm 的 0/19/20/22 帧

```bash
PYTHONPATH=src "$PY" scripts/extract_contact_air_profile.py \
  --config configs/contact_air/base.yaml --case tnt_50mm \
  --states 0 19 20 22 \
  --out outputs/contact_air_air_eligibility_20261109/profile_50mm_states
```

### 8.2 生产阈值下的 R_b/缓存（当前会硬失败）

```bash
PYTHONPATH=src "$PY" -m spherical_blast_pinn.cli.contact_air precheck \
  --config configs/contact_air/base.yaml --case tnt_50mm

PYTHONPATH=src "$PY" -m spherical_blast_pinn.cli.contact_air build-cache \
  --config configs/contact_air/base.yaml --case tnt_50mm \
  --profile-dir outputs/contact_air_air_eligibility_20261109/profile_50mm_states \
  --out outputs/contact_air_air_eligibility_20261109/tnt_50mm_states.npz
# 预期：t0 outside every continuous R_b segment（见第 6.1 节）
```

### 8.3 R_b 诊断（生产阈值 / 背景附近 / 1% 背景）

```bash
for spec in "rb_current_thresholds 0.10132 1.0e-9 1.0e-6" \
            "rb_background_thresholds 0.10132 1.225e-4 1.0e-5" \
            "rb_background_1pct 0.10132 1.225e-2 2.0e-3"; do
  set -- $spec
  PYTHONPATH=src "$PY" scripts/diagnose_contact_air_rb.py \
    --profile-dir outputs/contact_air_air_eligibility_20261109/profile_50mm_states \
    --out outputs/contact_air_air_eligibility_20261109/$1 \
    --states 0 19 20 22 \
    --tol-pressure $2 --tol-density $3 --tol-velocity $4
done
```

### 8.4 诊断缓存（仅验证链路，不是已验证物理）

```bash
PYTHONPATH=src "$PY" -m spherical_blast_pinn.cli.contact_air build-cache \
  --config configs/contact_air/diagnostic_background_thresholds.yaml \
  --case tnt_50mm \
  --profile-dir outputs/contact_air_air_eligibility_20261109/profile_50mm_states \
  --out outputs/contact_air_air_eligibility_20261109/diagnostic_cache/tnt_50mm_states.npz
```

### 8.5 回归测试

```bash
PYTHONPATH=src "$PY" -m pytest -q -rs
```

---

## 9. 剩余阻塞与结论

1. **空气污染作用范围已修正。** 产物/混合单元的无效状态不再整层否决空气
   rho/u/p；空气自身缺失贡献只否决 p/e；逐字段有效性和覆盖率保留。
2. **contact_air 空气资格已独立。** 使用 `material_phase`、接触带保护和空气
   自身质量；没有 confirmed shock 不再自动撤标；激波状态单独记录。真实 50 mm
   状态 20 在无 confirmed shock 时仍有 624 个壳层样本。
3. **R_b 仍然定不出来。** 生产密度/速度阈值低于实测数值背景 3–7 个数量级，
   三场候选不存在；`build-cache` 在 `t0` 处硬失败。把阈值放到背景之上后，
   候选由压力控制，但受扰壳层不连续、阈值敏感性大（75/125 组合，0.208–1.366 m），
   不能当作稳定物理读数。
4. **没有外推、填常数或移动时间强行通关。** 诊断缓存单独放在
   `diagnostic_cache/`，配置单独标注为诊断；生产 `base.yaml` 阈值未改。
5. 下一步若要继续，需要的数据侧决定是：明确接受哪个“未扰动”密度/速度背景
   （并给出依据），还是先修正远场壳平均密度的数值噪声（例如检查 per-material
   mass / alpha V 在粗网格上的量级）。在这两项决定之前，不应把 50 mm 空气监督
   当成已就绪。

---

## 10. 输出文件索引

```text
outputs/contact_air_air_eligibility_20261109/
├── REPORT.md                         # 本文件
├── summary.json                      # 机器可读汇总
├── profile_50mm_states/              # 本次真实提取（states 0/19/20/22）
├── deck_checks/{tnt_80mm,tnt_100mm,tnt_500mm}/  # state 0 装药半径核对
├── rb_current_thresholds/            # 生产阈值
├── rb_background_thresholds/         # 背景附近诊断阈值
├── rb_background_1pct/               # 1% 背景诊断阈值
└── diagnostic_cache/tnt_50mm_states.npz  # 仅诊断，不是已验证物理
```

每个 `rb_*` 目录包含：

- `rb_diagnostic.json`：逐帧候选、binding field、动态样本、背景、敏感性；
- `near_candidate_points.csv`：候选窗口内逐点 p/u/rho、三项判据、逐字段有效性、
  覆盖率、空气资格；
- `report.md`：人类可读摘要。
