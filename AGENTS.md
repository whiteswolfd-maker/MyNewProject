# 自由空气中球形 TNT 爆炸的双场 PINN

本仓库只有一套实现：`src/spherical_blast_pinn/`，一个连续耦合的一维球对称
Euler PINN（Python 3.10+、PyTorch、`float64`）。第一版顺序耦合实现整体保留在
`_legacy_v1_backup/`，新路径**不**引用它，也不要恢复它。

`README.md` 是方法、数据状态与已知限制的权威说明。

## 不得改变的物理选择

* **唯一初始化方案（2026-10-21 决策，2026-10-25 定稿）**：

  ``t_phys = 0`` 时是**均匀静止的 JWL 产物球 + 静止空气**（``R_0`` 是材料界面，
  不是爆轰波前）。产物用现行 JWL 与 ``E0``，空气用现行理想气体 EOS。它**不是**
  CJ 表面突破，也不引入 Taylor 内场。

  小正时间启动由**局部平面黎曼问题**给出：

  1. 产物侧用实际 JWL 等熵线积分（``de/dV = -p/rho0``、``du/dV = c/V``）；
  2. 空气侧用理想气体激波的三条 RH 关系；
  3. 压力/速度匹配得到 ``p_*``、``u_*``，并保留完整自相似稀疏波与星区；
  4. 把该自相似扇嵌入球面，得到 ``t_phys = epsilon`` 的
     :class:`~spherical_blast_pinn.physics.two_medium_startup_planar.PlanarStartupField`；
  5. 版本化保存为 ``handoff.npz`` + ``handoff.json``，可重新加载并逐项核对。

  实现：``physics/two_medium_initial.py``、``physics/two_medium_riemann.py``、
  ``physics/two_medium_startup_planar.py``、``physics/two_medium_startup_handoff.py``、
  ``sampling/planar_startup_initial.py``；决策记录见
  ``DECISION_2026-10-21_UNIFORM_PRODUCT_SPHERE.md``。

  **不做任何网格时间推进**：新链路里没有有限体积/差分/拉格朗日逐步推进，也没有
  旧的爆轰产物场与拉格朗日双材料前置求解器。

* ``delta = (R_s-R_c)/R_0 = 1/200`` 是当前候选值，**不是**已验证常数；
  ``epsilon = delta R_0/(D_s-u_*)``。不同 ``epsilon`` 对应不同物理时刻，
  不能相减后称为时间收敛。
* 新启动场固定 ``reference_field=False``、``training_qualified=False``，
  不是旧 ``ReferenceField`` 的替代品。

## 时间

* 新链路的时间约定**只有一个**：

  ``tau = t_phys - epsilon``，``s = U_* tau / (R_0 hat T_*)``，
  ``s = 0 ⇔ tau = 0 ⇔ t_phys = epsilon_field``。

  由 :class:`~spherical_blast_pinn.physics.similitude.PlanarStartupScales` 显式
  表达：``start_time()`` 就是该场自己的 ``epsilon``，``detonation_velocity=None``
  （新原点不含 ``R_0/D``）。它只能由
  ``PlanarStartupScales.for_startup_field(field, time_scale=...)`` 构造。
* 旧方案 B 的 ``SimilitudeScales.start_time() = T_h = R_0/D + epsilon(R_0)``
  仍保留在**旧对象**里，但新链路不使用、也不得把它传给初值批次
  （``build_planar_startup_initial_batch`` 会明确拒绝）。
* ``hat_t = U_* tau/R_0``，``s = hat_t/hat_T_*``；``hat_T_*`` 仍由**训练半径**
  冻结。外推判据在无量纲 ``s`` 上做，不在秒数上做。
* 物理时间与网络时间的换算只经由 ``physics.similitude``，而且只平移一次
  （``physical_time_from_network`` / ``network_time_from_physical`` 配对）。
* LS-DYNA 的原始时间轴 ``T`` 属于数据侧；它还没有迁移到新链路的时间原点
  （见"尚未接入"）。

## 坐标与 PDE

* 网络输入坐标一律是 ``(s, xi)`` 或 ``(s, eta)``；物理半径只在需要 ``1/r``
  几何源项、EOS 与数据标签的地方出现。
* 链式法则只实现一次，在 ``physics/moving_frame.py``。时间项系数是
  ``U_*/(R_0 hat_T_*)``，``R_0`` 随算例变，所以 ``scales`` 是必填参数。
* 接触面与激波在采样与损失里各取**单侧**状态；不在两帧之间做时间线性混合，
  不跨材料插值。
* Torch 与 NumPy 两条 EOS 路径必须给同一个数，包括声速；声速平方用**未裁剪**
  的函数判断有效性。

## 硬失败，不是警告

下面每一条都会让运行停下来。改成警告就等于把"方程已满足"写成"残差恰为零"：

* ``R_s <= R_c``、前沿或前沿速度非有限、配置点半径到达轴心（``R_c <= 0``）；
* 零个产物配置点，或大部分请求的产物点被丢弃；
* 产物 PDE 残差在**零个点**上求值（``mean`` 会把 ``nan`` 吞掉）；
* 检查点里的 ``hat_T_*`` 与当前运行不同；
* 检查点里没有采样器的 RNG 状态；
* 训练半径缺数据；
* 训练算例的数据兼容性不是 ``compatible``。

**``R_c < R_0`` 不在这一列**：它是监控量，不是非法移动域。

## 正式命令的当前状态

* 可运行：``python -m spherical_blast_pinn.cli.startup``（写出 handoff）、
  ``python -m spherical_blast_pinn.cli.validate_startup_planar``（delta 敏感性）、
  ``python -m spherical_blast_pinn.cli.check_startup_chain``（装配 + 一次反向传播），
  以及数据侧的 ``audit_dyna`` / ``audit_data`` / ``extract_profiles`` /
  ``validate_eos``。
* **已接入新启动场**：``sampling.sampler.RadiusSampler(field, scales, tau_end)``
  直接生成 IC / PDE / 接触面 / 激波点；IC 复用
  ``build_planar_startup_initial_batch``。``losses.objective.contact_pressure_scale(field)``
  取场自己的 ``p_*``。``evaluation.checks.build_check_set(field, scales, tau_end)``
  用同一启动场与同一压力尺度建立固定检查集。无数据监督时数据与几何批次为空，
  **不构造假算例**；把真实算例传给采样器会明确报未支持（数据时间轴尚未迁移）。
* **尚未接入**：``train`` / ``predict`` / ``compare``。它们会明确报
  “新启动场的训练接口尚未接入”并以退出码 3 结束，**不会**回退旧参考场。
  ``Trainer`` 构造时直接抛 ``TrainingNotConnected``：训练执行层仍按旧的多半径
  / 检查点语义，迁移它属于下一轮。不得用伪造的 ``ReferenceField``、伪造爆速或
  ``training_qualified=True`` 绕过。
* **检查点（新契约）**：``training.checkpoint`` 只读写
  ``spherical_blast_pinn.startup_checkpoint`` v1，绑定启动场的
  ``model_id`` / ``method_version`` / ``cache_key``、完整的
  ``PlanarStartupScales`` 参数、模型结构/EOS/dtype、模型参数、可选的优化器
  状态、步数、全局+采样器随机状态，以及**采样配置与动态时间窗口**
  （``tau_end``、各类点数、分层比例、``r_min_fraction``、当前窗口分数与
  历史）。恢复端必须逐项相同：``tau_end``/点数/分层不同，或接收采样器站在
  另一个启动场/尺度上，都会在修改模型、优化器和采样器**之前**明确拒绝——
  换一套采样设置继续跑不能称为精确续训。保存用
  ``save_startup_checkpoint(path, field=..., scales=..., model=..., optimizer=..., sampler=...)``，
  读取用 ``load_startup_checkpoint(path, handoff_json=..., model=..., sampler=...)``。
  旧检查点（format ``spherical_blast_pinn.checkpoint``、version ≤ 8、带爆速与
  ``T_h``）按格式/版本**明确拒绝**，不迁移、不猜测旧时间原点。
* 保留但未接线的通用组件：``models/``、``physics/pde.py``、
  ``physics/moving_frame.py``、``losses/terms.py``、``sampling/batch.py``、
  ``training/schedule.py``、``training/balancing.py``、``training/trainer.py``。

## 已退出的旧方案（只作历史）

2026-09-22 至 2026-10-21 之间使用的“方案 B”是：中心起爆、有限爆速传播形成
非均匀爆后产物场，波前到达装药表面后接空气，再用短时球对称拉格朗日双材料
前置求解到 ``T_h = R_0/D + epsilon``。对应的 ``physics/detonation.py``、
``physics/precursor.py``（含 ``ReferenceField`` 与验收缓存）、
``physics/two_medium_startup.py``、``cli/reference_field.py``、
``cli/validate_startup.py``、``cli/diagnose_startup.py`` 及其专属测试已删除。
旧缓存文件（``reference_field.npz``）不再被任何代码读取，也不得当成新启动场。

## 剖面提取

* 材料分区一律来自 d3plot 的 **ALE 材料体积分数**（`cli.extract_profiles`）。密度
  最陡下降那条启发式路径已经废弃，只在没有配套文件的旧导出上还会走，并且会被闸门
  判成 `incompatible`。
* **分类必须用全部单元的体积加权体积分数**（`shell_average.mixture_fraction_profile`），
  不是只对纯产物单元求平均的那个。后者在没有产物的壳层上是 `nan` 而不是 0，既过不了
  `>= 0.95` 也过不了 `<= 0.05`，空气会整个消失；而且它在产物侧恒等于 ~1，永远不穿过
  0.5，接触面根本定不出来。
* **标签必须蕴含状态。** "标成空气、场量却是空"不是少几个点的问题：采样器会把 `nan`
  标签逐个丢掉，数据损失里的空气组于是静默消失。提取器写盘前检查这一点，兼容性检查
  再独立验一遍，门槛是"零个"，不是"比例很低"。
* 比内能随剖面一起写出（`e`）。它是**重建量，不是 DYNA 输出量**：这套 d3plot 里实体
  单元根本没有比内能结果，所以只能按各材料的 EOS 反解。**反解是除法，不是求根**，
  `omega > 0`、`rho > 0` 时永远有有限实数值——不存在"没有实数解"那一档。会出现的
  是反解出**负**值，那说明该单元的 `p` 落在 JWL 冷压曲线之下（化学能还没放出来，
  未起爆或正被爆轰波扫过），本来就不该套产物 EOS。把 `nan` 说成"解不出来"会让人去查
  求解器，而问题在判据的语义上；实测 40 个状态 × 122 万单元里反解为非有限值的是
  **0 个**。
* **"标了材料"不等于"有状态"。** 没有状态的单元其比内能留空，且逐帧计数写入元数据；
  下游只能把缺口算成"反解为负"，不能算成"没有解"。壳层平均会**静默跳过** `nan`，
  所以污染必须按**体积**单独算一遍（`contaminated_shell_fraction`），任何非零比例
  让整个壳层作废——否则同一个壳层里剩下的单元会被平均成一个看着很正常的数。
* **没有激波就没有空气域。** `R_s` 定不出来时 `R_c < r < R_s` 没有边界，空气侧无法与
  环境态区分，而环境态按规范**无标签**。此时整个空气侧撤标，不能把两米以内的均匀
  大气当成空气监督送进训练。
* 前沿与场量用**同一套网格**。曾经前沿另用一套均匀分箱定位，近场分箱比场量壳层粗
  四倍——"加密了场量网格"于是不等于"前沿也加密了"。
* **激波标签用"压力梯度候选 + RH 核查"**（`data/shock_detect.py`，§10），不是任何
  形式的阈值判据。三类判据都试过并被否掉：
  * "最外侧超压点"：测到的是远场数值噪声（`p/p_amb` 在 1.000 上下摆动、幅度可达
    1.5%），50 mm 第 10 状态落在 94–101 mm 的噪声尖峰上，而真实超压区外缘在 58 mm；
  * "与接触面相连的连续超压区外缘"：稳健一些，但**那只是外缘，没有核验**；
  * 固定阈值：放松一点噪声就成了激波，收紧一点弱激波就漏掉。

  现在的做法是：在**连续有效的空气侧**数据上找 `-dp/dr` 正峰（用真实 `dr`，
  噪声底取中位数梯度的倍数与 `1e-3 × p / dr` 的较大者），跃迁带取**峰值梯度的一个
  比例**以上的连续区间——不用"高于噪声底"，因为衰减的爆炸波在波后一直有梯度，
  那样圈出来的"带"会把已经衰减掉的状态当成波后，RH 残差被撑到 30% 以上而否掉真激波。
  两侧状态取**紧邻跃迁带外的平台**。然后由**质量守恒**从实测两侧状态解 `Ds`，
  再核查动量与能量 RH、压缩方向、`Ds - u2 > 0`、熵不减。

  **质量守恒已经用来求 `Ds`，所以它不构成独立证据。** 两侧状态一律取实测数据，
  **不得先由 RH 造出波后状态再验证 RH**。弱激波的密度跳变埋在窗口自身散布里时，
  `Ds` 病态，标记为证据不足而不是给一个数。
* **检测状态与监督资格分开记**（`confirmed` / `doubtful` / `not_detected`）。跃迁带
  跨不过一个原始单元、或两侧状态不足以定 `Ds`，都只能记存疑，**不给监督资格**；
  没有监督资格就没有可靠位置范围，空气侧整体撤标。实测 50 mm：状态 778 被确认
  （`Ds = 594.5 m/s`、压缩比 1.87、动量残差 0.081），其余被诚实拒绝。
* **壳层网格的宽度不是分辨率。** 前沿位置和 `h = R_s - R_c` 的分辨率下限是**ALE 单元
  自己的径向厚度**，实测装药外第一个空气单元：50 mm 算例 8.9–14.1 mm，**80 mm 算例
  208–361 mm，100 mm 算例 260–452 mm，500 mm 算例 1303–2258 mm**。把壳层从 2500 加到
  5000 只是把同一个粗糙单元重采样得更细，"两套网格一致"因此只证明提取对分箱不敏感，
  **不证明网格够用**。
* **接触面也要判"分辨得出吗"，判据是穿越区间里的空壳层个数。** 0.5 穿越如果跨着一段
  没有数据的壳层插出来，报出的半径就是"两个壳层中间某处"：80/100/500 mm 三个算例的
  接触面因此被插到 131.97 / 164.96 / 824.80 mm，而装药半径是 80 / 100 / 500 mm——
  偏高 65%。**不要用"跨距 vs 单元厚度"比大小**：单个巨型单元内部处处是空的，跨距
  （104 mm）反而小于单元厚度（128 mm），比大小会把这一档整个漏掉。三种状态记在
  `material.npz` 的 `contact_state` 与 `shock_status` 里。
* **d3plot 文件数不等于状态数**（实测 780 文件 / 779 状态）。用 `_resolve_state_count`
  的探针拿权威值，**并用边界复核它**（`n-1` 必须读出恰好一个时间戳、`n` 必须读不出）。
  解析失败一律**停下**；回退到文件数会静默错位，而错位的时间戳不会报错，只会把每条
  曲线平移。探针依赖 LASSO 的报错措辞，复核不依赖。
* 读取状态要**分批**（`--batch-states`）。`buffered_reading=True` 不是"只留一帧"，
  LASSO 仍为每个被选中的状态分配数组。
* 材料映射**解析 `*ALE_MULTI-MATERIAL_GROUP` 的出现顺序**，不拿 part id 当列号。
  四个算例上两者恰好相等（psid = 1..7 且列也按 1..7 排），但那是卡片刚好这么写的：
  500 mm 的 part 名全是 `Solid Body N`，靠名字猜不可能。字段名因此是
  `product_ale_indices` / `air_ale_indices`，不是 `*_material_ids`。
* **球壳统计用单元与壳层的重叠体积**（`data/shell_average.py`，§8），不是"质心落在
  壳层里就整格算进去"。后者在壳层比单元细时是阶梯，而且**壳层之间的体积与质量不守恒**。
  重叠按**圆台重构**（`dV/dr ~ r^2`，对固定立体角精确；实测完全在内侧的单元重叠分数
  之和精确为 1）。`rho`、`p` 体积加权，**`u`、`e` 质量加权**。
  **重叠体积不增加分辨率**：它只把同一个单元在壳层之间重新分配。
  覆盖率的分母是**被建模的那一块**（八分之一球要除以 8），否则覆盖率会恰好小 8 倍。
  聚合走**稀疏矩阵乘**：2040 万条配对逐场量 `bincount` 是 2.83 s/帧，改成
  `A @ (w * v)` 后 1.26 s/帧；`A` 在欧拉网格上整个算例只建一次。
* **lasso-python 2.0.3 读不动 80 / 100 / 500 mm 的单元 history 变量**（50 mm 正常）。
  那三个算例的头块 `neiph = 22` 且 `has_element_strain` /
  `has_solid_shell_plastic_strain_tensor` 都是真，LASSO 把这两项各减 6 得到 10 列的
  容器，却往里面写 16 列，直接抛 `ValueError: could not broadcast ...`。
  `raw_reader._install_solid_history_allocation_fix` 把容器按数据的宽度加宽（**只增不
  减、不动数值**）。**取错列不可能同时满足 `_charge_radius_from_volume`**——它由体积分数
  与单元体积反推装药半径，四个算例分别给出 49.9979 / 79.9943 / 99.9929 / 499.9610 mm
  （相对偏差 ≤ 8e-5）。升级 LASSO 之后要重新验这条。

## 数据闸门

* `audit_dyna` 写 `data_compatibility.json`，训练入口读它。状态 `incompatible`
  或 `unknown` 一律拒绝；越过它必须显式写 `data.allow_unverified_cases: true`，
  并且检查点与运行清单会被打上 `data_compatibility: unverified`。
* 没有审查证据**不等于**通过。报告里没有的算例是 `unknown`。
* 没有从材料体积分数来的分区，就是没有；那一项独立判定，别的项审得再好也不放行。

## 检查与选模型

* 每 `training.check_every` 步在一套**固定**检查点上跑物理验收（`evaluation/checks.py`）。
  它一次采样、全程复用、独立种子、取开发帧、不进梯度。保存检查点**不等于**
  完成验收。
* 选模型的偏序在 `evaluation/checks.py:SELECTION_CRITERIA` 里逐字写死：先非物理
  状态计数，再物理/界面残差超出 §16 容差多少，最后才比开发集误差。**不按加权
  训练总损失选模型。**
* 数据在训练半径内按**连续帧组**留出约 10% 作开发检查（`ExtractedCase.frame_split`）。
  同一帧的相邻网格点高度相关，随机拆开会让"开发集误差"只是训练误差的另一种写法。
* 守恒诊断（产物质量漂移、固定控制体能量平衡）只报告，不进梯度。

## 契约

* 半径只在还原物理量时乘上去。损耗、采样、模型一律通过
  `losses.terms`/`sampling.batch` 携带的 `charge_radius` 取它，不读模块默认值。
* 网络输入坐标一律是 `(s, xi)` 或 `(s, eta)`，物理半径只在需要 `1/r` 几何源项、
  EOS 与数据标签的地方出现。
* 链式法则只实现一次，在 `physics/moving_frame.py`。任何地方若要把网络输出换
  成物理导数，必须走那里；重复叠加链式项是本仓库最容易犯的错误。
* 抽样帧一律返回**真实帧号**，不是 `usable_frames` 里的位置。两者都是合法整数，
  混用不会报错，只会安静地读错帧。
* Torch 与 NumPy 两条 EOS 路径必须给同一个数，包括声速。声速平方用**未裁剪**的
  那个函数判断有效性。
* `tests/test_moving_frame_and_pde.py` 用解析光滑函数把显式链式法则与自动求导
  对齐到机器精度；`tests/test_review_regressions.py` 逐条钉住审查清单里的每一个
  可复现差异。改动移动坐标、时间轴或损失分组前后都要跑它们。

## 规范

```bash
pytest -q
```

测试通过不等于完整爆炸预测通过。哪些项"已执行"、哪些"仅实现"、哪些"参考数据
不足以判断"，必须逐项说明，不能只报一个通过率。

## 规范

```bash
python -m pytest -q
```

干净环境安装与运行：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install .          # 或 pip install -r requirements.txt
.venv/Scripts/python -c "import spherical_blast_pinn, sys; print(spherical_blast_pinn.__file__)"
.venv/Scripts/python -m spherical_blast_pinn.cli.startup --output-dir outputs/planar_startup
```

测试通过不等于完整爆炸预测通过。哪些项"已执行"、哪些"仅实现"、哪些"参考数据
不足以判断"，必须逐项说明，不能只报一个通过率。
