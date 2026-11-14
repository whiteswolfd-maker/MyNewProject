# 仓库约定

本仓库只有一个当前方案：`src/spherical_blast_pinn/contact_air/`。共享
`ContactNet(t,R0)->R_c` 与 `AirNet(zeta,t,R0)->(rho,u,p)`，两个持久 Adam 交替
更新，几何为 `R_c(t) <= r <= R_s(t)`，`R_s` 是压力梯度识别出的外传空气激波。
前置为 CJ + JWL 等熵释放 + 空气 Hugoniot 匹配。

旧 ProductNet/FrontNet、旧联合总损失训练器、旧三维球壳提取、旧平面黎曼启动、
旧 `R_b` 环境边界、固定 500 mm 测试规则均已删除。不要恢复、不要重建第二套
入口、不要为兼容旧测试重新提供别名。

`README.md` 是方法、命令、数据状态和已知差异的权威说明。

## 不得改变的物理选择

- 时间 `t` 是物理秒；`t0=t_sep+epsilon`。半径输入不缩放，不乘 `R0/100mm`。
  唯一的可选输入缩放是 `networks.time_reference_s`：两个 MLP 的输入变为
  `tau = t / T_ref`（默认 1.0，即秒）。`T_ref` 是由训练组确定的固定值、所有
  算例共用；缩放必须留在自动求导图内，激波轨迹、速度、PDE 和边界条件仍按秒，
  速度仍是 m/s。
- 前置保留 CJ 状态、JWL 等熵释律、空气强冲击/精确 Hugoniot 匹配、
  `R_sep=R0 V_sep^(1/3)`、`t_sep` 和 `C` 的来源。不要替换成另一套前置能量。
- 初始薄层：

  ```text
  epsilon = R_sep / [epsilon_denominator_factor * (D_s_sep - u_c_sep)]
  t0      = t_sep + epsilon
  Rc0     = R_sep + u_c_sep * epsilon
  Rs0     = R_sep + D_s_sep * epsilon
  ```

  `epsilon_denominator_factor` 是配置值（本轮 200），真正参与计算。固定
  `epsilon_s` 输入拒绝。`D_s_sep > u_c_sep`、`epsilon > 0`、初始厚度为正都是
  硬检查。
- AirNet 初值只在 `0 < zeta < 1` 使用分离态 `(rho_a_sep,u_a_sep,p_a_sep)`；
  没有未扰动空气子区，没有旧 `zeta_s0` 分段初场。
- 右边界是有限马赫数 RH 波后状态；`D_s > c_inf`、`M_s > 1` 才能进入该项。
  不恢复静止环境 `rho_inf/0/p_inf`。
- ContactNet 初值只约束位置；独立初始速度损失已删除。
- 接触区间取与爆心主产物区外缘相连混合段的**实际单元边界**；不使用
  `alpha=0.5` 单点。不计算、不返回也不缓存区间中点；训练对象只有
  `[R_c,low, R_c,up]`。
- ContactNet 压力闭合用与前置一致的相对比容
  `V=V_sep (R_c/R_sep)^3`；不要与有量纲 `1/rho` 混用。
- 材料卡数值仍是未核验占位值；`material_source_verified` 未核验前不得当作
  已验证结果。

## 一维数据与几何契约

- 标准中间格式 `spherical_blast_pinn.standard_1d_source`（`.npz` 或目录）。
- 缓存必须记录生成时标准文件的 `standard_data_digest`
  （`sha256(npz member name|crc32|size)`，读 ZIP 目录，不重读数组）。复用
  缓存、`--cache-only`、审计都必须核对它；标准被重建时对应缓存必须重建。
  没有该指纹的旧缓存不得当作 current。
- LS-DYNA 导入必须保留全部原始输出帧：`frame_selection_policy="all_frames"`，
  `raw_frame_count == imported_frame_count`，`dropped_frame_count=0`，并记录
  完整 `raw_frame_keys`、首末时间和 `raw_input_fingerprint`。旧的
  `--dense-until-s`/`--frame-stride`/事件保帧窗口不属于正式数据路径；传入即
  报错。全帧导入不恢复未输出的求解器内部时间。
- metadata 必须显式提供 `units`、`coordinate`、`velocity_component`、
  `absolute_pressure=true`、`position_type`、`frame_times`、`cells_per_frame`。
  `coordinate`/`velocity_component` **没有静默默认**；缺键拒绝。
- `frame_times` 必须全部有限且严格递增；`frame_offsets` 必须是一维整数数组，
  首值 0、严格递增、末值等于总单元数，并与 `cells_per_frame` 一致。所有逐单元
  数组（含 `cell_id`）必须等长；不允许截断切片后静默接受。
- `cell_id` 必须为整数且每帧唯一。缺省生成 ID 时在 frame/case 的
  `source["cell_id_source"]`、`source["generated_cell_id_frames"]` 中记录为
  generated，不能伪称原始 ID。
- 读取后统一 SI；`u` 是径向速度分量，不是速度模长；`p` 是绝对压力。
- 逐字段掩码：`valid=True` 但有 NaN 时拒绝；缺失字段不填零/环境常数/前帧值。
- 两介质材料语义：product + air only。显式 `alpha_air` 时
  `alpha_product + alpha_air = 1 ± 1e-4`；空腔/第三介质不支持，禁止归一化、
  填充或修正。未提供 `alpha_air` 时，剩余分数按 `1-alpha_product` 解释为空气并
  记录来源。显式 `alpha_air` 是空气判定依据，不能仅凭 `alpha_product≈0` 认定
  纯空气。接触区间、空气掩码、激波检测、事件检测、观测筛选必须共用
  `RadialFrame.material_masks` / `air_mask`。
- 单元：`r_left/r_right/radius` 有限、径向坐标非负、`r_right>r_left`、代表位置
  在单元内、径向位置严格递增、单元不重叠。真实正缺口允许并作为分段。
- 节点源必须已经转换为单元代表值：
  `node_to_cell_convention="preconverted_cell_representatives"`。不支持任意
  约定字符串，也不做猜测式节点→单元转换。
- 连续段必须同时满足字段有效、材料连通和**真实单元边界相邻**。不能仅凭数组
  下标连续跨过空间缺口求梯度或统计纯空气区间。
- 激波只用 `-dp/dr` 正峰；不跨缺失点/混合带/几何缺口；要求压缩方向、距接触
  带距离、相邻有效帧外传。位置重复是 plateau，不是向内运动；位置严格回退
  拒绝。不自动跨失败帧插值，不暗中平滑。
- 事件对齐：`t_model = t0 + (t_data - t_event_data)`。事件帧必须**不早于**
  理论 `t_sep`；窗口内没有合格后续帧时明确报错，不退回旧帧。搜索窗口只作
  定位指导；原始时间/帧号/偏移必须保存；不缩放帧距、不平移半径。原始时间
  保持不变，模型时间仍执行该事件偏移。
- `ShockTrajectory` 用 PCHIP；单点无速度，两点只有区间平均速度；不外推。分段
  系数用 SciPy 预处理，运行时用 PyTorch 求值和解析导数；不得 detach `t` 后
  转 NumPy 查询并声称自动微分完整。

## 模型与损失契约

- 默认网络：ContactNet 5×50，AirNet 10×32；smoke 可显式缩小并标明是工程配置。
- `R_c = R_s sigmoid(raw)`；
  `R_c_dot = D_s s + R_s s(1-s) raw_t`；`L=R_s-R_c` 必须为正，不 clamp 厚度。
- 空气 PDE 三条残差在 `mapping.py` 只实现一次；两网络复用。能量尺度是单位体积
  总能量 `s_E=s_p/(gamma-1)+0.5 s_rho s_u^2`，`s_fE=s_E/T_star`。
- 数据坐标在损失时用**当前预测的** `R_c` 与 `R_s` 重算：
  `zeta=(r-R_c)/(R_s-R_c)`。域外点计数后排除，不 clamp；整批全筛掉硬失败。
- 预测/目标/掩码先规范形状再按掩码选择，禁止 `[N,1]` 与 `[N]` 广播成 `[N,N]`。
- 接触标签在自己的 `t_contact` 上重新计算 ContactNet；不借用 PDE 时刻预测。
- 两阶段交替：空气阶段只更新 AirNet；接触阶段只更新 ContactNet。接触阶段用
  参数冻结、输入导数保留的 functional AirNet，不用 `no_grad`/整体 detach 包住
  物理残差。
- 每次反传检查：目标网络有非零有限梯度、非目标网络无梯度、参数未移动。
- 两个优化器持久存在；不要每轮重建。
- loop 日志包含三个归一化空气 PDE 分项（`pde_*_normalised`）；它们只用于
  记录，不得改变损失、采样或梯度，也不得引入随机数调用。
- 接触区间监督权重 `training.contact_data_weight`（默认 1.0）只乘
  `lambda_data * L_c_data`；初值、界面、空气数据和 PDE 权重不变。历史记录保留
  未加权分项和实际权重。

## 硬失败，不是警告

- `t, R0, R_s, D_s` 形状不一致或非有限；`R_c`/`R_c_dot` 非有限。
- `L = R_s - R_c <= 0`；几何退化。
- `D_s <= c_inf` / `M_s <= 1` 的采样进入 RH 边界损失。
- `D_s_sep <= u_c_sep`、`epsilon <= 0`、初始厚度非正。
- 理论 `R_c0 >= R_s_data(t0)`（硬几何无法容纳初始位置）。
- 数据/接触/PDE 采样在零个有效点上求值；总损失或梯度非有限。
- 检查点数据/配置/验证划分/网络结构不匹配；旧缓存/旧检查点版本。网络
  `time_reference_s` 不一致也必须拒绝续训和预测（旧检查点缺该字段按 1.0，
  整个模型描述块缺失则拒绝）。检查点 `contact_data_weight` 与当前配置不一致
  也必须拒绝续训（旧检查点缺该字段按 1.0）。
- 一维源缺少必填 metadata、单位未知、掩码覆盖 NaN、节点约定未支持、
  单元重叠、时间非严格递增。

## 采样与命令

- `train --cases all` 或显式 `case_id` 集合；只选配置中 `role: train` 的算例。
  无固定 50/80/100；`role: test` 不得进入训练。
- 配置中整例 `role: validation` 的算例参与验证评分与 `best.pt` 选择，不进入
  训练采样；不能与训练算例重复 `case_id`。`validation.holdout_fraction: 0`
  表示训练算例全部帧可用，验证完全由该半径带提供。
- `build-cache --source <标准一维文件或目录> --case ... --radius ... --role ... --out ...`。
- 观测采样使用加载时建立的帧索引，但**不得改变抽样分布或随机数调用顺序**：
  仍是先均匀选帧、再均匀选帧内点，重复抽样规则不变；非单调数组走
  `_sample_scan()` 原扫描分支。预测 `R_c` 决定的域内筛选仍在损失阶段执行，
  不能提前缓存成固定掩码。`scripts/benchmark_sampling.py` 只做工程计时。
- `scripts/record_baseline.py` 和 `scripts/diagnose_contact_gradients.py` 是只读
  工具：前者记录代码/配置/缓存指纹/检查点/环境。后者必须区分
  `--mode training`（正式 10 训练算例预算与采样顺序，用于整体判断）和
  `--mode stratified`（早/中/晚窗口的补充定位；初值仍在 `t0`，空窗口明确
  不可评估，不能回退全时段）。静态梯度先对 10 个算例的损失取平均再求加权
  梯度；`--adam-trials` 从检查点内存副本做独立单步更新，不得改文件。参考指标
  必须同时给出算例等权 `macro_mse/rmse` 与帧数加权 `pooled_mse/rmse`，覆盖率
  用整数 `covered_count` 累加，禁止 `int(coverage*n)`。三分位边界只归入一个
  分段；空算例/分段标为不可评估，不能当零。零梯度方向标为不可定义；非有限
  梯度、静态/Adam 失败、缺少优化器状态或检查点哈希变化必须返回非零。汇总
  脚本同样不得用 None/NaN 参与统计，源诊断失败时汇总必须非零并标记。

- `scripts/contact_fullcontact_pair.py` 是配对单步诊断：A 用正式随机接触标签，
  B 用全部合法训练接触标签（每例均值、算例等权）。A/B 必须从同一检查点、
  同一非接触批次、同一 AirNet 更新开始，只替换 contact_data 标签；全标签
  预先生成，不额外消耗随机数；每个 A/B 对独立恢复检查点，检查点不得改写。
  验证算例不进入梯度或固定参考。A/B 的随机批次损失与全标签损失不能当作同一
  集合的数值比较。汇总按配对差值分种子、合并报告，不得只挑有利种子。
  汇总必须使用实验清单核对预期文件、检查点/配置哈希、种子和试验数，并核对
  requested/completed/failed 与 trials 实际状态、编号唯一性和逐例全标签非空；
  不能只信源文件 `ok=true`。参考帧数从记录计算，分别报告独立集合与累计评估点；
  共同 A 标签只作补充，不作为 B 优劣的独立否决条件。

- `scripts/contact_fullcontact_pair.py --analysis direction` 是实际 Adam 方向
  诊断（汇总 `scripts/summarize_contact_adam_direction.py`）：每个检查点/种子
  固定 20 个独立配对单步，沿用 B 全标签路径。记录四项加权梯度及总和、真实
  Adam 更新、`gC·δ`/`gJ·δ`、实际 C/J 变化、同方向 `a=1,1/2,1/4,1/8` 的内存
  副本评估和中心差分。若 `gC·δ` 为正，按实际 Adam 公式重构 δ，并在固定本次
  二阶矩分母下把分子拆成历史一阶矩与当前四项贡献；这是代数分解，不是删除
  某项重跑的因果实验。梯度裁剪、非零 weight_decay、AMSGrad、maximize 等
  不支持项必须明确拒绝。参数插值后必须恢复参数、优化器状态、AirNet 和 RNG；
  非目标网络保持冻结；检查点哈希不得变化。
- `scripts/prepare_fullframe_data.py --cases ...` 是薄编排器：调用现有导入器和
  `build-cache`，把全帧标准/缓存写到 `outputs/fullframes/`；不覆盖旧数据。
- `scripts/audit_reference_data.py --stage <manifest|imports|metrics|consistency|report|all>`
  是只读数据审计：独立解析原始 `.xy`、按容差核对导入值、从**初次入射脉冲**
  （到达阈值 → 严格回零 → 脉冲内最大值）提取固定半径峰值并与 Henrych 对比、
  用 `u(E+p) - D_s E` 检查 RH、在 t0 后早/中/晚多个有效帧上检查接触/空气
  压力与边界轨迹斜率。脉内振荡和后续正 excursion 只标为候选，不称为二次
  激波。不得修改缓存、检查点或训练配置；Kingery-Bulmash 在系数表核实前保持
  不可用，Henrych 系数标记暂定。导入完整性失败（漏帧/多余帧、原始键或时间
  不一致、缓存 raw_times/帧数不一致、逐值 mismatch）必须返回非零退出码；
  KB/Henrych 和 JWL 局部差异只记录，不作为失败。全帧验收用
  `--stage all --import-sample-frames 0`。`prepare_fullframe_data.py` 在跳过
  现有标准/缓存前会核对全帧键、frame_times、缓存 raw_times、原始输入指纹和
  缓存记录的标准数据指纹；本次重建了标准就强制重建缓存。
- `predict/evaluate --case ...`；预测只读 metadata 与 shock 轨迹，评估才读参考
  内部场。500 mm 与其他半径同一接口。
- 空气质量数据先抽帧再抽帧内点；右边界只在 `D_s > c_inf` 时刻采样。
- 预测是“给定测得的 `R_s` 轨迹条件下的预测”，不是无需轨迹的自主跨半径激波
  预测。

## 检查

```bash
python -m pytest -q
```

必须使用本仓库解释器并核对 `spherical_blast_pinn.__file__`。旧的 541 数字与
其他环境的输出不能作为本工作树的证据。另需在新进程执行 README 的
标准 `.npz → build-cache → train → resume → predict → evaluate` 全链路。

测试通过不等于真实数据验证；当前使用 LS-DYNA ASCII `*DATABASE_PROFILE` 输出
（不是二进制 ALE1D 槽位读取），二进制读取仍未实现。
