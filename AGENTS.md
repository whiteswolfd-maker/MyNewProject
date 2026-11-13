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

测试通过不等于真实 50 mm 数据验证；当前尚无真实一维 ALE1D 数据，二进制读取
未实现。
