# 仓库约定

跟随 `AGENTS.md`。方法与数据状态在 `README.md`。

本仓库只包含当前的一维球对称 ContactNet + AirNet 方案，位于
`src/spherical_blast_pinn/contact_air/`。旧 ProductNet/FrontNet、旧三维剖面
提取、旧启动场与固定 500 mm 规则已经删除，不要恢复它们，也不要在新路径里
重建两套实现。

改动物理、时间约定、几何映射或链式法则之前，先读 `AGENTS.md` 的“不得改变的
物理选择”与“硬失败，不是警告”两节。
