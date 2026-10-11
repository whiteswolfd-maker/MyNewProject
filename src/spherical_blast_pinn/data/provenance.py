"""材料分区的**结构化来源记录**（提示词 §5）。

以前这里只有一句自由文本，判据是 ``"volume fraction" in text.lower()``——一句
"not from volume fraction" 里也含这几个字，于是**否认被读成肯定**。改成只看一个
布尔量之后，判据又变成"必须有一个非空值"：数字 ``123``、``{"unrelated": "value"}`、
``"heuristic — not from volume fraction"`` 都能通过。

两种写法是同一个毛病的两面：**它们都没有核实来源，只是换了一种方式相信它**。

这里定义的是一份**有结构、有版本、有类型**的记录。它回答四个问题：

* **怎么来的**——``method`` 取自一个封闭枚举，不是自由文本；
* **谁写的、按哪一版约定**——``producer`` 与 ``producer_version``；
* **用了哪些输入**——``fields`` 与 ``input_identifier``；
* **参数是什么**——``detail`` 里的阈值等。

而且它必须与那个布尔声明**互相印证**：``method`` 说"体积极分"时布尔量必须是真，
反之亦然。两个字段各说各话时，读数的人会按自己先看到的那个去理解。

**这份校验证明的是"记录符合约定"，不是"物理算法正确"。** 一份自洽的来源记录
丝毫不能说明体积分数读对了列、或者提取器没有 bug；它只说明**这份导出说清楚了自己
是怎么来的**。物理正确性另有闸门。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: 分区方法的**封闭枚举**。加新方法要改这里，那是有意的：自由文本没法机器判。
METHOD_ALE_VOLUME_FRACTION = "ale_volume_fraction"
METHOD_HEURISTIC_DENSITY = "heuristic_density_detector"

PARTITION_METHODS: tuple[str, ...] = (
    METHOD_ALE_VOLUME_FRACTION,
    METHOD_HEURISTIC_DENSITY,
)

#: 只有从体积分数读出来的分区才算权威。
AUTHORITATIVE_METHODS: tuple[str, ...] = (METHOD_ALE_VOLUME_FRACTION,)

#: 支持的记录版本。读端只认这些；生产端写出别的版本时，读端**拒绝**而不是猜。
SUPPORTED_VERSIONS: tuple[int, ...] = (1,)

#: 来源记录必须带的键。缺一个就不成结构。
REQUIRED_KEYS: tuple[str, ...] = (
    "method", "producer", "producer_version", "fields", "input_identifier",
)


def make_partition_source(
    *,
    method: str,
    producer: str,
    fields: Sequence[str],
    input_identifier: str,
    detail: Mapping[str, Any] | None = None,
    producer_version: int = 1,
) -> dict[str, Any]:
    """生产端构造一份来源记录。写出来的东西必须能被 :func:`validate_partition_source` 接受。"""
    return {
        "method": str(method),
        "producer": str(producer),
        "producer_version": int(producer_version),
        "fields": [str(name) for name in fields],
        "input_identifier": str(input_identifier),
        "detail": dict(detail or {}),
    }


def validate_partition_source(record: Any) -> list[str]:
    """检查一份来源记录。返回**问题列表**（空列表表示通过）。

    只做结构、类型、版本与一致性检查；不判断物理对错。
    """
    problems: list[str] = []
    if record is None:
        return ["没有留下结构化的来源记录（material_partition_source）"]
    if not isinstance(record, Mapping):
        # 数字、字符串、列表都落在这里。以前它们只要有内容就算通过。
        return [
            f"来源记录必须是一个结构化映射，实际是 {type(record).__name__}"
            "（自由文本与数字不能当机器证据）"
        ]
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        problems.append(f"来源记录缺字段 {missing}")
    method = record.get("method")
    if not isinstance(method, str) or method not in PARTITION_METHODS:
        problems.append(
            f"来源记录的方法 {method!r} 不在允许的枚举 {list(PARTITION_METHODS)} 里"
        )
    producer = record.get("producer")
    if not isinstance(producer, str) or not producer.strip():
        problems.append(f"来源记录的生产者标识无效：{producer!r}")
    version = record.get("producer_version")
    if not isinstance(version, int) or isinstance(version, bool):
        problems.append(f"来源记录的版本号必须是整数，实际是 {version!r}")
    elif version not in SUPPORTED_VERSIONS:
        problems.append(
            f"来源记录的版本 {version} 不被支持（支持 {list(SUPPORTED_VERSIONS)}）；"
            "读端不猜另一个版本的字段含义"
        )
    fields = record.get("fields")
    if not isinstance(fields, (list, tuple)) or not fields or not all(
        isinstance(name, str) and name for name in fields
    ):
        problems.append(f"来源记录没有列出使用的输入字段：{fields!r}")
    identifier = record.get("input_identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        problems.append(
            f"来源记录没有输入标识（input_identifier）：{identifier!r}。"
            "输入不可访问时应当留空并说明，而不是编一个指纹"
        )
    return problems


def method_is_authoritative(method: Any) -> bool:
    return isinstance(method, str) and method in AUTHORITATIVE_METHODS


@dataclass
class PartitionSourceVerdict:
    """来源记录的判定结果。"""

    ok: bool
    method: str | None
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "method": self.method, "problems": list(self.problems)}


def evaluate_partition_source(
    record: Any, *, declared_volume_fraction: Any
) -> PartitionSourceVerdict:
    """结构检查 **加上** 与布尔声明的一致性。

    ``declared_volume_fraction`` 是提取器另写的那个布尔量。两者必须指同一件事：
    ``method`` 是体积极分而布尔量是假，说明这份导出自己都没说清是从哪来的。
    """
    problems = validate_partition_source(record)
    method = record.get("method") if isinstance(record, Mapping) else None
    if not problems or (isinstance(method, str) and method in PARTITION_METHODS):
        authoritative = method_is_authoritative(method)
        if declared_volume_fraction is not None:
            if bool(declared_volume_fraction) != authoritative:
                problems.append(
                    f"来源记录的方法 {method!r} 与布尔声明 "
                    f"material_partition_is_volume_fraction_based="
                    f"{declared_volume_fraction!r} 互相矛盾"
                )
    return PartitionSourceVerdict(ok=not problems, method=method, problems=problems)


__all__ = [
    "AUTHORITATIVE_METHODS",
    "METHOD_ALE_VOLUME_FRACTION",
    "METHOD_HEURISTIC_DENSITY",
    "PARTITION_METHODS",
    "REQUIRED_KEYS",
    "SUPPORTED_VERSIONS",
    "PartitionSourceVerdict",
    "evaluate_partition_source",
    "make_partition_source",
    "method_is_authoritative",
    "validate_partition_source",
]
