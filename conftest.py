"""pytest 配置：把 ``src`` 放到路径上，并把默认 dtype 固定为 float64。

固定 dtype 不是洁癖：网络、EOS 与损失全部以 ``float64`` 为契约，而 PyTorch 的
默认是 ``float32``。测试里不设它，一个 ``torch.zeros`` 就可能悄悄把整条链路降
到单精度，然后在别处以一个 dtype 不匹配的形式失败——离真正的原因很远。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for entry in (str(SRC), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


@pytest.fixture(autouse=True)
def _float64_default():
    torch = pytest.importorskip("torch")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)
