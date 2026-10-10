"""复现用户报告的三处漏洞（临时脚本，验证后删除）。"""
import numpy as np

from spherical_blast_pinn.cli._common import (
    ReferenceNotAccepted,
    _field_content_hash,
    assert_reference_accepted,
    record_acceptance,
)
from spherical_blast_pinn.physics.precursor import (
    ACCEPTANCE_TOLERANCES,
    EVIDENCE_DEPENDENT_CHECKS,
    PHYSICS_BASIS_VERIFIED,
)


class Field:
    def __init__(self):
        self.times = np.array([0.0, 1e-6])
        self.cell_radii = np.array([[0.0, 0.5], [0.0, 0.6]])
        self.density = np.array([[100.0, 1.0], [101.0, 1.0]])
        self.velocity = np.array([[10.0, 0.0], [11.0, 0.0]])
        self.specific_energy = np.array([[1.0e5, 2.0e5], [1.0e5, 2.0e5]])
        self.material = np.array([0, 1])
        self.contact_radius = np.array([0.4, 0.5])
        self.shock_radius = np.array([0.6, 0.7])
        self.reference_radius = 0.05
        self.metadata = {}

    def accept_metadata(self):
        self.metadata.update({
            "acceptance_metrics": {k: 0.0 for k in ACCEPTANCE_TOLERANCES},
            "acceptance_evidence": {n: True for n in EVIDENCE_DEPENDENT_CHECKS},
            "physics_basis_status": PHYSICS_BASIS_VERIFIED,
            "data_contract_status": "ready",
        })


print("=" * 70)
print("HOLE 1: record_acceptance() 把旧指标重新绑定到新场")
print("=" * 70)
f = Field()
f.accept_metadata()
f.metadata["acceptance_field_hash"] = _field_content_hash(f)  # 旧记录绑定旧场
old = f.metadata["acceptance_field_hash"]
# 现在把场改坏：密度出现负值（真实违反 min_density > 0）
f.density = f.density.copy()
f.density[0, 0] = -5.0
new = _field_content_hash(f)
print(f"  旧场哈希 {old[:16]}...  新场哈希 {new[:16]}...  相同={old == new}")
try:
    assert_reference_accepted(f)
    print("  assert_reference_accepted: 通过（预期拒绝）")
except ReferenceNotAccepted as exc:
    print(f"  assert_reference_accepted: 拒绝 -> {str(exc)[:40]}")
verdict = record_acceptance(f, {n: True for n in EVIDENCE_DEPENDENT_CHECKS})
print(f"  record_acceptance 判定 = {verdict['acceptance_status']}")
print(f"  写回后 acceptance_field_hash = {f.metadata['acceptance_field_hash'][:16]}...")
f.metadata["field_content_hash"] = _field_content_hash(f)
try:
    assert_reference_accepted(f)
    print("  >>> 闸门通过：被改坏的场现在带着'验收通过'。漏洞成立。")
except ReferenceNotAccepted as exc:
    print(f"  闸门拒绝 -> {str(exc)[:60]}")

print()
print("=" * 70)
print("HOLE 2: 元数据缺 field_content_hash / acceptance_field_hash 时直接放行")
print("=" * 70)
g = Field()
g.accept_metadata()  # 有指标、有证据，但一个哈希都没有
try:
    assert_reference_accepted(g)
    print("  >>> 放行：没有内容哈希的元数据照样通过验收闸门。漏洞成立。")
except ReferenceNotAccepted as exc:
    print(f"  拒绝 -> {str(exc)[:60]}")

print()
print("=" * 70)
print("HOLE 3: 形状变化不影响内容哈希")
print("=" * 70)
h = Field()
h1 = _field_content_hash(h)
h.times = h.times.reshape(1, 2)          # 只改形状，字节不变
h2 = _field_content_hash(h)
h.density = h.density.reshape(4)          # 2x2 -> 4
h3 = _field_content_hash(h)
print(f"  times (2,) -> (1,2):     哈希相同 = {h1 == h2}")
print(f"  density (2,2) -> (4,):   哈希相同 = {h2 == h3}")
print(f"  >>> 形状不是内容的一部分，(T,N) 与 (N*T,) 不可区分。漏洞成立 = {h1 == h3}")
