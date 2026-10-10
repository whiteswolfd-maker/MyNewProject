"""Streaming parser for LS-DYNA keyword decks.

The decks in this study are ~200 MB each and are dominated by ``*NODE`` and
``*ELEMENT_SOLID`` blocks, so this parser **streams** the file and keeps only
what it needs.  It never builds a full in-memory token list.

Two format quirks are handled because the real Workbench-generated decks
contain them:

* ``*PART`` may carry a free-format part-name line *before* the numeric card
  (present in the 50 mm deck, absent in the others);
* ``*ELEMENT_SOLID`` may write each element as a **two-line** card
  (``eid pid`` then the eight node ids), which is what Workbench 2022 R1 emits
  for ``elform=11`` ALE multi-material solids.

Unit systems are **never** inferred from a deck.  ``*MAT_*`` cards contain bare
numbers; this parser returns them as written and the caller attaches units from
the case metadata (see :mod:`spherical_blast_pinn.data.units`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .schema import DataSchemaError, charge_mass_from_radius


class KeywordDeckError(DataSchemaError):
    """Raised when a keyword deck is missing something the pipeline needs."""


# --------------------------------------------------------------------------
# Low-level tokenisation
# --------------------------------------------------------------------------

_KEYWORD_RE = re.compile(r"^\*[A-Za-z0-9_\-]+")


def _is_keyword(line: str) -> bool:
    return bool(_KEYWORD_RE.match(line))


def _is_comment(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("$") or not stripped


#: LS-DYNA 定宽格式的字段宽度。
FIXED_FIELD_WIDTH = 10


def _fixed_width_fields(line: str) -> list[str]:
    """按 10 字符把一行切成字段，模拟 LS-DYNA 自己的定宽读取。"""
    padded = line.rstrip("\n").ljust(FIXED_FIELD_WIDTH)
    count = max(1, (len(padded) + FIXED_FIELD_WIDTH - 1) // FIXED_FIELD_WIDTH)
    return [padded[i * FIXED_FIELD_WIDTH:(i + 1) * FIXED_FIELD_WIDTH] for i in range(count)]


def _floats(line: str, minimum: int = 1) -> list[float] | None:
    """All leading numeric fields of ``line``, or ``None`` if fewer than ``minimum``.

    Returning *every* leading value rather than a fixed count is what lets an
    LS-DYNA card be read without knowing in advance how many fields the writer
    put on the line — Workbench splits the same card differently between
    releases.

    Two layouts have to be handled, and the deck does not announce which it uses:

    * **free format** — fields separated by whitespace or commas;
    * **fixed width** — 10-character fields, which run together with no
      separator whenever a value fills its field.  A ``*EOS_JWL`` line can then
      read ``1 3.712E+113.7471E+09`` and whitespace splitting silently yields
      one number instead of two.

    Whitespace is tried first because it is unambiguous when it works; the fixed
    field width is the fallback, and it is what LS-DYNA itself uses for these
    decks.
    """
    values = _parse_all(line.replace(",", " ").split())
    if values is not None and len(values) >= minimum:
        return values

    # 只要有一个空白字段读不出数字，就认为这一行是定宽的。判据必须是"**全部**
    # 字段都能读"，而不是"读出来的够多就行"：定宽行 `1 3.712E+113.7471E+09`
    # 在空白切分下会得到一个漂亮但错误的 [1.0]，只按数量判断就永远回退不了。
    values = _parse_all(_fixed_width_fields(line), skip_blank=True)
    if values is None or len(values) < minimum:
        return None
    return values


def _parse_all(fields: Sequence[str], *, skip_blank: bool = False) -> list[float] | None:
    """把一组字段全部读成浮点；任何一个读不出就返回 ``None``。"""
    values: list[float] = []
    for field in fields:
        text = field.strip()
        if not text:
            if skip_blank:
                continue
            return None
        try:
            values.append(float(text))
        except ValueError:
            return None
    return values


def _ints(line: str, minimum: int = 1) -> list[int] | None:
    floats = _floats(line, minimum=minimum)
    if floats is None:
        return None
    return [int(round(value)) for value in floats]


def iter_keyword_blocks(path: Path | str, wanted: Sequence[str] | None = None) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(keyword, data_lines)`` for each block in the deck.

    ``wanted`` restricts the output to a set of upper-case keyword names, which
    is what makes parsing a 200 MB deck cheap: the large node and element
    blocks are skipped without being materialised.
    """
    wanted_set = {name.upper() for name in wanted} if wanted else None
    keyword: str | None = None
    buffer: list[str] = []

    with open(path, "r", errors="replace", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if _is_keyword(line):
                if keyword is not None and (wanted_set is None or keyword in wanted_set):
                    yield keyword, buffer
                keyword = line.strip().upper()
                buffer = []
                continue
            if keyword is None or _is_comment(line):
                continue
            buffer.append(line)
        if keyword is not None and (wanted_set is None or keyword in wanted_set):
            yield keyword, buffer


# --------------------------------------------------------------------------
# Parsed containers
# --------------------------------------------------------------------------


@dataclass
class MaterialCard:
    """``*MAT_HIGH_EXPLOSIVE_BURN`` or ``*MAT_NULL``."""

    material_id: int
    kind: str
    density: float | None = None
    detonation_velocity: float | None = None
    p_cj: float | None = None
    nuclear_alpha_beta: float | None = None
    raw: dict[str, float] = field(default_factory=dict)


@dataclass
class EosCard:
    """``*EOS_JWL`` or ``*EOS_LINEAR_POLYNOMIAL``."""

    eos_id: int
    kind: str
    parameters: dict[str, float] = field(default_factory=dict)

    @property
    def volume_energy(self) -> float | None:
        return self.parameters.get("E0")

    @property
    def initial_relative_volume(self) -> float | None:
        return self.parameters.get("V0")


@dataclass
class PartCard:
    part_id: int
    section_id: int | None
    material_id: int | None
    eos_id: int | None
    name: str | None = None


@dataclass
class AleGroup:
    """`*ALE_MULTI-MATERIAL_GROUP` 的一条。

    ``psid`` 与 ``idtype`` 直接来自卡片，不加解释；解释在
    :meth:`DeckSummary.ale_material_part_ids` 里显式做，并把结论记下来。
    """

    psid: int
    idtype: int
    parts: tuple[int, ...] = ()
    resolved_as: str = "unresolved"


@dataclass
class DeckSummary:
    """Everything the pipeline extracts from one deck."""

    path: Path
    title: str | None = None
    materials: dict[int, MaterialCard] = field(default_factory=dict)
    eos: dict[int, EosCard] = field(default_factory=dict)
    parts: dict[int, PartCard] = field(default_factory=dict)
    sections: dict[int, int] = field(default_factory=dict)
    #: `*ALE_MULTI-MATERIAL_GROUP` 的条目，**按卡片出现顺序**。
    #:
    #: 顺序就是 ALE 材料号（1 基）：d3plot 的 history 体积分数列按这个顺序排。
    #: 拿 part id 当列号是巧合，不是约定——参 :meth:`ale_material_part_ids`。
    ale_groups: list[AleGroup] = field(default_factory=list)
    #: `*SET_PART` 的集合，用于 idtype 表示"part set"时解析。
    part_sets: dict[int, tuple[int, ...]] = field(default_factory=dict)
    initial_detonation: tuple[float, float, float] | None = None
    initial_detonation_time: float | None = None
    termination_time: float | None = None
    d3plot_interval: float | None = None
    control_ale: dict[str, float] = field(default_factory=dict)
    database_format: dict[str, float] = field(default_factory=dict)
    extent_binary: dict[str, float] = field(default_factory=dict)
    source_unit_system: str | None = None
    warnings: list[str] = field(default_factory=list)

    # -- helpers ----------------------------------------------------------
    def explosive_materials(self) -> dict[int, MaterialCard]:
        return {k: v for k, v in self.materials.items() if v.kind == "high_explosive_burn"}

    def air_materials(self) -> dict[int, MaterialCard]:
        return {k: v for k, v in self.materials.items() if v.kind == "null"}

    def explosive_material_ids(self) -> list[int]:
        return sorted(self.explosive_materials())

    def air_material_ids(self) -> list[int]:
        return sorted(self.air_materials())

    def require_explosive(self) -> MaterialCard:
        cards = self.explosive_materials()
        if not cards:
            raise KeywordDeckError(f"no *MAT_HIGH_EXPLOSIVE_BURN card found in {self.path}")
        return cards[sorted(cards)[0]]

    def require_jwl(self) -> EosCard:
        jwl = {k: v for k, v in self.eos.items() if v.kind == "jwl"}
        if not jwl:
            raise KeywordDeckError(f"no *EOS_JWL card found in {self.path}")
        return jwl[sorted(jwl)[0]]

    def material_names(self) -> dict[int, str]:
        return {pid: (part.name or "") for pid, part in self.parts.items()}

    # -- ALE 组 -> history 体积分数列 -------------------------------------

    def ale_material_part_ids(self) -> list[tuple[int, ...]]:
        """每个 ALE 材料号包含哪些 part，**按卡片顺序**。

        返回值第 ``k`` 项（0 基）对应 ALE 材料号 ``k + 1``，也就是 history 体积分数
        的第 ``k`` 列。这是材料映射的唯一来源：**不能用 ``part_id - 1`` 去猜列号**。
        四个算例上两者恰好一致（psid = 1..7、列也按 1..7 排），但那是因为卡片刚好
        这么写的，不是格式约定——换个从 0 编号或跳过号的 deck 就会静默读错列。

        ``idtype`` 的处理是**由卡片内容决定、并把结论记下来**，不是照抄手册极性：
        先按 ``idtype`` 指的类型找，找不到就试另一种；两种都不成立才报错。本机四个
        deck 是 ``psid = 1..7, idtype = 1``，且**没有任何 `*SET_PART`**，所以按 part
        set 解释会全部落空，只能按 part id 解释——这一点记在 ``resolved_as`` 里。
        """
        resolved: list[tuple[int, ...]] = []
        for index, group in enumerate(self.ale_groups, start=1):
            as_set = self.part_sets.get(group.psid)
            as_part = group.psid if group.psid in self.parts else None
            # idtype == 0 按手册是 part id；其余先试 part set，再退回 part id。
            prefer_set = group.idtype != 0
            if prefer_set and as_set:
                group.parts, group.resolved_as = as_set, "part set"
            elif as_part is not None:
                group.parts, group.resolved_as = (as_part,), "part id"
            elif as_set:
                group.parts, group.resolved_as = as_set, "part set"
            else:
                raise KeywordDeckError(
                    f"*ALE_MULTI-MATERIAL_GROUP entry {index} (psid={group.psid}, "
                    f"idtype={group.idtype}) resolves to neither a *SET_PART nor a *PART "
                    f"in {self.path}; the volume-fraction column order cannot be "
                    "established, and guessing it would silently mis-label materials"
                )
            resolved.append(group.parts)
        return resolved

    def ale_material_indices_for_parts(
        self, part_ids: Sequence[int]
    ) -> list[int]:
        """给定 part id，返回它们的 ALE 材料号（1 基）列表。"""
        wanted = set(int(p) for p in part_ids)
        out: list[int] = []
        for index, parts in enumerate(self.ale_material_part_ids(), start=1):
            if wanted.intersection(parts):
                out.append(index)
        return out

    def ale_material_kind(self, index: int) -> str:
        """ALE 材料号 ``index``（1 基）的介质类别：``product`` / ``air`` / ``mixed`` / ``unknown``。"""
        parts = self.ale_material_part_ids()[int(index) - 1]
        kinds = set()
        for pid in parts:
            part = self.parts.get(pid)
            material = self.materials.get(part.material_id) if part and part.material_id else None
            if material is None:
                kinds.add("unknown")
            elif material.kind == "high_explosive_burn":
                kinds.add("product")
            elif material.kind == "null":
                kinds.add("air")
            else:
                kinds.add(material.kind)
        if len(kinds) != 1:
            return "mixed"
        return kinds.pop()


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_MATERIAL_KEYS = ("ro", "d", "pcj", "beta", "k", "g", "sigy")
_JWL_KEYS = ("A", "B", "R1", "R2", "omeg", "E0", "V0")
_POLY_KEYS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")


def parse_deck(
    path: Path | str,
    *,
    geometry: bool = True,
    sample_nodes: int | None = None,
) -> DeckSummary:
    """Parse the constitutive, control and (optionally) geometric blocks."""
    target = Path(path)
    if not target.is_file():
        raise KeywordDeckError(f"keyword deck not found: {target}")

    summary = DeckSummary(path=target)
    wanted = [
        "*TITLE", "*MAT_HIGH_EXPLOSIVE_BURN", "*MAT_NULL", "*EOS_JWL",
        "*EOS_LINEAR_POLYNOMIAL", "*PART", "*SECTION_SOLID", "*INITIAL_DETONATION",
        "*CONTROL_TERMINATION", "*DATABASE_BINARY_D3PLOT", "*CONTROL_ALE",
        "*DATABASE_FORMAT", "*DATABASE_EXTENT_BINARY",
        "*ALE_MULTI-MATERIAL_GROUP", "*SET_PART",
    ]
    if geometry:
        wanted += ["*NODE", "*ELEMENT_SOLID"]

    for keyword, lines in iter_keyword_blocks(target, wanted):
        try:
            if keyword == "*TITLE":
                summary.title = lines[0].strip() if lines else None
            elif keyword == "*MAT_HIGH_EXPLOSIVE_BURN":
                _parse_material_block(lines, "high_explosive_burn", summary)
            elif keyword == "*MAT_NULL":
                _parse_material_block(lines, "null", summary)
            elif keyword == "*EOS_JWL":
                _parse_eos_block(lines, "jwl", _JWL_KEYS, summary)
            elif keyword == "*EOS_LINEAR_POLYNOMIAL":
                _parse_eos_block(lines, "linear_polynomial", _POLY_KEYS + ("E0", "V0"), summary)
            elif keyword == "*PART":
                _parse_part_block(lines, summary)
            elif keyword == "*ALE_MULTI-MATERIAL_GROUP":
                _parse_ale_group_block(lines, summary)
            elif keyword == "*SET_PART":
                _parse_set_part_block(lines, summary)
            elif keyword == "*SECTION_SOLID":
                _parse_section_block(lines, summary)
            elif keyword == "*INITIAL_DETONATION":
                _parse_initial_detonation(lines, summary)
            elif keyword == "*CONTROL_TERMINATION":
                values = _first_numeric_line(lines)
                if values is not None:
                    summary.termination_time = values[0]
            elif keyword == "*DATABASE_BINARY_D3PLOT":
                values = _first_numeric_line(lines)
                if values is not None:
                    summary.d3plot_interval = values[0]
            elif keyword == "*CONTROL_ALE":
                summary.control_ale = _named_first_line(lines)
            elif keyword == "*DATABASE_FORMAT":
                summary.database_format = _named_first_line(lines)
            elif keyword == "*DATABASE_EXTENT_BINARY":
                summary.extent_binary = _parse_extent_binary(lines)
            elif keyword in {"*NODE", "*ELEMENT_SOLID"}:
                # Handled by the dedicated streaming helpers below; skipped
                # here so the generic loop stays allocation-free.
                continue
        except (ValueError, IndexError) as exc:  # pragma: no cover - defensive
            summary.warnings.append(f"could not parse {keyword}: {exc}")

    if not summary.materials:
        summary.warnings.append("no material cards found")
    if not summary.eos:
        summary.warnings.append("no EOS cards found")
    return summary


def _first_numeric_line(lines: Sequence[str]) -> list[float] | None:
    for line in lines:
        values = _floats(line, 1)
        if values is not None:
            return values
    return None


_EXTENT_BINARY_FIELDS: tuple[str, ...] = (
    "neiph", "neips", "maxint", "strflg", "sigflg", "epsflg", "rltflg", "engflg",
    "cmpflg", "ieverp", "beamip", "dcomp", "shge", "stssz", "n3thdt", "ialemat",
    "nintsld", "pkp_sen", "sclp", "hydro", "msscl", "therm", "intout", "nodout",
)


def _parse_extent_binary(lines: Sequence[str]) -> dict[str, float]:
    """Name the ``*DATABASE_EXTENT_BINARY`` fields in their documented order.

    ``neiph`` matters: a zero value means no per-solid-element history
    variables were requested, which is how this study's decks are configured —
    the ALE volume fractions still appear, but no extra state is written.
    """
    flat: list[float] = []
    for line in lines:
        values = _floats(line, 1)
        if values:
            flat.extend(values)
    return {name: flat[i] for i, name in enumerate(_EXTENT_BINARY_FIELDS) if i < len(flat)}


def _named_first_line(lines: Sequence[str]) -> dict[str, float]:
    """Return the leading numeric values of a block keyed by column position."""
    values = _first_numeric_line(lines)
    if values is None:
        return {}
    return {f"field_{i}": v for i, v in enumerate(values)}


def _parse_material_block(lines: Sequence[str], kind: str, summary: DeckSummary) -> None:
    values = _first_numeric_line(lines)
    if values is None or len(values) < 2:
        return
    material_id = int(values[0])
    card = MaterialCard(material_id=material_id, kind=kind)
    if kind == "high_explosive_burn":
        # ID RO D PCJ BETA K G SIGY
        card.density = values[1]
        card.detonation_velocity = values[2] if len(values) > 2 else None
        card.p_cj = values[3] if len(values) > 3 else None
        for index, name in enumerate(("ro", "d", "pcj", "beta", "k", "g", "sigy"), start=1):
            if index < len(values):
                card.raw[name] = values[index]
    else:
        card.density = values[1]
        for index, name in enumerate(("ro", "pc", "mu", "terod", "cerod", "ym", "pr"), start=1):
            if index < len(values):
                card.raw[name] = values[index]
    summary.materials[material_id] = card


def _parse_eos_block(
    lines: Sequence[str], kind: str, keys: Sequence[str], summary: DeckSummary
) -> None:
    """Parse an EOS card.

    The card is ``ID param1 param2 ...`` and the writer may put all parameters
    on the header line (both decks in this study do), split them across lines,
    or both.  The ID is always the first value; every remaining value on every
    numeric line is consumed in order.
    """
    numeric_lines = []
    for line in lines:
        values = _floats(line, 1)
        if values:
            numeric_lines.append(values)
    if not numeric_lines:
        return

    eos_id = int(numeric_lines[0][0])
    parameters: dict[str, float] = {}
    for line_index, values in enumerate(numeric_lines):
        for value in values[1:] if line_index == 0 else values:
            if len(parameters) >= len(keys):
                break
            parameters[keys[len(parameters)]] = value
        if len(parameters) >= len(keys):
            break
    summary.eos[eos_id] = EosCard(eos_id=eos_id, kind=kind, parameters=parameters)


def _parse_ale_group_block(lines: Sequence[str], summary: DeckSummary) -> None:
    """Parse one ``*ALE_MULTI-MATERIAL_GROUP`` block.

    每条是 ``psid idtype``。**顺序即 ALE 材料号**，所以只按出现顺序追加，不排序、
    不去重——重排一次就会把体积分数的列号整体错开。
    """
    for line in lines:
        values = _ints(line, 2)
        if values is None:
            continue
        summary.ale_groups.append(AleGroup(psid=int(values[0]), idtype=int(values[1])))


def _parse_set_part_block(lines: Sequence[str], summary: DeckSummary) -> None:
    """Parse one ``*SET_PART`` block: ``sid`` then a list of part ids.

    本机四个 deck 都没有这个卡片，但 ALE 组的 ``idtype`` 允许指向 part set，
    所以必须能解析——否则遇到那种 deck 就只能靠猜。
    """
    sid: int | None = None
    members: list[int] = []
    for line in lines:
        values = _ints(line, 1)
        if values is None:
            continue
        if sid is None:
            sid = int(values[0])
            members.extend(int(v) for v in values[1:])
        else:
            members.extend(int(v) for v in values)
    if sid is not None:
        summary.part_sets[sid] = tuple(members)


def _parse_part_block(lines: Sequence[str], summary: DeckSummary) -> None:
    """Parse one ``*PART`` block.

    The block is ``*PART`` followed by one or more parts; each part nominally has
    an optional free-format name line then the ``ID secid mid eosid hgid grav
    adopt tmid`` card.  The name line is detected by *failing* to parse as the
    numeric card, which is what distinguishes the two deck variants.
    """
    pending_name: str | None = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        # A name line never starts with an integer ID followed by 2+ integers.
        numeric = _ints(line, 4)
        if numeric is None:
            pending_name = line.strip() or None
            continue
        part_id, section_id, material_id = numeric[0], numeric[1], numeric[2]
        eos_id = numeric[3] if len(numeric) > 3 else None
        summary.parts[part_id] = PartCard(
            part_id=part_id,
            section_id=section_id,
            material_id=material_id,
            eos_id=eos_id,
            name=pending_name,
        )
        pending_name = None


def _parse_section_block(lines: Sequence[str], summary: DeckSummary) -> None:
    values = _first_numeric_line(lines)
    if values is None or len(values) < 2:
        return
    summary.sections[int(values[0])] = int(values[1])


def _parse_initial_detonation(lines: Sequence[str], summary: DeckSummary) -> None:
    values = _first_numeric_line(lines)
    if values is None:
        return
    if len(values) >= 4 and not any(values[:3]):
        # pid 0 with coordinates 0 0 0 is the common "centre of the charge" form.
        summary.initial_detonation = (values[1], values[2], values[3])
        if len(values) > 4:
            summary.initial_detonation_time = values[4]
        return
    if len(values) >= 4:
        summary.initial_detonation = (values[1], values[2], values[3])
        if len(values) > 4:
            summary.initial_detonation_time = values[4]


# --------------------------------------------------------------------------
# Streaming geometry
# --------------------------------------------------------------------------


@dataclass
class GeometrySummary:
    """Minimal geometric facts needed to describe a case."""

    n_nodes: int = 0
    n_solid_elements: int = 0
    bounding_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    radius_min: float = float("inf")
    radius_max: float = 0.0
    total_volume: float = 0.0
    counts_by_part: dict[int, int] = field(default_factory=dict)
    radius_by_part: dict[int, tuple[float, float]] = field(default_factory=dict)
    volume_by_part: dict[int, float] = field(default_factory=dict)
    element_format: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_solid_elements": self.n_solid_elements,
            "bounding_box": self.bounding_box,
            "radius_min": self.radius_min,
            "radius_max": self.radius_max,
            "total_volume": self.total_volume,
            "counts_by_part": dict(self.counts_by_part),
            "radius_by_part": {k: list(v) for k, v in self.radius_by_part.items()},
            "volume_by_part": dict(self.volume_by_part),
            "element_format": self.element_format,
        }


def scan_geometry(path: Path | str, *, max_nodes: int | None = None) -> GeometrySummary:
    """Stream ``*NODE`` and ``*ELEMENT_SOLID`` and summarise the mesh.

    ``*ELEMENT_SOLID`` is accepted in both the one-line and the two-line card
    format; the format actually found is reported in
    :attr:`GeometrySummary.element_format` so a silent mis-parse is impossible.
    """
    target = Path(path)
    summary = GeometrySummary()
    node_radius: dict[int, float] = {}
    node_coords: dict[int, tuple[float, float, float]] = {}
    elements: list[tuple[int, int, list[int]]] = []
    format_counts = {"one_line": 0, "two_line": 0}

    keyword: str | None = None
    pending: list[int] | None = None

    with open(target, "r", errors="replace", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if _is_keyword(line):
                keyword = line.strip().upper()
                pending = None
                if keyword == "*END":
                    break
                if keyword not in {"*NODE", "*ELEMENT_SOLID"}:
                    keyword = None
                continue
            if keyword is None or _is_comment(line):
                continue

            if keyword == "*NODE":
                values = line.split()
                if len(values) >= 4:
                    try:
                        node_id = int(values[0])
                        x, y, z = (float(values[1]), float(values[2]), float(values[3]))
                    except ValueError:
                        continue
                    node_radius[node_id] = float(np.sqrt(x * x + y * y + z * z))
                    if max_nodes is None or len(node_coords) < max_nodes:
                        node_coords[node_id] = (x, y, z)
                    summary.n_nodes += 1
            elif keyword == "*ELEMENT_SOLID":
                values = line.split()
                if pending is not None:
                    # Second line of a two-line card: the eight node ids.
                    nodes = _ints(line, 8)
                    if nodes is None:
                        continue
                    eid, pid = pending
                    elements.append((eid, pid, nodes))
                    format_counts["two_line"] += 1
                    pending = None
                    continue
                if len(values) == 2:
                    # First line of a two-line card: `eid pid`.
                    pair = _ints(line, 2)
                    if pair is not None:
                        pending = pair
                        continue
                if len(values) >= 10:
                    # One-line card: `eid pid n1 .. n8`.
                    ten = _ints(line, 10)
                    if ten is None:
                        continue
                    elements.append((ten[0], ten[1], ten[2:10]))
                    format_counts["one_line"] += 1

    summary.n_solid_elements = len(elements)
    if node_coords:
        xs = [c[0] for c in node_coords.values()]
        ys = [c[1] for c in node_coords.values()]
        zs = [c[2] for c in node_coords.values()]
        summary.bounding_box = (
            (min(xs), max(xs)),
            (min(ys), max(ys)),
            (min(zs), max(zs)),
        )
    if node_radius:
        radii = np.fromiter(node_radius.values(), dtype=np.float64)
        summary.radius_min = float(radii.min())
        summary.radius_max = float(radii.max())

    hex8 = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    for _eid, pid, nodes in elements:
        summary.counts_by_part[pid] = summary.counts_by_part.get(pid, 0) + 1
        present = [node_radius[n] for n in nodes if n in node_radius]
        if present:
            low, high = min(present), max(present)
            current = summary.radius_by_part.get(pid)
            summary.radius_by_part[pid] = (
                min(low, current[0]) if current else low,
                max(high, current[1]) if current else high,
            )
        if len(nodes) == 8 and all(n in node_coords for n in nodes):
            points = np.array([node_coords[n] for n in nodes], dtype=np.float64)
            element_volume = _hex8_volume(points, hex8)
            summary.total_volume += element_volume
            summary.volume_by_part[pid] = summary.volume_by_part.get(pid, 0.0) + element_volume

    if format_counts["two_line"] and format_counts["one_line"]:
        summary.element_format = "mixed"
    elif format_counts["two_line"]:
        summary.element_format = "two_line"
    elif format_counts["one_line"]:
        summary.element_format = "one_line"
    del elements
    return summary


def _hex8_volume(points: np.ndarray, tets: Sequence[tuple[int, int, int, int]]) -> float:
    """Volume of a trilinear hex8 via a six-tetrahedron decomposition."""
    total = 0.0
    for i, j, k, l in tets:
        a, b, c, d = points[i], points[j], points[k], points[l]
        total += abs(float(np.dot(np.cross(b - a, c - a), d - a))) / 6.0
    return total


def charge_radius_from_geometry(
    geometry: GeometrySummary,
    explosive_part_ids: Sequence[int],
) -> float:
    """Infer the charge radius from the outermost node of the explosive parts."""
    radii = [geometry.radius_by_part[pid][1] for pid in explosive_part_ids if pid in geometry.radius_by_part]
    if not radii:
        raise KeywordDeckError(
            "cannot infer the charge radius: none of the explosive part ids "
            f"{list(explosive_part_ids)} appear in the element block"
        )
    return float(max(radii))


def explosive_part_ids(summary: DeckSummary) -> list[int]:
    """Part ids whose material is an explosive."""
    explosive = set(summary.explosive_materials())
    ids = [pid for pid, part in summary.parts.items() if part.material_id in explosive]
    if not ids:
        raise KeywordDeckError("no *PART references an explosive material")
    return sorted(ids)


def air_part_ids(summary: DeckSummary) -> list[int]:
    air = set(summary.air_materials())
    return sorted(pid for pid, part in summary.parts.items() if part.material_id in air)


def extract_jwl_parameters(summary: DeckSummary) -> dict[str, float]:
    """Return JWL parameters **as written in the deck** (deck units, not SI)."""
    materials = summary.explosive_materials()
    if not materials:
        raise KeywordDeckError(f"no explosive material in {summary.path}")
    material = materials[sorted(materials)[0]]
    jwl = summary.require_jwl()
    parameters = jwl.parameters
    missing = [key for key in ("A", "B", "R1", "R2", "omeg") if key not in parameters]
    if missing:
        raise KeywordDeckError(f"*EOS_JWL is missing {missing}")
    if material.density is None:
        raise KeywordDeckError("*MAT_HIGH_EXPLOSIVE_BURN is missing the density field")
    return {
        "A": parameters["A"],
        "B": parameters["B"],
        "R1": parameters["R1"],
        "R2": parameters["R2"],
        "omega": parameters["omeg"],
        "rho0": material.density,
        "E0": parameters.get("E0", 0.0),
        "V0": parameters.get("V0", 1.0),
        "detonation_velocity": material.detonation_velocity or 0.0,
        "p_cj": material.p_cj or 0.0,
    }


def extract_air_parameters(summary: DeckSummary) -> dict[str, Any]:
    """Return the air material + EOS as written in the deck."""
    materials = summary.air_materials()
    if not materials:
        raise KeywordDeckError(f"no *MAT_NULL (air) material in {summary.path}")
    material = materials[sorted(materials)[0]]
    polynomial = {k: v for k, v in summary.eos.items() if v.kind == "linear_polynomial"}
    result: dict[str, Any] = {"rho": material.density, "eos": None, "pressure": None}
    if polynomial:
        card = polynomial[sorted(polynomial)[0]]
        result["eos"] = card.kind
        result["coefficients"] = dict(card.parameters)
        volume_energy = card.parameters.get("E0")
        c4 = card.parameters.get("C4", 0.0)
        if volume_energy is not None and c4 > 0.0:
            # p0 = C4 * E0 for the ideal-gas-equivalent polynomial card.
            result["pressure"] = c4 * volume_energy
    return result


def summarise_case(
    deck_path: Path | str,
    *,
    unit_resolution,
    geometry: bool = True,
) -> dict[str, Any]:
    """Deck-derived case description in **SI**, ready for the audit report.

    ``unit_resolution`` is a :class:`spherical_blast_pinn.data.units.UnitResolution`.
    """
    summary = parse_deck(deck_path, geometry=False)
    jwl_deck = extract_jwl_parameters(summary)
    air_deck = extract_air_parameters(summary)

    jwl_si = {
        "A": jwl_deck["A"] * unit_resolution.factor("pressure"),
        "B": jwl_deck["B"] * unit_resolution.factor("pressure"),
        "R1": jwl_deck["R1"],
        "R2": jwl_deck["R2"],
        "omega": jwl_deck["omega"],
        "rho0": jwl_deck["rho0"] * unit_resolution.factor("density"),
        "E0": jwl_deck["E0"] * unit_resolution.factor("energy_per_volume"),
        "detonation_velocity": jwl_deck["detonation_velocity"] * unit_resolution.factor("velocity"),
        "p_cj": jwl_deck["p_cj"] * unit_resolution.factor("pressure"),
    }
    air_si = {
        "rho": (air_deck["rho"] or 0.0) * unit_resolution.factor("density"),
        "pressure": (air_deck["pressure"] or 0.0) * unit_resolution.factor("pressure"),
        "coefficients": air_deck.get("coefficients"),
    }

    result: dict[str, Any] = {
        "deck": str(Path(deck_path)),
        "jwl_si": jwl_si,
        "air_si": air_si,
        "termination_time_s": (
            summary.termination_time * unit_resolution.factor("time")
            if summary.termination_time is not None
            else None
        ),
        "d3plot_interval_s": (
            summary.d3plot_interval * unit_resolution.factor("time")
            if summary.d3plot_interval is not None
            else None
        ),
        "detonation_centre_m": (
            tuple(v * unit_resolution.factor("length") for v in summary.initial_detonation)
            if summary.initial_detonation
            else None
        ),
        "explosive_part_ids": explosive_part_ids(summary),
        "air_part_ids": air_part_ids(summary),
        "warnings": list(summary.warnings),
    }
    if geometry:
        geom = scan_geometry(deck_path)
        result["geometry"] = geom.as_dict()
        charge_radius = charge_radius_from_geometry(geom, result["explosive_part_ids"])
        result["charge_radius_m"] = charge_radius * unit_resolution.factor("length")
        if jwl_si["rho0"] > 0.0:
            result["charge_mass_kg"] = charge_mass_from_radius(
                result["charge_radius_m"], jwl_si["rho0"]
            )
    return result


__all__ = [
    "DeckSummary",
    "EosCard",
    "GeometrySummary",
    "KeywordDeckError",
    "MaterialCard",
    "PartCard",
    "air_part_ids",
    "charge_radius_from_geometry",
    "explosive_part_ids",
    "extract_air_parameters",
    "extract_jwl_parameters",
    "iter_keyword_blocks",
    "parse_deck",
    "scan_geometry",
    "summarise_case",
]
