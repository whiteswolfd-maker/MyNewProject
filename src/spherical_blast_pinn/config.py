"""Configuration loading, validation and snapshotting.

The configuration is a nested mapping loaded from YAML.  Experiment files may
declare ``inherits: base.yaml`` and are deep-merged on top of the inherited
document, which keeps the ablation configs in ``configs/experiments/`` short
and makes the difference between two experiments reviewable.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

_MISSING = object()

PLACEHOLDER_MARKERS: tuple[str, ...] = ("EDIT_ME", "CHANGEME", "<", "TODO_CONFIG")


class ConfigError(Exception):
    """Raised when the configuration is missing, malformed or unconfirmed."""


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def find_repo_root(start: Path | str | None = None) -> Path:
    """Walk upwards from ``start`` looking for ``pyproject.toml``.

    Falls back to the directory that contains ``src/spherical_blast_pinn`` and
    finally to ``start`` itself, so the function is total.
    """
    here = Path(start or __file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    for candidate in (here, *here.parents):
        if (candidate / "src" / "spherical_blast_pinn").is_dir():
            return candidate
    return here


# --------------------------------------------------------------------------
# Deep merge
# --------------------------------------------------------------------------


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` onto a copy of ``base``.

    Lists are replaced wholesale, not concatenated: an experiment that wants a
    different ``stages`` schedule must be able to shorten it.  Mappings are
    merged key by key.
    """
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = {k: copy.deepcopy(v) for k, v in base.items()}
        for key, value in override.items():
            merged[key] = deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _coerce_scalar(text: str) -> Any:
    """Parse ``--set`` values with YAML semantics (bools, nulls, numbers)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _get_dotted(source: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    node: Any = source
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is _MISSING:
                raise ConfigError(f"missing required configuration key: {dotted!r}")
            return default
        node = node[part]
    return node


# --------------------------------------------------------------------------
# Config object
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Immutable-by-convention wrapper around a resolved configuration mapping."""

    values: dict[str, Any]
    path: Path
    root: Path
    overrides: dict[str, Any] = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        path: Path | str | None = None,
        root: Path | str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "Config":
        path = Path(path) if path is not None else Path.cwd() / "config.yaml"
        root = Path(root) if root is not None else find_repo_root(path.parent)
        return cls(copy.deepcopy(dict(values)), path, root, dict(overrides or {}))

    # -- access -----------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        return _get_dotted(self.values, dotted, default)

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING or value is None:
            raise ConfigError(f"configuration key {dotted!r} is required but missing/null")
        return value

    def set(self, dotted: str, value: Any) -> "Config":
        """Return a modified copy (the original is untouched)."""
        clone = self.copy()
        _set_dotted(clone.values, dotted, value)
        _set_dotted(clone.overrides, dotted, value)
        return clone

    def copy(self) -> "Config":
        return Config(
            copy.deepcopy(self.values),
            self.path,
            self.root,
            copy.deepcopy(self.overrides),
        )

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration section {name!r} is not a mapping")
        return dict(value)

    def with_defaults(self, defaults: Mapping[str, Any]) -> "Config":
        """Return a copy where ``defaults`` fill in anything not already set."""
        return Config(
            deep_merge(defaults, self.values),
            self.path,
            self.root,
            dict(self.overrides),
        )

    # -- paths ------------------------------------------------------------
    def path_of(self, dotted: str, default: Any = _MISSING) -> Path | None:
        """Resolve a configured path relative to the repository root."""
        raw = self.get(dotted, default)
        if raw is None:
            return None
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate

    def output_dir(self) -> Path:
        return self.path_of("training.output_dir", "outputs/default") or (
            self.root / "outputs" / "default"
        )

    # -- introspection ----------------------------------------------------
    def iter_leaves(self) -> Iterator[tuple[str, Any]]:
        def walk(node: Any, prefix: str) -> Iterator[tuple[str, Any]]:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    yield from walk(value, f"{prefix}.{key}" if prefix else str(key))
            else:
                yield prefix, node

        yield from walk(self.values, "")

    def placeholders(self) -> list[str]:
        """Return dotted keys whose value still looks like an unfilled template."""
        found: list[str] = []
        for key, value in self.iter_leaves():
            if isinstance(value, str) and any(m in value.upper() for m in PLACEHOLDER_MARKERS):
                found.append(key)
        return found

    def unconfirmed_fallbacks(self, required: Iterable[str] | None = None) -> list[str]:
        """Keys listed in ``confirmed_values``-style metadata that are still false.

        The detector is deliberately generic: any boolean key ending in
        ``verified_against_keyword`` or ``confirmed`` that is ``False`` is
        reported, together with the physical keys it governs.
        """
        unconfirmed: list[str] = []
        for key, value in self.iter_leaves():
            leaf = key.rsplit(".", 1)[-1]
            if isinstance(value, bool) and not value and (
                "verified" in leaf or "confirmed" in leaf
            ):
                unconfirmed.append(key)
        return sorted(unconfirmed)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.values, sort_keys=True, allow_unicode=True, default_flow_style=False)

    def sha256(self) -> str:
        payload = json.dumps(_jsonable(self.values), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml(), encoding="utf-8")
        return target

    # -- numeric helpers --------------------------------------------------
    def float_of(self, dotted: str, default: Any = _MISSING) -> float:
        value = self.get(dotted, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"configuration key {dotted!r} must be numeric, got {value!r}") from exc

    def int_of(self, dotted: str, default: Any = _MISSING) -> int:
        return int(self.float_of(dotted, default))

    def bool_of(self, dotted: str, default: Any = _MISSING) -> bool:
        value = self.get(dotted, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def list_of(self, dotted: str, default: Any = _MISSING) -> list[Any]:
        value = self.get(dotted, default)
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ConfigError(f"configuration key {dotted!r} must be a list, got {type(value).__name__}")


def _jsonable(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {str(k): _jsonable(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(v) for v in node]
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    return repr(node)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_yaml(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"configuration file not found: {target}")
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"configuration file {target} must contain a mapping at the top level")
    return dict(loaded)


def _resolve_inherits(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"circular `inherits` chain detected at {path}")
    document = load_yaml(path)
    parent = document.pop("inherits", None)
    if parent is None:
        return document
    if isinstance(parent, (list, tuple)):
        merged: dict[str, Any] = {}
        for entry in parent:
            child_path = (path.parent / str(entry)).resolve()
            merged = deep_merge(merged, _resolve_inherits(child_path, seen + (resolved,)))
        return deep_merge(merged, document)
    parent_path = (path.parent / str(parent)).resolve()
    base = _resolve_inherits(parent_path, seen + (resolved,))
    return deep_merge(base, document)


def load_config(
    path: Path | str,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    *,
    root: Path | str | None = None,
) -> Config:
    """Load a YAML configuration, applying ``inherits`` and then overrides.

    ``overrides`` may be a mapping or a sequence of ``"dotted.key=value"``
    strings (as produced by ``--set`` on the command line).
    """
    target = Path(path)
    values = _resolve_inherits(target)
    applied: dict[str, Any] = {}

    if overrides:
        items: list[tuple[str, Any]] = []
        if isinstance(overrides, Mapping):
            items = list(overrides.items())
        else:
            for entry in overrides:
                if "=" not in entry:
                    raise ConfigError(f"--set override {entry!r} must look like key.subkey=value")
                key, raw = entry.split("=", 1)
                items.append((key.strip(), _coerce_scalar(raw.strip())))
        for key, value in items:
            _set_dotted(values, key, value)
            _set_dotted(applied, key, value)

    resolved_root = Path(root) if root is not None else find_repo_root(target.parent)
    return Config(values, target, resolved_root, applied)


# --------------------------------------------------------------------------
# Small validation helpers used across the package
# --------------------------------------------------------------------------


def require_positive(config: Config, dotted: str) -> float:
    value = config.float_of(dotted)
    if not math.isfinite(value) or value <= 0.0:
        raise ConfigError(f"configuration key {dotted!r} must be finite and positive, got {value}")
    return value


def require_fraction(config: Config, dotted: str) -> float:
    value = config.float_of(dotted)
    if not (0.0 <= value <= 1.0):
        raise ConfigError(f"configuration key {dotted!r} must lie in [0, 1], got {value}")
    return value


__all__ = [
    "Config",
    "ConfigError",
    "deep_merge",
    "find_repo_root",
    "load_config",
    "load_yaml",
    "require_fraction",
    "require_positive",
]
