"""Grid configuration, host environment capability probing, and effort/budget gating."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


SETUP_PROFILES: Tuple[str, ...] = (
    "stdlib_airgap",
    "local_neural",
    "local_runner_swarm",
    "hybrid_fabric_sync",
    "thin_remote_client",
)


class AdaptiveNodeGrid:
    """Multi-axis Configuration Settings Grid with cross-parameter constraints and clamping."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        self.grid_id: str = str(spec["grid_id"])
        self.generation: int = int(spec["generation"])
        self.default_regime: str = str(spec["default_regime"])
        self.axes: List[Dict[str, Any]] = [dict(a) for a in spec["axes"]]
        self._axis_map: Dict[str, Dict[str, Any]] = {str(a["name"]): dict(a) for a in self.axes}
        self.constraints: List[Dict[str, Any]] = [dict(c) for c in spec.get("constraints", [])]
        self.regimes: Dict[str, Dict[str, Any]] = {
            str(k): dict(v) for k, v in spec["regimes"].items()
        }
        self.runtime_constants: Dict[str, Any] = dict(spec["runtime_constants"])

    @classmethod
    def from_json_path(cls, path: Path) -> "AdaptiveNodeGrid":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw)

    def clamp_cell(self, candidate: Mapping[str, Any]) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for name, axis in self._axis_map.items():
            kind = str(axis["kind"])
            val = candidate.get(name)
            if kind == "integer_range":
                low = int(axis["low"])
                high = int(axis["high"])
                step = max(1, int(axis.get("step", 1)))
                num = int(val) if val is not None else low
                clamped = max(low, min(high, num))
                offset = ((clamped - low) // step) * step
                resolved[name] = low + offset
            elif kind == "choice":
                choices = list(axis["choices"])
                resolved[name] = val if val in choices else choices[0]
            elif kind == "boolean":
                resolved[name] = bool(axis.get("default", False)) if val is None else bool(val)
            else:
                resolved[name] = val
        for rule in self.constraints:
            left = str(rule["left"])
            right = str(rule["right"])
            op = str(rule["op"])
            if left in resolved and right in resolved:
                if op == "le" and int(resolved[left]) > int(resolved[right]):
                    resolved[left] = int(resolved[right])
        return resolved

    def resolve_cell(
        self,
        regime: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = str(regime or self.default_regime)
        base = dict(self.regimes.get(key) or self.regimes[self.default_regime])
        if overrides:
            for k, v in overrides.items():
                if v is not None and k in self._axis_map:
                    base[k] = v
        return self.clamp_cell(base)


class EnvironmentCapabilityProbe:
    """Detects local compute/storage/model capabilities and resolves optimal setup profile."""

    def __init__(self, grid: AdaptiveNodeGrid) -> None:
        self.grid = grid

    @staticmethod
    def _has_module(mod_name: str, masked_modules: Optional[Tuple[str, ...]] = None) -> bool:
        if masked_modules and mod_name in masked_modules:
            return False
        try:
            return importlib.util.find_spec(mod_name) is not None
        except Exception:
            return False

    @staticmethod
    def _check_sqlite_capabilities(data_dir: Optional[Path], force_readonly: bool = False) -> Dict[str, bool]:
        writable = False
        fts5_ok = False
        wal_ok = False
        if data_dir is not None and not force_readonly:
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                probe_file = data_dir / ".capability_probe.tmp"
                probe_file.write_text("ok", encoding="utf-8")
                probe_file.unlink(missing_ok=True)
                writable = True
            except Exception:
                writable = False
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(body)")
            conn.execute("DROP TABLE _fts_probe")
            fts5_ok = True
            wal_ok = writable
            conn.close()
        except Exception:
            fts5_ok = False
        return {
            "writable_data_dir": writable,
            "sqlite_wal_supported": wal_ok,
            "sqlite_fts5_supported": fts5_ok,
        }

    @staticmethod
    def _host_memory_kib() -> int:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                for line in meminfo.read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) > 1:
                            return int(parts[1])
            except Exception:
                return 0
        return 0

    def _probe_tcp_endpoint(self, raw_url: str) -> bool:
        cleaned = raw_url.strip()
        if not cleaned:
            return False
        try:
            from urllib.parse import urlparse

            parsed = urlparse(cleaned)
            host = parsed.hostname
            if not host:
                return False
            default_port = int(self.grid.runtime_constants["default_local_runner_port"])
            if parsed.scheme == "https":
                port = parsed.port or int(self.grid.runtime_constants["https_default_port"])
            else:
                port = parsed.port or default_port
            timeout_s = (
                float(self.grid.runtime_constants["loopback_probe_timeout_ms"])
                / float(self.grid.runtime_constants["ms_per_second"])
            )
            sock = socket.create_connection((host, int(port)), timeout=timeout_s)
            sock.close()
            return True
        except Exception:
            return False

    def probe(
        self,
        data_dir: Optional[Path],
        requested_profile: str = "auto",
        local_runner_url: str = "",
        upstream_endpoint: str = "",
        allow_socket_probe: bool = False,
        masked_modules: Optional[Tuple[str, ...]] = None,
        force_readonly: bool = False,
        simulated_runner_reachable: Optional[bool] = None,
        simulated_upstream_reachable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        sql_caps = self._check_sqlite_capabilities(data_dir, force_readonly=force_readonly)
        has_numpy = self._has_module("numpy", masked_modules)
        has_torch = self._has_module("torch", masked_modules)
        has_safetensors = self._has_module("safetensors", masked_modules)
        cpu_cores = int(os.cpu_count() or 1)
        mem_kib = self._host_memory_kib()

        if simulated_runner_reachable is not None:
            runner_ok = bool(simulated_runner_reachable)
        elif allow_socket_probe and local_runner_url.strip():
            runner_ok = self._probe_tcp_endpoint(local_runner_url)
        else:
            runner_ok = False

        upstream_configured = bool(upstream_endpoint.strip())
        if simulated_upstream_reachable is not None:
            upstream_ok = bool(simulated_upstream_reachable)
        elif allow_socket_probe and upstream_configured:
            upstream_ok = True
        else:
            upstream_ok = upstream_configured

        req = str(requested_profile or "auto").strip()
        if req in SETUP_PROFILES:
            selected = req
        else:
            if not sql_caps["writable_data_dir"] and upstream_ok:
                selected = "thin_remote_client"
            elif upstream_ok and sql_caps["writable_data_dir"]:
                selected = "hybrid_fabric_sync"
            elif runner_ok and sql_caps["writable_data_dir"]:
                selected = "local_runner_swarm"
            elif (has_numpy and has_torch) and sql_caps["writable_data_dir"]:
                selected = "local_neural"
            else:
                selected = "stdlib_airgap"

        active_features = self._feature_matrix_for_profile(
            selected=selected,
            sql_caps=sql_caps,
            has_numpy=has_numpy,
            has_torch=has_torch,
            has_safetensors=has_safetensors,
            runner_ok=runner_ok,
            upstream_ok=upstream_ok,
        )
        return {
            "requested_profile": req,
            "selected_profile": selected,
            "available_profiles": list(SETUP_PROFILES),
            "capabilities": {
                **sql_caps,
                "cpu_cores": cpu_cores,
                "available_ram_kib": mem_kib,
                "numpy_installed": has_numpy,
                "torch_installed": has_torch,
                "safetensors_installed": has_safetensors,
                "local_runner_reachable": runner_ok,
                "upstream_configured": upstream_configured,
                "upstream_reachable": upstream_ok,
            },
            "active_features": active_features,
        }

    @staticmethod
    def _feature_matrix_for_profile(
        selected: str,
        sql_caps: Mapping[str, bool],
        has_numpy: bool,
        has_torch: bool,
        has_safetensors: bool,
        runner_ok: bool,
        upstream_ok: bool,
    ) -> Dict[str, Any]:
        is_thin = selected == "thin_remote_client"
        neural_ready = (selected in ("local_neural", "local_runner_swarm", "hybrid_fabric_sync")) and has_numpy and has_torch
        proxy_ready = selected in ("local_runner_swarm", "hybrid_fabric_sync")
        sync_ready = selected in ("hybrid_fabric_sync", "thin_remote_client") and upstream_ok
        return {
            "storage_mode": "ephemeral_remote_bridge" if is_thin else ("sqlite_wal" if sql_caps["sqlite_wal_supported"] else "sqlite_memory"),
            "fenced_cas_task_queue": True,
            "multi_task_affinity_bundle_dispatch": True,
            "enriched_trace_envelopes": True,
            "attenuation_circuit_breaker": True,
            "maker_checker_object_store": True,
            "fts5_rrf_hybrid_search": bool(sql_caps["sqlite_fts5_supported"]) and not is_thin,
            "macro_and_action_lattice_dag": True,
            "parameterized_node_grid_and_rl_canary": True,
            "stdlib_relu_ridge_micro_compiler": not is_thin,
            "lora_xs_subspace_cartridges": not is_thin,
            "thinking_off_confound_gate": not is_thin,
            "pytorch_multihead_and_svd_nullspace_edit": neural_ready,
            "wanda_trace_channel_pruning": neural_ready,
            "safetensors_export": neural_ready and has_safetensors,
            "loopback_distilling_swarm_proxy": proxy_ready,
            "local_runner_schema_boildown": proxy_ready and runner_ok,
            "selective_upstream_fabric_sync": sync_ready,
        }


class EffortAndBudgetGate:
    """Zero-network <0.2 ms request effort calibrator and spend-envelope guard."""

    def __init__(self, grid: AdaptiveNodeGrid) -> None:
        self.grid = grid
        self.spent_micro_units: int = 0
        self.budget_cap_micro_units: int = int(grid.runtime_constants["default_budget_micro_units"])

    def configure_budget(self, cap_micro_units: int, reset_spent: bool = False) -> Dict[str, int]:
        self.budget_cap_micro_units = max(0, int(cap_micro_units))
        if reset_spent:
            self.spent_micro_units = 0
        return {
            "budget_cap_micro_units": self.budget_cap_micro_units,
            "spent_micro_units": self.spent_micro_units,
            "remaining_micro_units": max(0, self.budget_cap_micro_units - self.spent_micro_units),
        }

    def check_and_debit(self, cost_micro_units: int) -> Dict[str, Any]:
        debit = max(0, int(cost_micro_units))
        if self.spent_micro_units + debit > self.budget_cap_micro_units:
            return {
                "ok": False,
                "status_code": int(self.grid.runtime_constants["http_status_payment_required"]),
                "error": "budget_envelope_exhausted",
                "spent_micro_units": self.spent_micro_units,
                "budget_cap_micro_units": self.budget_cap_micro_units,
            }
        self.spent_micro_units += debit
        return {
            "ok": True,
            "status_code": int(self.grid.runtime_constants["http_status_ok"]),
            "spent_micro_units": self.spent_micro_units,
            "remaining_micro_units": self.budget_cap_micro_units - self.spent_micro_units,
        }

    def calibrate_llm_request(
        self,
        request: Mapping[str, Any],
        regime: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        updated = dict(request)
        cell = self.grid.resolve_cell(regime)
        messages = updated.get("messages") or []
        last_user = ""
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, Mapping) and str(msg.get("role") or "") == "user":
                    content = msg.get("content")
                    last_user = str(content or "") if not isinstance(content, list) else json.dumps(content)
                    break

        char_len = len(last_user.strip())
        trivial_cap = int(cell["trivial_prompt_char_cap"])
        med_cap = int(self.grid.runtime_constants["medium_prompt_char_threshold"])
        complex_cap = int(self.grid.runtime_constants["complex_prompt_char_threshold"])
        json_floor = int(cell["structured_output_token_floor"])

        resp_fmt = updated.get("response_format")
        wants_json = isinstance(resp_fmt, Mapping) and str(resp_fmt.get("type") or "") in (
            "json_object",
            "json_schema",
        )

        if not wants_json and char_len <= trivial_cap:
            effort = "minimal"
        elif char_len <= med_cap and not wants_json:
            effort = "low"
        elif char_len >= complex_cap or wants_json:
            effort = "high" if char_len >= complex_cap else "medium"
        else:
            effort = "medium"

        if "reasoning_effort" not in updated:
            updated["reasoning_effort"] = effort

        current_max = updated.get("max_tokens")
        if wants_json and (current_max is None or int(current_max) < json_floor):
            updated["max_tokens"] = json_floor

        return updated, {
            "prompt_chars": char_len,
            "wants_structured_json": wants_json,
            "selected_effort": updated.get("reasoning_effort"),
            "max_tokens": updated.get("max_tokens"),
        }
