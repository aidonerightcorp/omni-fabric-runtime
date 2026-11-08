"""Hermes Agent unified adaptive plugin: omni-fabric-runtime.

Packages the complete local-to-federated agent fabric runtime behind 5 automatic
environment setup profiles:
1. stdlib_airgap        - Pure Python stdlib + SQLite WAL/FTS5 (0 pip packages, 0 network)
2. local_neural         - Unlocks NumPy/PyTorch multi-head micro-nets, 0-epoch SVD nullspace surgery, and Wanda trace channel pruning
3. local_runner_swarm   - Unlocks loopback /v1/chat/completions + /swarm/* distilling proxy and schema-constrained local open models
4. hybrid_fabric_sync   - Local SQLite WAL cell + selective sync of sync_ready objects & frontier tasks with upstream_endpoint
5. thin_remote_client   - Ephemeral zero-disk container bridge to upstream_endpoint (/q/* and /mcp)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .boil_down_and_swarm import (
    AdaptiveBoilDownEngine,
    FederatedFabricBridge,
    LoopbackSwarmProxyManager,
)
from .fabric_substrate import LocalAdaptiveFabricStore
from .grid_and_env import (
    SETUP_PROFILES,
    AdaptiveNodeGrid,
    EffortAndBudgetGate,
    EnvironmentCapabilityProbe,
)

_GRID_PATH = Path(__file__).resolve().parent / "config" / "adaptive_fabric_grid.json"


class OmniFabricPluginRuntime:
    """Lazy environment-adaptive plugin state scoped to ctx.state.data_dir."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self.grid = AdaptiveNodeGrid.from_json_path(_GRID_PATH)
        self.probe_engine = EnvironmentCapabilityProbe(self.grid)
        self.budget_gate = EffortAndBudgetGate(self.grid)
        self._store: Optional[LocalAdaptiveFabricStore] = None
        self._boil_engine: Optional[AdaptiveBoilDownEngine] = None
        self._proxy_mgr: Optional[LoopbackSwarmProxyManager] = None
        self._bridge: Optional[FederatedFabricBridge] = None
        self._override_profile: Optional[str] = None
        self._last_probe: Optional[Dict[str, Any]] = None
        self.uncheckpointed_leases: set[str] = set()

    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            val = self._ctx.get_config(key, default)
            return default if val is None else val
        except Exception:
            return default

    def data_dir(self) -> Optional[Path]:
        try:
            raw = getattr(getattr(self._ctx, "state", None), "data_dir", None)
            return Path(str(raw)) if raw else None
        except Exception:
            return None

    def active_regime(self) -> str:
        return str(self._cfg("regime", self.grid.default_regime))

    def upstream_endpoint(self) -> str:
        return str(self._cfg("upstream_endpoint", "")).strip()

    def local_runner_url(self) -> str:
        return str(self._cfg("local_runner_url", "http://127.0.0.1:11434")).strip()

    def resolve_environment(self, allow_socket_probe: bool = False, **sim_kwargs: Any) -> Dict[str, Any]:
        req_prof = self._override_profile or str(self._cfg("setup_profile", "auto"))
        info = self.probe_engine.probe(
            data_dir=self.data_dir(),
            requested_profile=req_prof,
            local_runner_url=self.local_runner_url(),
            upstream_endpoint=self.upstream_endpoint(),
            allow_socket_probe=allow_socket_probe,
            **sim_kwargs,
        )
        self._last_probe = info
        return info

    @property
    def store(self) -> LocalAdaptiveFabricStore:
        if self._store is None:
            env_info = self._last_probe or self.resolve_environment(allow_socket_probe=False)
            thin = env_info["selected_profile"] == "thin_remote_client"
            ddir = self.data_dir()
            db_file = (ddir / "omni_fabric_runtime.sqlite3") if (ddir is not None and not thin) else None
            self._store = LocalAdaptiveFabricStore(
                grid=self.grid,
                db_path=db_file,
                ephemeral_memory=thin or (db_file is None),
            )
        return self._store

    @property
    def boil_engine(self) -> AdaptiveBoilDownEngine:
        if self._boil_engine is None:
            self._boil_engine = AdaptiveBoilDownEngine(self.grid, self.store)
        return self._boil_engine

    @property
    def proxy_mgr(self) -> LoopbackSwarmProxyManager:
        if self._proxy_mgr is None:
            self._proxy_mgr = LoopbackSwarmProxyManager(
                grid=self.grid,
                store=self.store,
                boil_engine=self.boil_engine,
                budget_gate=self.budget_gate,
                local_runner_url=self.local_runner_url(),
            )
        return self._proxy_mgr

    @property
    def bridge(self) -> FederatedFabricBridge:
        if self._bridge is None:
            self._bridge = FederatedFabricBridge(self.grid, self.store, self.data_dir())
        return self._bridge


def register(ctx: Any) -> None:
    """Entry point invoked by Hermes PluginManager."""
    rt = OmniFabricPluginRuntime(ctx)

    # ── 1. Eight Unified Tools ────────────────────────────────────────────

    def _tool_probe_and_configure(args: Dict[str, Any], **_kw: Any) -> str:
        requested = args.get("setup_profile")
        if requested:
            rt._override_profile = str(requested)
            rt._store = None
            rt._boil_engine = None
            rt._proxy_mgr = None
            rt._bridge = None
        if args.get("budget_cap_micro_units") is not None:
            rt.budget_gate.configure_budget(int(args["budget_cap_micro_units"]))
        masked = tuple(args["masked_modules"]) if isinstance(args.get("masked_modules"), list) else None
        info = rt.resolve_environment(
            allow_socket_probe=bool(args.get("probe_sockets", False)),
            masked_modules=masked,
            force_readonly=bool(args.get("force_readonly", False)),
            simulated_runner_reachable=args.get("simulated_runner_reachable"),
            simulated_upstream_reachable=args.get("simulated_upstream_reachable"),
        )
        return json.dumps(info, sort_keys=True)

    ctx.register_tool(
        name="fabric_probe_and_configure",
        toolset="omni_fabric_runtime",
        description="Auto-detect local SQLite WAL/FTS5, CPU/RAM, NumPy/PyTorch, loopback runner, and optional upstream Fabric to activate one of 5 setup profiles.",
        schema={
            "name": "fabric_probe_and_configure",
            "description": "Probe host environment capabilities and select or override the active setup profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "setup_profile": {
                        "type": "string",
                        "enum": ["auto", *SETUP_PROFILES],
                    },
                    "probe_sockets": {"type": "boolean"},
                    "budget_cap_micro_units": {"type": "integer"},
                },
            },
        },
        handler=_tool_probe_and_configure,
    )

    def _tool_task_queue(args: Dict[str, Any], **_kw: Any) -> str:
        action = str(args.get("action") or "enqueue")
        regime = rt.active_regime()
        if action == "enqueue":
            res = rt.store.enqueue_task(
                task_id=str(args["task_id"]),
                namespace=str(args.get("namespace") or "default"),
                operation=str(args.get("operation") or "execute"),
                payload=dict(args.get("payload") or {}),
                contract_tag=str(args.get("contract_tag") or "v1"),
                depends_on=args.get("depends_on"),
                visibility=str(args.get("visibility") or rt._cfg("default_visibility", "local_only")),
            )
            return json.dumps(res, sort_keys=True)
        if action == "claim":
            res = rt.store.claim_task(
                worker_id=str(args["worker_id"]),
                namespace=args.get("namespace"),
                task_id=args.get("task_id"),
                regime=regime,
            )
            if res.get("ok"):
                rt.uncheckpointed_leases.add(str(res["task_id"]))
            return json.dumps(res, sort_keys=True)
        if action == "checkpoint":
            res = rt.store.checkpoint_task(
                task_id=str(args["task_id"]),
                worker_id=str(args["worker_id"]),
                lease_epoch=int(args["lease_epoch"]),
                fence_token=str(args["fence_token"]),
                checkpoint=dict(args.get("checkpoint") or {}),
                completed=bool(args.get("completed", False)),
                regime=regime,
            )
            if res.get("ok"):
                rt.uncheckpointed_leases.discard(str(args["task_id"]))
            return json.dumps(res, sort_keys=True)
        if action == "reclaim_stale":
            res = rt.store.reclaim_stale_leases(worker_id_filter=args.get("worker_id"))
            return json.dumps(res, sort_keys=True)
        return json.dumps({"ok": False, "error": f"unsupported_action:{action}"}, sort_keys=True)

    ctx.register_tool(
        name="fabric_task_queue",
        toolset="omni_fabric_runtime",
        description="Monotonic CAS lease_epoch + HMAC fence_token task queue with topological depends_on DAG and crash-safe resume packets.",
        schema={
            "name": "fabric_task_queue",
            "description": "Enqueue, claim, checkpoint, or reclaim fenced tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["enqueue", "claim", "checkpoint", "reclaim_stale"]},
                    "task_id": {"type": "string"},
                    "worker_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "operation": {"type": "string"},
                    "contract_tag": {"type": "string"},
                    "payload": {"type": "object"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "lease_epoch": {"type": "integer"},
                    "fence_token": {"type": "string"},
                    "checkpoint": {"type": "object"},
                    "completed": {"type": "boolean"},
                    "visibility": {"type": "string"},
                },
                "required": ["action"],
            },
        },
        handler=_tool_task_queue,
    )

    def _tool_object_store(args: Dict[str, Any], **_kw: Any) -> str:
        action = str(args.get("action") or "hybrid_search")
        if action in ("stage", "promote_review"):
            res = rt.store.stage_or_promote_object(
                task_id=str(args["task_id"]),
                maker_id=str(args["maker_id"]),
                checker_id=args.get("checker_id"),
                namespace=str(args.get("namespace") or "default"),
                task_key=args.get("task_key"),
                contract_tag=str(args.get("contract_tag") or "v1"),
                visibility=str(args.get("visibility") or rt._cfg("default_visibility", "local_only")),
                title=str(args.get("title") or args["task_id"]),
                summary_text=str(args.get("summary_text") or ""),
                payload=dict(args.get("payload") or {}),
                valid_from=args.get("valid_from"),
                valid_to=args.get("valid_to"),
            )
            return json.dumps(res, sort_keys=True)
        if action == "lookup_reuse":
            found = rt.store.lookup_reusable_object(
                task_key=str(args["task_key"]),
                contract_tag=str(args.get("contract_tag") or "v1"),
                as_of=args.get("as_of"),
            )
            return json.dumps({"ok": found is not None, "object": found}, sort_keys=True)
        res = rt.store.hybrid_search_objects(
            query=str(args.get("query") or ""),
            namespace=args.get("namespace"),
            promoted_only=bool(args.get("promoted_only", True)),
            unseen_only_for=args.get("unseen_only_for"),
            limit=args.get("limit"),
        )
        return json.dumps(res, sort_keys=True)

    ctx.register_tool(
        name="fabric_object_store",
        toolset="omni_fabric_runtime",
        description="Bitemporal maker!=checker Task Object store with 0-token reuse lookup, FTS5 RRF hybrid search, and unseen_only novelty filtering.",
        schema={
            "name": "fabric_object_store",
            "description": "Stage, independently promote, lookup 0-token reuse, or hybrid-search Task Objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["stage", "promote_review", "lookup_reuse", "hybrid_search"]},
                    "task_id": {"type": "string"},
                    "task_key": {"type": "string"},
                    "contract_tag": {"type": "string"},
                    "maker_id": {"type": "string"},
                    "checker_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "visibility": {"type": "string"},
                    "title": {"type": "string"},
                    "summary_text": {"type": "string"},
                    "payload": {"type": "object"},
                    "query": {"type": "string"},
                    "unseen_only_for": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["action"],
            },
        },
        handler=_tool_object_store,
    )

    def _tool_compile_workflow(args: Dict[str, Any], **_kw: Any) -> str:
        res = rt.store.compile_workflow_dag(
            workflow_id=str(args["workflow_id"]),
            goal=str(args["goal"]),
            target_task_count=int(args.get("target_task_count") or rt.grid.runtime_constants["macro_dag_phase_count"]),
            namespace=str(args.get("namespace") or "macro_workflow"),
            enqueue_immediately=bool(args.get("enqueue_immediately", True)),
        )
        return json.dumps(res, sort_keys=True)

    ctx.register_tool(
        name="fabric_compile_workflow_dag",
        toolset="omni_fabric_runtime",
        description="Compile macro objectives or occupational role workflows (3..250 tasks across 6 universal action phases) into dependency-ordered fenced tasks.",
        schema={
            "name": "fabric_compile_workflow_dag",
            "description": "Decompose a goal into a topological 6-phase fenced task DAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string"},
                    "goal": {"type": "string"},
                    "target_task_count": {"type": "integer"},
                    "namespace": {"type": "string"},
                    "enqueue_immediately": {"type": "boolean"},
                },
                "required": ["workflow_id", "goal"],
            },
        },
        handler=_tool_compile_workflow,
    )

    def _tool_grid_and_rl(args: Dict[str, Any], **_kw: Any) -> str:
        action = str(args.get("action") or "propose_batch")
        if action == "define_space":
            res = rt.store.define_optimization_space(
                campaign_id=str(args["campaign_id"]),
                axes=list(args.get("axes") or []),
                constraints=args.get("constraints"),
            )
            return json.dumps(res, sort_keys=True)
        if action == "propose_batch":
            res = rt.store.propose_optimization_batch(
                campaign_id=str(args["campaign_id"]),
                strategy=args.get("strategy"),
                batch_size=args.get("batch_size"),
            )
            return json.dumps(res, sort_keys=True)
        if action == "record_outcome":
            res = rt.store.record_optimization_outcome(
                campaign_id=str(args["campaign_id"]),
                params=dict(args.get("params") or {}),
                reward=float(args["reward"]),
                cost=float(args.get("cost") or 0.0),
                agent_id=str(args["agent_id"]),
                auto_expand=bool(args.get("auto_expand", True)),
            )
            return json.dumps(res, sort_keys=True)
        res = rt.store.canary_policy_step(
            policy_id=str(args.get("policy_id") or args.get("campaign_id") or "default_policy"),
            mode=str(args.get("mode") or "select"),
            baseline_arm=str(args.get("baseline_arm") or "reuse_then_verify"),
            canary_arm=str(args.get("canary_arm") or "boil_down_micro_net"),
            used_canary=bool(args.get("used_canary", False)),
            reward=float(args.get("reward") or 0.0),
            sample_index=int(args.get("sample_index") or 0),
        )
        return json.dumps(res, sort_keys=True)

    ctx.register_tool(
        name="fabric_grid_and_rl_optimize",
        toolset="omni_fabric_runtime",
        description="Parameterized NodeGrid optimizer (grid_search, latin_maximin, ucb_surrogate, successive_halving), boundary auto-expansion, and RL canary gate.",
        schema={
            "name": "fabric_grid_and_rl_optimize",
            "description": "Manage NodeGrid optimization spaces, propose batches, promote cells with maker!=checker quorum, and step RL canary policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["define_space", "propose_batch", "record_outcome", "canary_policy_step"]},
                    "campaign_id": {"type": "string"},
                    "axes": {"type": "array", "items": {"type": "object"}},
                    "constraints": {"type": "array", "items": {"type": "object"}},
                    "strategy": {"type": "string"},
                    "batch_size": {"type": "integer"},
                    "params": {"type": "object"},
                    "reward": {"type": "number"},
                    "cost": {"type": "number"},
                    "agent_id": {"type": "string"},
                    "policy_id": {"type": "string"},
                    "mode": {"type": "string"},
                    "used_canary": {"type": "boolean"},
                    "sample_index": {"type": "integer"},
                },
                "required": ["action"],
            },
        },
        handler=_tool_grid_and_rl,
    )

    def _tool_boil_down(args: Dict[str, Any], **_kw: Any) -> str:
        action = str(args.get("action") or "compile_stdlib")
        ns = str(args.get("namespace") or "default")
        if action == "record_trace":
            rt.store.record_execution_trace(
                namespace=ns,
                input_text=str(args["input_text"]),
                target_label=str(args["target_label"]),
                tokens_used=int(args.get("tokens_used") or 0),
                latency_ms=float(args.get("latency_ms") or 0.0),
            )
            return json.dumps({"ok": True, "namespace": ns}, sort_keys=True)
        if action == "predict":
            res = rt.boil_engine.predict_micro_router(ns, str(args["input_text"]))
            return json.dumps(res, sort_keys=True)
        if action in ("neural_prune_or_synthesize", "nullspace_surgery"):
            op = "nullspace_surgery" if action == "nullspace_surgery" else "wanda_prune"
            ddir = rt.data_dir()
            res = rt.boil_engine.run_neural_surgery_or_pruning(
                operation=op,
                namespace=ns,
                samples=args.get("samples"),
                anchor_vectors=args.get("anchor_vectors"),
                edit_vectors=args.get("edit_vectors"),
                edit_targets=args.get("edit_targets"),
                export_dir=(ddir / "models") if ddir else None,
                regime=rt.active_regime(),
            )
            return json.dumps(res, sort_keys=True)
        res = rt.boil_engine.compile_stdlib_micro_router(
            namespace=ns,
            samples=args.get("samples"),
            regime=rt.active_regime(),
        )
        return json.dumps(res, sort_keys=True)

    ctx.register_tool(
        name="fabric_boil_down_and_edit",
        toolset="omni_fabric_runtime",
        description="Compile traces into 0-token stdlib ReLU/ridge micro-routers, or run PyTorch Wanda channel pruning (-77% RAM) and 0-epoch SVD nullspace weight surgery.",
        schema={
            "name": "fabric_boil_down_and_edit",
            "description": "Progressive trace distillation, weight pruning, nullspace weight surgery, and 0-token inference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["record_trace", "compile_stdlib", "neural_prune_or_synthesize", "nullspace_surgery", "predict"],
                    },
                    "namespace": {"type": "string"},
                    "input_text": {"type": "string"},
                    "target_label": {"type": "string"},
                    "tokens_used": {"type": "integer"},
                    "samples": {"type": "array", "items": {"type": "object"}},
                    "anchor_vectors": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "edit_vectors": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "edit_targets": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                },
                "required": ["action"],
            },
        },
        handler=_tool_boil_down,
    )

    def _tool_swarm_proxy(args: Dict[str, Any], **_kw: Any) -> str:
        action = str(args.get("action") or "status")
        if action == "start":
            res = rt.proxy_mgr.start(
                host="127.0.0.1",
                port=int(args.get("port") or 0),
            )
            return json.dumps(res, sort_keys=True)
        if action == "stop":
            return json.dumps(rt.proxy_mgr.stop(), sort_keys=True)
        if action == "debit_budget":
            return json.dumps(rt.budget_gate.check_and_debit(int(args.get("cost_micro_units") or 0)), sort_keys=True)
        return json.dumps(rt.proxy_mgr.status(), sort_keys=True)

    ctx.register_tool(
        name="fabric_swarm_proxy_control",
        toolset="omni_fabric_runtime",
        description="Manage the loopback 127.0.0.1 OpenAI-compatible (/v1/chat/completions + /swarm/*) distilling swarm proxy and metered budget envelope.",
        schema={
            "name": "fabric_swarm_proxy_control",
            "description": "Start, inspect, or stop the loopback multi-agent swarm proxy or check spend budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop", "status", "debit_budget"]},
                    "port": {"type": "integer"},
                    "cost_micro_units": {"type": "integer"},
                },
                "required": ["action"],
            },
        },
        handler=_tool_swarm_proxy,
    )

    def _tool_federated_sync(args: Dict[str, Any], **_kw: Any) -> str:
        action = str(args.get("action") or "pulse")
        endpoint = str(args.get("upstream_endpoint") or rt.upstream_endpoint())
        if action == "pull_unseen":
            res = rt.bridge.pull_unseen_frontier(
                base_url=endpoint,
                worker_id=str(args.get("worker_id") or "hermes-local-cell"),
                query=str(args.get("query") or ""),
            )
            return json.dumps(res, sort_keys=True)
        if action == "push_promoted":
            return json.dumps(rt.bridge.push_sync_ready_objects(endpoint), sort_keys=True)
        if action == "remote_mcp":
            res = rt.bridge.remote_mcp_list_or_call(
                base_url=endpoint,
                tool_name=args.get("tool_name"),
                arguments=args.get("arguments"),
            )
            return json.dumps(res, sort_keys=True)
        return json.dumps(rt.bridge.check_upstream_pulse(endpoint), sort_keys=True)

    ctx.register_tool(
        name="fabric_federated_sync",
        toolset="omni_fabric_runtime",
        description="Selective bidirectional sync (pulse, pull_unseen, push_promoted, remote_mcp) with a configured upstream Fabric endpoint while keeping local_only data private.",
        schema={
            "name": "fabric_federated_sync",
            "description": "Check pulse, pull unseen frontier tasks, push sync_ready promoted objects, or invoke upstream MCP tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pulse", "pull_unseen", "push_promoted", "remote_mcp"]},
                    "upstream_endpoint": {"type": "string"},
                    "worker_id": {"type": "string"},
                    "query": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["action"],
            },
        },
        handler=_tool_federated_sync,
    )

    # ── 2. LLM Request Middleware ─────────────────────────────────────────

    def _llm_request_middleware(request: Dict[str, Any], **_kw: Any) -> Optional[Dict[str, Any]]:
        updated, diag = rt.budget_gate.calibrate_llm_request(request, regime=rt.active_regime())
        if updated != request:
            return {"request": updated, "source": "omni-fabric-runtime", "diagnostics": diag}
        return None

    ctx.register_middleware("llm_request", _llm_request_middleware)

    # ── 3. Ten Lifecycle, Subagent & Kanban Hooks (all **kwargs) ──────────

    def _on_session_start(**_kwargs: Any) -> None:
        rt.resolve_environment(allow_socket_probe=False)

    def _on_pre_llm_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
        msg = str(kwargs.get("user_message") or "").strip()
        cell = rt.grid.resolve_cell(rt.active_regime())
        if len(msg) <= int(cell["trivial_prompt_char_cap"]) or not rt.uncheckpointed_leases:
            return None
        cap = int(rt.grid.runtime_constants["pre_llm_context_char_cap"])
        active = ", ".join(sorted(rt.uncheckpointed_leases))
        prof = (rt._last_probe or {}).get("selected_profile", "auto")
        note = f"[omni-fabric:{prof}] Active leased task(s) awaiting fenced checkpoint: {active}"
        return {"context": note[:cap]}

    def _on_pre_verify(**_kwargs: Any) -> Optional[Dict[str, Any]]:
        if not bool(rt._cfg("enable_pre_verify_gate", True)) or not rt.uncheckpointed_leases:
            return None
        active = ", ".join(sorted(rt.uncheckpointed_leases))
        return {
            "action": "continue",
            "message": f"Leased task(s) ({active}) lack a fenced checkpoint. Call fabric_task_queue(action='checkpoint') before finishing.",
        }

    def _on_post_api_request(**kwargs: Any) -> None:
        usage = kwargs.get("usage")
        if isinstance(usage, dict):
            toks = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
            rt.budget_gate.check_and_debit(toks)

    def _on_post_tool_call(**kwargs: Any) -> None:
        tname = str(kwargs.get("tool_name") or "").strip()
        if tname and not tname.startswith("fabric_"):
            rt.store.record_execution_trace(
                namespace="tool_trace",
                input_text=str(kwargs.get("arguments") or tname),
                target_label=tname,
                tokens_used=0,
            )

    def _on_subagent_start(**kwargs: Any) -> None:
        wid = str(kwargs.get("subagent_id") or kwargs.get("agent_id") or "subagent").strip()
        role = str(kwargs.get("role") or "subagent")
        rt.store.update_subagent_worker(wid, role, "running")

    def _on_subagent_stop(**kwargs: Any) -> None:
        wid = str(kwargs.get("subagent_id") or kwargs.get("agent_id") or "subagent").strip()
        role = str(kwargs.get("role") or "subagent")
        rt.store.update_subagent_worker(wid, role, "stopped")

    def _on_kanban_claimed(**kwargs: Any) -> None:
        kid = str(kwargs.get("task_id") or "").strip()
        if kid:
            rt.store.enqueue_task(
                task_id=f"kanban:{kid}",
                namespace=str(kwargs.get("board") or "kanban"),
                operation="kanban_execute",
                payload={"assignee": kwargs.get("assignee")},
            )

    def _on_kanban_completed(**kwargs: Any) -> None:
        kid = str(kwargs.get("task_id") or "").strip()
        if kid:
            rt.uncheckpointed_leases.discard(f"kanban:{kid}")

    def _on_kanban_stale_claim(**kwargs: Any) -> None:
        wid = str(kwargs.get("worker_id") or kwargs.get("assignee") or "").strip() or None
        rt.store.reclaim_stale_leases(worker_id_filter=wid)

    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_verify", _on_pre_verify)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_hook("kanban_task_claimed", _on_kanban_claimed)
    ctx.register_hook("kanban_task_completed", _on_kanban_completed)
    ctx.register_hook("on_kanban_worker_stale_claim", _on_kanban_stale_claim)

    # ── 4. Slash Commands (/fabric-status, /fabric-mode) ──────────────────

    def _cmd_fabric_status(_raw: str = "") -> str:
        env = rt._last_probe or rt.resolve_environment(allow_socket_probe=False)
        st = rt.store.summarize_status()
        px = rt.proxy_mgr.status()
        return (
            f"omni-fabric-runtime profile={env['selected_profile']} "
            f"tasks={st['tasks_by_status']} promoted={st['promoted_objects']} "
            f"traces={st['execution_traces']} routers={len(rt.boil_engine.compiled_routers)} "
            f"proxy_running={px['running']}"
        )

    def _cmd_fabric_mode(raw: str = "") -> str:
        chosen = raw.strip() or "auto"
        if chosen not in ("auto", *SETUP_PROFILES):
            return f"Unknown profile '{chosen}'. Valid: auto, {', '.join(SETUP_PROFILES)}"
        rt._override_profile = chosen
        rt._store = None
        env = rt.resolve_environment(allow_socket_probe=False)
        return f"Switched setup_profile -> {env['selected_profile']} (requested={chosen})"

    ctx.register_command(
        name="fabric-status",
        handler=_cmd_fabric_status,
        description="Inspect detected host capabilities, active setup profile, fenced tasks, promoted objects, and boil-down routers.",
    )
    ctx.register_command(
        name="fabric-mode",
        handler=_cmd_fabric_mode,
        description="Switch active environment profile (auto, stdlib_airgap, local_neural, local_runner_swarm, hybrid_fabric_sync, thin_remote_client).",
    )
