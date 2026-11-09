# omni-fabric-runtime (v0.4.0)

Environment-adaptive runtime plugin for Hermes Agent (`>=0.20.0`, Manifest v2) and OpenAI-v1 agent swarms (`GrokBot`, `OpenCode`, subagent pools). Unites durable task fencing, multi-task affinity bundle dispatch, enriched trace telemetry, distinct `maker != checker` object reuse, hybrid `FTS5` search, a unified **6-stage weight-space compiler across 6 model classes**, an **`Actor × {Transfer, Transform, Save} × Horizon` check-gated DAG decomposer**, a **leak-free holdout recursive RL trial engine**, a loopback OpenAI-v1 swarm proxy, and optional federated Fabric sync in one zero-dependency package.

## Automatic Host Capability Detection & 5 Setup Profiles

At session start (or via `fabric_probe_and_configure` / `/fabric-mode`), the plugin inspects what can run locally inside the current environment and selects one of five operational setups:

| Profile | Local Requirements | Active Local & Federated Capabilities |
|---|---|---|
| `stdlib_airgap` | Python 3 stdlib + SQLite WAL/FTS5 (`0` pip packages, `0` network) | Monotonic `CAS lease_epoch` queue, `1..4` affinity bundle dispatch, enriched `trace_envelopes`, `maker != checker` object cache, `FTS5` BM25 + RRF search, 6-stage weight-space compiler across 6 model classes, `Actor × {Transfer, Transform, Save} × Horizon` DAG decomposer, leak-free recursive RL trial engine, and `256B` `LoRA-XS` cartridges |
| `local_neural` | `+ numpy` & `torch` (`+ safetensors` optional), `0` network | Everything in `stdlib_airgap` plus multi-head `.safetensors` synthesis, NumPy SVD nullspace weight surgery (`0.0` anchor drift), and Wanda trace channel pruning |
| `local_runner_swarm` | `+ loopback open-model runner` (`127.0.0.1:11434`) | Everything above plus on-demand loopback `/v1/models`, `/v1/chat/completions`, `/swarm/claim-bundle`, and `/swarm/trace` proxy for `Hermes` subagents and `GrokBot` workers |
| `hybrid_fabric_sync` | Local SQLite WAL cell `+ upstream_endpoint` | Runs private `local_only` work, subagent swarms, and weight compilation locally while syncing `unseen_only` frontier tasks, `sync_ready` promoted objects, and `/mcp` calls with a remote Fabric endpoint |
| `thin_remote_client` | Ephemeral / read-only container `+ upstream_endpoint` | Zero disk footprint (`:memory:` buffer); bridges queue, object, optimization, and MCP operations directly to `upstream_endpoint` |

## What Is New in v0.4.0

1. **Unified 6-Stage Weight-Space Pipeline Across 6 Model Classes (`fabric_compile_weight_unit`):**
   - Supports `embedding`, `search_retrieval`, `decision_router`, `lora_xs_cartridge`, `memory_as_lora`, and `steered_llm`.
   - Executes all six stages end-to-end on historical SQLite WAL traces or `.safetensors` cartridges:
     1) **Symbolic Invariant Induction & Exact ReLU Polytope Skeleton (`W_sym`)** (`0` gradient epochs, `0.0` rule violations)
     2) **Historical-Trace Activation Channel Pruning + Closed-Form Ridge Recalibration** (up to `-95.3%` embedding dimension memory reduction and `-72.0%` search posting prune)
     3) **Singular-Subspace `LoRA-XS` (`256B` `r x r`) Fine-Tuning** (hot-swappable `float32` `.bin` and spec-compliant `.safetensors` export)
     4) **Memory-as-LoRA Episodic Trace-to-Weight Slot Compilation** (Gram-Schmidt orthogonal episodic slots eliminating `8,192+` repeated multi-turn context tokens per slot)
     5) **SVD Anchor-Nullspace Projection (`Null(W_sym)`)** (`0.0` drift on prior invariants and anchors)
     6) **Matched Direct-Baseline (`thinking=off`) Gate + Disjoint `maker_lineage != checker_lineage` Canary Promotion & Instant Auto-Rollback**
2. **Universal `Actor × {Transfer, Transform, Save} × Horizon` Auditable DAG Decomposer (`fabric_decompose_horizon`):**
   - Catalogs all `27` 3-step `Transfer × Transform × Save` information triads and decomposes long-horizon `person`, `machine`, and `swarm` workflows (`1..96` steps) across general and competitive data/code archetypes (`bitemporal_tabular_oof`, `symbolic_grid_synthesis`, `combinatorial_packing`, `repository_patch_repair`).
   - Inserts disjoint `maker_lineage != checker_lineage` bitemporal `save` checkpoints that reset cumulative step attenuation back to `1.0`.
3. **Leak-Free Holdout Recursive RL Trial Engine (`fabric_recursive_trial`):**
   - Combines UCB exploration + softmax policy-gradient updates over `(NodeGrid cell, weight cartridge)` pairs while enforcing disjoint maker/checker lineages, sealed holdout generalization-gap bounds, and matched `thinking=off` baseline controls so recursive self-improvement never overfits.

## Public Extension Surfaces

- **12 Tools:** `fabric_probe_and_configure`, `fabric_task_queue`, `fabric_claim_bundle`, `fabric_object_store`, `fabric_compile_workflow_dag`, `fabric_grid_and_rl_optimize`, `fabric_boil_down_and_edit`, `fabric_swarm_proxy_control`, `fabric_federated_sync`, `fabric_compile_weight_unit`, `fabric_decompose_horizon`, `fabric_recursive_trial`
- **10 Lifecycle, Subagent & Kanban Hooks:** `on_session_start`, `pre_llm_call`, `pre_verify`, `post_api_request`, `post_tool_call`, `subagent_start`, `subagent_stop`, `kanban_task_claimed`, `kanban_task_completed`, `on_kanban_worker_stale_claim`
- **1 Request Middleware:** `llm_request` (`<0.2 ms` reasoning-effort calibration + structured JSON `max_tokens` floor)
- **2 Slash Commands:** `/fabric-status`, `/fabric-mode`

## Hermes Agent & GrokBot Quick-Check Commands

```bash
# 1. Run Hermes Plugin Doctor CI check (0 required pip deps, 0 network calls on load)
hermes plugins doctor --ci ./omni-fabric-runtime

# 2. Install into Hermes Agent plugins directory
mkdir -p ~/.hermes/plugins && cp -R ./omni-fabric-runtime ~/.hermes/plugins/omni-fabric-runtime

# 3. Point GrokBot or any OpenAI-v1 client at the loopback swarm proxy started via fabric_swarm_proxy_control
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
export OPENAI_API_KEY="local"
curl -s "$OPENAI_BASE_URL/models"
curl -s -X POST "http://127.0.0.1:8787/swarm/claim-bundle" \
  -H "Content-Type: application/json" \
  -d '{"worker_id":"grokbot-1","bundle_size":3,"dispatch_mode":"back_to_back_sequential"}'
```

## Privacy & Network Disclosure

- Default `upstream_endpoint` is `""` (local-only): `0` external network requests, `0` telemetry, and `local_only` default visibility for staged Task Objects.
- Remote synchronization (`fabric_federated_sync`) runs only when an operator explicitly configures `upstream_endpoint`.
