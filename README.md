# omni-fabric-runtime (v0.3.0)

Environment-adaptive runtime plugin for Hermes Agent (`>=0.20.0`, Manifest v2) and OpenAI-v1 agent swarms (`GrokBot`, `OpenCode`, subagent pools). Unites durable task fencing, multi-task affinity bundle dispatch, enriched trace telemetry, distinct `maker != checker` object reuse, hybrid `FTS5` search, attenuation-gated Macro-DAG compilation, parameterized `NodeGrid` + contextual RL canary optimization, `256-byte` `LoRA-XS` subspace weight cartridges with mandatory `thinking=off` matched-baseline gating, a loopback OpenAI-v1 swarm proxy, and optional federated Fabric sync in one zero-dependency package.

## Automatic Host Capability Detection & 5 Setup Profiles

At session start (or via `fabric_probe_and_configure` / `/fabric-mode`), the plugin inspects what can run locally inside the current environment and selects one of five operational setups:

| Profile | Local Requirements | Active Local & Federated Capabilities |
|---|---|---|
| `stdlib_airgap` | Python 3 stdlib + SQLite WAL/FTS5 (`0` pip packages, `0` network) | Monotonic `CAS lease_epoch` queue, `1..4` affinity bundle dispatch, enriched `trace_envelopes`, `maker != checker` object cache, `FTS5` BM25 + RRF search, attenuation-gated `3..250` Macro-DAG compiler, `NodeGrid` + RL canary optimizer, and `256B` `LoRA-XS` / ReLU-ridge compilers with `thinking=off` confound gating |
| `local_neural` | `+ numpy` & `torch` (`+ safetensors` optional), `0` network | Everything in `stdlib_airgap` plus multi-head `.safetensors` micro-net synthesis, `0`-epoch SVD nullspace weight surgery (`0.0` anchor drift), and Wanda trace channel pruning |
| `local_runner_swarm` | `+ loopback open-model runner` (`127.0.0.1:11434`) | Everything above plus on-demand loopback `/v1/models`, `/v1/chat/completions`, `/swarm/claim-bundle`, and `/swarm/trace` proxy for `Hermes` subagents and `GrokBot` workers |
| `hybrid_fabric_sync` | Local SQLite WAL cell `+ upstream_endpoint` | Runs private `local_only` work, subagent swarms, and boil-down locally while syncing `unseen_only` frontier tasks, `sync_ready` promoted objects, and `/mcp` calls with a remote Fabric endpoint |
| `thin_remote_client` | Ephemeral / read-only container `+ upstream_endpoint` | Zero disk footprint (`:memory:` buffer); bridges queue, object, optimization, and MCP operations directly to `upstream_endpoint` |

## What Is New in v0.3.0

1. **Multi-Task Worker Bundle Dispatch (`fabric_claim_bundle` & `POST /swarm/claim-bundle`):**
   - Atomically leases `1..4` ready tasks in one SQLite WAL transaction across four modes: `back_to_back_sequential` (clusters shared `prefix_signature` and `capability_tag` back-to-back to maximize KV-cache and `LoRA-XS` cartridge locality), `simultaneous_batched_pack` (packs compatible `tool_schema_tag` tasks into one turn), `speculative_pipeline`, and `single_isolated`.
   - Returns explicit `estimated_kv_prefix_reuse_ratio` and a slot-by-slot `recommended_execution_schedule`.
2. **Enriched Trace Envelopes (`trace_envelopes` & `POST /swarm/trace`):**
   - Records `prefix_signature`, `reasoning_mode` (`off | minimal | adaptive | deep`), `thinking_tokens`, `answer_tokens`, `activation_norm_digest`, `attenuation_confidence_milli`, and `seed_id` during result and trace submissions.
3. **Attenuation-Tracked Decomposition & Circuit-Breaker (`compile_macro_dag`):**
   - Tracks `origin_id`, `dag_width`, `dag_depth`, `granularity_tier`, and `cumulative_attenuation_survival` across parent-to-child hops. Automatically halts downstream tasks (`attenuation_halted`) when cumulative survival drops below `attenuation_circuit_breaker_milli`, and resets survival to `1.0` once an independent `maker != checker` review promotes the upstream task.
4. **`256-Byte` `LoRA-XS` Subspace Cartridges + Matched `thinking=off` Confound Gate:**
   - Synthesizes rank-`8` (`8 x 8` `float32` = `256` bytes) `LoRA-XS` singular subspace mixing matrices on frozen deterministic bases (`0` SGD epochs, hot-swappable in `<1 ms`).
   - Evaluates candidate weight operators across multiple deterministic seeds against **both** `verbose_baseline` and `direct_thinking_off_baseline` (`matched_control_gain_over_thinking_off`), blocking false promotions when `thinking=off` alone accounts for the benchmark gain.

## Public Extension Surfaces

- **9 Tools:** `fabric_probe_and_configure`, `fabric_task_queue`, `fabric_claim_bundle`, `fabric_object_store`, `fabric_compile_workflow_dag`, `fabric_grid_and_rl_optimize`, `fabric_boil_down_and_edit`, `fabric_swarm_proxy_control`, `fabric_federated_sync`
- **10 Lifecycle, Subagent & Kanban Hooks:** `on_session_start`, `pre_llm_call`, `pre_verify`, `post_api_request`, `post_tool_call`, `subagent_start`, `subagent_stop`, `kanban_task_claimed`, `kanban_task_completed`, `on_kanban_worker_stale_claim`
- **1 Request Middleware:** `llm_request` (`<0.2 ms` reasoning-effort calibration + structured JSON `max_tokens` floor)
- **2 Slash Commands:** `/fabric-status`, `/fabric-mode`

## Hermes Agent & GrokBot Judge Quick-Check Commands

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
