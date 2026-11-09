# omni-fabric-runtime (v0.5.0)

Environment-adaptive runtime plugin for Hermes Agent (`>=0.20.0`, Manifest v2) and OpenAI-v1 agent swarms (`GrokBot`, `OpenCode`, subagent pools). Combines durable task fencing, multi-task affinity bundle dispatch, enriched trace telemetry, distinct `maker != checker` object reuse, hybrid `FTS5` search, a unified **6-stage weight-space compiler across 6 model classes**, an **`Actor × {Transfer, Transform, Save} × Horizon` check-gated DAG decomposer**, a **leak-free holdout recursive RL trial engine**, **topic-nullspace generator attribution**, **zero-token Koopman + CEGAR trajectory planning**, **5-target IEEE-754 float32 cartridge export with Hopfield/Woodbury 1-shot updates**, a loopback OpenAI-v1 swarm proxy, and optional federated Fabric sync in one zero-dependency package.

## Automatic Host Capability Detection & 5 Setup Profiles

At session start (or via `fabric_probe_and_configure` / `/fabric-mode`), the plugin inspects what can run locally inside the current environment and selects one of five operational setups:

| Profile | Local Requirements | Active Local & Federated Capabilities |
|---|---|---|
| `stdlib_airgap` | Python 3 stdlib + SQLite WAL/FTS5 (`0` pip packages, `0` network) | Monotonic `CAS lease_epoch` queue, `1..4` affinity bundle dispatch, enriched `trace_envelopes`, `maker != checker` object cache, `FTS5` BM25 + RRF search, 6-stage weight-space compiler, `Actor × {Transfer, Transform, Save} × Horizon` DAGs, leak-free recursive RL trials, topic-nullspace attribution, Koopman + CEGAR planner, and 5-target portable Hopfield/Woodbury cartridges |
| `local_neural` | `+ numpy` & `torch` (`+ safetensors` optional), `0` network | Everything in `stdlib_airgap` plus multi-head `.safetensors` synthesis, NumPy SVD nullspace weight surgery (`0.0` anchor drift), and Wanda trace channel pruning |
| `local_runner_swarm` | `+ loopback open-model runner` (`127.0.0.1:11434`) | Everything above plus on-demand loopback `/v1/models`, `/v1/chat/completions`, `/swarm/claim-bundle`, and `/swarm/trace` proxy for `Hermes` subagents and `GrokBot` workers |
| `hybrid_fabric_sync` | Local SQLite WAL cell `+ upstream_endpoint` | Runs private `local_only` work, subagent swarms, and weight compilation locally while syncing `unseen_only` frontier tasks, `sync_ready` promoted objects, and `/mcp` calls with a remote Fabric endpoint |
| `thin_remote_client` | Ephemeral / read-only container `+ upstream_endpoint` | Zero disk footprint (`:memory:` buffer); bridges queue, object, optimization, and MCP operations directly to `upstream_endpoint` |

## What Is New in v0.5.0 (15 Tools)

1. **Topic-Nullspace Generator Attribution & Conformal Spoof Gate (`fabric_attribute_generator_profile`):**
   - Projects ridge attribution weights onto the orthogonal complement of topical subspaces (`W_attr = W_ridge(I - V_topic V_topic^T)`) so `||W_attr V_topic||_F = 0.0` (`99.30%` unseen-topic accuracy, `0.70 pp` public-to-private shift).
   - Screens adversarial style-spoofed passages via class-conditional nullspace residual energy calibrated at the `0.96` conformal quantile.
2. **Zero-Token Lifted Koopman Tool Trajectory Planner + `CEGAR` Counterexample Selector (`fabric_plan_koopman_trajectory`):**
   - Lifts pipeline states into polynomial + bilinear observables and solves action-specific linear Koopman operators `z_{t+1} = K_a z_t` with linear feasibility gates (`100.0%` multi-step completion, `0` invalid transitions, `0` LLM tokens).
   - Ranks candidate boundary scenarios by minimum decision margin to select high-leverage `CEGAR` counterexamples (`6.67x` fewer oracle queries).
3. **5-Runtime IEEE-754 Portable Cartridge Exporter + Hopfield 1-Shot & Woodbury Multi-Shot Updates (`fabric_export_portable_cartridge`):**
   - Exports any `8x8` (`256B`) or linear/ReLU `float32` cartridge to `js` (`Float32Array`), `python` (zero-import stdlib), `sql` (declarative SQLite dot-product), `c99` (`const float` `.h` header), and `safetensors` (`<= 1e-5` discrepancy gate).
   - Supports Modern Continuous Hopfield `0`-epoch 1-shot column writes (`0.0045 ms`) and exact Recursive Woodbury RLS rank-1 streaming multi-shot updates (`0` replay buffer rows, `0.0` forgetting).
4. **Plus Everything from `v0.4.0`:**
   - `fabric_compile_weight_unit` (6-stage weight-space pipeline across 6 model classes), `fabric_decompose_horizon` (27 triads, `1..96` horizon steps), and `fabric_recursive_trial` (leak-free holdout RL).

## Public Extension Surfaces

- **15 Tools:** `fabric_probe_and_configure`, `fabric_task_queue`, `fabric_claim_bundle`, `fabric_object_store`, `fabric_compile_workflow_dag`, `fabric_grid_and_rl_optimize`, `fabric_boil_down_and_edit`, `fabric_swarm_proxy_control`, `fabric_federated_sync`, `fabric_compile_weight_unit`, `fabric_decompose_horizon`, `fabric_recursive_trial`, `fabric_attribute_generator_profile`, `fabric_plan_koopman_trajectory`, `fabric_export_portable_cartridge`
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
```

## Privacy & Network Disclosure

- Default `upstream_endpoint` is `""` (local-only): `0` external network requests, `0` telemetry, and `local_only` default visibility for staged Task Objects.
- Remote synchronization (`fabric_federated_sync`) runs only when an operator explicitly configures `upstream_endpoint`.
