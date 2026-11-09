# omni-fabric-runtime (v0.6.0)

Environment-adaptive runtime plugin for Hermes Agent (`>=0.20.0`, Manifest v2) and OpenAI-v1 agent swarms (`GrokBot`, `OpenCode`, subagent pools). Combines durable task fencing, multi-task affinity bundle dispatch, enriched trace telemetry, distinct `maker != checker` object reuse, hybrid `FTS5` search, a unified **6-stage weight-space compiler across 6 model classes**, an **`Actor × {Transfer, Transform, Save} × Horizon` check-gated DAG decomposer**, a **leak-free holdout recursive RL trial engine**, **topic-nullspace generator attribution**, **zero-token Koopman + CEGAR trajectory planning**, **7-target IEEE-754 float32 cartridge export (including in-memory WebAssembly v1 `.wasm`/`.wat` and WebGPU `.wgsl` compute shaders)**, **`(epsilon, delta)`-DP Nullspace Woodbury federated swarm merging**, a DNS-pinned loopback OpenAI-v1 swarm proxy, and owner-configured federated Fabric sync in one zero-dependency package.

## Automatic Host Capability Detection & 5 Setup Profiles

At session start (or via `fabric_probe_and_configure` / `/fabric-mode`), the plugin inspects what can run locally inside the current environment and selects one of five operational setups:

| Profile | Local Requirements | Active Local & Federated Capabilities |
|---|---|---|
| `stdlib_airgap` | Python 3 stdlib + SQLite WAL/FTS5 (`0` pip packages, `0` network) | Monotonic `CAS lease_epoch` queue, `1..4` affinity bundle dispatch, enriched `trace_envelopes`, `maker != checker` object cache, `FTS5` BM25 + RRF search, 6-stage weight-space compiler, `Actor × {Transfer, Transform, Save} × Horizon` DAGs, leak-free recursive RL trials, topic-nullspace attribution, Koopman + CEGAR planner, 7-target portable Hopfield/Woodbury cartridges (`.safetensors`, `js`, `python`, `sql`, `c99`, `.wasm`/`.wat`, `.wgsl`), and `(epsilon, delta)`-DP Nullspace Woodbury merges |
| `local_neural` | `+ numpy` & `torch` (`+ safetensors` optional), `0` network | Everything in `stdlib_airgap` plus multi-head `.safetensors` synthesis, NumPy SVD nullspace weight surgery (`0.0` anchor drift), and Wanda trace channel pruning |
| `local_runner_swarm` | `+ loopback open-model runner` (`127.0.0.1:11434`, `allow_local_runner: true`) | Everything above plus on-demand loopback `/v1/models`, `/v1/chat/completions`, `/swarm/claim-bundle`, and `/swarm/trace` proxy with `Host` and `Origin` loopback guards plus `atexit` clean shutdown |
| `hybrid_fabric_sync` | Local SQLite WAL cell `+ owner-configured upstream_endpoint` | Runs private `local_only` work, subagent swarms, and weight compilation locally while syncing `unseen_only` frontier tasks, `sync_ready` promoted objects, and `/mcp` calls over DNS-pinned TLS connections |
| `thin_remote_client` | Ephemeral / read-only container `+ owner-configured upstream_endpoint` | Zero disk footprint (`:memory:` buffer); bridges queue, object, optimization, and MCP operations directly to `upstream_endpoint` |

## What Is New in v0.6.0 (17 Tools)

1. **In-Memory IEEE-754 WebAssembly v1 & WebGPU WGSL Cartridge Compiler (`fabric_compile_wasm_wgsl_kernel`):**
   - Compiles any 1-layer (`W1, b1`) or 2-layer (`W1, b1, W2, b2`) IEEE-754 `float32` cartridge strictly in memory into valid WebAssembly v1 bytecode (`wasm_base64` + `wasm_byte_length`, writing `0` binary files to disk), human-readable `.wat`, and `@compute @workgroup_size` WebGPU `.wgsl` shader source.
2. **`(epsilon, delta)`-DP Nullspace Woodbury Federated Merge (`fabric_dp_federated_merge`):**
   - Projects per-worker Recursive Woodbury updates onto `Null(W_anchor)`, clips Frobenius L2 sensitivity to `C`, adds calibrated Gaussian noise inside `Null(W_anchor)`, and tracks cumulative `(epsilon, delta)` privacy spend with `0.0` anchor drift and `0` raw record transfers.
3. **Strict SSRF, DNS-Rebinding & Loopback CSRF Hardening:**
   - `fabric_federated_sync` reads exclusively the owner-configured `upstream_endpoint` (never a model-supplied URL), resolves all candidate IPs via `socket.getaddrinfo`, blocks cloud metadata / link-local / CGNAT / multicast / reserved ranges (`169.254.0.0/16`, `100.64.0.0/10`, `fd00:ec2::254`, etc.), blocks private/loopback upstream targets unless `allow_local_upstream: true`, pins the TCP socket to the vetted IP while preserving TLS SNI, never follows redirects, enforces `127.0.0.1`/`localhost` `Host` + `Origin` headers (`403` on cross-origin) on `_SwarmHandler`, and registers `atexit` shutdown.
4. **Plus Everything from `v0.5.0` & `v0.4.0`:**
   - `fabric_attribute_generator_profile`, `fabric_plan_koopman_trajectory`, `fabric_export_portable_cartridge`, `fabric_compile_weight_unit`, `fabric_decompose_horizon`, and `fabric_recursive_trial`.

## Public Extension Surfaces

- **17 Tools:** `fabric_probe_and_configure`, `fabric_task_queue`, `fabric_claim_bundle`, `fabric_object_store`, `fabric_compile_workflow_dag`, `fabric_grid_and_rl_optimize`, `fabric_boil_down_and_edit`, `fabric_swarm_proxy_control`, `fabric_federated_sync`, `fabric_compile_weight_unit`, `fabric_decompose_horizon`, `fabric_recursive_trial`, `fabric_attribute_generator_profile`, `fabric_plan_koopman_trajectory`, `fabric_export_portable_cartridge`, `fabric_compile_wasm_wgsl_kernel`, `fabric_dp_federated_merge`
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

- Default `upstream_endpoint` is `""` (local-only), `allow_local_upstream` is `false`, and `allow_local_runner` is `false`: `0` external network requests, `0` telemetry, and `local_only` default visibility for staged Task Objects.
- Remote synchronization (`fabric_federated_sync`) runs only when an operator explicitly configures `upstream_endpoint` in host settings.
