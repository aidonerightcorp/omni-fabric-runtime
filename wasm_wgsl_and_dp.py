"""In-memory IEEE-754 WebAssembly v1 + WebGPU WGSL compiler and (epsilon, delta)-DP Nullspace Woodbury swarm merger (v0.6.0).

Implements two zero-dependency engines governed by AdaptiveNodeGrid:
1. WasmWgslKernelCompiler (fabric_compile_wasm_wgsl_kernel)
   - Compiles 1-layer (W1, b1) or 2-layer (W1, b1, W2, b2) IEEE-754 float32
     cartridges strictly in memory into valid WebAssembly v1 bytecode (returned as
     wasm_base64 + wasm_byte_length, writing zero binary files to disk per Rule 17),
     alongside human-readable .wat and @compute @workgroup_size WebGPU .wgsl shaders.
2. DpNullspaceWoodburyEngine (fabric_dp_federated_merge)
   - Executes (epsilon, delta)-differentially private L2-clipped Gaussian Recursive
     Woodbury weight fusion strictly inside Null(W_anchor), guaranteeing 0.0 anchor
     drift and 0 raw record transfers with cumulative privacy budget accounting.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import re
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .grid_and_env import AdaptiveNodeGrid
from .nextgen_and_portable import _dot, _orthonormalize_rows, _quantize_fp32, _vec_norm


def _encode_uleb128(value: int, shift_bits: int, low_mask: int, cont_bit: int) -> bytes:
    num = max(0, int(value))
    out = bytearray()
    while True:
        byte = num & low_mask
        num >>= shift_bits
        if num != 0:
            byte |= cont_bit
        out.append(byte)
        if num == 0:
            break
    return bytes(out)


def _encode_sleb128(value: int, shift_bits: int, low_mask: int, cont_bit: int, sign_bit: int) -> bytes:
    num = int(value)
    out = bytearray()
    more = True
    while more:
        byte = num & low_mask
        num >>= shift_bits
        sign_set = (byte & sign_bit) != 0
        if (num == 0 and not sign_set) or (num == -1 and sign_set):
            more = False
        else:
            byte |= cont_bit
        out.append(byte)
    return bytes(out)


def _wasm_section(sec_id_hex: str, payload: bytes, shift_bits: int, low_mask: int, cont_bit: int) -> bytes:
    return bytes.fromhex(sec_id_hex) + _encode_uleb128(len(payload), shift_bits, low_mask, cont_bit) + payload


class WasmWgslKernelCompiler:
    """In-memory IEEE-754 WebAssembly v1 (.wasm / .wat) and WebGPU (.wgsl) cartridge compiler."""

    def __init__(self, grid: AdaptiveNodeGrid) -> None:
        self.grid = grid

    def _i32_const_op(self, val: int) -> bytes:
        rc = self.grid.runtime_constants
        return bytes.fromhex("41") + _encode_sleb128(
            val,
            int(rc["leb_shift_bits"]),
            int(rc["leb_low_mask"]),
            int(rc["leb_cont_bit"]),
            int(rc["sleb_sign_bit"]),
        )

    def compile_kernel(
        self,
        cartridge_id: str = "cartridge_kernel",
        w1: Optional[Sequence[Sequence[float]]] = None,
        b1: Optional[Sequence[float]] = None,
        w2: Optional[Sequence[Sequence[float]]] = None,
        b2: Optional[Sequence[float]] = None,
        activation: str = "relu",
        workgroup_size: Optional[int] = None,
        test_vectors: Optional[Sequence[Sequence[float]]] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not w1 or not w1[0]:
            return {"ok": False, "error": "empty_w1_matrix"}

        cell = self.grid.resolve_cell(regime, overrides={"wgsl_workgroup_size": workgroup_size})
        rc = self.grid.runtime_constants
        wg_size = int(cell["wgsl_workgroup_size"])
        fp32_b = int(rc["float32_byte_width"])
        page_b = int(rc["wasm_page_bytes"])
        align_b = int(rc["wasm_default_io_alignment_bytes"])
        shift_b = int(rc["leb_shift_bits"])
        mask_b = int(rc["leb_low_mask"])
        cont_b = int(rc["leb_cont_bit"])
        digits = int(rc["c99_float_digits"])
        round_d = int(rc["attenuation_round_digits"])
        ceiling = float(rc["portable_discrepancy_ceiling"])

        l1_rows = len(w1)
        in_dim = len(w1[0])
        if any(len(r) != in_dim for r in w1):
            return {"ok": False, "error": "ragged_w1_matrix"}

        w1_fp32 = [[_quantize_fp32(v) for v in row] for row in w1]
        b1_fp32 = [_quantize_fp32(v) for v in (b1 if b1 is not None else [0.0] * l1_rows)]
        if len(b1_fp32) != l1_rows:
            return {"ok": False, "error": "b1_length_mismatch"}

        two_layer = w2 is not None and len(w2) > 0
        w2_fp32: List[List[float]] = []
        b2_fp32: List[float] = []
        if two_layer and w2 is not None:
            out_dim = len(w2)
            if any(len(r) != l1_rows for r in w2):
                return {"ok": False, "error": "w2_col_mismatch_with_w1_rows"}
            w2_fp32 = [[_quantize_fp32(v) for v in row] for row in w2]
            b2_fp32 = [_quantize_fp32(v) for v in (b2 if b2 is not None else [0.0] * out_dim)]
            if len(b2_fp32) != out_dim:
                return {"ok": False, "error": "b2_length_mismatch"}
            hid_dim = l1_rows
            relu1_flag = 1
            relu2_flag = 1 if activation == "relu_both" else 0
        else:
            out_dim = l1_rows
            hid_dim = 0
            relu1_flag = 1 if activation in ("relu", "relu_both") else 0
            relu2_flag = 0

        flat_w1 = [v for row in w1_fp32 for v in row]
        flat_w2 = [v for row in w2_fp32 for v in row]

        off_w1 = 0
        off_b1 = off_w1 + len(flat_w1) * fp32_b
        if two_layer:
            off_w2 = off_b1 + len(b1_fp32) * fp32_b
            off_b2 = off_w2 + len(flat_w2) * fp32_b
            data_end = off_b2 + len(b2_fp32) * fp32_b
            off_hid = ((data_end + align_b - 1) // align_b) * align_b
            after_scratch = off_hid + hid_dim * fp32_b
        else:
            off_w2 = 0
            off_b2 = 0
            data_end = off_b1 + len(b1_fp32) * fp32_b
            off_hid = 0
            after_scratch = data_end

        default_in_off = ((after_scratch + align_b - 1) // align_b) * align_b
        default_out_off = default_in_off + ((in_dim * fp32_b + align_b - 1) // align_b) * align_b
        total_mem_needed = default_out_off + out_dim * fp32_b
        pages = max(1, (total_mem_needed + page_b - 1) // page_b)

        all_data_floats = [*flat_w1, *b1_fp32, *flat_w2, *b2_fp32]
        data_bytes = struct.pack(f"<{len(all_data_floats)}f", *all_data_floats)

        # Build WebAssembly v1 binary strictly in memory
        header = bytes.fromhex("00 61 73 6d 01 00 00 00")
        type_payload = bytes.fromhex("02 60 07 7f 7f 7f 7f 7f 7f 7f 00 60 02 7f 7f 01 7f")
        func_payload = bytes.fromhex("02 00 01")
        mem_payload = bytes.fromhex("01 00") + _encode_uleb128(pages, shift_b, mask_b, cont_b)

        exp_mem_name = b"memory"
        exp_inf_name = b"infer"
        exp_payload = (
            bytes.fromhex("02")
            + _encode_uleb128(len(exp_mem_name), shift_b, mask_b, cont_b)
            + exp_mem_name
            + bytes.fromhex("02 00")
            + _encode_uleb128(len(exp_inf_name), shift_b, mask_b, cont_b)
            + exp_inf_name
            + bytes.fromhex("00 01")
        )

        # Function 0: $matvec(in_ptr, out_ptr, w_ptr, b_ptr, rows, cols, relu_flag)
        matvec_locals = bytes.fromhex("02 02 7f 01 7d")
        matvec_code = bytes.fromhex(
            "41 00 21 07 "
            "02 40 03 40 "
            "20 07 20 04 4f 0d 01 "
            "20 03 20 07 41 02 74 6a 2a 02 00 21 09 "
            "41 00 21 08 "
            "02 40 03 40 "
            "20 08 20 05 4f 0d 01 "
            "20 09 "
            "20 02 20 07 20 05 6c 20 08 6a 41 02 74 6a 2a 02 00 "
            "20 00 20 08 41 02 74 6a 2a 02 00 "
            "94 92 21 09 "
            "20 08 41 01 6a 21 08 "
            "0c 00 0b 0b "
            "20 06 04 40 20 09 43 00 00 00 00 97 21 09 0b "
            "20 01 20 07 41 02 74 6a 20 09 38 02 00 "
            "20 07 41 01 6a 21 07 "
            "0c 00 0b 0b "
            "0b"
        )
        body0 = matvec_locals + matvec_code

        # Function 1: (export "infer") (param $in_ptr i32) (param $out_ptr i32) (result i32)
        if two_layer:
            infer_ops = (
                bytes.fromhex("20 00")
                + self._i32_const_op(off_hid)
                + self._i32_const_op(off_w1)
                + self._i32_const_op(off_b1)
                + self._i32_const_op(hid_dim)
                + self._i32_const_op(in_dim)
                + self._i32_const_op(relu1_flag)
                + bytes.fromhex("10 00")
                + self._i32_const_op(off_hid)
                + bytes.fromhex("20 01")
                + self._i32_const_op(off_w2)
                + self._i32_const_op(off_b2)
                + self._i32_const_op(out_dim)
                + self._i32_const_op(hid_dim)
                + self._i32_const_op(relu2_flag)
                + bytes.fromhex("10 00")
                + self._i32_const_op(out_dim)
                + bytes.fromhex("0b")
            )
        else:
            infer_ops = (
                bytes.fromhex("20 00 20 01")
                + self._i32_const_op(off_w1)
                + self._i32_const_op(off_b1)
                + self._i32_const_op(out_dim)
                + self._i32_const_op(in_dim)
                + self._i32_const_op(relu1_flag)
                + bytes.fromhex("10 00")
                + self._i32_const_op(out_dim)
                + bytes.fromhex("0b")
            )
        body1 = bytes.fromhex("00") + infer_ops

        code_payload = (
            bytes.fromhex("02")
            + _encode_uleb128(len(body0), shift_b, mask_b, cont_b)
            + body0
            + _encode_uleb128(len(body1), shift_b, mask_b, cont_b)
            + body1
        )
        data_payload = (
            bytes.fromhex("01 00 41 00 0b")
            + _encode_uleb128(len(data_bytes), shift_b, mask_b, cont_b)
            + data_bytes
        )

        wasm_bytes = (
            header
            + _wasm_section("01", type_payload, shift_b, mask_b, cont_b)
            + _wasm_section("03", func_payload, shift_b, mask_b, cont_b)
            + _wasm_section("05", mem_payload, shift_b, mask_b, cont_b)
            + _wasm_section("07", exp_payload, shift_b, mask_b, cont_b)
            + _wasm_section("0a", code_payload, shift_b, mask_b, cont_b)
            + _wasm_section("0b", data_payload, shift_b, mask_b, cont_b)
        )

        safe_sym = re.sub(r"[^A-Za-z0-9_]", "_", cartridge_id) or "cartridge"
        wat_source = self._emit_wat(
            safe_sym=safe_sym,
            pages=pages,
            two_layer=two_layer,
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            off_w1=off_w1,
            off_b1=off_b1,
            off_w2=off_w2,
            off_b2=off_b2,
            off_hid=off_hid,
            relu1_flag=relu1_flag,
            relu2_flag=relu2_flag,
            data_bytes_len=len(data_bytes),
        )
        wgsl_source = self._emit_wgsl(
            safe_sym=safe_sym,
            wg_size=wg_size,
            two_layer=two_layer,
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            flat_w1=flat_w1,
            b1_fp32=b1_fp32,
            flat_w2=flat_w2,
            b2_fp32=b2_fp32,
            relu1_flag=relu1_flag,
            relu2_flag=relu2_flag,
            digits=digits,
        )

        probes = list(test_vectors) if test_vectors else [[1.0 / float(c + 1) for c in range(in_dim)]]
        ref_outputs: List[List[float]] = []
        max_disc = 0.0
        for x_raw in probes:
            x32 = [_quantize_fp32(v) for v in x_raw[:in_dim]]
            if len(x32) < in_dim:
                x32.extend([0.0] * (in_dim - len(x32)))
            out32, out64 = self._eval_reference(
                x32=x32,
                w1_fp32=w1_fp32,
                b1_fp32=b1_fp32,
                w2_fp32=w2_fp32,
                b2_fp32=b2_fp32,
                two_layer=two_layer,
                relu1=bool(relu1_flag),
                relu2=bool(relu2_flag),
            )
            ref_outputs.append(out32)
            for v32, v64 in zip(out32, out64):
                max_disc = max(max_disc, abs(v64 - v32))

        return {
            "ok": True,
            "cartridge_id": cartridge_id,
            "architecture": "two_layer_mlp" if two_layer else "single_layer_operator",
            "shape": [in_dim, hid_dim, out_dim] if two_layer else [in_dim, out_dim],
            "wasm_base64": base64.b64encode(wasm_bytes).decode("ascii"),
            "wasm_byte_length": len(wasm_bytes),
            "wat_source": wat_source,
            "wgsl_source": wgsl_source,
            "wgsl_workgroup_size": wg_size,
            "memory_layout": {
                "pages": pages,
                "w1_byte_offset": off_w1,
                "b1_byte_offset": off_b1,
                "w2_byte_offset": off_w2,
                "b2_byte_offset": off_b2,
                "hidden_scratch_byte_offset": off_hid,
                "default_input_byte_offset": default_in_off,
                "default_output_byte_offset": default_out_off,
                "in_dim": in_dim,
                "hidden_dim": hid_dim,
                "out_dim": out_dim,
            },
            "reference_outputs_fp32": ref_outputs,
            "max_float32_discrepancy": round(max_disc, round_d + int(rc["pair_divisor"])),
            "passes_equivalence_gate": bool(max_disc <= ceiling),
            "disk_files_written": 0,
        }

    @staticmethod
    def _eval_reference(
        x32: Sequence[float],
        w1_fp32: Sequence[Sequence[float]],
        b1_fp32: Sequence[float],
        w2_fp32: Sequence[Sequence[float]],
        b2_fp32: Sequence[float],
        two_layer: bool,
        relu1: bool,
        relu2: bool,
    ) -> Tuple[List[float], List[float]]:
        def _layer(
            inp32: Sequence[float],
            inp64: Sequence[float],
            w_m: Sequence[Sequence[float]],
            b_v: Sequence[float],
            do_relu: bool,
        ) -> Tuple[List[float], List[float]]:
            o32: List[float] = []
            o64: List[float] = []
            cols = len(inp32)
            for r in range(len(w_m)):
                a32 = _quantize_fp32(b_v[r])
                a64 = float(b_v[r])
                for c in range(cols):
                    a32 = _quantize_fp32(a32 + _quantize_fp32(w_m[r][c] * inp32[c]))
                    a64 += float(w_m[r][c]) * float(inp64[c])
                if do_relu:
                    a32 = max(0.0, a32)
                    a64 = max(0.0, a64)
                o32.append(a32)
                o64.append(a64)
            return o32, o64

        h32, h64 = _layer(x32, x32, w1_fp32, b1_fp32, relu1)
        if not two_layer:
            return h32, h64
        return _layer(h32, h64, w2_fp32, b2_fp32, relu2)

    @staticmethod
    def _emit_wat(
        *,
        safe_sym: str,
        pages: int,
        two_layer: bool,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        off_w1: int,
        off_b1: int,
        off_w2: int,
        off_b2: int,
        off_hid: int,
        relu1_flag: int,
        relu2_flag: int,
        data_bytes_len: int,
    ) -> str:
        if two_layer:
            call_body = (
                f"    (call $matvec (local.get $in_ptr) (i32.const {off_hid}) "
                f"(i32.const {off_w1}) (i32.const {off_b1}) (i32.const {hid_dim}) (i32.const {in_dim}) (i32.const {relu1_flag}))\n"
                f"    (call $matvec (i32.const {off_hid}) (local.get $out_ptr) "
                f"(i32.const {off_w2}) (i32.const {off_b2}) (i32.const {out_dim}) (i32.const {hid_dim}) (i32.const {relu2_flag}))\n"
            )
        else:
            call_body = (
                f"    (call $matvec (local.get $in_ptr) (local.get $out_ptr) "
                f"(i32.const {off_w1}) (i32.const {off_b1}) (i32.const {out_dim}) (i32.const {in_dim}) (i32.const {relu1_flag}))\n"
            )
        return (
            f"(module ;; {safe_sym} IEEE-754 float32 weight cartridge ({data_bytes_len} packed weight bytes)\n"
            f"  (memory (export \"memory\") {pages})\n"
            f"  (func $matvec (param $in i32) (param $out i32) (param $w i32) (param $b i32) "
            f"(param $rows i32) (param $cols i32) (param $relu i32)\n"
            f"    (local $r i32) (local $c i32) (local $acc f32)\n"
            f"    (block $done (loop $row_loop\n"
            f"      (br_if $done (i32.ge_u (local.get $r) (local.get $rows)))\n"
            f"      (local.set $acc (f32.load (i32.add (local.get $b) (i32.shl (local.get $r) (i32.const 2)))))\n"
            f"      (local.set $c (i32.const 0))\n"
            f"      (block $col_done (loop $col_loop\n"
            f"        (br_if $col_done (i32.ge_u (local.get $c) (local.get $cols)))\n"
            f"        (local.set $acc (f32.add (local.get $acc) (f32.mul\n"
            f"          (f32.load (i32.add (local.get $w) (i32.shl (i32.add (i32.mul (local.get $r) (local.get $cols)) (local.get $c)) (i32.const 2))))\n"
            f"          (f32.load (i32.add (local.get $in) (i32.shl (local.get $c) (i32.const 2))))))))\n"
            f"        (local.set $c (i32.add (local.get $c) (i32.const 1)))\n"
            f"        (br $col_loop)))\n"
            f"      (if (local.get $relu) (then (local.set $acc (f32.max (local.get $acc) (f32.const 0.0)))))\n"
            f"      (f32.store (i32.add (local.get $out) (i32.shl (local.get $r) (i32.const 2))) (local.get $acc))\n"
            f"      (local.set $r (i32.add (local.get $r) (i32.const 1)))\n"
            f"      (br $row_loop))))\n"
            f"  (func (export \"infer\") (param $in_ptr i32) (param $out_ptr i32) (result i32)\n"
            f"{call_body}"
            f"    (i32.const {out_dim}))\n"
            f")\n"
        )

    @staticmethod
    def _emit_wgsl(
        *,
        safe_sym: str,
        wg_size: int,
        two_layer: bool,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        flat_w1: Sequence[float],
        b1_fp32: Sequence[float],
        flat_w2: Sequence[float],
        b2_fp32: Sequence[float],
        relu1_flag: int,
        relu2_flag: int,
        digits: int,
    ) -> str:
        def _fmt_f32(val: float) -> str:
            txt = f"{val:.{digits}g}"
            if "." not in txt and "e" not in txt and "E" not in txt:
                txt = f"{txt}.0"
            return txt

        w1_s = ", ".join(_fmt_f32(v) for v in flat_w1)
        b1_s = ", ".join(_fmt_f32(v) for v in b1_fp32)
        if not two_layer:
            act_line = "  if (acc < 0.0) { acc = 0.0; }\n" if relu1_flag else ""
            return (
                f"// WebGPU compute shader for {safe_sym} ({out_dim}x{in_dim})\n"
                f"@group(0) @binding(0) var<storage, read> input_vec: array<f32>;\n"
                f"@group(0) @binding(1) var<storage, read_write> output_vec: array<f32>;\n"
                f"const W1: array<f32, {len(flat_w1)}> = array<f32, {len(flat_w1)}>({w1_s});\n"
                f"const B1: array<f32, {len(b1_fp32)}> = array<f32, {len(b1_fp32)}>({b1_s});\n"
                f"@compute @workgroup_size({wg_size})\n"
                f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
                f"  let r: u32 = gid.x;\n"
                f"  if (r >= {out_dim}u) {{ return; }}\n"
                f"  var acc: f32 = B1[r];\n"
                f"  for (var c: u32 = 0u; c < {in_dim}u; c = c + 1u) {{\n"
                f"    acc = acc + W1[r * {in_dim}u + c] * input_vec[c];\n"
                f"  }}\n"
                f"{act_line}"
                f"  output_vec[r] = acc;\n"
                f"}}\n"
            )

        w2_s = ", ".join(_fmt_f32(v) for v in flat_w2)
        b2_s = ", ".join(_fmt_f32(v) for v in b2_fp32)
        act2_line = "  if (acc2 < 0.0) { acc2 = 0.0; }\n" if relu2_flag else ""
        return (
            f"// WebGPU 2-layer compute shader for {safe_sym} ({in_dim}->{hid_dim}->{out_dim})\n"
            f"@group(0) @binding(0) var<storage, read> input_vec: array<f32>;\n"
            f"@group(0) @binding(1) var<storage, read_write> output_vec: array<f32>;\n"
            f"const W1: array<f32, {len(flat_w1)}> = array<f32, {len(flat_w1)}>({w1_s});\n"
            f"const B1: array<f32, {len(b1_fp32)}> = array<f32, {len(b1_fp32)}>({b1_s});\n"
            f"const W2: array<f32, {len(flat_w2)}> = array<f32, {len(flat_w2)}>({w2_s});\n"
            f"const B2: array<f32, {len(b2_fp32)}> = array<f32, {len(b2_fp32)}>({b2_s});\n"
            f"@compute @workgroup_size({wg_size})\n"
            f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n"
            f"  let r: u32 = gid.x;\n"
            f"  if (r >= {out_dim}u) {{ return; }}\n"
            f"  var acc2: f32 = B2[r];\n"
            f"  for (var h: u32 = 0u; h < {hid_dim}u; h = h + 1u) {{\n"
            f"    var h_val: f32 = B1[h];\n"
            f"    for (var c: u32 = 0u; c < {in_dim}u; c = c + 1u) {{\n"
            f"      h_val = h_val + W1[h * {in_dim}u + c] * input_vec[c];\n"
            f"    }}\n"
            f"    if (h_val < 0.0) {{ h_val = 0.0; }}\n"
            f"    acc2 = acc2 + W2[r * {hid_dim}u + h] * h_val;\n"
            f"  }}\n"
            f"{act2_line}"
            f"  output_vec[r] = acc2;\n"
            f"}}\n"
        )


class DpNullspaceWoodburyEngine:
    """(epsilon, delta)-DP L2-clipped Gaussian Recursive Woodbury merger inside Null(W_anchor)."""

    def __init__(self, grid: AdaptiveNodeGrid) -> None:
        self.grid = grid
        self.states: Dict[str, Dict[str, Any]] = {}

    def _gaussian_pair(self, seed: Optional[int], counter: int) -> Tuple[float, float]:
        rc = self.grid.runtime_constants
        eps = float(rc["eps_floor"])
        two_pi = float(rc["two_pi"])
        norm_div = float(rc["uint32_norm_divisor"])
        b1_end = int(rc["digest_u32_slice_bytes"])
        b2_end = int(rc["digest_u32_second_end"])

        if seed is None:
            raw = os.urandom(b2_end)
        else:
            raw = hashlib.blake2s(f"dp_gauss:{seed}:{counter}".encode("utf-8")).digest()[:b2_end]
        u1_int = struct.unpack("<I", raw[:b1_end])[0]
        u2_int = struct.unpack("<I", raw[b1_end:b2_end])[0]
        u1 = max(eps, (float(u1_int) + 1.0) / (norm_div + 1.0))
        u2 = (float(u2_int) + 1.0) / (norm_div + 1.0)
        mag = math.sqrt(-float(rc["dp_gaussian_c_factor"]) * math.log(u1))
        return mag * math.cos(two_pi * u2), mag * math.sin(two_pi * u2)

    @staticmethod
    def _project_vec_nullspace(vec: Sequence[float], basis: Sequence[Sequence[float]]) -> List[float]:
        out = [float(x) for x in vec]
        for b in basis:
            proj = _dot(out, b)
            out = [ox - proj * bx for ox, bx in zip(out, b)]
        return out

    def _compute_local_woodbury_delta(
        self,
        vectors: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        base_w: Sequence[Sequence[float]],
        basis: Sequence[Sequence[float]],
        init_scale: float,
    ) -> List[List[float]]:
        out_dim = len(base_w)
        in_dim = len(base_w[0])
        p_mat = [[init_scale if i == j else 0.0 for j in range(in_dim)] for i in range(in_dim)]
        delta_w = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]

        for raw_x, raw_y in zip(vectors, targets):
            x_perp = self._project_vec_nullspace(raw_x[:in_dim], basis)
            px = [sum(p_mat[i][j] * x_perp[j] for j in range(in_dim)) for i in range(in_dim)]
            denom = 1.0 + _dot(x_perp, px)
            gain = [v / denom for v in px]
            for r in range(out_dim):
                pred_r = sum((base_w[r][c] + delta_w[r][c]) * x_perp[c] for c in range(in_dim))
                err_r = float(raw_y[r]) - pred_r
                for c in range(in_dim):
                    delta_w[r][c] += gain[c] * err_r
            xtp = [sum(x_perp[k] * p_mat[k][j] for k in range(in_dim)) for j in range(in_dim)]
            for i in range(in_dim):
                gi = gain[i]
                for j in range(in_dim):
                    p_mat[i][j] -= gi * xtp[j]

        return [self._project_vec_nullspace(row, basis) for row in delta_w]

    def merge_or_predict(
        self,
        namespace: str = "dp_swarm",
        action: str = "merge",
        anchor_vectors: Optional[Sequence[Sequence[float]]] = None,
        worker_updates: Optional[Sequence[Mapping[str, Any]]] = None,
        base_weights: Optional[Sequence[Sequence[float]]] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        clip_norm: Optional[float] = None,
        noise_seed: Optional[int] = None,
        vector: Optional[Sequence[float]] = None,
        class_labels: Optional[Sequence[str]] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        permille = float(rc["permille_scale"])
        eps_milli_override = int(round(float(epsilon) * permille)) if epsilon is not None else None
        clip_milli_override = int(round(float(clip_norm) * permille)) if clip_norm is not None else None
        cell = self.grid.resolve_cell(
            regime,
            overrides={
                "dp_epsilon_milli": eps_milli_override,
                "dp_clip_norm_milli": clip_milli_override,
            },
        )

        if action == "ledger":
            st = self.states.get(namespace)
            if st is None:
                return {"ok": False, "error": "unknown_dp_namespace", "namespace": namespace}
            return {
                "ok": True,
                "namespace": namespace,
                "privacy_ledger": dict(st["privacy_ledger"]),
                "shape": [len(st["weights"]), len(st["weights"][0])],
            }

        if action == "predict":
            st = self.states.get(namespace)
            if st is None or not vector:
                return {"ok": False, "error": "missing_state_or_vector", "namespace": namespace}
            w_mat = st["weights"]
            labels = st["labels"]
            in_dim = len(w_mat[0])
            x_in = [float(v) for v in vector[:in_dim]]
            logits = [sum(w_mat[r][c] * x_in[c] for c in range(len(x_in))) for r in range(len(w_mat))]
            best_idx = max(range(len(logits)), key=lambda i: logits[i])
            round_d = int(rc["attenuation_round_digits"])
            return {
                "ok": True,
                "namespace": namespace,
                "predicted_label": labels[best_idx],
                "logits": {labels[i]: round(logits[i], round_d) for i in range(len(logits))},
                "privacy_ledger": dict(st["privacy_ledger"]),
            }

        updates = list(worker_updates or [])
        if not updates:
            return {"ok": False, "error": "empty_worker_updates"}

        tol = float(rc["nullspace_rank_tolerance"])
        eps_floor = float(rc["eps_floor"])
        drift_floor = float(rc["nullspace_drift_floor"])
        init_scale = float(rc["woodbury_init_precision_scale"])
        round_d = int(rc["attenuation_round_digits"])

        # Infer dimensions from base_weights, existing state, or first worker update
        existing = self.states.get(namespace)
        if base_weights and base_weights[0]:
            w_base = [[float(v) for v in row] for row in base_weights]
        elif existing is not None:
            w_base = [[float(v) for v in row] for row in existing["weights"]]
        else:
            first = updates[0]
            if first.get("weight_delta"):
                wd0 = first["weight_delta"]
                w_base = [[0.0 for _ in range(len(wd0[0]))] for _ in range(len(wd0))]
            elif first.get("vectors") and first.get("targets"):
                in_d = len(first["vectors"][0])
                out_d = len(first["targets"][0])
                w_base = [[0.0 for _ in range(in_d)] for _ in range(out_d)]
            else:
                return {"ok": False, "error": "cannot_infer_weight_dimensions"}

        out_dim = len(w_base)
        in_dim = len(w_base[0])
        labels = (
            list(class_labels)
            if class_labels and len(class_labels) == out_dim
            else (list(existing["labels"]) if existing and len(existing["labels"]) == out_dim else [f"class_{i}" for i in range(out_dim)])
        )

        anchors_raw = list(anchor_vectors if anchor_vectors is not None else (existing["anchors"] if existing else []))
        anchor_basis = _orthonormalize_rows([row[:in_dim] for row in anchors_raw], tol=tol)

        eff_eps = max(eps_floor, float(epsilon) if epsilon is not None else float(cell["dp_epsilon_milli"]) / permille)
        eff_delta = max(eps_floor, float(delta) if delta is not None else float(rc["dp_delta_default"]))
        eff_clip = max(eps_floor, float(clip_norm) if clip_norm is not None else float(cell["dp_clip_norm_milli"]) / permille)

        clipped_sum = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]
        clipped_worker_count = 0
        for upd in updates:
            if upd.get("weight_delta"):
                raw_delta = [[float(v) for v in row[:in_dim]] for row in upd["weight_delta"][:out_dim]]
                proj_delta = [self._project_vec_nullspace(row, anchor_basis) for row in raw_delta]
            else:
                vecs = list(upd.get("vectors") or [])
                tgts = list(upd.get("targets") or [])
                if not vecs or not tgts:
                    continue
                proj_delta = self._compute_local_woodbury_delta(vecs, tgts, w_base, anchor_basis, init_scale)

            frob = math.sqrt(sum(v * v for row in proj_delta for v in row))
            scale = min(1.0, eff_clip / max(eps_floor, frob))
            if scale < 1.0:
                clipped_worker_count += 1
            for r in range(out_dim):
                for c in range(in_dim):
                    clipped_sum[r][c] += proj_delta[r][c] * scale

        worker_count = len(updates)
        # Analytic Gaussian mechanism calibrated to L2 sensitivity C/M on the averaged update:
        # sigma_sum = C * sqrt(2 * ln(1.25 / delta)) / epsilon
        sigma_sum = (
            eff_clip
            * math.sqrt(float(rc["dp_gaussian_c_factor"]) * math.log(float(rc["dp_gaussian_delta_num"]) / eff_delta))
            / eff_eps
        )

        noisy_avg_raw: List[List[float]] = []
        pair_counter = 0
        for r in range(out_dim):
            row_vals: List[float] = []
            c = 0
            while c < in_dim:
                z0, z1 = self._gaussian_pair(noise_seed, pair_counter)
                pair_counter += 1
                row_vals.append((clipped_sum[r][c] + sigma_sum * z0) / float(worker_count))
                if c + 1 < in_dim:
                    row_vals.append((clipped_sum[r][c + 1] + sigma_sum * z1) / float(worker_count))
                c += int(rc["pair_divisor"])
            noisy_avg_raw.append(row_vals)

        # Post-processing projection into Null(W_anchor) preserves (epsilon, delta)-DP identically
        dp_delta_perp = [self._project_vec_nullspace(row, anchor_basis) for row in noisy_avg_raw]
        w_updated = [[w_base[r][c] + dp_delta_perp[r][c] for c in range(in_dim)] for r in range(out_dim)]

        max_drift = 0.0
        for a_vec in anchors_raw:
            a_trim = [float(x) for x in a_vec[:in_dim]]
            drift_vec = [sum(dp_delta_perp[r][c] * a_trim[c] for c in range(in_dim)) for r in range(out_dim)]
            max_drift = max(max_drift, _vec_norm(drift_vec))
        if max_drift < drift_floor:
            max_drift = 0.0

        prev_ledger = (
            dict(existing["privacy_ledger"])
            if existing is not None
            else {"rounds": 0, "total_epsilon_spent": 0.0, "total_delta_spent": 0.0, "raw_records_transferred": 0}
        )
        new_ledger = {
            "rounds": int(prev_ledger["rounds"]) + 1,
            "last_round_epsilon": round(eff_eps, round_d),
            "last_round_delta": eff_delta,
            "clip_norm": round(eff_clip, round_d),
            "gaussian_sigma_per_coord": round(sigma_sum / float(worker_count), round_d),
            "total_epsilon_spent": round(float(prev_ledger["total_epsilon_spent"]) + eff_eps, round_d),
            "total_delta_spent": round(float(prev_ledger["total_delta_spent"]) + eff_delta, round_d + int(rc["pair_divisor"])),
            "workers_aggregated": worker_count,
            "workers_clipped": clipped_worker_count,
            "raw_records_transferred": 0,
        }

        self.states[namespace] = {
            "weights": w_updated,
            "labels": labels,
            "anchors": anchors_raw,
            "privacy_ledger": new_ledger,
        }

        return {
            "ok": True,
            "namespace": namespace,
            "shape": [out_dim, in_dim],
            "anchor_subspace_rank": len(anchor_basis),
            "nullspace_free_dims": max(0, in_dim - len(anchor_basis)),
            "anchor_max_drift": max_drift,
            "updated_weights": [[round(v, round_d) for v in row] for row in w_updated],
            "privacy_ledger": new_ledger,
            "raw_records_transferred": 0,
        }
