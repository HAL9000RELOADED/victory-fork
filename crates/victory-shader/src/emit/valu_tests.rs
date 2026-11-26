//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the LDS
//! float atomics `ds_min_f32`/`ds_max_f32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
pub(super) const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
pub(super) const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false)
}

/// VOP3P `op vdst, src0, src1, src2` (default opsel_hi: high halves for second element).
const fn vop3p(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x33 << 26 | (op as u32 & 0x7F) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18 | (3 << 27),
    ]
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

pub(super) const fn sgpr(n: u8) -> u16 {
    n as u16
}

pub(super) const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
pub(super) fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 1, 0)
}

/// [`build_valu_cs`] with `threads` threads in x (thread index in `v0`
/// on entry, before `body`) and `lds_bytes` of LDS (a multiple of 512).
fn build_cs(body: &[u32], threads: u32, lds_bytes: u32) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15), // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
        half_views: lowered.half_views,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
pub(super) fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    devices: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let (mut ran, mut no_32_lanes) = (0, 0);
    for &(label, warp) in devices {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(made) => made,
            // A program that declares `WaveSize(32)` (one thread with LDS, say) on a device
            // without 32-lane waves (#356: 16 at most): refused, as it should be; not run.
            Err(e) if !host.runs_waves_of(32) => {
                let (min, max) = host.wave_lanes().unwrap_or_default();
                eprintln!("{label}: waves of {min} to {max} lanes, not 32; skipped ({e})");
                no_32_lanes += 1;
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    if ran == 0 && devices.iter().all(|&(_, warp)| !warp) {
        // A case for the hardware adapter alone, on a machine without one.
        eprintln!("no SM 6.6 hardware device; a hardware-only case not run");
        return;
    }
    assert!(ran + no_32_lanes > 0, "no SM 6.6 device available ({devices:?})");
}

/// Each device says which wave sizes it runs (what the skip above goes by): a power-of-two
/// range, which its own ends are in.
#[test]
fn devices_report_their_wave_sizes() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (min, max) = host.wave_lanes().expect("OPTIONS1");
        eprintln!("{label}: waves of {min} to {max} lanes");
        assert!(min.is_power_of_two() && max.is_power_of_two() && min <= max, "{min}..{max}");
        assert!(host.runs_waves_of(min) && host.runs_waves_of(max));
        assert!(!host.runs_waves_of(max * 2) && !host.runs_waves_of(min / 2));
    }
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

/// The scalar bit searches and counts, ported from TuxBox (brcly, Apache-2.0,
/// `shader: restore s_flbit_i32_b64 and s_ff0_i32 lowering dropped by a merge`,
/// 1a50eb9): SFU-free, none of them writes SCC, and the 64-bit forms look at
/// the high word first.
#[test]
fn scalar_bit_searches_match_the_console() {
    // SOP1 `s_<op> s16, s0` (`src0` 0 is s[0:1] for a two-dword source).
    let sop1 = |op: u32, src0: u32| 0xBE80_0000 | 16 << 16 | op << 8 | src0;
    let read_back = |op: u32, src0: u32| [sop1(op, src0), vop1(0x001, 0, sgpr(16))];

    // s_flbit_i32_b64: 31 less the high word's first set bit from the top, else
    // 32 past the low word's, ~0 for a zero pair.
    let flbit = |[lo, hi, _, _]: [u32; 4]| {
        if hi != 0 {
            hi.leading_zeros()
        } else if lo != 0 {
            32 + lo.leading_zeros()
        } else {
            u32::MAX
        }
    };
    let cases = [
        [0, 1, 0, 0],
        [0, 0x8000_0000, 0, 0],
        [0x1234_5678, 0, 0, 0],
        [1, 0, 0, 0],
        [0x8000_0000, 0, 0, 0],
        [0, 0, 0, 0],
        [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0],
    ];
    let shader = build_valu_cs(&read_back(0x16, 0));
    run_cases(&shader, &cases, |backend, _i, inp, got| {
        assert_eq!(got, flbit(inp), "{backend}: s_flbit_i32_b64{s:?}", s = inp);
    });

    // s_ff0_i32_b32: the first zero bit from the bottom, ~0 when there is none.
    let ff0_b32 = |[a, _, _, _]: [u32; 4]| {
        if a == u32::MAX {
            u32::MAX
        } else {
            (!a).trailing_zeros()
        }
    };
    let cases = [[0, 0, 0, 0], [1, 0, 0, 0], [0x8000_0001, 0, 0, 0], [0xFFFF_FFFF, 0, 0, 0], [0xFFFF_FFFE, 0, 0, 0]];
    let shader = build_valu_cs(&read_back(0x11, 0));
    run_cases(&shader, &cases, |backend, _i, inp, got| {
        assert_eq!(got, ff0_b32(inp), "{backend}: s_ff0_i32_b32{s:?}", s = inp);
    });

    // s_ff0_i32_b64: the same over the pair, low word first.
    let ff0_b64 = |[lo, hi, _, _]: [u32; 4]| {
        if !lo != 0 {
            (!lo).trailing_zeros()
        } else if !hi != 0 {
            32 + (!hi).trailing_zeros()
        } else {
            u32::MAX
        }
    };
    let cases = [
        [0, 0, 0, 0],
        [0xF000_0000, 0, 0, 0],
        [0xFFFF_FFFF, 0, 0, 0],
        [0xFFFF_FFFF, 0x7FFF_FFFF, 0, 0],
        [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0],
    ];
    let shader = build_valu_cs(&read_back(0x12, 0));
    run_cases(&shader, &cases, |backend, _i, inp, got| {
        assert_eq!(got, ff0_b64(inp), "{backend}: s_ff0_i32_b64{s:?}", s = inp);
    });
}

/// The float `v_cvt_f32_f64` gives for `x`: Rust's rounding (to nearest
/// even), with results below the smallest normal float flushed to signed
/// zero as the console's default float mode does.
fn cvt_f32_f64_reference(x: f64) -> u32 {
    let r = x as f32;
    if r.is_subnormal() || (r == 0.0 && x != 0.0) {
        r.to_bits() & 0x8000_0000
    } else {
        r.to_bits()
    }
}

#[test]
#[ignore = "expects results below the smallest normal float flushed to zero; the IEEE lowering (v_cvt_f32_f64_matches_ieee754) is kept until the console's mode is checked"]
fn v_cvt_f32_f64_rounds_to_nearest_even() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x0f)
    let shader = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    let tie_even = 1.0f64 + f64::from(f32::EPSILON) / 2.0; // ties to 1.0
    let tie_odd = 1.0f64 + 1.5 * f64::from(f32::EPSILON); // ties up to 1 + 2 ulp
    let values = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        std::f64::consts::PI,
        1.0e-3,
        123_456_789.123,
        tie_even,
        tie_odd,
        1.0 + f64::from(f32::EPSILON) * 0.75,
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_000_1, // rounds to infinity
        1.0e300,
        -1.0e300,
        1.0e-40, // a float denormal: flushed
        -1.0e-300,
        f64::from(f32::MIN_POSITIVE),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let x = values[i];
        if x.is_nan() {
            assert!(f32::from_bits(got).is_nan(), "{backend}: cvt({x}) = {got:#010x}, want NaN");
            return;
        }
        let want = cvt_f32_f64_reference(x);
        assert_eq!(got, want, "{backend}: cvt({x:e}) = {got:#010x}, want {want:#010x}");
    });
}

/// The lane `v_permlane16_b32` (`cross` false) or `v_permlanex16_b32` reads
/// for `lane`, with selector dwords `s1` (lanes 0-7 of a row) and `s2`.
fn permlane_source(lane: u32, s1: u32, s2: u32, cross: bool) -> u32 {
    let sel = u64::from(s1) | u64::from(s2) << 32;
    let pick = (sel >> (4 * (lane & 15)) & 15) as u32;
    let row = if cross {
        (lane & !31) | ((lane & 16) ^ 16)
    } else {
        lane & !15
    };
    row | pick
}

#[test]
fn v_permlane16_and_x16_read_the_lanes_their_selectors_pick() {
    // 32 lanes, each with its lane number in v0; v1 = permute(v0, s0, s1);
    // lane `read` of v1 into s12, then into v0 for the store.
    let selectors: [(u32, u32); 3] = [
        (0x7654_3210, 0xfedc_ba98), // identity within the row
        (0x89ab_cdef, 0x0123_4567), // reversed
        (0x1111_1111, 0xf0f0_f0f0), // repeats
    ];
    for (op, cross) in [(0x377u16, false), (0x378, true)] {
        for read in [0u8, 5, 15, 16, 22, 31] {
            let mut body = Vec::new();
            body.extend_from_slice(&vop3(op, 1, vgpr(0), sgpr(0), sgpr(1)));
            // v_readlane_b32 s12, v1, read (inline constant 128 + read)
            body.extend_from_slice(&vop3(0x360, 12, vgpr(1), 128 + u16::from(read), 0));
            body.push(vop1(0x001, 0, sgpr(12))); // v_mov_b32 v0, s12
            let shader = build_cs(&body, 32, 0);
            let inputs: Vec<[u32; 4]> = selectors.iter().map(|&(a, b)| [a, b, 0, 0]).collect();
            // Cross-lane code declares WaveSize(32), which WARP's pipeline
            // creation refuses: hardware only.
            run_cases_on(&shader, &inputs, &[("hardware", false)], |backend, i, inp, got| {
                let want = permlane_source(u32::from(read), inp[0], inp[1], cross);
                assert_eq!(
                    got, want,
                    "{backend}: op {op:#x} selectors {:08x?}, lane {read}",
                    selectors[i]
                );
            });
        }
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }

    // f16: -0 = 0x8000, +0 = 0x0000. Low 16 bits hold the f16 result.
    let (pz16, nz16) = (0x0000_u32, 0x8000_u32);
    let cases16 = [
        [nz16, pz16, nz16, 0],
        [pz16, nz16, nz16, 0],
        [nz16, pz16, pz16, 0],
        [pz16, nz16, pz16, 0],
    ];
    let ops16 = [
        ("v_max_f16", 0x139, pz16),
        ("v_min_f16", 0x13A, nz16),
        ("v_max3_f16", 0x354, pz16),
        ("v_min3_f16", 0x351, nz16),
    ];
    for (name, op, want) in ops16 {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases16, |backend, _, inp, got| {
            assert_eq!(got & 0xFFFF, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

/// `v_div_fixup_f16` as the RDNA2 ISA lists it; `None` for a NaN.
fn div_fixup_f16_reference(q: u16, den: u16, num: u16) -> Option<u16> {
    let nan = |x: u16| is_f16_nan(x);
    let abs = |x: u16| x & 0x7FFF;
    let sign = (den ^ num) & 0x8000;
    if nan(num) || nan(den) {
        return None;
    }
    if (abs(den) == 0 && abs(num) == 0) || (abs(den) == 0x7C00 && abs(num) == 0x7C00) {
        return None;
    }
    if abs(den) == 0 || abs(num) == 0x7C00 {
        return Some(sign | 0x7C00);
    }
    if abs(den) == 0x7C00 || abs(num) == 0 {
        return Some(sign);
    }
    // An all-ones exponent overflows, so a NaN quotient gives infinity
    // (gfx1036 hardware).
    if abs(q) >= 0x7C00 {
        return Some(sign | 0x7C00);
    }
    Some(sign | abs(q))
}

#[test]
fn v_div_fixup_f16_follows_the_isa_special_cases() {
    // v_mov_b32 v0, 0; v_div_fixup_f16 v0, v1, v2, v3 (VOP3 opcode 0x35f):
    // v1 the quotient, v2 the denominator, v3 the numerator.
    let mut body = vec![vop1(0x001, 0, 128)];
    body.extend_from_slice(&vop3(0x35f, 0, vgpr(1), vgpr(2), vgpr(3)));
    let shader = build_valu_cs(&body);
    // 0, -0, 1, -1.5, 2^-24, max finite, inf, -inf, NaN, -NaN.
    let values = [
        0x0000u16, 0x8000, 0x3C00, 0xBE00, 0x0001, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01,
    ];
    let mut cases = Vec::new();
    for &q in &[0x4248u16, 0xC248, 0x7E00, 0x7C01] {
        for &den in &values {
            for &num in &values {
                cases.push([u32::from(q), u32::from(den), u32::from(num), 0]);
            }
        }
    }
    run_cases(&shader, &cases, |backend, _, inp, got| {
        let (q, den, num) = (inp[0] as u16, inp[1] as u16, inp[2] as u16);
        let got_h = got as u16;
        assert_eq!(got >> 16, 0, "{backend}: the high half changed");
        match div_fixup_f16_reference(q, den, num) {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_div_fixup_f16({q:#06x}, {den:#06x}, {num:#06x}) = {got_h:#06x}, \
                 want {want:#06x}",
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_div_fixup_f16({q:#06x}, {den:#06x}, {num:#06x}) = {got_h:#06x}, \
                 want NaN",
            ),
        }
    });
}

/// 64-bit integer compares order by the high dword (signed for i64), then
/// by the low one. AMD's D3D12 compiler runs an ordered i64 `icmp` as if the
/// dwords were swapped (RX 9070 and gfx1036 alike), so the lowering compares
/// the dwords; the values are picked so that the two orders disagree.
#[test]
fn v_cmp_u64_and_i64_order_by_the_high_dword() {
    type Order = fn(u64, u64) -> bool;
    let ops: [(u16, &str, Order); 12] = [
        (0x0a1, "v_cmp_lt_i64", |a, b| (a as i64) < (b as i64)),
        (0x0a2, "v_cmp_eq_i64", |a, b| a == b),
        (0x0a3, "v_cmp_le_i64", |a, b| (a as i64) <= (b as i64)),
        (0x0a4, "v_cmp_gt_i64", |a, b| (a as i64) > (b as i64)),
        (0x0a5, "v_cmp_ne_i64", |a, b| a != b),
        (0x0a6, "v_cmp_ge_i64", |a, b| (a as i64) >= (b as i64)),
        (0x0e1, "v_cmp_lt_u64", |a, b| a < b),
        (0x0e2, "v_cmp_eq_u64", |a, b| a == b),
        (0x0e3, "v_cmp_le_u64", |a, b| a <= b),
        (0x0e4, "v_cmp_gt_u64", |a, b| a > b),
        (0x0e5, "v_cmp_ne_u64", |a, b| a != b),
        (0x0e6, "v_cmp_ge_u64", |a, b| a >= b),
    ];
    let values: [u64; 10] = [
        0,
        1,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0x1_FFFF_FFFF,
        0x0000_0040_C010_FBFF,
        0x0001_8000_0000_0040,
        0x7FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0001,
        0xFFFF_FFFF_0000_0000,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .flat_map(|&a| values.iter().map(move |&b| (a, b)))
        .map(|(a, b)| [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32])
        .collect();
    for (op, name, order) in ops {
        // op s20, v[1:2], v[3:4] and, sharing v2, op s20, v[1:2], v[2:3];
        // v_cndmask_b32_e64 v0, 0, 1, s20
        for second in [3, 2] {
            let mut body = vop3(op, 20, vgpr(1), vgpr(second), 0x80).to_vec();
            body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
            let shader = build_valu_cs(&body);
            let at = usize::from(second) - 1;
            run_cases(&shader, &inputs, |backend, _, inp, got| {
                let a = u64::from(inp[0]) | u64::from(inp[1]) << 32;
                let b = u64::from(inp[at]) | u64::from(inp[at + 1]) << 32;
                let want = u32::from(order(a, b));
                assert_eq!(got, want, "{backend}: {name}({a:#018x}, {b:#018x})");
            });
        }
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// An f32 result: exactly these bits, any NaN, or within the 1 ULP the
/// RDNA2 ISA gives the transcendental ops (same sign, finite).
#[derive(Clone, Copy, Debug)]
enum F32Want {
    Bits(u32),
    Nan,
    Ulp1(f32),
}

impl F32Want {
    fn holds(self, got: u32) -> bool {
        match self {
            Self::Bits(b) => got == b,
            Self::Nan => f32::from_bits(got).is_nan(),
            Self::Ulp1(v) => {
                let v = v.to_bits();
                f32::from_bits(got).is_finite() && (got ^ v) >> 31 == 0 && got.abs_diff(v) <= 1
            }
        }
    }
}

#[test]
fn rcp_rsq_sqrt_log_exp_f32_special_values_match_rdna2() {
    use F32Want::{Bits, Nan, Ulp1};
    let f = f32::to_bits;
    let (pinf, ninf, nz, nan) = (0x7F80_0000, 0xFF80_0000, 0x8000_0000, 0x7FC0_0000);
    let rcp: &[(u32, F32Want)] = &[
        (0, Bits(pinf)),
        (nz, Bits(ninf)),
        (pinf, Bits(0)),
        (ninf, Bits(nz)),
        (nan, Nan),
        (f(2.0), Ulp1(0.5)),
        (f(-4.0), Ulp1(-0.25)),
        (f(3.0), Ulp1(1.0 / 3.0)),
        (f(1e-30), Ulp1((1.0 / f64::from(1e-30f32)) as f32)),
    ];
    // (VOP1 opcode, name, cases); v_log_f32 is log2, v_exp_f32 is 2^x.
    type Cases<'a> = &'a [(u32, F32Want)];
    let ops: &[(u16, &str, Cases)] = &[
        (0x02a, "v_rcp_f32", rcp),
        (0x02b, "v_rcp_iflag_f32", rcp),
        (
            0x02e,
            "v_rsq_f32",
            &[
                (0, Bits(pinf)),
                (nz, Bits(ninf)),
                (pinf, Bits(0)),
                (ninf, Nan),
                (f(-1.0), Nan),
                (nan, Nan),
                (f(4.0), Ulp1(0.5)),
                (f(0.25), Ulp1(2.0)),
                (f(2.0), Ulp1(std::f32::consts::FRAC_1_SQRT_2)),
            ],
        ),
        (
            0x033,
            "v_sqrt_f32",
            &[
                (0, Bits(0)),
                (nz, Bits(nz)),
                (pinf, Bits(pinf)),
                (ninf, Nan),
                (f(-1.0), Nan),
                (f(-1e-30), Nan),
                (nan, Nan),
                (f(4.0), Ulp1(2.0)),
                (f(2.0), Ulp1(std::f32::consts::SQRT_2)),
                (f(1e30), Ulp1(f64::from(1e30f32).sqrt() as f32)),
            ],
        ),
        (
            0x027,
            "v_log_f32",
            &[
                (0, Bits(ninf)),
                (nz, Bits(ninf)),
                (pinf, Bits(pinf)),
                (ninf, Nan),
                (f(-1.0), Nan),
                (nan, Nan),
                (f(1.0), Ulp1(0.0)),
                (f(8.0), Ulp1(3.0)),
                (f(0.125), Ulp1(-3.0)),
                (f(3.0), Ulp1(3f64.log2() as f32)),
            ],
        ),
        (
            0x025,
            "v_exp_f32",
            &[
                (ninf, Bits(0)),
                (pinf, Bits(pinf)),
                (nan, Nan),
                (0, Ulp1(1.0)),
                (nz, Ulp1(1.0)),
                (f(3.0), Ulp1(8.0)),
                (f(-3.0), Ulp1(0.125)),
                (f(0.5), Ulp1(std::f32::consts::SQRT_2)),
                (f(128.0), Bits(pinf)),
                (f(-200.0), Bits(0)),
            ],
        ),
    ];
    for &(op, name, cases) in ops {
        let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert!(
                want.holds(got),
                "{backend}: {name}({:#010x} = {:?}) = {got:#010x} ({:?}), want {want:?}",
                inp[0],
                f32::from_bits(inp[0]),
                f32::from_bits(got),
            );
        });
    }
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_cvt_i32_and_u32_f32_truncate_saturate_and_take_nan_as_zero() {
    // RDNA2: toward zero, out-of-range values (infinities too) saturate, NaN
    // gives 0 – what Rust's `as` does.
    let f = f32::to_bits;
    let inputs = [
        0,
        0x8000_0000, // -0
        f(1.5),
        f(-1.5),
        f(-0.75),
        f(0.999_999_94),
        f(16_777_215.0),
        f(2_147_483_520.0), // largest f32 below 2^31
        f(2_147_483_648.0), // 2^31
        f(-2_147_483_648.0),
        f(-2_147_483_904.0), // next f32 below -2^31
        f(4_294_967_040.0),  // largest f32 below 2^32
        f(4_294_967_296.0),  // 2^32
        f(1e30),
        f(-1e30),
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0000, // NaN
        0xFFC0_0000, // -NaN
        0x7F80_0001, // signalling NaN
        0x0000_0001, // smallest denormal
    ];
    let cases: Vec<[u32; 4]> = inputs.iter().map(|&x| [x, 0, 0, 0]).collect();
    for (op, name) in [(0x008, "v_cvt_i32_f32"), (0x007, "v_cvt_u32_f32")] {
        let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let x = f32::from_bits(inp[0]);
            let want = if op == 0x008 {
                x as i32 as u32
            } else {
                x as u32
            };
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x} = {x:?}) = {got:#010x}, want {want:#010x}",
                inp[0],
            );
        });
    }
}

#[test]
fn v_cvt_f32_i32_u32_and_ubyte_round_to_nearest_even() {
    // Integers past 2^24 round to nearest even (the MODE default), as Rust's
    // `as` does; the ubyte forms convert one byte of the source.
    let inputs = [
        0,
        1,
        0xFF,
        0x0100_0001, // 2^24 + 1: tie, rounds down to even
        0x0100_0003, // 2^24 + 3: tie, rounds up to even
        0x0100_0005,
        0x7FFF_FFBF, // just below the halfway point to 2^31
        0x7FFF_FFC0, // halfway, rounds to 2^31
        0x7FFF_FFFF,
        0x8000_0000,
        0x8000_0001,
        0xFFFF_FF7F,
        0xFFFF_FF80, // u32: halfway, rounds to 2^32
        0xFFFF_FFFF,
        0x80FF_7F01,
        0x12AB_00FE,
    ];
    let ops = [
        (0x005, "v_cvt_f32_i32"),
        (0x006, "v_cvt_f32_u32"),
        (0x011, "v_cvt_f32_ubyte0"),
        (0x012, "v_cvt_f32_ubyte1"),
        (0x013, "v_cvt_f32_ubyte2"),
        (0x014, "v_cvt_f32_ubyte3"),
    ];
    let cases: Vec<[u32; 4]> = inputs.iter().map(|&x| [x, 0, 0, 0]).collect();
    for (op, name) in ops {
        let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let x = inp[0];
            let want = match op {
                0x005 => x as i32 as f32,
                0x006 => x as f32,
                _ => f32::from((x >> (8 * (op - 0x011))) as u8),
            }
            .to_bits();
            // WARP's uitofp is not correctly rounded above 2^31 (it gave
            // 2^32 for 0xffffff7f); the hardware result must be exact.
            if backend == "warp" && op == 0x006 && x >= 1 << 31 && got != want {
                assert!(
                    got.abs_diff(want) <= 1,
                    "warp: {name}({x:#010x}) = {got:#010x}"
                );
                eprintln!("warp: {name}({x:#010x}) = {got:#010x}, RDNA2 gives {want:#010x}");
                return;
            }
            assert_eq!(
                got,
                want,
                "{backend}: {name}({:#010x}) = {got:#010x} ({:?}), want {want:#010x} ({:?})",
                inp[0],
                f32::from_bits(got),
                f32::from_bits(want),
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    // s_bfe_u32 s6, s0, s1 (SOP2 opcode 0x27); v_mov_b32 v0, s6
    let s_unsigned = build_valu_cs(&[sop2(0x27, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // s_bfe_u32 width is 7-bit (src1[22:16]), so width=32 keeps all remaining bits without sign extension.
    let mut su_cases = unsigned_cases.to_vec();
    su_cases.push(([0x8000_0000, 4, 32, 0], 0x0800_0000));
    let su_in: Vec<[u32; 4]> = su_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_unsigned, &su_in, |backend, i, inp, got| {
        let want = su_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

/// SOPC `op ssrc0, ssrc1`.
const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SOPK `op sdst, imm16`.
const fn sopk(op: u8, sdst: u8, imm: u16) -> u32 {
    0xB000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | imm as u32
}

/// `s_cselect_b32 s6, 1, 0` (SCC to a dword), then `v_mov_b32 v0, s6`.
fn scc_to_v0() -> [u32; 2] {
    [sop2(0x0a, 6, 129, 128), vop1(0x001, 0, sgpr(6))]
}

#[test]
fn scalar_compares_set_scc_as_the_rdna2_document_defines() {
    // SOPC compares of s0 with s1, the bit tests, and SOPK compares of s0
    // with a 16-bit immediate (sign-extended for the signed forms,
    // zero-extended for the unsigned ones).
    #[allow(clippy::type_complexity)]
    let conditions: &[(&str, u8, fn(u32, u32) -> bool)] = &[
        ("s_cmp_eq_i32", 0x00, |a, b| a == b),
        ("s_cmp_lg_i32", 0x01, |a, b| a != b),
        ("s_cmp_gt_i32", 0x02, |a, b| (a as i32) > (b as i32)),
        ("s_cmp_ge_i32", 0x03, |a, b| (a as i32) >= (b as i32)),
        ("s_cmp_lt_i32", 0x04, |a, b| (a as i32) < (b as i32)),
        ("s_cmp_le_i32", 0x05, |a, b| (a as i32) <= (b as i32)),
        ("s_cmp_eq_u32", 0x06, |a, b| a == b),
        ("s_cmp_lg_u32", 0x07, |a, b| a != b),
        ("s_cmp_gt_u32", 0x08, |a, b| a > b),
        ("s_cmp_ge_u32", 0x09, |a, b| a >= b),
        ("s_cmp_lt_u32", 0x0a, |a, b| a < b),
        ("s_cmp_le_u32", 0x0b, |a, b| a <= b),
        // Bit (s1 & 31) of s0 is clear / set.
        ("s_bitcmp0_b32", 0x0c, |a, b| a >> (b & 31) & 1 == 0),
        ("s_bitcmp1_b32", 0x0d, |a, b| a >> (b & 31) & 1 == 1),
    ];
    let values = [
        0u32,
        1,
        2,
        31,
        32,
        0x7fff_fffe,
        0x7fff_ffff,
        0x8000_0000,
        0x8000_0001,
        0xffff_fffe,
        0xffff_ffff,
    ];
    let pairs: Vec<[u32; 4]> = values
        .iter()
        .flat_map(|&a| values.iter().map(move |&b| [a, b, 0, 0]))
        .collect();
    let mut failures = Vec::new();
    for &(name, op, reference) in conditions {
        let [c, m] = scc_to_v0();
        let shader = build_valu_cs(&[sopc(op, 0, 1), c, m]);
        run_cases(&shader, &pairs, |backend, _, inp, got| {
            let want = u32::from(reference(inp[0], inp[1]));
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#x}, {:#x}) = {got}, want {want}",
                    inp[0], inp[1]
                ));
            }
        });
    }
    #[allow(clippy::type_complexity)]
    let immediates: &[(&str, u8, fn(u32, u16) -> bool)] = &[
        ("s_cmpk_eq_i32", 0x03, |a, k| {
            a as i32 == i32::from(k as i16)
        }),
        ("s_cmpk_lg_i32", 0x04, |a, k| {
            a as i32 != i32::from(k as i16)
        }),
        ("s_cmpk_gt_i32", 0x05, |a, k| a as i32 > i32::from(k as i16)),
        ("s_cmpk_ge_i32", 0x06, |a, k| {
            a as i32 >= i32::from(k as i16)
        }),
        ("s_cmpk_lt_i32", 0x07, |a, k| {
            (a as i32) < i32::from(k as i16)
        }),
        ("s_cmpk_le_i32", 0x08, |a, k| {
            a as i32 <= i32::from(k as i16)
        }),
        ("s_cmpk_eq_u32", 0x09, |a, k| a == u32::from(k)),
        ("s_cmpk_lg_u32", 0x0a, |a, k| a != u32::from(k)),
        ("s_cmpk_gt_u32", 0x0b, |a, k| a > u32::from(k)),
        ("s_cmpk_ge_u32", 0x0c, |a, k| a >= u32::from(k)),
        ("s_cmpk_lt_u32", 0x0d, |a, k| a < u32::from(k)),
        ("s_cmpk_le_u32", 0x0e, |a, k| a <= u32::from(k)),
    ];
    let singles: Vec<[u32; 4]> = values.iter().map(|&a| [a, 0, 0, 0]).collect();
    for &(name, op, reference) in immediates {
        for k in [0u16, 1, 0x7fff, 0x8000, 0xffff] {
            let [c, m] = scc_to_v0();
            let shader = build_valu_cs(&[sopk(op, 0, k), c, m]);
            run_cases(&shader, &singles, |backend, _, inp, got| {
                let want = u32::from(reference(inp[0], k));
                if got != want && failures.len() < 40 {
                    failures.push(format!(
                        "{backend}: {name}({:#x}, {k:#x}) = {got}, want {want}",
                        inp[0]
                    ));
                }
            });
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// SOP1 `op sdst, ssrc0`.
const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

#[test]
fn scalar_64_bit_operations_match_the_rdna2_definitions() {
    // The pair s[0:1] and s[2:3] as 64-bit operands (and s2 as a count), the
    // result in s[6:7], read back as the low and then the high dword.
    #[allow(clippy::type_complexity)]
    let operations: &[(&str, u32, fn(u64, u64) -> u64)] = &[
        ("s_and_b64", sop2(0x0f, 6, 0, 2), |a, b| a & b),
        ("s_or_b64", sop2(0x11, 6, 0, 2), |a, b| a | b),
        ("s_xor_b64", sop2(0x13, 6, 0, 2), |a, b| a ^ b),
        ("s_andn2_b64", sop2(0x15, 6, 0, 2), |a, b| a & !b),
        ("s_orn2_b64", sop2(0x17, 6, 0, 2), |a, b| a | !b),
        ("s_nand_b64", sop2(0x19, 6, 0, 2), |a, b| !(a & b)),
        ("s_nor_b64", sop2(0x1b, 6, 0, 2), |a, b| !(a | b)),
        ("s_xnor_b64", sop2(0x1d, 6, 0, 2), |a, b| !(a ^ b)),
        // The shift count is the low six bits of the second operand's low dword.
        ("s_lshl_b64", sop2(0x1f, 6, 0, 2), |a, b| a << (b & 63)),
        ("s_lshr_b64", sop2(0x21, 6, 0, 2), |a, b| a >> (b & 63)),
        ("s_ashr_i64", sop2(0x23, 6, 0, 2), |a, b| {
            ((a as i64) >> (b & 63)) as u64
        }),
        ("s_not_b64", sop1(0x08, 6, 0), |a, _| !a),
        ("s_brev_b64", sop1(0x0c, 6, 0), |a, _| a.reverse_bits()),
        // Counts and finds: one dword in s6 (-1 for none), the high one unused.
        ("s_bcnt1_i32_b64", sop1(0x10, 6, 0), |a, _| u64::from(a.count_ones())),
        ("s_ff1_i32_b64", sop1(0x14, 6, 0), |a, _| {
            if a == 0 {
                0xffff_ffff
            } else {
                u64::from(a.trailing_zeros())
            }
        }),
    ];
    let halves = [0u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x1234_5678];
    let mut cases = Vec::new();
    for &a_lo in &halves {
        for &a_hi in &halves {
            for &(b_lo, b_hi) in &[
                (0u32, 0u32),
                (1, 0),
                (33, 7),
                (63, 0xffff_ffff),
                (0xffff_ffc1, 0x8000_0000),
            ] {
                cases.push([a_lo, a_hi, b_lo, b_hi]);
            }
        }
    }
    let mut failures = Vec::new();
    for &(name, word, reference) in operations {
        let low = build_valu_cs(&[word, vop1(0x001, 0, sgpr(6))]);
        let high = build_valu_cs(&[word, vop1(0x001, 0, sgpr(7))]);
        // The counts and finds write one dword.
        let one_dword = name.contains("_i32_b64");
        for (half, shader) in [("low", &low), ("high", &high)] {
            if one_dword && half == "high" {
                continue;
            }
            run_cases(shader, &cases, |backend, _, inp, got| {
                let a = u64::from(inp[1]) << 32 | u64::from(inp[0]);
                let b = u64::from(inp[3]) << 32 | u64::from(inp[2]);
                let want = reference(a, b);
                let want = if half == "low" {
                    want as u32
                } else {
                    (want >> 32) as u32
                };
                if got != want && failures.len() < 40 {
                    failures.push(format!(
                        "{backend}: {name}({a:#x}, {b:#x}) {half} dword = {got:#x}, want {want:#x}"
                    ));
                }
            });
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    let mul2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    let mul4 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    let div2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    let div2_clamp = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000, -0.5=0xB800
    run_cases(
        &mul2,
        &[[0x3400, 0x3800, 0, 0]], // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 mul:2");
        },
    );
    run_cases(
        &mul4,
        &[[0x3400, 0x3400, 0, 0]], // (0.25 + 0.25) * 4 = 2.0 (0x4000)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x4000, "{backend}: v_add_f16 mul:4");
        },
    );
    run_cases(
        &div2,
        &[[0x3E00, 0x3E00, 0, 0]], // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 div:2");
        },
    );
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00),
    // whereas clamp before div:2 would give 0.5 (0x3800).
    let clamp_cases: &[(u16, u16, u16)] = &[
        (0x3A00, 0x3A00, 0x3A00), // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        (0x3E00, 0x3E00, 0x3C00), // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        (0xB800, 0xB800, 0x0000), // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    ];
    let inputs: Vec<[u32; 4]> = clamp_cases
        .iter()
        .map(|&(a, b, _)| [u32::from(a), u32::from(b), 0, 0])
        .collect();
    run_cases(&div2_clamp, &inputs, |backend, i, _, got| {
        let want = clamp_cases[i].2;
        assert_eq!(
            got as u16, want,
            "{backend}: v_add_f16 div:2 clamp case {i}: got {:#06x}, want {want:#06x}",
            got as u16,
        );
    });
}

#[test]
fn cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    let cvt_u16 = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_u16_abs = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    let cvt_i16 = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_i16_neg = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    let u16_cases: &[(u16, u16)] = &[
        (0xBC00, 0),      // -1.0 -> 0
        (0xFC00, 0),      // -inf -> 0
        (0x7E00, 0),      // NaN -> 0
        (0x57BC, 123),    // 123.75 -> 123
        (0x7BFF, 65504),  // 65504.0 (max finite f16) -> 65504
        (0x7C00, 0xFFFF), // +inf -> 65535 (saturated)
    ];
    let u16_in: Vec<[u32; 4]> = u16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_u16, &u16_in, |backend, i, inp, got| {
        let want = u16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_u16_abs,
        &[[0xD7BC, 0, 0, 0]], // |-123.75| -> 123
        |backend, _, _, got| {
            assert_eq!(got as u16, 123, "{backend}: v_cvt_u16_f16 |src0|");
        },
    );

    let i16_cases: &[(u16, u16)] = &[
        (0xFC00, 0x8000),       // -inf -> -32768
        (0xF8E2, 0x8000),       // -40000.0 -> -32768 (saturated, not wrapped)
        (0xD7BC, (-123i16) as u16), // -123.75 -> -123
        (0x7E00, 0),            // NaN -> 0
        (0x57BC, 123),          // 123.75 -> 123
        (0x78E2, 0x7FFF),       // +40000.0 -> 32767 (saturated, not wrapped)
        (0x7C00, 0x7FFF),       // +inf -> 32767
    ];
    let i16_in: Vec<[u32; 4]> = i16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_i16, &i16_in, |backend, i, inp, got| {
        let want = i16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_i16_neg,
        &[[0x57BC, 0, 0, 0], [0x78E2, 0, 0, 0]], // -(123.75) -> -123; -(40000.0) -> -32768
        |backend, i, _, got| {
            let want = [(-123i16) as u16, 0x8000][i];
            assert_eq!(got as u16, want, "{backend}: v_cvt_i16_f16 -src0 case {i}");
        },
    );
}

const S_BARRIER: u32 = 0xBF8A_0000;

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_MIN_F32: u8 = 0x12;
const DS_MAX_F32: u8 = 0x13;
const DS_MIN_RTN_F32: u8 = 0x32;
const DS_MAX_RTN_F32: u8 = 0x33;
const DS_READ_B32: u8 = 0x36;

/// What RDNA2 writes for `ds_max_f32` / `ds_min_f32` over a dword holding
/// `old` with `data` (bits), as measured on gfx1036 with f32 denormals kept:
/// an sNaN, data first, then memory, is written quieted; otherwise a NaN
/// loses to a number (two quiet NaNs give the data); numbers compare as
/// floats with -0 below +0.
fn lds_min_max_ref(old: u32, data: u32, max: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let snan = |x: u32| nan(x) && x & 0x40_0000 == 0;
    let key = |x: u32| {
        let b = x as i32;
        if b < 0 { b ^ 0x7FFF_FFFF } else { b }
    };
    if snan(data) {
        return data | 0x40_0000;
    }
    if snan(old) {
        return old | 0x40_0000;
    }
    match (nan(data), nan(old)) {
        (true, true) | (false, true) => return data,
        (true, false) => return old,
        (false, false) => {}
    }
    let take = if max {
        key(data) > key(old)
    } else {
        key(data) < key(old)
    };
    if take { data } else { old }
}

#[test]
fn ds_min_max_f32_order_floats_and_return_the_old_value() {
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        1e-30,
        -1e-30,
        f32::from_bits(1), // denormal
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    // Quiet and signalling NaNs of both signs, in memory and as data.
    let nans = [
        0x7FC0_0000u32,
        0xFFC0_0000,
        0x7F80_0001,
        0xFFA0_0000,
        0x7FBF_FFFF,
    ];
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).chain(nans).collect();
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &old in &bits {
        for &data in &bits {
            cases.push([old, data, 0, 0]);
        }
    }
    for (rtn, plain, max, what) in [
        (DS_MAX_RTN_F32, DS_MAX_F32, true, "ds_max"),
        (DS_MIN_RTN_F32, DS_MIN_F32, false, "ds_min"),
    ] {
        // v5 = 0; LDS[v5] = v1; the atomic with v2; v0 = the old value
        // (rtn) or the dword afterwards (read back).
        let mut returns_old = vec![vop1(0x001, 5, 0x80)];
        returns_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_old.extend(ds(rtn, 0, 5, 2));
        let mut returns_new = vec![vop1(0x001, 5, 0x80)];
        returns_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_new.extend(ds(plain, 0, 5, 2));
        returns_new.extend(ds(DS_READ_B32, 0, 5, 0));
        for (body, old_result) in [(returns_old, true), (returns_new, false)] {
            // One thread with LDS declares WaveSize(32), which WARP's
            // pipeline creation refuses: hardware only (the 64-thread test
            // below runs on WARP too).
            let shader = build_cs(&body, 1, 512);
            run_cases_on(
                &shader,
                &cases,
                &[("hardware", false)],
                |backend, _, inp, got| {
                    let (old, data) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
                    let want = if old_result {
                        inp[0]
                    } else {
                        lds_min_max_ref(inp[0], inp[1], max)
                    };
                    assert_eq!(
                        got,
                        want,
                        "{backend}: {what}_f32 on {old:e} [{:#010x}] with {data:e} [{:#010x}] gave {got:#010x}, want {want:#010x} ({})",
                        inp[0],
                        inp[1],
                        if old_result {
                            "old value"
                        } else {
                            "dword after"
                        },
                    );
                },
            );
        }
    }
}

#[test]
fn ds_min_max_f32_are_atomic_across_waves() {
    // 64 threads (two host waves of 32 or one of 64): LDS[0] = v1, then
    // every thread applies (thread - 32) as a float; v0 = LDS[0].
    for (op, max, init, want) in [
        (DS_MAX_F32, true, f32::NEG_INFINITY, 31.0f32),
        (DS_MAX_F32, true, 100.0, 100.0),
        (DS_MIN_F32, false, f32::INFINITY, -32.0),
        (DS_MIN_F32, false, -0.0, -32.0),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(ds(DS_WRITE_B32, 0, 5, 1));
        body.push(S_BARRIER);
        body.push(vop2(0x26, 6, vgpr(0), 3)); // v_sub_nc_u32 v6, v0, v3
        body.push(vop1(0x005, 6, vgpr(6))); // v_cvt_f32_i32
        body.extend(ds(op, 0, 5, 6));
        body.push(S_BARRIER);
        body.extend(ds(DS_READ_B32, 0, 5, 0));
        let shader = build_cs(&body, 64, 512);
        run_cases(
            &shader,
            &[[init.to_bits(), 0, 32, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    f32::from_bits(got),
                    want,
                    "{backend}: {} from {init} over 64 threads",
                    if max { "max" } else { "min" },
                );
            },
        );
    }
}

/// DS `op vdst, addr, data0, data1` at offset 0.
const fn ds2(op: u8, vdst: u8, addr: u8, data0: u8, data1: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data1 as u32) << 16 | (data0 as u32) << 8 | addr as u32,
    ]
}

/// The 32-bit LDS atomics against the instruction set document: for each,
/// the new content of the dword and (the `_rtn` form) the old content it
/// returns, from an initial value and two operands. One thread, one dword.
#[test]
fn lds_atomics_match_the_rdna2_definitions() {
    // (name, opcode of the form without a result, reference of the new
    // content from the old one and the two operands). The `_rtn` opcode is
    // 0x20 above.
    #[allow(clippy::type_complexity)]
    let operations: &[(&str, u8, fn(u32, u32, u32) -> u32)] = &[
        ("ds_add_u32", 0x00, |m, a, _| m.wrapping_add(a)),
        ("ds_sub_u32", 0x01, |m, a, _| m.wrapping_sub(a)),
        ("ds_min_i32", 0x05, |m, a, _| {
            (m as i32).min(a as i32) as u32
        }),
        ("ds_max_i32", 0x06, |m, a, _| {
            (m as i32).max(a as i32) as u32
        }),
        ("ds_min_u32", 0x07, |m, a, _| m.min(a)),
        ("ds_max_u32", 0x08, |m, a, _| m.max(a)),
        ("ds_and_b32", 0x09, |m, a, _| m & a),
        ("ds_or_b32", 0x0a, |m, a, _| m | a),
        ("ds_xor_b32", 0x0b, |m, a, _| m ^ a),
        // Stores the second operand when the content equals the first.
        ("ds_cmpst_b32", 0x10, |m, a, b| if m == a { b } else { m }),
    ];
    let values = [
        0u32,
        1,
        2,
        5,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_fffe,
        0xffff_ffff,
    ];
    let mut cases = Vec::new();
    for &m in &values {
        for &a in &values {
            for b in [0u32, 0xf0f0_f0f0] {
                cases.push([m, a, b, 0]);
            }
        }
    }
    // One thread with LDS declares WaveSize(32), which WARP's pipeline
    // creation refuses: hardware only, as the other single-thread LDS tests.
    let hardware = &[("hardware", false)];
    if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") {
        eprintln!("the hardware adapter is WARP; skipped");
        return;
    }
    let mut failures = Vec::new();
    for &(name, opcode, reference) in operations {
        // v5 = 0 (the address); LDS[0] = v1; op (v2, v3); then either the
        // result register or a read of the dword.
        let prefix = |op: u32, result: u8| {
            let mut body = vec![vop1(0x001, 5, 0x80)];
            body.extend(ds(DS_WRITE_B32, 0, 5, 1));
            body.push(S_BARRIER);
            body.extend(ds2(op as u8, result, 5, 2, 3));
            body.push(S_BARRIER);
            body
        };
        let mut new_content = prefix(u32::from(opcode), 0);
        new_content.extend(ds(DS_READ_B32, 0, 5, 0));
        let new_content = build_cs(&new_content, 1, 512);
        let old_content = build_cs(&prefix(u32::from(opcode) + 0x20, 0), 1, 512);
        run_cases_on(&new_content, &cases, hardware, |backend, _, inp, got| {
            let want = reference(inp[0], inp[1], inp[2]);
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#x}; {:#x}, {:#x}) leaves {got:#x}, want {want:#x}",
                    inp[0], inp[1], inp[2]
                ));
            }
        });
        run_cases_on(&old_content, &cases, hardware, |backend, _, inp, got| {
            if got != inp[0] && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}_rtn({:#x}; {:#x}, {:#x}) returns {got:#x}, want the old {:#x}",
                    inp[0], inp[1], inp[2], inp[0]
                ));
            }
        });
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn mul_hi_matches_rdna2_without_int64() {
    const VALUES: [u32; 12] = [
        0,
        1,
        2,
        0xFFFF,
        0x1_0000,
        0x007F_FFFF,
        0x0080_0000,
        0x00FF_FFFF,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_0001,
        0xFFFF_FFFF,
    ];
    let mut inputs = Vec::with_capacity(VALUES.len() * VALUES.len());
    for a in VALUES {
        for b in VALUES {
            inputs.push([a, b, 0, 0]);
        }
    }
    let check_sfi0 = |shader: &HostShader, label: &str| {
        let container = crate::Container::parse(&shader.dxil).expect("DXBC container");
        let features = container.part(crate::FourCC::SFI0).expect("SFI0 part");
        let flags = u64::from_le_bytes(features.data.try_into().expect("u64 SFI0"));
        assert_eq!(flags & 0x8000, 0, "{label} must not set SFI0 INT64");
    };
    type Eval = fn(u32, u32) -> u32;
    let cases: [(&str, Vec<u32>, Eval); 4] = [
        (
            "v_mul_hi_u32",
            vop3(0x16A, 0, vgpr(1), vgpr(2), 0).to_vec(),
            |a, b| ((u64::from(a) * u64::from(b)) >> 32) as u32,
        ),
        (
            "v_mul_hi_i32",
            vop3(0x16C, 0, vgpr(1), vgpr(2), 0).to_vec(),
            |a, b| ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32,
        ),
        (
            "s_mul_hi_u32",
            vec![sop2(0x35, 16, 0, 1), vop1(0x001, 0, sgpr(16))],
            |a, b| ((u64::from(a) * u64::from(b)) >> 32) as u32,
        ),
        (
            "s_mul_hi_i32",
            vec![sop2(0x36, 16, 0, 1), vop1(0x001, 0, sgpr(16))],
            |a, b| ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32,
        ),
    ];
    for (name, body, eval) in cases {
        let shader = build_valu_cs(&body);
        check_sfi0(&shader, name);
        run_cases(&shader, &inputs, |backend, _, inp, got| {
            let want = eval(inp[0], inp[1]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1]
            );
        });
    }
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn packed_dot_products_and_clamps_execute_cleanly() {
    let [iw0, iw1] = vop3p(0x016, 0, vgpr(1), vgpr(2), vgpr(3));
    let dot4_i8_shader = build_valu_cs(&[iw0, iw1]);
    let inputs = [[0x0403_0201u32, 0x0504_0302u32, 10u32, 0]];
    run_cases(&dot4_i8_shader, &inputs, |backend, _i, _inp, got| {
        assert_eq!(got, 50, "{backend}: v_dot4_i32_i8 got {got}, want 50");
    });

    // v_mov_b32 v0, v3; v_dot4c_i32_i8 v0, v1, v2
    let d4c_copy = vop1(0x001, 0, vgpr(3));
    let d4c = (0x0Du32 << 25) | (2u32 << 9) | (vgpr(1) as u32);
    let dot4c_i8_shader = build_valu_cs(&[d4c_copy, d4c]);
    run_cases(&dot4c_i8_shader, &inputs, |backend, _i, _inp, got| {
        assert_eq!(got, 50, "{backend}: v_dot4c_i32_i8 got {got}, want 50");
    });

    let [uw0, uw1] = vop3p(0x017, 0, vgpr(1), vgpr(2), vgpr(3));
    let dot4_u8_shader = build_valu_cs(&[uw0, uw1]);
    run_cases(&dot4_u8_shader, &inputs, |backend, _i, _inp, got| {
        assert_eq!(got, 50, "{backend}: v_dot4_u32_u8 got {got}, want 50");
    });

    // opsel_hi / opsel swapping the halves of s0 on v_dot4_u32_u8 (opsel=0b001, opsel_hi=0b110)
    let mut u8_op = vop3p(0x017, 0, vgpr(1), vgpr(2), vgpr(3));
    u8_op[0] |= 1 << 11;
    u8_op[1] = (u8_op[1] & !(3 << 27)) | (2 << 27);
    let dot4_u8_opsel_shader = build_valu_cs(&u8_op);
    run_cases(&dot4_u8_opsel_shader, &inputs, |backend, _i, _inp, got| {
        assert_eq!(got, 42, "{backend}: v_dot4_u32_u8 opsel got {got}, want 42");
    });

    let [d2i_w0, d2i_w1] = vop3p(0x014, 0, vgpr(1), vgpr(2), vgpr(3));
    let dot2_i16_shader = build_valu_cs(&[d2i_w0, d2i_w1]);
    run_cases(
        &dot2_i16_shader,
        &[[0x0003_FFFEu32, 0xFFFB_0004u32, 10u32, 0]],
        |backend, _i, _inp, got| {
            assert_eq!(got, 0xFFFF_FFF3, "{backend}: v_dot2_i32_i16 got {got:#x}");
        },
    );

    let [d2u_w0, d2u_w1] = vop3p(0x015, 0, vgpr(1), vgpr(2), vgpr(3));
    let dot2_u16_shader = build_valu_cs(&[d2u_w0, d2u_w1]);
    run_cases(
        &dot2_u16_shader,
        &[[0x0003_0002u32, 0x0005_0004u32, 10u32, 0]],
        |backend, _i, _inp, got| {
            assert_eq!(got, 33, "{backend}: v_dot2_u32_u16 got {got}, want 33");
        },
    );

    let [d8u_w0, d8u_w1] = vop3p(0x019, 0, vgpr(1), vgpr(2), vgpr(3));
    let dot8_u4_shader = build_valu_cs(&[d8u_w0, d8u_w1]);
    run_cases(
        &dot8_u4_shader,
        &[[0x1111_1111u32, 0x2222_2222u32, 10u32, 0]],
        |backend, _i, _inp, got| {
            assert_eq!(got, 26, "{backend}: v_dot8_u32_u4 got {got}, want 26");
        },
    );

    let [mut d2f_w0, d2f_w1] = vop3p(0x013, 0, vgpr(1), vgpr(2), vgpr(3));
    let d2f_in = [[0x4000_3C00u32, 0x4400_4200u32, 5.0f32.to_bits(), 0]];
    let dot2_f16_shader = build_valu_cs(&[d2f_w0, d2f_w1]);
    run_cases(&dot2_f16_shader, &d2f_in, |backend, _i, _inp, got| {
        assert_eq!(got, 16.0f32.to_bits(), "{backend}: v_dot2_f32_f16 got {got:#x}");
    });
    d2f_w0 |= 1 << 15;
    let dot2_f16_clamp_shader = build_valu_cs(&[d2f_w0, d2f_w1]);
    run_cases(&dot2_f16_clamp_shader, &d2f_in, |backend, _i, _inp, got| {
        assert_eq!(got, 1.0f32.to_bits(), "{backend}: v_dot2_f32_f16 clamp got {got:#x}");
    });

    // VOP2 v_dot2c_f32_f16 v3, v1, v2 (opcode 0x02): 1.0*3.0 + 2.0*4.0 + 5.0 = 16.0
    let d2c = (0x02u32 << 25) | (3u32 << 17) | ((2u32) << 9) | (vgpr(1) as u32);
    let copy_v3_v0 = vop1(0x001, 0, vgpr(3));
    let dot2c_f16_shader = build_valu_cs(&[d2c, copy_v3_v0]);
    run_cases(&dot2c_f16_shader, &d2f_in, |backend, _i, _inp, got| {
        assert_eq!(got, 16.0f32.to_bits(), "{backend}: v_dot2c_f32_f16 got {got:#x}");
    });

    // Clamped integer dot products saturate instead of wrapping.
    let [mut cu_w0, cu_w1] = vop3p(0x017, 0, vgpr(1), vgpr(2), vgpr(3));
    cu_w0 |= 1 << 15;
    let clamp_u8_shader = build_valu_cs(&[cu_w0, cu_w1]);
    run_cases(
        &clamp_u8_shader,
        &[[0xFFFF_FFFFu32, 0xFFFF_FFFFu32, 0xFFFF_FFF0u32, 0]],
        |backend, _i, _inp, got| {
            assert_eq!(got, 0xFFFF_FFFFu32, "{backend}: v_dot4_u32_u8 clamp got {got:#x}");
        },
    );

    let [mut ci_w0, ci_w1] = vop3p(0x016, 0, vgpr(1), vgpr(2), vgpr(3));
    ci_w0 |= 1 << 15;
    let clamp_i8_shader = build_valu_cs(&[ci_w0, ci_w1]);
    let ci_in = [
        [0x7F7F_7F7Fu32, 0x7F7F_7F7Fu32, 0x7FFF_FFF0u32, 0],
        [0x7F7F_7F7Fu32, 0xFFFF_FFFFu32, 0x8000_0010u32, 0],
    ];
    run_cases(&clamp_i8_shader, &ci_in, |backend, i, _inp, got| {
        let want = if i == 0 { 0x7FFF_FFFFu32 } else { 0x8000_0000u32 };
        assert_eq!(got, want, "{backend}: v_dot4_i32_i8 clamp case {i} got {got:#x}");
    });

    // v_dot2_u32_u16 with opsel picking high half of v1 for pair 0 (opsel[0] = 1, opsel_hi[0] = 0).
    let mut u16_op = vop3p(0x015, 0, vgpr(1), vgpr(2), vgpr(3));
    u16_op[0] |= 1 << 11;
    u16_op[1] &= !(1 << 27);
    let opsel_u16_shader = build_valu_cs(&u16_op);
    run_cases(
        &opsel_u16_shader,
        &[[0x0003_0002u32, 0x0005_0004u32, 10u32, 0]],
        |backend, _i, _inp, got| {
            assert_eq!(got, 32, "{backend}: v_dot2_u32_u16 opsel got {got}, want 32");
        },
    );

    // v_dot2_f32_f16 with negated accumulator (v.neg[2]=1) and negated low src0 (v.neg[0]=1, v.neg[2]=1).
    for (words, want) in [
        ([0xCC13_4000u32, 0x9C0E_0501u32], 2.0f32.to_bits()),
        ([0xCC13_4000u32, 0xBC0E_0501u32], (-4.0f32).to_bits()),
    ] {
        let shader = build_valu_cs(&words);
        run_cases(
            &shader,
            &[[0x4000_3C00u32, 0x4400_4200u32, 9.0f32.to_bits(), 0]],
            |backend, _i, _inp, got| {
                assert_eq!(got, want, "{backend}: v_dot2_f32_f16 neg {words:08x?} got {got:#x}, want {want:#x}");
            },
        );
    }

    // v_dot2_f32_f16 preserves f16 subnormals through legacyF16ToF32:
    // 0x03FF (largest f16 subnormal) + 0x0001 (smallest) each times 1.0, into zero acc.
    // The opposite-sign case is not tested here because RDNA2 V_DOT2_F32_F16 does not
    // guarantee exact cancellation when both halves are subnormal: the two chained FMAs
    // may accumulate a rounding difference, as measured on RX 7900 XTX (#808).
    let denorm_in = [[0x03FF_0001u32, 0x3C00_3C00u32, 0.0f32.to_bits(), 0]];
    run_cases(&dot2_f16_shader, &denorm_in, |backend, _i, _inp, got| {
        let want = (1024.0f32 * 2.0f32.powi(-24)).to_bits();
        assert_eq!(got, want, "{backend}: v_dot2_f32_f16 denorm got {got:#x}, want {want:#x}");
    });

    // Constant multiplicand cases across dot4 forms (guards against Arc IGC 16-bit constant miscompilation).
    let [d4i_w0, d4i_w1] = vop3p(0x016, 0, vgpr(1), vgpr(2), vgpr(3));
    let [d4u_w0, d4u_w1] = vop3p(0x017, 0, vgpr(1), vgpr(2), vgpr(3));
    let const_dot4_cases: &[(&[u32], [u32; 4], u32)] = &[
        (
            &[vop1(0x001, 2, 0xFF), 0x0000_8000, d4i_w0, d4i_w1],
            [0x0403_0201u32, 0, 10u32, 0],
            (10i32 + 2i32 * (-128i32)) as u32,
        ),
        (
            &[vop1(0x001, 2, 0xFF), 0xFFFF_8000, d4u_w0, d4u_w1],
            [0x0403_0201u32, 0, 10u32, 0],
            10 + 4 * 0xFF + 3 * 0xFF + 2 * 0x80,
        ),
        (
            &[vop1(0x001, 2, 0xFF), 0x0000_4000, vop1(0x001, 0, vgpr(3)), 0x0D << 25 | 2 << 9 | vgpr(1) as u32],
            [0x0403_0201u32, 0, 10u32, 0],
            10 + 2 * 0x40,
        ),
    ];
    for &(body, inp, want) in const_dot4_cases {
        let shader = build_valu_cs(body);
        run_cases(&shader, &[inp], |backend, _i, _inp, got| {
            assert_eq!(got, want, "{backend}: const dot4 {body:08x?} got {got:#x}, want {want:#x}");
        });
    }
}

/// `v_mad_legacy_f32` as RDNA 2 runs it: fused (`v_fma_legacy_f32`), with
/// flushed subnormal sources and the legacy zero-factor rule.
#[test]
fn legacy_mad_fuses_its_product_and_flushes_its_sources() {
    let mad = build_valu_cs(&vop3(0x140, 0, vgpr(1), vgpr(2), vgpr(3)));
    let cases = [
        // (1 + 2^-23) * (1 - 2^-23) - 1 is -2^-46 exactly: kept by one
        // rounding, lost when the product is rounded before the add.
        [0x3f80_0001, 0x3f7f_fffe, 0xbf80_0000, 0],
        // A subnormal source flushes to zero: subnormal * 1.0 + 0.0 == 0.0.
        [0x0000_0001, 0x3f80_0000, 0x0000_0000, 0],
        // A flushed subnormal factor triggers the legacy zero-factor rule with +inf.
        [0x0000_0001, 0x7f80_0000, 0x3f80_0000, 0],
        // A zero factor gives a +0 product even with an infinite factor.
        [0, 0x7f80_0000, 0x3f80_0000, 0],
    ];
    let want = [0xa880_0000u32, 0x0000_0000, 0x3f80_0000, 0x3f80_0000];
    if Host::new(false).is_err() && Host::new(true).is_err() {
        eprintln!("no SM 6.6 host or WARP device; skipped");
        return;
    }
    run_cases(&mad, &cases, |backend, i, _, got| {
        if backend == "warp" && i == 0 {
            // WARP rounds the product before the add; hardware's single
            // rounding is the reference for the fused form.
            eprintln!("{backend}: fused rounding differs on WARP; skipped");
            return;
        }
        assert_eq!(got, want[i], "{backend}: case {i}");
    });
}

/// A dispatch spread across z past 65,535 groups in x
/// (`USER_DISPATCH_SPREAD`): the program sees the group id the title asked
/// for, `(x + z * row)`, and the groups past the asked count do nothing.
/// One thread a group stores 1 at dword `group id x` of a raw UAV. Run
/// unspread (10 groups), spread with surplus (10 asked as 4 x 1 x 3) and
/// spread exactly (10 asked as 5 x 1 x 2), on the hardware adapter and
/// WARP. (After upstream VIctory pull request #2312.)
#[test]
fn a_spread_dispatch_runs_each_asked_group_once() {
    use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
    let words: Vec<u32> = [
        vec![s_mov_b32(5, 0x80)], // s5 = 0: the table address's high half
        s_load_dwordx4(8, 4).to_vec(), // s[8:11] = table entry 0 (raw UAV)
        vec![
            vop1(0x001, 1, sgpr(6)),  // v_mov_b32 v1, s6 (group id x)
            0x3404_0282,              // v_lshlrev_b32 v2, 2, v1
            vop1(0x001, 3, 129),      // v_mov_b32 v3, 1
            0xBF8C_C07F,              // s_waitcnt lgkmcnt(0)
            0x38 << 26 | 0x1C << 18 | 1 << 12, // buffer_store_dword v3, v2, s[8:11], 0 offen
            0x80 << 24 | (8 >> 2) << 16 | 3 << 8 | 2,
            S_ENDPGM,
        ],
    ]
    .concat();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 0,
        }],
        // 6 user SGPRs (s0..s5), then the group id x in s6 (TGID_X_EN).
        io: StageIo::Cs(ComputeIo { rsrc2: (6 << 1) | (1 << 7), threads: [1, 1, 1] }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
        half_views: lowered.half_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let spread = RootType::Other(super::USER_DISPATCH_SPREAD);
    let row_at = shader.user_data.iter().find(|u| u.root == spread && u.dword == 0).expect("the row word").index;
    let asked_at = shader.user_data.iter().find(|u| u.root == spread && u.dword == 1).expect("the asked word").index;
    assert_eq!(asked_at, row_at + 1, "the two words are adjacent (one root-constant argument sets both)");
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = Vec::new();
    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 {backend} device ({e}); skipped");
                continue;
            }
        };
        ran.push(backend);
        let (root, pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("{backend}: {e}"));
        // (row, asked, groups): unspread, spread with surplus, spread exactly.
        for (row, asked, groups) in [(0u32, 10u32, [10u32, 1, 1]), (4, 10, [4, 1, 3]), (5, 10, [5, 1, 2])] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            ud[row_at as usize] = row;
            ud[asked_at as usize] = if row == 0 { 0 } else { asked };
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(16)],
                        samplers: &[],
                        groups,
                    },
                )
                .unwrap_or_else(|e| panic!("{backend} row {row}: {e}"));
            let want: Vec<u32> = (0..16).map(|i| u32::from(i < asked)).collect();
            assert_eq!(out[0], want, "{backend}: row {row}, groups {groups:?}");
        }
    }
    eprintln!("ran on {ran:?}");
    assert!(!ran.is_empty(), "no device ran the test");
}

/// A program that moves a typed buffer's 16-bit channels
/// (`buffer_load_format_d16_xyzw` from the table's first view,
/// `buffer_store_format_d16_xyzw` to its second) names both places for the
/// host (`HostShader::half_views`) and moves every pattern as it is through
/// unsigned views: half floats, the not-a-number ones among them, and
/// plain 16-bit integers (the title keeps both in such buffers; the
/// network that moves its faces reads indices this way). A program that
/// moves no halves names nothing.
#[test]
fn halves_go_through_a_table_as_they_are_and_name_their_views() {
    use super::host::Stride;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_UINT;
    /// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 idxen`.
    const fn mubuf_idxen(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | 1 << 13,
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }
    const LOAD_D16_XYZW: u32 = 0x83;
    const STORE_D16_XYZW: u32 = 0x87;
    const THREADS: u32 = 32;
    // The table's second descriptor, 0x20 bytes on.
    let mut second = s_load_dwordx4(12, 4);
    second[1] |= 0x20;
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(&second);
    // v[1:2] = first[thread] (four halves), second[thread] = v[1:2].
    words.extend_from_slice(&mubuf_idxen(LOAD_D16_XYZW, 1, 0, 8));
    words.extend_from_slice(&mubuf_idxen(STORE_D16_XYZW, 1, 0, 12));
    words.push(S_ENDPGM);
    let view = |kind: ResourceType, table_offset_dwords: u32| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: table_offset_dwords / 8,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![view(ResourceType::Srv, 0), view(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    assert_eq!(lowered.half_views, [(1, 0), (1, 8)]);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
        half_views: lowered.half_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut dxil = shader.dxil.clone();
        v.validate(&mut dxil).expect("DXIL validation failed");
    }
    // A raw store through the same table: no halves, nothing named.
    assert!(build_valu_cs(&[]).half_views.is_empty());

    // Four halves a thread: numbers, both zeros, subnormals, infinities,
    // not-a-numbers of both signs, and integers past the largest number's
    // pattern, as indices are.
    const SPECIAL: [u16; 16] = [
        0x0000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0x7c01, 0x7e00, 0x7fff, 0x8000,
        0x8001, 0xbc00, 0xfc00, 0xfc01, 0xffff,
    ];
    let halves: Vec<u16> = (0..4 * THREADS as usize)
        .map(|i| match i {
            0..16 => SPECIAL[i],
            _ => (i as u16).wrapping_mul(0x0203).wrapping_add(0x7b00),
        })
        .collect();
    let data: Vec<u32> = halves
        .chunks(2)
        .map(|pair| u32::from(pair[0]) | u32::from(pair[1]) << 16)
        .collect();
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 1) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::Buffer {
                            data: &data,
                            stride: Stride::Format(DXGI_FORMAT_R16G16B16A16_UINT, 8),
                        },
                        View::HalfUintUav(THREADS),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(
            out[0], data,
            "{label}: the halves stored are the halves loaded"
        );
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// The places a program that moves halves names are the descriptors it
/// loads, not the start of the range their used-resource entry describes:
/// with one entry for a range from dword 0, a load at byte 0x20 names
/// dword 8 (a host view there in a 16-bit float format is the one shown
/// as its unsigned twin). A descriptor picked at run time (the load's
/// offset in an SGPR) names nothing and keeps the float path: which slot
/// it reads, and in which format, is known only then.
#[test]
fn half_places_are_the_descriptors_the_program_loads() {
    /// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 idxen`.
    const fn mubuf_idxen(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | 1 << 13,
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }
    const LOAD_D16_XYZW: u32 = 0x83;
    const STORE_D16_XYZW: u32 = 0x87;
    let lowered = |words: &[u32]| {
        let range = |kind: ResourceType, table_offset_dwords: u32| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: table_offset_dwords / 8,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            // One entry for a range of views from dword 0, one at 16.
            used: vec![range(ResourceType::Srv, 0), range(ResourceType::Uav, 16)],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res =
            resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
        let io = io::raise(&p, &native).expect("io::raise");
        lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower")
    };
    // `s_load_dwordx4 s[dst:dst+3], s[4:5], offset` (soffset null), or with
    // the offset in `soffset` instead.
    let at = |dst: u8, offset: u32| {
        let mut w = s_load_dwordx4(dst, 4);
        w[1] |= offset;
        w
    };
    let picked = |dst: u8, soffset: u8| {
        let mut w = s_load_dwordx4(dst, 4);
        w[1] = u32::from(soffset) << 25;
        w
    };
    let mut fixed = vec![s_mov_b32(5, 0x80)];
    fixed.extend_from_slice(&at(8, 0x20));
    fixed.extend_from_slice(&at(12, 0x40));
    fixed.extend_from_slice(&mubuf_idxen(LOAD_D16_XYZW, 1, 0, 8));
    fixed.extend_from_slice(&mubuf_idxen(STORE_D16_XYZW, 1, 0, 12));
    fixed.push(S_ENDPGM);
    assert_eq!(lowered(&fixed).half_views, [(1, 8), (1, 16)]);
    // The load's offset in s20, from the first lane's thread index
    // (`v_readfirstlane_b32 s20, v0`): picked at run time. The pick may land
    // on the store's place too, so no place of that table is named: the d16
    // accesses there take the float path, as before half views.
    let mut runtime = vec![s_mov_b32(5, 0x80), vop1(0x02, 20, vgpr(0))];
    runtime.extend_from_slice(&picked(8, 20));
    runtime.extend_from_slice(&at(12, 0x40));
    runtime.extend_from_slice(&mubuf_idxen(LOAD_D16_XYZW, 1, 0, 8));
    runtime.extend_from_slice(&mubuf_idxen(STORE_D16_XYZW, 1, 0, 12));
    runtime.push(S_ENDPGM);
    assert!(lowered(&runtime).half_views.is_empty());
}

/// A format load of whole dwords (`buffer_load_format_x`, not d16) whose
/// result the program only reads one half of as a half float
/// (`v_cvt_f32_f16`) is moving 16-bit integers: the head program reads the
/// face network's output so, from an `R16_UINT` view. It names its view for
/// the host and loads the integer: 0x3c00 comes through as 1.0. Used as a
/// whole float instead (`v_add_f32`), the load stays a float load and names
/// nothing; so does a load of halves beside a load of the same view read as
/// a whole float (a view of numbers the program also picks one half float
/// out of, as hundreds of the title's pixel programs do).
#[test]
fn a_format_load_read_as_halves_loads_the_integer() {
    use super::host::Stride;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT;
    /// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 idxen`.
    const fn mubuf_idxen(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | 1 << 13,
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }
    const LOAD_FORMAT_X: u32 = 0x00;
    const V_CVT_F32_F16: u16 = 0x0B;
    const V_ADD_F32: u8 = 0x03;
    let lowered = |body: &[u32]| {
        let view = |kind: ResourceType, table_offset_dwords: u32, stride: u32| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: table_offset_dwords / 8,
            stride,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let mut second = s_load_dwordx4(12, 4);
        second[1] |= 0x20;
        let mut words = vec![s_mov_b32(5, 0x80)];
        words.extend_from_slice(&s_load_dwordx4(8, 4));
        words.extend_from_slice(&second);
        // `body` leaves v2, stored raw.
        words.extend_from_slice(body);
        words.extend_from_slice(&buffer_store_dword(2, 12, 0));
        words.push(S_ENDPGM);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![view(ResourceType::Srv, 0, 4), view(ResourceType::Uav, 8, 0)],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res =
            resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
        let io = io::raise(&p, &native).expect("io::raise");
        let l = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        HostShader {
            stage: HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_indexed_views: res.root_indexed_views,
            half_views: l.half_views,
        }
    };
    // v1 = first[v0] (one channel).
    let load = mubuf_idxen(LOAD_FORMAT_X, 1, 0, 8);
    let half = vop1(V_CVT_F32_F16, 2, vgpr(1));
    let as_halves = lowered(&[&load[..], &[half]].concat());
    assert_eq!(as_halves.half_views, [(1, 0)]);
    let as_float = lowered(&[&load[..], &[vop2(V_ADD_F32, 2, vgpr(1), 1)]].concat());
    assert!(as_float.half_views.is_empty());
    // v3 = first[v0] too, read as a whole float: v2 = half(v1) + v3.
    let mut beside = load.to_vec();
    beside.extend_from_slice(&mubuf_idxen(LOAD_FORMAT_X, 3, 0, 8));
    beside.push(half);
    beside.push(vop2(V_ADD_F32, 2, vgpr(3), 2));
    assert!(lowered(&beside).half_views.is_empty());
    // Behind a branch, as the head program has it: where the lanes meet
    // again the loaded value goes on through a phi, and a later write of
    // v1 under EXEC reads it only to keep it in the lanes it leaves alone.
    const SAVE_EXEC: u32 = 0xBE90_037E; // s_mov_b32 s16, exec_lo
    const ONLY_LANE_0: u32 = 0x7DA4_0080; // v_cmpx_eq_u32 exec_lo, 0, v0
    const SKIP_3: u32 = 0xBF88_0003; // s_cbranch_execz over the next 3 words
    const RESTORE_EXEC: u32 = 0xBEFE_0310; // s_mov_b32 exec_lo, s16
    let zero = vop1(0x01, 2, 128); // v_mov_b32 v2, 0
    let rewrite = vop1(0x01, 1, 128); // v_mov_b32 v1, 0
    let branched = lowered(
        &[
            &[zero, SAVE_EXEC, ONLY_LANE_0, SKIP_3][..],
            &load,
            &[
                half,
                RESTORE_EXEC,
                SAVE_EXEC,
                ONLY_LANE_0,
                rewrite,
                RESTORE_EXEC,
            ],
        ]
        .concat(),
    );
    assert_eq!(branched.half_views, [(1, 0)]);
    // A d16 load names its place where it is the only access there...
    const LOAD_FORMAT_D16_X: u32 = 0x80;
    let d16 = mubuf_idxen(LOAD_FORMAT_D16_X, 1, 0, 8);
    let d16_alone = lowered(&[&d16[..], &[half]].concat());
    assert_eq!(d16_alone.half_views, [(1, 0)]);
    // ...and not beside a whole float read through the same place, which
    // would see the unsigned twin: v2 = half(d16 v1) + v3.
    let mut d16_beside = d16.to_vec();
    d16_beside.extend_from_slice(&mubuf_idxen(LOAD_FORMAT_X, 3, 0, 8));
    d16_beside.push(half);
    d16_beside.push(vop2(V_ADD_F32, 2, vgpr(3), 2));
    assert!(lowered(&d16_beside).half_views.is_empty());
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        for shader in [&as_halves, &branched, &d16_alone] {
            let mut dxil = shader.dxil.clone();
            v.validate(&mut dxil).expect("DXIL validation failed");
        }
    }
    // Element 0 of an R16_UINT buffer holds the half 1.0 (0x3c00).
    let data = [0x0000_3c00u32, 0];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (name, shader) in [("straight", &as_halves), ("branched", &branched)] {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[
                            View::Buffer {
                                data: &data,
                                stride: Stride::Format(DXGI_FORMAT_R16_UINT, 2),
                            },
                            View::RawUav(1),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(
                out[0][0],
                1.0f32.to_bits(),
                "{label}, {name}: the half read as an integer's bits"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// SMEM `s_load_dwordx8 s[dst:dst+7], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx8(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x03 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// MIMG `image_store vdata, vaddr, s[srsrc:srsrc+7] dmask:dmask dim:2D unorm [d16]`.
const fn image_store_2d(vdata: u8, vaddr: u8, srsrc: u8, dmask: u8, d16: bool) -> [u32; 2] {
    let w0 = 0x3C << 26 | 0x08 << 18 | 1 << 12 | (dmask as u32 & 0xF) << 8 | 1 << 3;
    let w1 = (d16 as u32) << 31
        | ((srsrc as u32 >> 2) & 0x1F) << 16
        | (vdata as u32) << 8
        | (vaddr as u32);
    [w0, w1]
}

fn try_build_image_store_cs(dmask: u8, d16: bool) -> Result<HostShader, crate::ShaderError> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx8(8, 4));
    // Data in v1..v4 from root constants s0..s3; 2D texel coordinate (0, 0) in v5..v6.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.push(vop1(0x001, 5, 0x80)); // v5 = 0
    words.push(vop1(0x001, 6, 0x80)); // v6 = 0
    words.extend_from_slice(&image_store_2d(1, 5, 8, dmask, d16));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
        half_views: lowered.half_views,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

#[test]
fn image_store_partial_dmask_writes_expected_channels() {
    // (dmask, d16, input root constants, expected RGBA texel)
    let cases: &[(u8, bool, [u32; 4], [f32; 4])] = &[
        (0x0, false, [1.5f32.to_bits(), 0, 0, 0], [0.0, 0.0, 0.0, 0.0]),
        (0x1, false, [1.5f32.to_bits(), 0, 0, 0], [1.5, 0.0, 0.0, 0.0]),
        (
            0x3,
            false,
            [1.5f32.to_bits(), (-2.25f32).to_bits(), 0, 0],
            [1.5, -2.25, 0.0, 0.0],
        ),
        (
            0x7,
            false,
            [
                1.5f32.to_bits(),
                (-2.25f32).to_bits(),
                4.0f32.to_bits(),
                0,
            ],
            [1.5, -2.25, 4.0, 0.0],
        ),
        // Sparse masks: RDNA2 packs enabled channels consecutively and zeros disabled ones.
        (
            0x5,
            false,
            [1.5f32.to_bits(), (-2.25f32).to_bits(), 0, 0],
            [1.5, 0.0, -2.25, 0.0],
        ),
        (
            0x9,
            false,
            [1.5f32.to_bits(), 4.0f32.to_bits(), 0, 0],
            [1.5, 0.0, 0.0, 4.0],
        ),
        // d16: 1.5 (0x3E00) in low half, -2.0 (0xC000) in high half, 4.0 (0x4400) in next VGPR.
        (0x1, true, [0x0000_3E00, 0, 0, 0], [1.5, 0.0, 0.0, 0.0]),
        (0x3, true, [0xC000_3E00, 0, 0, 0], [1.5, -2.0, 0.0, 0.0]),
        (
            0x7,
            true,
            [0xC000_3E00, 0x0000_4400, 0, 0],
            [1.5, -2.0, 4.0, 0.0],
        ),
        (
            0xD,
            true,
            [0xC000_3E00, 0x0000_4400, 0, 0],
            [1.5, 0.0, -2.0, 4.0],
        ),
    ];

    let shaders: Vec<HostShader> = cases
        .iter()
        .map(|&(dmask, d16, ..)| {
            try_build_image_store_cs(dmask, d16)
                .unwrap_or_else(|e| panic!("dmask={dmask:#x} d16={d16}: {e}"))
        })
        .collect();

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (&(dmask, d16, inputs, want), shader) in cases.iter().zip(&shaders) {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: dmask={dmask:#x} d16={d16}: {e}"));
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::TextureUav(1, 1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let got: [f32; 4] = [
                f32::from_bits(out[0][0]),
                f32::from_bits(out[0][1]),
                f32::from_bits(out[0][2]),
                f32::from_bits(out[0][3]),
            ];
            assert_eq!(
                got, want,
                "{label}: image_store dmask={dmask:#x} d16={d16} wrote {got:?}, want {want:?}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

/// EXP `exp target, v0, v1, v2, v3 [done] [vm]` (`en`: 4-bit channel mask).
const fn exp_inst(target: u8, en: u8, v: [u8; 4], done: bool, vm: bool) -> [u32; 2] {
    let w0 = 0x3E << 26
        | (vm as u32) << 12
        | (done as u32) << 11
        | (target as u32 & 0x3F) << 4
        | (en as u32 & 0xF);
    let w1 = (v[3] as u32) << 24 | (v[2] as u32) << 16 | (v[1] as u32) << 8 | (v[0] as u32);
    [w0, w1]
}

#[test]
fn vs_rt_and_viewport_index_exports_validate_and_create_pso() {
    use super::container::SigElement;
    use crate::dxbc::{Container, FourCC};

    let build_stage = |stage: HwStage,
                       words: &[u32],
                       layout: Vec<RootParam>,
                       stage_io: StageIo,
                       outputs: Vec<SigElement>|
     -> HostShader {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage,
            wave64: false,
            vgprs: 8,
            layout,
            used: Vec::new(),
            io: stage_io,
            code,
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res =
            resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
        let io = io::raise(&p, &native).expect("io::raise");
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs,
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        let shader = HostShader {
            stage,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: res.root_indexed_views,
            half_views: lowered.half_views,
        };
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy).expect("DXIL validation failed");
            assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        }
        shader
    };

    // Minimal pixel shader writing SV_Target0 = (1, 0, 0, 1).
    let ps_words = [
        vop1(0x001, 0, 0xF2), // v0 = 1.0
        vop1(0x001, 1, 0x80), // v1 = 0.0
        vop1(0x001, 2, 0x80), // v2 = 0.0
        vop1(0x001, 3, 0xF2), // v3 = 1.0
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[0],
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[1],
        S_ENDPGM,
    ];
    let ps = build_stage(
        HwStage::Ps,
        &ps_words,
        Vec::new(),
        StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        vec![SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }],
    );

    // Three VS variants: RT index only, viewport index only, and both + clip distance in pos2.
    let configs = [
        (true, false, false),
        (false, true, false),
        (true, true, true),
    ];
    let mut vs_shaders = Vec::new();
    for (has_rt, has_vp, has_clip) in configs {
        let mut words = vec![
            vop1(0x001, 0, 0x80),    // pos.x = 0.0
            vop1(0x001, 1, 0x80),    // pos.y = 0.0
            vop1(0x001, 2, 0x80),    // pos.z = 0.0
            vop1(0x001, 3, 0xF2),    // pos.w = 1.0
            vop1(0x001, 4, sgpr(0)), // pos1.z = packed (viewport << 16) | rt
            vop1(0x001, 5, sgpr(1)), // pos2.x = clip distance
        ];
        words.extend_from_slice(&exp_inst(12, 0xF, [0, 1, 2, 3], false, false)); // pos0
        words.extend_from_slice(&exp_inst(13, 0x4, [0, 0, 4, 0], !has_clip, false)); // pos1.z
        if has_clip {
            words.extend_from_slice(&exp_inst(14, 0x1, [5, 0, 0, 0], true, false)); // pos2.x
        }
        words.push(S_ENDPGM);

        let mut outputs = vec![SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }];
        let mut next_row = 1u32;
        if has_rt {
            outputs.push(SigElement {
                name: "SV_RenderTargetArrayIndex".into(),
                index: 0,
                kind: 4,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_vp {
            outputs.push(SigElement {
                name: "SV_ViewportArrayIndex".into(),
                index: 0,
                kind: 5,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_clip {
            outputs.push(SigElement {
                name: "SV_ClipDistance".into(),
                index: 0,
                kind: 6,
                comp: 9,
                interp: 2,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
        }

        let vs = build_stage(
            HwStage::Vs,
            &words,
            vec![RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            StageIo::Vs {
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            outputs,
        );
        let c = Container::parse(&vs.dxil).expect("parse emitted VS DXIL");
        let sfi0 = u64::from_le_bytes(
            c.part(FourCC::SFI0).expect("SFI0").data[..8]
                .try_into()
                .unwrap(),
        );
        assert_ne!(
            sfi0 & 0x2000,
            0,
            "SFI0 should set ViewPortAndRTArrayIndexFromAnyShaderFeedingRasterizer"
        );
        vs_shaders.push(vs);
    }

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (i, vs) in vs_shaders.iter().enumerate() {
            host.pipeline(vs, &ps)
                .unwrap_or_else(|e| panic!("{label}: graphics pipeline case {i} failed: {e}"));
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

#[test]
fn ps_sample_coverage_input_reads_coverage_mask() {
    use super::bitcode::{Attr, Term};
    use super::container::{self, SigElement};
    use super::host::Draw;
    use super::shader::Shader;
    use crate::dxbc::{Container, FourCC};

    // SPI_PS_INPUT_ADDR bit 14 puts the sample coverage mask in v0.
    // Convert u32 -> f32 into v1 and export mrt0 = (v1, v1, v1, v1).
    let words = [
        vop1(0x006, 1, vgpr(0)), // v_cvt_f32_u32 v1, v0
        0xF800_180F,
        0x0101_0101, // exp mrt0 v1, v1, v1, v1 done vm
        S_ENDPGM,
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 2,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 1 << 14,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    // Include SV_Coverage (kind 14) in console ISG1 to verify emit() strips it
    // (NotInSig in DXIL) while keeping SV_Target in OSG1.
    let isg1 = container::signature(
        &[SigElement {
            name: "SV_Coverage".into(),
            index: 0,
            kind: 14,
            comp: 5,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        }],
        true,
    );
    let osg1 = container::signature(
        &[SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        }],
        false,
    );
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).expect("Container::parse");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let ps = super::emit(&c, &native, &p, &st, &io, &res).expect("emit");

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = ps.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    // Minimal pass-through VS: POSITION -> SV_Position.
    let mut s = Shader::new(false);
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    for col in 0..4u8 {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), undef];
        let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let out_args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &out_args);
    }
    s.inputs.push(SigElement {
        name: "POSITION".into(),
        index: 0,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    });
    s.outputs.push(SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    });
    s.m.terminate(b, Term::Ret);
    let vs = HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
        half_views: Vec::new(),
    };

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .pipeline(&vs, &ps)
            .unwrap_or_else(|e| panic!("{label}: graphics pipeline failed: {e}"));
        let pixels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![
                        (0, Root::Constants(vec![0])),
                        (1, Root::Constants(vec![0])),
                    ],
                    textures: &[],
                    samplers: &[],
                    vertices: &[
                        -1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0,
                    ],
                    stride: 16,
                    size: (2, 2),
                    mesh_groups: 0,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
        // 1x sample target: every covered pixel reports coverage mask = 1 (bit 0).
        assert_eq!(pixels, vec![[1.0; 4]; 4], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn v_med3_min3_max3_integer_matches_rdna2() {
    let make = |op: u16| build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));

    let min3_i = make(0x152); // v_min3_i32
    let min3_u = make(0x153); // v_min3_u32
    let max3_i = make(0x155); // v_max3_i32
    let max3_u = make(0x156); // v_max3_u32
    let med3_i = make(0x158); // v_med3_i32
    let med3_u = make(0x159); // v_med3_u32

    // (inputs [a, b, c, 0], min_i32, max_i32, med_i32, min_u32, max_u32, med_u32)
    let cases: &[([u32; 4], u32, u32, u32, u32, u32, u32)] = &[
        // Plain ascending order
        ([1, 2, 3, 0], 1, 3, 2, 1, 3, 2),
        // Reversed
        ([3, 2, 1, 0], 1, 3, 2, 1, 3, 2),
        // All equal
        ([7, 7, 7, 0], 7, 7, 7, 7, 7, 7),
        // Two equal (low)
        ([1, 1, 5, 0], 1, 5, 1, 1, 5, 1),
        // Two equal (high)
        ([5, 9, 9, 0], 5, 9, 9, 5, 9, 9),
        // Signed vs unsigned with negative numbers
        (
            [(-3i32) as u32, (-1i32) as u32, 2, 0],
            (-3i32) as u32,
            2,
            (-1i32) as u32,
            2,
            (-1i32) as u32,
            (-3i32) as u32,
        ),
        // Signed extremes
        (
            [0x8000_0000, 0x7FFF_FFFF, 0, 0],
            0x8000_0000,
            0x7FFF_FFFF,
            0,
            0,
            0x8000_0000,
            0x7FFF_FFFF,
        ),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();

    let check = |name: &str,
                 shader: &HostShader,
                 extract: fn(&([u32; 4], u32, u32, u32, u32, u32, u32)) -> u32| {
        run_cases(shader, &inputs, |backend, i, inp, got| {
            let want = extract(&cases[i]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2]
            );
        });
    };

    check("v_min3_i32", &min3_i, |c| c.1);
    check("v_max3_i32", &max3_i, |c| c.2);
    check("v_med3_i32", &med3_i, |c| c.3);
    check("v_min3_u32", &min3_u, |c| c.4);
    check("v_max3_u32", &max3_u, |c| c.5);
    check("v_med3_u32", &med3_u, |c| c.6);
}

#[test]
fn v_cmp_class_f32_matches_rdna2() {
    // v_cmp_class_f32_e64 s6, v1, v2 (VOP3 encoding of VOPC 0x088)
    // v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp = build_valu_cs(&[
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary32 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x40_0000 clear)
        (0xFFBF_FFFF, 0), // -sNaN
        (0x7FC0_0000, 1), // +qNaN (quiet bit set)
        (0xFFC0_0001, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal (-1.0)
        (0x8000_0001, 4), // -denormal
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // +denormal
        (0x3F80_0000, 8), // +normal (+1.0)
        (0x7F80_0000, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    cases.push(([0x3F80_0000, 0, 0, 0], 0u32));
    cases.push(([0x3F80_0000, 0xFFFF_FC00, 0, 0], 0u32));

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32({:#010x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

/// The integer VALU operations whose result is a pure function of up to three
/// dwords, each with the RDNA2 instruction set document's definition written
/// out in Rust: (name, VOP3 opcode, reference).
#[allow(clippy::type_complexity)]
const INTEGER_OPS: &[(&str, u16, fn([u32; 3]) -> u32)] = &[
    ("v_bfi_b32", 0x14a, |[a, b, c]| a & b | !a & c),
    ("v_alignbit_b32", 0x14e, |[a, b, c]| {
        ((u64::from(a) << 32 | u64::from(b)) >> (c & 31)) as u32
    }),
    ("v_alignbyte_b32", 0x14f, |[a, b, c]| {
        ((u64::from(a) << 32 | u64::from(b)) >> (8 * (c & 3))) as u32
    }),
    ("v_bfm_b32", 0x363, |[a, b, _]| {
        (1u32 << (a & 31)).wrapping_sub(1) << (b & 31)
    }),
    ("v_bcnt_u32_b32", 0x364, |[a, b, _]| {
        a.count_ones().wrapping_add(b)
    }),
    ("v_bfrev_b32", 0x1b8, |[a, _, _]| a.reverse_bits()),
    ("v_not_b32", 0x1b7, |[a, _, _]| !a),
    ("v_xnor_b32", 0x11e, |[a, b, _]| !(a ^ b)),
    ("v_xor3_b32", 0x178, |[a, b, c]| a ^ b ^ c),
    ("v_or3_b32", 0x372, |[a, b, c]| a | b | c),
    ("v_and_or_b32", 0x371, |[a, b, c]| a & b | c),
    ("v_xad_u32", 0x345, |[a, b, c]| (a ^ b).wrapping_add(c)),
    ("v_add3_u32", 0x36d, |[a, b, c]| {
        a.wrapping_add(b).wrapping_add(c)
    }),
    ("v_lshl_add_u32", 0x346, |[a, b, c]| {
        (a << (b & 31)).wrapping_add(c)
    }),
    ("v_add_lshl_u32", 0x347, |[a, b, c]| {
        a.wrapping_add(b) << (c & 31)
    }),
    ("v_lshl_or_b32", 0x36f, |[a, b, c]| a << (b & 31) | c),
    ("v_lshrrev_b32", 0x116, |[a, b, _]| b >> (a & 31)),
    ("v_lshlrev_b32", 0x11a, |[a, b, _]| b << (a & 31)),
    ("v_ashrrev_i32", 0x118, |[a, b, _]| {
        ((b as i32) >> (a & 31)) as u32
    }),
    ("v_mul_lo_u32", 0x169, |[a, b, _]| a.wrapping_mul(b)),
    ("v_mul_hi_u32", 0x16a, |[a, b, _]| {
        ((u64::from(a) * u64::from(b)) >> 32) as u32
    }),
    ("v_mul_hi_i32", 0x16c, |[a, b, _]| {
        ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32
    }),
    ("v_mul_u32_u24", 0x10b, |[a, b, _]| {
        (a & 0xff_ffff).wrapping_mul(b & 0xff_ffff)
    }),
    ("v_mul_hi_u32_u24", 0x10c, |[a, b, _]| {
        ((u64::from(a & 0xff_ffff) * u64::from(b & 0xff_ffff)) >> 32) as u32
    }),
    ("v_mul_i32_i24", 0x109, |[a, b, _]| {
        (sext24(a) * sext24(b)) as u32
    }),
    ("v_mul_hi_i32_i24", 0x10a, |[a, b, _]| {
        ((sext24(a) * sext24(b)) >> 32) as u32
    }),
    ("v_mad_u32_u24", 0x143, |[a, b, c]| {
        (a & 0xff_ffff).wrapping_mul(b & 0xff_ffff).wrapping_add(c)
    }),
    ("v_mad_i32_i24", 0x142, |[a, b, c]| {
        (sext24(a) * sext24(b)).wrapping_add(i64::from(c)) as u32
    }),
    ("v_min3_i32", 0x152, |[a, b, c]| {
        (a as i32).min(b as i32).min(c as i32) as u32
    }),
    ("v_min3_u32", 0x153, |[a, b, c]| a.min(b).min(c)),
    ("v_max3_i32", 0x155, |[a, b, c]| {
        (a as i32).max(b as i32).max(c as i32) as u32
    }),
    ("v_max3_u32", 0x156, |[a, b, c]| a.max(b).max(c)),
    ("v_med3_i32", 0x158, |[a, b, c]| {
        let mut v = [a as i32, b as i32, c as i32];
        v.sort_unstable();
        v[1] as u32
    }),
    ("v_med3_u32", 0x159, |[a, b, c]| {
        let mut v = [a, b, c];
        v.sort_unstable();
        v[1]
    }),
    // {S0, S1} as eight bytes, S1 below: selectors 8-11 replicate the sign
    // bit of bytes 1, 3, 5 and 7, 12 is zero, above it 0xff.
    ("v_perm_b32", 0x344, |[a, b, c]| {
        let bytes = u64::from(a) << 32 | u64::from(b);
        (0..4).fold(0u32, |out, i| {
            let sel = c >> (8 * i) & 0xff;
            let byte = match sel {
                0..=7 => (bytes >> (8 * sel)) as u8,
                8..=11 => {
                    if bytes >> (8 * (2 * (sel - 8) + 1) + 7) & 1 != 0 {
                        0xff
                    } else {
                        0
                    }
                }
                12 => 0,
                _ => 0xff,
            };
            out | u32::from(byte) << (8 * i)
        })
    }),
    ("v_ffbl_b32", 0x1ba, |[a, _, _]| {
        if a == 0 { u32::MAX } else { a.trailing_zeros() }
    }),
    ("v_ffbh_u32", 0x1b9, |[a, _, _]| {
        if a == 0 { u32::MAX } else { a.leading_zeros() }
    }),
    ("v_ffbh_i32", 0x1bb, |[a, _, _]| {
        if a == 0 || a == u32::MAX {
            u32::MAX
        } else if a >> 31 != 0 {
            (!a).leading_zeros()
        } else {
            a.leading_zeros()
        }
    }),
];

fn sext24(x: u32) -> i64 {
    i64::from((x << 8) as i32 >> 8)
}

/// Dwords that sit on the edges of the operations above: shifts and field
/// widths around 0, 31 and 32, the 24-bit and sign boundaries, runs of ones.
const INTEGER_EDGES: &[u32] = &[
    0,
    1,
    2,
    3,
    4,
    7,
    8,
    12,
    13,
    31,
    32,
    33,
    0xff,
    0x100,
    0xffff,
    0x1_0000,
    0x7f_ffff,
    0x80_0000,
    0xff_ffff,
    0x100_0000,
    0x7fff_ffff,
    0x8000_0000,
    0x8000_0001,
    0xffff_fffe,
    0xffff_ffff,
    0xdead_beef,
    0x1234_5678,
    0x8765_4321,
    0xff00_ff00,
    0x00ff_00ff,
    0x0f0f_0f0f,
    0x0302_0100,
];

#[test]
fn integer_operations_match_the_rdna2_definitions() {
    // Every edge as each operand with the others varied, then a spread of
    // pairs and triples from a fixed sequence.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        INTEGER_EDGES[(seed >> 33) as usize % INTEGER_EDGES.len()]
    };
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &e in INTEGER_EDGES {
        cases.push([e, next(), next(), 0]);
        cases.push([next(), e, next(), 0]);
        cases.push([next(), next(), e, 0]);
    }
    for _ in 0..32 {
        cases.push([next(), next(), next(), 0]);
    }
    let mut failures = Vec::new();
    for &(name, opcode, reference) in INTEGER_OPS {
        let shader = build_valu_cs(&vop3(opcode, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = reference([inp[0], inp[1], inp[2]]);
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#x}, {:#x}, {:#x}) = {got:#x}, want {want:#x}",
                    inp[0], inp[1], inp[2]
                ));
            }
        });
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn mbcnt_of_the_first_lane_counts_no_lane_below_it() {
    // The one thread is lane 0 of a wave, so no lane is below it and the
    // result is the addend. The shaders ask for a 32-lane wave: not run on
    // WARP, whose waves are narrower (and `VICTORY_WARP=1` makes the
    // "hardware" adapter WARP).
    if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") {
        eprintln!("the hardware adapter is WARP; skipped");
        return;
    }
    for (name, opcode) in [("v_mbcnt_lo_u32_b32", 0x365), ("v_mbcnt_hi_u32_b32", 0x366)] {
        let shader = build_valu_cs(&vop3(opcode, 0, vgpr(1), vgpr(2), vgpr(3)));
        let cases = [
            [u32::MAX, 0, 0, 0],
            [u32::MAX, 5, 0, 0],
            [0x8000_0001, 0xffff_fff0, 0, 0],
        ];
        run_cases_on(
            &shader,
            &cases,
            &[("hardware", false)],
            |backend, _, inp, got| {
                assert_eq!(
                    got, inp[1],
                    "{backend}: {name}({:#x}, {:#x})",
                    inp[0], inp[1]
                );
            },
        );
    }
}

/// The scalar integer operations on one or two dwords, as the RDNA2
/// instruction set document defines them: (name, encoding, reference); the
/// encoding is `Sop1(opcode)` or `Sop2(opcode)`. The bit-field extracts take
/// their operand pair as the instruction does (offset in bits 4:0, width in
/// 22:16) and are tested on widths of up to 32.
#[allow(clippy::type_complexity)]
const SCALAR_OPS: &[(&str, ScalarOp, fn([u32; 2]) -> u32)] = &[
    ("s_add_u32", Sop2(0x00), |[a, b]| a.wrapping_add(b)),
    ("s_sub_u32", Sop2(0x01), |[a, b]| a.wrapping_sub(b)),
    ("s_add_i32", Sop2(0x02), |[a, b]| a.wrapping_add(b)),
    ("s_sub_i32", Sop2(0x03), |[a, b]| a.wrapping_sub(b)),
    ("s_min_i32", Sop2(0x06), |[a, b]| {
        (a as i32).min(b as i32) as u32
    }),
    ("s_min_u32", Sop2(0x07), |[a, b]| a.min(b)),
    ("s_max_i32", Sop2(0x08), |[a, b]| {
        (a as i32).max(b as i32) as u32
    }),
    ("s_max_u32", Sop2(0x09), |[a, b]| a.max(b)),
    ("s_and_b32", Sop2(0x0e), |[a, b]| a & b),
    ("s_or_b32", Sop2(0x10), |[a, b]| a | b),
    ("s_xor_b32", Sop2(0x12), |[a, b]| a ^ b),
    ("s_andn2_b32", Sop2(0x14), |[a, b]| a & !b),
    ("s_orn2_b32", Sop2(0x16), |[a, b]| a | !b),
    ("s_nand_b32", Sop2(0x18), |[a, b]| !(a & b)),
    ("s_nor_b32", Sop2(0x1a), |[a, b]| !(a | b)),
    ("s_xnor_b32", Sop2(0x1c), |[a, b]| !(a ^ b)),
    ("s_lshl_b32", Sop2(0x1e), |[a, b]| a << (b & 31)),
    ("s_lshr_b32", Sop2(0x20), |[a, b]| a >> (b & 31)),
    ("s_ashr_i32", Sop2(0x22), |[a, b]| {
        ((a as i32) >> (b & 31)) as u32
    }),
    ("s_bfm_b32", Sop2(0x24), |[a, b]| {
        (1u32 << (a & 31)).wrapping_sub(1) << (b & 31)
    }),
    ("s_mul_i32", Sop2(0x26), |[a, b]| a.wrapping_mul(b)),
    ("s_bfe_u32", Sop2(0x27), |[a, b]| bfe(a, b, false)),
    ("s_bfe_i32", Sop2(0x28), |[a, b]| bfe(a, b, true)),
    ("s_absdiff_i32", Sop2(0x2c), |[a, b]| {
        (a as i32).wrapping_sub(b as i32).unsigned_abs()
    }),
    ("s_lshl1_add_u32", Sop2(0x2e), |[a, b]| {
        (a << 1).wrapping_add(b)
    }),
    ("s_lshl2_add_u32", Sop2(0x2f), |[a, b]| {
        (a << 2).wrapping_add(b)
    }),
    ("s_lshl3_add_u32", Sop2(0x30), |[a, b]| {
        (a << 3).wrapping_add(b)
    }),
    ("s_lshl4_add_u32", Sop2(0x31), |[a, b]| {
        (a << 4).wrapping_add(b)
    }),
    ("s_pack_ll_b32_b16", Sop2(0x32), |[a, b]| {
        a & 0xffff | b << 16
    }),
    ("s_pack_lh_b32_b16", Sop2(0x33), |[a, b]| {
        a & 0xffff | b & 0xffff_0000
    }),
    ("s_pack_hh_b32_b16", Sop2(0x34), |[a, b]| {
        a >> 16 | b & 0xffff_0000
    }),
    ("s_mul_hi_u32", Sop2(0x35), |[a, b]| {
        ((u64::from(a) * u64::from(b)) >> 32) as u32
    }),
    ("s_mul_hi_i32", Sop2(0x36), |[a, b]| {
        ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32
    }),
    ("s_not_b32", Sop1(0x07), |[a, _]| !a),
    ("s_brev_b32", Sop1(0x0b), |[a, _]| a.reverse_bits()),
    ("s_bcnt1_i32_b32", Sop1(0x0f), |[a, _]| a.count_ones()),
    ("s_ff1_i32_b32", Sop1(0x13), |[a, _]| {
        if a == 0 { u32::MAX } else { a.trailing_zeros() }
    }),
    ("s_flbit_i32_b32", Sop1(0x15), |[a, _]| {
        if a == 0 { u32::MAX } else { a.leading_zeros() }
    }),
    ("s_flbit_i32", Sop1(0x17), |[a, _]| {
        if a == 0 || a == u32::MAX {
            u32::MAX
        } else if a >> 31 != 0 {
            (!a).leading_zeros()
        } else {
            a.leading_zeros()
        }
    }),
    ("s_abs_i32", Sop1(0x34), |[a, _]| (a as i32).unsigned_abs()),
];

#[derive(Clone, Copy)]
enum ScalarOp {
    Sop1(u8),
    Sop2(u8),
}
use ScalarOp::{Sop1, Sop2};

/// `s_bfe_u32` / `s_bfe_i32` of `a` by the field `field` (offset in bits 4:0,
/// width in 22:16): zero for a width of 0; a field reaching past bit 31 is
/// what is left of `a` shifted down, sign-extended from bit 31 when signed.
fn bfe(a: u32, field: u32, signed: bool) -> u32 {
    let (offset, width) = (field & 31, field >> 16 & 0x7f);
    if width == 0 {
        return 0;
    }
    let shifted = if signed {
        ((a as i32) >> offset) as u32
    } else {
        a >> offset
    };
    if width >= 32 - offset {
        return shifted;
    }
    let kept = shifted & ((1u32 << width) - 1);
    if signed && kept >> (width - 1) != 0 {
        kept | !((1u32 << width) - 1)
    } else {
        kept
    }
}

#[test]
fn scalar_integer_operations_match_the_rdna2_definitions() {
    let edges: Vec<u32> = INTEGER_EDGES.to_vec();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        edges[(seed >> 33) as usize % edges.len()]
    };
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &e in INTEGER_EDGES {
        cases.push([e, next(), 0, 0]);
        cases.push([next(), e, 0, 0]);
    }
    for _ in 0..32 {
        cases.push([next(), next(), 0, 0]);
    }
    let mut failures = Vec::new();
    for &(name, op, reference) in SCALAR_OPS {
        // `op s6, s0, s1`, then v_mov_b32 v0, s6.
        let first = match op {
            Sop2(code) => sop2(code, 6, 0, 1),
            Sop1(code) => 0xBE80_0000 | 6 << 16 | u32::from(code) << 8,
        };
        let shader = build_valu_cs(&[first, vop1(0x001, 0, sgpr(6))]);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = reference([inp[0], inp[1]]);
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#x}, {:#x}) = {got:#x}, want {want:#x}",
                    inp[0], inp[1]
                ));
            }
        });
    }
    assert!(
        failures.is_empty(),
        "{}",
        failures.join(
            "
"
        )
    );
}

#[test]
fn v_cmp_f_and_t_64bit_match_rdna2() {
    use crate::dxbc::{Container, FourCC};

    // v_cmp_{f,t}_{i64,u64}_e64 s6, v[1:2], v[3:4]; v_cndmask_b32_e64 v0, 0, 1, s6
    for (op, name, want) in [
        (0x0a0u16, "v_cmp_f_i64", 0u32),
        (0x0a7, "v_cmp_t_i64", 1),
        (0x0e0, "v_cmp_f_u64", 0),
        (0x0e7, "v_cmp_t_u64", 1),
    ] {
        let cmp = vop3(op, 6, vgpr(1), vgpr(3), 0);
        let sel = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        let shader = build_valu_cs(&[cmp[0], cmp[1], sel[0], sel[1]]);
        // Constant 64-bit compares must not require the Int64 feature bit (0x8000).
        let host = Container::parse(&shader.dxil).expect("Container::parse");
        let sfi0 = host.part(FourCC::SFI0).expect("SFI0").data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_eq!(
            features & 0x8000,
            0,
            "{name} should not set Int64 SFI0 flag: {features:#x}"
        );
        run_cases(
            &shader,
            &[[0, 0, 0, 0], [0xFFFF_FFFF, 0x7FFF_FFFF, 1, 0x8000_0000]],
            |backend, _, inp, got| {
                assert_eq!(
                    got, want,
                    "{backend}: {name}({inp:08x?}) = {got}, want {want}"
                );
            },
        );
    }
}

fn assert_no_int64(shader: &HostShader, label: &str) {
    let container = crate::Container::parse(&shader.dxil).expect("DXBC container");
    let features = container.part(crate::FourCC::SFI0).expect("SFI0 part");
    let flags = u64::from_le_bytes(features.data.try_into().expect("u64 SFI0"));
    assert_eq!(flags & 0x8000, 0, "{label} must not set SFI0 INT64");
}

#[derive(Clone, Copy)]
enum Shift64Kind {
    Shl,
    LShr,
    AShr,
}

impl Shift64Kind {
    fn eval(self, val: u64, shift: u32) -> u64 {
        let s = shift & 63;
        match self {
            Self::Shl => val << s,
            Self::LShr => val >> s,
            Self::AShr => ((val as i64) >> s) as u64,
        }
    }
}

fn shift64_inputs() -> Vec<[u32; 4]> {
    const VALUES: [u64; 9] = [
        0,
        1,
        u64::MAX,
        i64::MIN as u64,
        i64::MAX as u64,
        0x8000_0000_0000_0001,
        0x0000_0000_8000_0000,
        0x1234_5678_9ABC_DEF0,
        0xFEDC_BA98_7654_3210,
    ];
    const SHIFTS: [u32; 14] = [0, 1, 7, 15, 31, 32, 33, 47, 63, 64, 65, 95, 96, u32::MAX];
    let mut cases = Vec::with_capacity(VALUES.len() * SHIFTS.len());
    for v in VALUES {
        for s in SHIFTS {
            cases.push([v as u32, (v >> 32) as u32, s, 0]);
        }
    }
    cases
}

#[test]
fn v_shift_64_matches_rdna2_without_int64() {
    let inputs = shift64_inputs();
    for (name, op, kind) in [
        ("v_lshlrev_b64", 0x2FFu16, Shift64Kind::Shl),
        ("v_lshrrev_b64", 0x300, Shift64Kind::LShr),
        ("v_ashrrev_i64", 0x301, Shift64Kind::AShr),
    ] {
        let [w0, w1] = vop3(op, 6, vgpr(3), vgpr(1), 0);
        let lo_shader = build_valu_cs(&[w0, w1, vop1(0x001, 0, vgpr(6))]);
        let hi_shader = build_valu_cs(&[w0, w1, vop1(0x001, 0, vgpr(7))]);
        assert_no_int64(&lo_shader, name);
        assert_no_int64(&hi_shader, name);

        for (half, shader) in [("lo", &lo_shader), ("hi", &hi_shader)] {
            run_cases(shader, &inputs, |backend, _, inp, got| {
                let val = u64::from(inp[0]) | (u64::from(inp[1]) << 32);
                let want_64 = kind.eval(val, inp[2]);
                let want = if half == "lo" {
                    want_64 as u32
                } else {
                    (want_64 >> 32) as u32
                };
                assert_eq!(
                    got, want,
                    "{backend}: {name} {half}({val:#018x}, {}) = {got:#010x}, want {want:#010x}",
                    inp[2]
                );
            });
        }
    }

    // Inline negative constant (-1 = 193) and negative 32-bit literal (-100):
    // _i64 sign-extends the literal while _b64 zero-extends it.
    const MINUS_100: u32 = -100_i32 as u32;
    let shift_inputs: Vec<[u32; 4]> = [0u32, 1, 4, 31, 32, 33, 63, 64]
        .into_iter()
        .map(|s| [s, 0, 0, 0])
        .collect();
    for (name, op, kind, src1, extra, val) in [
        (
            "v_lshrrev_b64 inline -1",
            0x300u16,
            Shift64Kind::LShr,
            193u16,
            None,
            u64::MAX,
        ),
        (
            "v_ashrrev_i64 inline -1",
            0x301,
            Shift64Kind::AShr,
            193,
            None,
            u64::MAX,
        ),
        (
            "v_lshrrev_b64 lit -100",
            0x300,
            Shift64Kind::LShr,
            255,
            Some(MINUS_100),
            u64::from(MINUS_100),
        ),
        (
            "v_ashrrev_i64 lit -100",
            0x301,
            Shift64Kind::AShr,
            255,
            Some(MINUS_100),
            (-100_i64) as u64,
        ),
    ] {
        let [w0, w1] = vop3(op, 6, vgpr(1), src1, 0);
        let mut prefix = vec![w0, w1];
        if let Some(lit) = extra {
            prefix.push(lit);
        }
        let mut lo_body = prefix.clone();
        lo_body.push(vop1(0x001, 0, vgpr(6)));
        let mut hi_body = prefix;
        hi_body.push(vop1(0x001, 0, vgpr(7)));
        let lo_shader = build_valu_cs(&lo_body);
        let hi_shader = build_valu_cs(&hi_body);
        assert_no_int64(&lo_shader, name);
        assert_no_int64(&hi_shader, name);
        for (half, shader) in [("lo", &lo_shader), ("hi", &hi_shader)] {
            run_cases(shader, &shift_inputs, |backend, _, inp, got| {
                let want_64 = kind.eval(val, inp[0]);
                let want = if half == "lo" {
                    want_64 as u32
                } else {
                    (want_64 >> 32) as u32
                };
                assert_eq!(got, want, "{backend}: {name} {half} shift={}", inp[0]);
            });
        }
    }
}

#[test]
fn s_shift_64_matches_rdna2_without_int64() {
    let inputs = shift64_inputs();
    for (name, op, kind) in [
        ("s_lshl_b64", 0x1Fu8, Shift64Kind::Shl),
        ("s_lshr_b64", 0x21, Shift64Kind::LShr),
        ("s_ashr_i64", 0x23, Shift64Kind::AShr),
    ] {
        let shift = sop2(op, 16, 0, 2);
        let lo_shader = build_valu_cs(&[shift, vop1(0x001, 0, sgpr(16))]);
        let hi_shader = build_valu_cs(&[shift, vop1(0x001, 0, sgpr(17))]);
        let scc_shader = build_valu_cs(&[
            shift,
            sop2(0x0A, 18, 129, 128), // s_cselect_b32 s18, 1, 0
            vop1(0x001, 0, sgpr(18)),
        ]);
        assert_no_int64(&lo_shader, name);
        assert_no_int64(&hi_shader, name);
        assert_no_int64(&scc_shader, name);

        for (part, shader) in [("lo", &lo_shader), ("hi", &hi_shader), ("scc", &scc_shader)] {
            run_cases(shader, &inputs, |backend, _, inp, got| {
                let val = u64::from(inp[0]) | (u64::from(inp[1]) << 32);
                let want_64 = kind.eval(val, inp[2]);
                let want = match part {
                    "lo" => want_64 as u32,
                    "hi" => (want_64 >> 32) as u32,
                    _ => u32::from(want_64 != 0),
                };
                assert_eq!(
                    got, want,
                    "{backend}: {name} {part}({val:#018x}, {}) = {got:#010x}, want {want:#010x}",
                    inp[2]
                );
            });
        }
    }
}

#[test]
fn vop3_unsigned_add_sub_clamp() {
    let cases = [
        [0, 0, 1, u32::MAX],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [7, 4, 9, 0],
        [u32::MAX, 0, 1, u32::MAX],
        [u32::MAX, 2, u32::MAX, 0],
        [u32::MAX, u32::MAX, u32::MAX, 0],
    ];
    for (op, add, reverse, carry_out) in [
        (0x125, true, false, false),  // v_add_nc_u32
        (0x126, false, false, false), // v_sub_nc_u32
        (0x127, false, true, false),  // v_subrev_nc_u32
        (0x30f, true, false, true),   // v_add_co_u32
        (0x310, false, false, true),  // v_sub_co_u32
        (0x319, false, true, true),   // v_subrev_co_u32
        (0x128, true, false, true),   // v_add_co_ci_u32
        (0x129, false, false, true),  // v_sub_co_ci_u32
        (0x12a, false, true, true),   // v_subrev_co_ci_u32
        (0x36d, true, false, false),  // v_add3_u32
    ] {
        for clamp in [false, true] {
            let add3 = op == 0x36d;
            let carry_in = matches!(op, 0x128..=0x12a);
            let src2 = if carry_in {
                sgpr(20)
            } else if add3 {
                vgpr(3)
            } else {
                0
            };
            let mut body = Vec::new();
            if carry_in {
                // v_add_co_u32 v5, s20, v4, 1
                let mut seed = vop3(0x30f, 5, vgpr(4), 129, 0);
                seed[0] |= 20 << 8;
                body.extend(seed);
            }
            let mut inst = vop3(op, 0, vgpr(1), vgpr(2), src2);
            inst[0] |= u32::from(clamp) << 15;
            body.extend(inst);
            let exact = |[a, b, d, seed]: [u32; 4]| {
                let (a, b) = if reverse { (b, a) } else { (a, b) };
                let carry = i64::from(carry_in && seed == u32::MAX);
                if add {
                    i64::from(a) + i64::from(b) + carry + if add3 { i64::from(d) } else { 0 }
                } else {
                    i64::from(a) - i64::from(b) - carry
                }
            };
            let saturate = |result: i64| result.clamp(0, i64::from(u32::MAX));
            run_cases(&build_valu_cs(&body), &cases, |backend, _, inputs, got| {
                let result = exact(inputs);
                let expected = if clamp { saturate(result) } else { result } as u32;
                assert_eq!(
                    got, expected,
                    "{backend}: op {op:#x}, clamp {clamp}, {inputs:?}"
                );
            });
            if carry_out {
                // Read the carry/borrow mask in s0 independently of the result.
                body.extend(vop3(0x101, 0, 128, 129, sgpr(0))); // v_cndmask_b32
                run_cases(&build_valu_cs(&body), &cases, |backend, _, inputs, got| {
                    let result = exact(inputs);
                    let expected = u32::from(saturate(result) != result);
                    assert_eq!(
                        got, expected,
                        "{backend}: op {op:#x}, clamp {clamp}, carry for {inputs:?}"
                    );
                });
            }
        }
    }
}

/// SOPK `op sdst, simm16`.
const fn sopk_i16(op: u8, sdst: u8, simm16: i16) -> u32 {
    0xB000_0000 | (op as u32 & 0x1F) << 23 | (sdst as u32 & 0x7F) << 16 | (simm16 as u16) as u32
}

#[test]
fn signed_scalar_add_sub_scc_matches_rdna2() {
    // s_add_i32 s6, s0, s1 (SOP2 0x02); s_cselect_b32 s7, 1, 0 (SOP2 0x0A); v_mov_b32 v0, s7
    let add_scc = build_valu_cs(&[
        sop2(0x02, 6, 0, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_sub_i32 s6, s0, s1 (SOP2 0x03); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let sub_scc = build_valu_cs(&[
        sop2(0x03, 6, 0, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_mov_b32 s6, s0; s_addk_i32 s6, +1 (SOPK 0x0F); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let addk_pos1 = build_valu_cs(&[
        s_mov_b32(6, 0),
        sopk_i16(0x0F, 6, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_mov_b32 s6, s0; s_addk_i32 s6, -32768 (SOPK 0x0F); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let addk_neg_min = build_valu_cs(&[
        s_mov_b32(6, 0),
        sopk_i16(0x0F, 6, i16::MIN),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);

    let add_cases: &[(i32, i32)] = &[
        (0, 0),
        (10, 20),
        (-1, 1), // unsigned carry, no signed overflow
        (i32::MAX, 0),
        (i32::MAX, 1),        // positive overflow
        (i32::MAX, i32::MAX), // positive overflow
        (i32::MIN, 0),
        (i32::MIN, -1),       // negative overflow
        (i32::MIN, i32::MIN), // negative overflow (result 0)
        (i32::MIN, i32::MAX), // -1, no overflow
    ];
    let add_in: Vec<[u32; 4]> = add_cases
        .iter()
        .map(|&(a, b)| [a as u32, b as u32, 0, 0])
        .collect();
    run_cases(&add_scc, &add_in, |backend, i, _, got| {
        let (a, b) = add_cases[i];
        let want = u32::from(a.overflowing_add(b).1);
        assert_eq!(
            got, want,
            "{backend}: s_add_i32({a}, {b}) SCC = {got}, want {want}",
        );
    });

    let sub_cases: &[(i32, i32)] = &[
        (0, 0),
        (0, 1),        // unsigned borrow, no signed overflow
        (0, i32::MIN), // 0 - (-2^31) overflows i32
        (i32::MIN, 0),
        (i32::MIN, 1),        // negative overflow
        (i32::MIN, i32::MIN), // 0, no overflow
        (i32::MAX, -1),       // positive overflow
        (i32::MAX, i32::MIN), // positive overflow
        (-5, -3),             // -2, no overflow
        (-1, i32::MAX),       // -2^31 (i32::MIN), no overflow
    ];
    let sub_in: Vec<[u32; 4]> = sub_cases
        .iter()
        .map(|&(a, b)| [a as u32, b as u32, 0, 0])
        .collect();
    run_cases(&sub_scc, &sub_in, |backend, i, _, got| {
        let (a, b) = sub_cases[i];
        let want = u32::from(a.overflowing_sub(b).1);
        assert_eq!(
            got, want,
            "{backend}: s_sub_i32({a}, {b}) SCC = {got}, want {want}",
        );
    });

    let addk_pos_cases: &[i32] = &[0, -1, i32::MAX - 1, i32::MAX, i32::MIN];
    let addk_pos_in: Vec<[u32; 4]> = addk_pos_cases
        .iter()
        .map(|&a| [a as u32, 0, 0, 0])
        .collect();
    run_cases(&addk_pos1, &addk_pos_in, |backend, i, _, got| {
        let a = addk_pos_cases[i];
        let want = u32::from(a.overflowing_add(1).1);
        assert_eq!(
            got, want,
            "{backend}: s_addk_i32({a}, +1) SCC = {got}, want {want}",
        );
    });

    let addk_neg_cases: &[i32] = &[0, i32::MAX, i32::MIN + 32768, i32::MIN + 32767, i32::MIN];
    let addk_neg_in: Vec<[u32; 4]> = addk_neg_cases
        .iter()
        .map(|&a| [a as u32, 0, 0, 0])
        .collect();
    run_cases(&addk_neg_min, &addk_neg_in, |backend, i, _, got| {
        let a = addk_neg_cases[i];
        let want = u32::from(a.overflowing_add(i32::from(i16::MIN)).1);
        assert_eq!(
            got, want,
            "{backend}: s_addk_i32({a}, -32768) SCC = {got}, want {want}",
        );
    });
}

#[test]
fn typed_float_loads_preserve_nan_bits() {
    // The low channel is a signed half, an unsigned 11-bit float, or an
    // unsigned 10-bit float. Compare the returned VGPR bits, including the
    // quiet bit that a floating-point is_nan check would miss.
    for (format, cases) in [
        (
            13u32,
            &[
                (0x7c01, 0x7f80_2000),
                (0xfc01, 0xff80_2000),
                (0xfe55, 0xffca_a000),
                (0xfc00, 0xff80_0000),
                (0x8000, 0x8000_0000),
            ][..],
        ),
        (
            36,
            &[(0x7c1, 0x7f82_0000), (0x7e1, 0x7fc2_0000), (1, 0x3580_0000)][..],
        ),
        (
            43,
            &[(0x3e1, 0x7f84_0000), (0x3f1, 0x7fc4_0000), (1, 0x3600_0000)][..],
        ),
    ] {
        let mut body = buffer_store_dword(1, 8, 0).to_vec();
        // tbuffer_load_format_x v0, off, s[8:11], 0 format:N
        body.extend([0xe800_0000 | format << 19, 0x8002_0000]);
        let shader = build_valu_cs(&body);
        let inputs: Vec<_> = cases.iter().map(|&(raw, _)| [raw, 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |host, i, input, got| {
            assert_eq!(
                got, cases[i].1,
                "{host}: format {format}, input {:#x}",
                input[0]
            );
        });
    }
}

#[test]
fn ds_swizzle_reads_only_active_compute_lanes() {
    let value = 0xdead_beef;
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (offset, threads, narrow, expected) in [
        (0x041f, 1, false, 0), // bitmask-mode swap adjacent lanes (lane 1 absent)
        (0x041f, 2, true, 0),  // bitmask-mode swap adjacent lanes (lane 1 masked by EXEC)
        (0x80b1, 2, true, 0),  // quad-mode swap adjacent lanes (lane 1 masked by EXEC)
        (0x041f, 2, false, value),
        (0x80b1, 2, false, value),
    ] {
        let mut words = vec![vop1(0x001, 5, 128)];
        if narrow {
            words.push(0x7DA4_0080); // v_cmpx_eq_u32 0, v0 (local thread ID)
        }
        words.extend([0xD8D4_0000 | offset, 0x0500_0001]); // ds_swizzle_b32 v5, v1
        // Only one lane writes the output, even if both ran the swizzle.
        words.extend([0x7DA4_0080, vop1(0x001, 0, vgpr(5))]);
        let shader = build_cs(&words, threads, 0);
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            let (root, pso) = match host.compute_pipeline(&shader) {
                Ok(p) => p,
                // WARP does not support fixed WaveSize(32) on SM 6.6.
                Err(e) if warp => {
                    eprintln!("{label}: ds_swizzle {offset:#x} skipped ({e})");
                    continue;
                }
                Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
            };
            let inputs = [value, 0, 0, 0];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(
                out[0][0], expected,
                "{label}: ds_swizzle offset {offset:#x}, {threads} threads, narrowed {narrow}",
            );
        }
    }
}

/// RDNA2 ignores VOP3 `op_sel` on 16-bit compares (reading the low halves
/// regardless), while non-compare 16-bit ops such as `v_add_nc_u16` honour it.
#[test]
fn cmp16_ignores_op_sel_as_rdna2_does() {
    let vop3_opsel = |op: u16, dst: u8, s0: u16, s1: u16, opsel: u8| -> [u32; 2] {
        let [w0, w1] = vop3(op, dst, s0, s1, 0);
        [w0 | (u32::from(opsel) << 11), w1]
    };

    // High and low halves give opposite outcomes for every non-zero source op_sel.
    let f16_cases: &[([u32; 4], u32)] = &[
        // hi 2.0 vs 1.0, lo 1.0 < 1.5 -> true only when both read the low half
        ([0x4000_3C00, 0x3C00_3E00, 0, 0], 1),
        // hi 1.0 vs 3.0, lo 2.0 < 1.5 -> false only when both read the low half
        ([0x3C00_4000, 0x4200_3E00, 0, 0], 0),
    ];
    let u16_cases: &[([u32; 4], u32)] = &[
        // hi 10 vs 1, lo 1 < 2 -> true
        ([0x000A_0001, 0x0001_0002, 0, 0], 1),
        // hi 1 vs 20, lo 10 < 2 -> false
        ([0x0001_000A, 0x0014_0002, 0, 0], 0),
    ];
    let i16_cases: &[([u32; 4], u32)] = &[
        // hi 1 vs -2, lo -1 < 0 -> true
        ([0x0001_FFFF, 0xFFFE_0000, 0, 0], 1),
        // hi -2 vs 2, lo 1 < -1 -> false
        ([0xFFFE_0001, 0x0002_FFFF, 0, 0], 0),
    ];

    let mut cmp_shaders = Vec::new();
    for (op, what, cases) in [
        (0x0c9u16, "v_cmp_lt_f16", f16_cases),
        (0x0a9, "v_cmp_lt_u16", u16_cases),
        (0x089, "v_cmp_lt_i16", i16_cases),
    ] {
        for opsel in [0b0001u8, 0b0010, 0b0011] {
            let mut body = vop3_opsel(op, 6, vgpr(1), vgpr(2), opsel).to_vec();
            body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(6)));
            cmp_shaders.push((what, opsel, build_valu_cs(&body), cases));
        }
    }

    // v_cmpx_lt_f16_e64 (0x0d9) writing EXEC directly also ignores op_sel.
    let [w0, w1] = vop3_opsel(0x0d9, 126, vgpr(1), vgpr(2), 0b0001);
    let cmpx = build_valu_cs(&[
        s_mov_b32(6, 126),    // save exec_lo
        vop1(0x001, 0, 0x80), // v0 = 0
        w0,
        w1,                   // v_cmpx_lt_f16_e64 exec_lo, v1, v2 op_sel:[1,0]
        vop1(0x001, 0, 0x81), // v0 = 1 if active
        s_mov_b32(126, 6),    // restore exec_lo
    ]);

    // Control: v_add_nc_u16 (0x303) still honours op_sel.
    let [w0, w1] = vop3_opsel(0x303, 0, vgpr(1), vgpr(2), 0b0001);
    let add_hi = build_valu_cs(&[vop1(0x001, 0, 0x80), w0, w1]);

    for (what, opsel, shader, cases) in &cmp_shaders {
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what} opsel={opsel:#b} ({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }

    let f16_in: Vec<[u32; 4]> = f16_cases.iter().map(|c| c.0).collect();
    run_cases(&cmpx, &f16_in, |backend, i, inp, got| {
        let want = f16_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmpx_lt_f16 opsel=0b1 ({:#010x}, {:#010x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    run_cases(
        &add_hi,
        &[[0x000A_0001, 0x0014_0002, 0, 0]],
        |backend, _, _, got| {
            assert_eq!(
                got, 12,
                "{backend}: v_add_nc_u16 op_sel:[1,0] = {got}, want 12"
            );
        },
    );
}

#[test]
fn cmp16_sdwa_preserves_word_selects() {
    // llvm-mc -mcpu=gfx1030: v_cmp_lt_{f16,i16,u16}_sdwa vcc_lo, v1, v2.
    let instructions = [
        (0x7d92_04f9, [0x4000_3c00, 0x3800_3e00, 0, 0]),
        (0x7d12_04f9, [0x0003_ffff, 0xfffe_0000, 0, 0]),
        (0x7d52_04f9, [0x0003_0001, 0x0000_0002, 0, 0]),
    ];
    let select = vop3(0x101, 0, 0x80, 0x81, sgpr(106));
    for (instruction, input) in instructions {
        // Low/low gives true. Selecting either high word gives false.
        for (extension, expected) in [(0x0404_0001, 1), (0x0405_0001, 0), (0x0504_0001, 0)] {
            let shader = build_valu_cs(&[instruction, extension, select[0], select[1]]);
            run_cases(&shader, &[input], |backend, _, _, actual| {
                assert_eq!(
                    actual, expected,
                    "{backend}: {instruction:#x} {extension:#x}"
                );
            });
        }
    }
}

/// VOP3P `op vdst, src0, src1, src2` with standard packed lane selects
/// (`opsel = 0b000`, `opsel_hi = 0b111`) and optional `clamp`.
const fn vop3p_clamp(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16, clamp: bool) -> [u32; 2] {
    [
        0x33 << 26
            | (op as u32 & 0x7F) << 16
            | (clamp as u32) << 15
            | (1 << 14) // opsel_hi[2] = 1
            | vdst as u32,
        (0b11 << 27) // opsel_hi[1:0] = 0b11
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

#[test]
fn v_pk_int16_clamp_saturates_as_rdna2() {
    let pack_u = |lo: u16, hi: u16| u32::from(lo) | (u32::from(hi) << 16);
    let pack_i = |lo: i16, hi: i16| pack_u(lo as u16, hi as u16);

    // VOP3P opcodes: 0x00=mad_i16, 0x02=add_i16, 0x03=sub_i16, 0x09=mad_u16, 0x0a=add_u16, 0x0b=sub_u16.
    let add_u16 = build_valu_cs(&vop3p_clamp(0x0a, 0, vgpr(1), vgpr(2), 0, true));
    let sub_u16 = build_valu_cs(&vop3p_clamp(0x0b, 0, vgpr(1), vgpr(2), 0, true));
    let mad_u16 = build_valu_cs(&vop3p_clamp(0x09, 0, vgpr(1), vgpr(2), vgpr(3), true));
    let add_i16 = build_valu_cs(&vop3p_clamp(0x02, 0, vgpr(1), vgpr(2), 0, true));
    let sub_i16 = build_valu_cs(&vop3p_clamp(0x03, 0, vgpr(1), vgpr(2), 0, true));
    let mad_i16 = build_valu_cs(&vop3p_clamp(0x00, 0, vgpr(1), vgpr(2), vgpr(3), true));

    // add_u16 clamp: lo overflows (60000+10000 -> 65535), hi fits (100+200 -> 300)
    run_cases(
        &add_u16,
        &[
            [pack_u(60000, 100), pack_u(10000, 200), 0, 0],
            [pack_u(u16::MAX, 0), pack_u(u16::MAX, 0), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_u(u16::MAX, 300), pack_u(u16::MAX, 0)][i];
            assert_eq!(got, want, "{backend}: v_pk_add_u16 clamp case {i}");
        },
    );

    // sub_u16 clamp: lo underflows (5-10 -> 0), hi fits (500-200 -> 300)
    run_cases(
        &sub_u16,
        &[
            [pack_u(5, 500), pack_u(10, 200), 0, 0],
            [pack_u(0, u16::MAX), pack_u(u16::MAX, 0), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_u(0, 300), pack_u(0, u16::MAX)][i];
            assert_eq!(got, want, "{backend}: v_pk_sub_u16 clamp case {i}");
        },
    );

    // mad_u16 clamp: lo max*max+max -> 65535 (tests no u32 wrap!), hi 20*30+40 -> 640
    run_cases(
        &mad_u16,
        &[[
            pack_u(u16::MAX, 20),
            pack_u(u16::MAX, 30),
            pack_u(u16::MAX, 40),
            0,
        ]],
        |backend, _, _, got| {
            let want = pack_u(u16::MAX, 640);
            assert_eq!(got, want, "{backend}: v_pk_mad_u16 clamp");
        },
    );

    // add_i16 clamp: lo +overflow (30000+10000 -> 32767), hi -overflow (-30000-10000 -> -32768)
    run_cases(
        &add_i16,
        &[
            [pack_i(30000, -30000), pack_i(10000, -10000), 0, 0],
            [pack_i(100, -50), pack_i(-25, 20), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MAX, i16::MIN), pack_i(75, -30)][i];
            assert_eq!(got, want, "{backend}: v_pk_add_i16 clamp case {i}");
        },
    );

    // sub_i16 clamp: lo -overflow (-30000-10000 -> -32768), hi +overflow (30000-(-10000) -> 32767)
    run_cases(
        &sub_i16,
        &[
            [pack_i(-30000, 30000), pack_i(10000, -10000), 0, 0],
            [pack_i(0, -100), pack_i(i16::MIN, -200), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MIN, i16::MAX), pack_i(i16::MAX, 100)][i];
            assert_eq!(got, want, "{backend}: v_pk_sub_i16 clamp case {i}");
        },
    );

    // mad_i16 clamp: lo (-32768)*(-32768)+32767 -> 32767, hi (-32768)*(32767)-32768 -> -32768
    run_cases(
        &mad_i16,
        &[
            [
                pack_i(i16::MIN, i16::MIN),
                pack_i(i16::MIN, i16::MAX),
                pack_i(i16::MAX, i16::MIN),
                0,
            ],
            [pack_i(-10, 5), pack_i(20, -4), pack_i(50, -3), 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MAX, i16::MIN), pack_i(-150, -23)][i];
            assert_eq!(got, want, "{backend}: v_pk_mad_i16 clamp case {i}");
        },
    );
}
