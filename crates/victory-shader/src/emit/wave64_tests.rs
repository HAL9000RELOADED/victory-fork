//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
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
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
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
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
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
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
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
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
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
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    whole_uav(shader, constants, 1).map(|out| out[0])
}

/// As [`on_the_card`], and gives the first `dwords` dwords of the UAV.
fn whole_uav(shader: &HostShader, constants: [u32; 2], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

#[test]
fn local_waves_select_their_console_bits_from_a_constant_mask() {
    let words = [
        0xD760_000E,
        0x0001_0100, // v_readlane_b32 s14, v0, 0: require console lane numbering
        0xBE8C_0380, // s_mov_b32 s12, 0
        0xBE8D_03C1, // s_mov_b32 s13, -1
        0xD501_0001,
        0x0031_0280, // v_cndmask_b32 v1, 0, 1, s[12:13]
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8002_0103, // buffer_store_dword v1, v3, s[8:11], 0 offen
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lower::with_local_waves_any(|| lowered(&native, &p, &HashSet::new()))
        .expect("lower a constant mask on two host waves");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return;
    };
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords() as usize]))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch the constant mask");
    assert_eq!(&out[0][..32], &[0; 32]);
    assert_eq!(&out[0][32..], &[1; 32]);
    eprintln!("ran on the hardware device: each half selected its own mask bits");
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
const MASK_OR_NUMBER: [u32; 8] = [
    0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
    0xBE94_036A, // s_mov_b32 s20, vcc_lo
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0001, // s_cbranch_scc1 over the next
    0xBE94_0385, // s_mov_b32 s20, 5
    0x7E00_0214, // v_mov_b32 v0, s20
    0x38 << 26 | 0x1C << 18,
    0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
];

/// The program above reads a lane mask as a number (on the path that kept it), so it needs
/// the console's 32 lanes in a host wave: it declares `WaveSize(32)`. Without it, a host of
/// 16-lane waves gave only 16 of the mask's bits (#356).
#[test]
fn a_lane_mask_joined_with_a_number_needs_the_consoles_wave_size() {
    let native = native_of(&MASK_OR_NUMBER, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
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
    });
    lower::TEST_HOST_WAVE.set(was);
    assert_eq!(lowered.expect("lower").wave_size, Some(32));
}

#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let native = native_of(&MASK_OR_NUMBER, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// Fork pull request #646 (mynzjerko): a spilled 64-thread wave64 compute program, as the
/// structure builder's, dispatched by a `DispatchX` that counts threads
/// (`USER_DISPATCH_THREADS`). It stores `(bit count of EXEC on entry, wave OR of
/// 1 << (lane >> 1))` at `uav[lane * 2..]` from every lane. The threads at or past the count
/// must store nothing and count in neither; 0 is no count (all 64). On a host of 64-lane
/// waves they leave at entry; on one of 32 (both halves exchange) they stay, left out of the
/// ballot, the reduction and the store, and no barrier hangs with an empty upper half.
#[test]
fn threads_past_a_dispatch_x_count_do_nothing() {
    let mut words = vec![
        // The spilled table's descriptor-table offset (user data 16) from s[0:1] + 0.
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, 0xff
        0xBE82_0300, // s_mov_b32 s2, s0
        0x3D << 26 | 4 << 6 | 1,
        125 << 25,   // s_load_dword s4, s[2:3], 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25,   // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x3402_0083, // v_lshlrev_b32 v1, 3, v0 (8 bytes a lane)
        0x7E04_0206, // v_mov_b32 v2, s6
        0x7E06_0207, // v_mov_b32 v3, s7
        0x38 << 26 | 0x1D << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 2 << 8 | 1, // buffer_store_dwordx2 v[2:3], v1, s[8:11], 0 offen
        S_ENDPGM,
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 16,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1, // s[0:1]: the spilled table's pointer
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    assert!(crate::spill::is_spilled(&native));
    let p = crate::spill::build(&native).expect("spill::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).ok();
    let counts = RootType::Other(super::USER_DISPATCH_THREADS);
    for host_wave in [64u32, 32] {
        let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
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
        });
        lower::TEST_HOST_WAVE.set(was);
        let lowered = lowered.unwrap_or_else(|e| panic!("lower (waves of {host_wave}): {e}"));
        assert!(
            lowered.user_data.iter().any(|u| u.root == counts && u.dword == 0),
            "waves of {host_wave}: the program does not take the thread count of x"
        );
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: res.root_indexed_views.clone(),
            half_views: lowered.half_views,
        };
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).unwrap_or_else(|e| panic!("DXIL validation failed (waves of {host_wave}): {e}"));
        }
        let Some(host) = host.as_ref() else { continue };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take the waves-of-{host_wave} form; not run");
            continue;
        };
        // (count, threads that run, their wave OR: bits 0..ceil(n / 2))
        for (count, active, or) in [(45u32, 45usize, 0x007F_FFFFu32), (1, 1, 1), (0, 64, u32::MAX)] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in shader.user_data.iter().filter(|u| u.root == counts && u.dword == 0) {
                ud[u.index as usize] = count;
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(128)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("dispatch failed (waves of {host_wave}, count {count}): {e}"));
            for lane in 0..64 {
                let got = (out[0][lane * 2], out[0][lane * 2 + 1]);
                let expected = if lane < active { (active as u32, or) } else { (0, 0) };
                assert_eq!(got, expected, "waves of {host_wave}, count {count}, lane {lane}");
            }
        }
        eprintln!("ran on the hardware device (waves of {host_wave}): counts 45, 1 and none exact");
    }
}

/// A wave64 program of 128 threads (two console waves) that leaves early when no lane of its
/// wave is below s0, as the console's compiler does when the end of the program is too far for
/// a branch: `s_cbranch_execnz` over a far jump to the end. `extra` are words that go into the
/// far jump's block before it, and `work_at_end` puts an instruction between the jump's target
/// and the end of the program. Then the gate: every thread stores 1, or 2 when some lane of its
/// console wave that is still running has the thread number s1 (the lanes that fail a test).
fn far_early_out(extra: &[u32], work_at_end: bool) -> NativeProgram {
    let gate = [
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE90_106A, // s_bcnt1_i32_b64 s16, vcc
        0xBF06_8010, // s_cmp_eq_u32 s16, 0
        0x8511_8281, // s_cselect_b32 s17, 1, 2
        0x7E02_0211, // v_mov_b32 v1, s17
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 1 << 8 | 2, // buffer_store_dword v1, v2, s[8:11], 0 offen
    ];
    let jump = [
        0xBE8E_1F00, // s_getpc_b64 s[14:15]
        0x800E_FF0E,
        4 * (4 + gate.len() as u32), // s_add_u32 s14, s14, the way to the end
        0x820F_800F, // s_addc_u32 s15, s15, 0
        0xBE80_200E, // s_setpc_b64 s[14:15]
    ];
    let mut words = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF89_0000 | (extra.len() + jump.len()) as u32, // s_cbranch_execnz over the jump
    ];
    words.extend_from_slice(extra);
    words.extend_from_slice(&jump);
    words.extend_from_slice(&gate);
    if work_at_end {
        words.push(0x7E06_0281); // v_mov_b32 v3, 1
    }
    native_of(&words, true, 128)
}

fn branches_named(p: &ir::Program, name: &str) -> usize {
    p.blocks.iter().filter(|b| b.insts.last().is_some_and(|i| i.name() == name)).count()
}

/// What the gate of [`far_early_out`] stores in each of 128 threads.
fn gate_expected(below: u32, failing: u32) -> Vec<u32> {
    (0..128u32)
        .map(|t| {
            let some_fail = failing < below && failing / 64 == t / 64;
            match (t < below, some_fail) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => 2,
            }
        })
        .collect()
}

#[test]
fn a_far_early_out_is_left_out_and_the_gate_counts_the_lanes_of_both_halves() {
    let native = far_early_out(&[], false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1);
    lower::take_needs();
    let kept = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&kept, Err(ShaderError::Unsupported { what, .. }) if what.ends_with("where not every lane runs")),
        "{:?}",
        kept.err()
    );
    for (name, form) in [("early-outs", ir::without_early_outs(&p)), ("skips", ir::without_skips(&p, false))] {
        let form = form.unwrap_or_else(|| panic!("without {name}: the far early-out stays"));
        assert_eq!(branches_named(&form, "s_cbranch_execnz"), 0, "without {name}");
    }
    let form = ir::without_early_outs(&p).expect("left out");
    let shader = lowered(&native, &form, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (threads below s0 are in bounds; the one thread s1 fails the test) A failing thread of the
    // upper half gives 2 to the lower half's threads as well, and a failing thread out of
    // bounds counts for nothing
    for (below, failing) in [(128u32, 40u32), (128, 100), (128, 63), (40, 50), (40, 10), (33, 32), (1, 0), (0, 5)] {
        let Some(got) = whole_uav(&shader, [below, failing], 128) else { return };
        assert_eq!(got, gate_expected(below, failing), "{below} threads in bounds, thread {failing} fails");
    }
    eprintln!("ran on the hardware device: the gate's count over both halves of each console wave");
}

#[test]
fn a_far_jump_that_does_work_is_not_an_early_out() {
    // work in the jump's block, and work between the jump's target and the end
    for (name, native) in [
        ("work in the jump", far_early_out(&[0xBE94_0385], false)), // s_mov_b32 s20, 5
        ("work at the target", far_early_out(&[], true)),
    ] {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1, "{name}");
        assert!(ir::without_early_outs(&p).is_none(), "{name}");
        assert!(ir::without_skips(&p, false).is_none(), "{name}");
        assert!(ir::without_skips(&p, true).is_none(), "{name}");
    }
}

/// `if (t < s0) { park s[20:21]; if (s1 == 0) {..} else {..}; take it back }` and a read of
/// s[20:21] after the join, as the console's compiler leaves a parked scalar across an inner
/// branch. `then` is the inner branch's first arm; the second one leaves the parked scalar alone.
fn parked_across_an_inner_branch(then: u32) -> ir::Program {
    let body = [
        0xBE96_0414, // s_mov_b64 s[22:23], s[20:21]
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0002, // s_cbranch_scc1 to the second arm
        then,
        0xBF82_0001, // s_branch to the join
        0xBE98_0382, // s_mov_b32 s24, 2
        0xBE94_0416, // s_mov_b64 s[20:21], s[22:23]
    ];
    let mut words = vec![
        0xBE94_0406, // s_mov_b64 s[20:21], s[6:7]
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | body.len() as u32, // s_cbranch_execz past the body
    ];
    words.extend_from_slice(&body);
    words.extend_from_slice(&[
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_0214, // v_mov_b32 v1, s20
    ]);
    let native = native(&words);
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

#[test]
fn a_skip_whose_parked_scalars_pass_an_inner_branch_unchanged_is_left_out() {
    let p = parked_across_an_inner_branch(0xBE96_0314); // s_mov_b32 s22, s20
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    // (s22 comes back through a phi of the inner join whose two arguments are copies of one
    // value: only `deep` follows it)
    assert!(ir::without_skips(&p, false).is_none());
    let deep = ir::without_skips(&p, true).expect("left out");
    assert_eq!(branches_named(&deep, "s_cbranch_execz"), 0);
}

#[test]
fn a_skip_whose_parked_scalar_an_inner_branch_changes_is_kept() {
    let p = parked_across_an_inner_branch(0xBE96_0387); // s_mov_b32 s22, 7
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    assert!(ir::without_skips(&p, false).is_none());
    assert!(ir::without_skips(&p, true).is_none());
}

/// `v_writelane_b32 vN, sX, K`.
fn writelane(n: u32, x: u32, lane: u32) -> Vec<u32> {
    vec![0xD761_0000 | n, (128 + lane) << 9 | x]
}

/// `v_readlane_b32 sX, vN, K`.
fn readlane(x: u32, n: u32, lane: u32) -> Vec<u32> {
    vec![0xD760_0000 | x, (128 + lane) << 9 | (256 + n)]
}

fn parked(words: &[Vec<u32>]) -> ir::Program {
    let native = native(&words.concat());
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

fn instructions<'a>(p: &'a ir::Program, name: &str) -> Vec<&'a ir::Inst> {
    p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).collect()
}

fn regs<T>(of: &[(ir::Reg, T)]) -> Vec<ir::Reg> {
    of.iter().map(|r| r.0).collect()
}

#[test]
fn a_pair_parked_in_lanes_that_hold_other_scalars_elsewhere_moves_as_a_pair() {
    // s[20:21] goes into lanes 4 and 5 of v10 and comes back into s[22:23]; later lane 4
    // holds s30 for a while: no lane of the two is the pair's alone
    let p = parked(&[
        writelane(10, 20, 4),
        writelane(10, 21, 5),
        vec![0x7E04_0281], // v_mov_b32 v2, 1
        readlane(22, 10, 4),
        readlane(23, 10, 5),
        writelane(10, 30, 4),
        readlane(31, 10, 4),
    ]);
    let wide = instructions(&p, "s_mov_b64");
    let [save, restore] = wide[..] else { panic!("one pair save and one pair restore: {} moves", wide.len()) };
    assert_eq!(regs(&save.reads), [ir::Reg::S(20), ir::Reg::S(21)]);
    assert_eq!(regs(&restore.writes), [ir::Reg::S(22), ir::Reg::S(23)]);
    assert_eq!(regs(&restore.reads), regs(&save.writes));
    // the single scalar has its own lane register, not the pair's
    let narrow = instructions(&p, "s_mov_b32");
    let lone_save = narrow.iter().find(|i| regs(&i.reads) == [ir::Reg::S(30)]).expect("the save of s30");
    let lone_restore = narrow.iter().find(|i| regs(&i.writes) == [ir::Reg::S(31)]).expect("the restore into s31");
    assert_eq!(regs(&lone_restore.reads), regs(&lone_save.writes));
    assert!(regs(&lone_save.writes).iter().all(|r| !regs(&save.writes).contains(r)));
}

#[test]
fn a_parked_pair_another_save_or_restore_shares_stays_two_scalars() {
    let cases = [
        (
            "a lone restore of the lane",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
                readlane(24, 10, 4),
            ],
        ),
        (
            "another save of the lane on one path to the restore",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                vec![0xBF06_8001, 0xBF85_0002], // s_cmp_eq_u32 s1, 0 and s_cbranch_scc1 over the save
                writelane(10, 30, 4),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
            ],
        ),
    ];
    for (name, words) in cases {
        let p = parked(&words);
        assert!(instructions(&p, "s_mov_b64").is_empty(), "{name}");
        assert!(!instructions(&p, "s_mov_b32").is_empty(), "{name}");
    }
}

/// Runs a compute shader on the hardware device and returns `uav_dwords` dwords
/// from UAV 0, or `None` if no hardware device with wave32 support is present.
fn on_the_card_uav(
    shader: &HostShader,
    constants: [u32; 2],
    uav_dwords: u32,
) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(p) => p,
        Err(e) => panic!("hardware pipeline refused: {e}"),
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(uav_dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

/// Builds a wave64 compute program with `threads` threads in x:
/// - `v1 = v0 + 1000` (each thread's payload)
/// - `v2 = (v0 ^ s0) << 2` (source lane byte address)
/// - `v3 = 0x5555_5555` (sentinel kept by guarded-off destination lanes)
/// - Under `v0 < s1` (`guard`, or a `s_cbranch_execz` skip when `branch` is set):
///   `ds_bpermute_b32 v3, v2, v1 offset:offset`
/// - Every thread stores `v3` to `uav[v0]`.
fn bpermute_program(threads: u32, offset: u16, branch: bool, lds_bytes: u32) -> NativeProgram {
    let mut words = vec![
        0x4A02_00FF,
        1000, // v_add_nc_u32 v1, 1000, v0
        0x3A04_0000, // v_xor_b32 v2, s0, v0
        0x3404_0482, // v_lshlrev_b32 v2, 2, v2
        0x7E06_02FF,
        0x5555_5555, // v_mov_b32 v3, 0x5555_5555
        0x7D88_0001, // v_cmp_gt_u32 vcc, s1, v0  (v0 < s1)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    ];
    if branch {
        words.push(0xBF88_0002); // s_cbranch_execz over the 2-dword ds_bpermute_b32
    }
    words.extend_from_slice(&[
        0xDACC_0000 | u32::from(offset),
        (3u32 << 24) | (1u32 << 8) | 2u32, // ds_bpermute_b32 v3, v2, v1 offset:offset
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0  (byte offset = v0 * 4)
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 3 << 8 | 4, // buffer_store_dword v3, v4, s[8:11], 0 offen
    ]);
    let mut prog = native_of(&words, true, threads);
    if let StageIo::Cs(cs) = &mut prog.io {
        cs.rsrc2 |= (lds_bytes / 512) << 15;
    }
    prog
}

fn expected_bpermute(threads: u32, offset: u16, xor_mask: u32, active_below: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            if t >= active_below {
                return 0x5555_5555;
            }
            let wave_off = t & !63;
            let addr = (t ^ xor_mask).wrapping_shl(2).wrapping_add(u32::from(offset));
            let src_lane = (addr >> 2) & 63;
            let src_thread = wave_off + src_lane;
            if src_thread < threads && src_thread < active_below {
                src_thread + 1000
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn bpermute_small_wave64_group_returns_zero_for_upper_lanes() {
    let prog = bpermute_program(32, 0, false, 0);
    let p = ir::build(&prog.code, prog.wave64, prog.vgprs).expect("ir::build");
    let shader = lowered(&prog, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (xor_mask, active_below) in [(32u32, 32u32), (1, 32), (1, 16), (32, 16)] {
        let Some(got) = on_the_card_uav(&shader, [xor_mask, active_below], 32) else {
            return;
        };
        let want = expected_bpermute(32, 0, xor_mask, active_below);
        assert_eq!(got, want, "small group [32]: xor={xor_mask}, active<{active_below}");
    }
}

#[test]
fn bpermute_cross_half_on_wave32_exchanges_all_64_lanes_and_guards() {
    for (threads, offset) in [(64u32, 0u16), (64, 4), (128, 0)] {
        let prog = bpermute_program(threads, offset, false, 0);
        let p = ir::build(&prog.code, prog.wave64, prog.vgprs).expect("ir::build");
        let shader = lowered(&prog, &p, &HashSet::new()).expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        for (xor_mask, active_below) in [(32u32, threads), (63, threads), (32, threads / 2 + 7)] {
            let Some(got) = on_the_card_uav(&shader, [xor_mask, active_below], threads) else {
                return;
            };
            let want = expected_bpermute(threads, offset, xor_mask, active_below);
            assert_eq!(
                got, want,
                "threads={threads}, offset={offset}, xor={xor_mask}, active<{active_below}"
            );
        }
    }
}

#[test]
fn bpermute_partial_tail_wave_returns_zero_for_missing_lanes() {
    let prog = bpermute_program(96, 0, false, 0);
    let p = ir::build(&prog.code, prog.wave64, prog.vgprs).expect("ir::build");
    let shader = lowered(&prog, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (xor_mask, active_below) in [(32u32, 96u32), (63, 96), (32, 80)] {
        let Some(got) = on_the_card_uav(&shader, [xor_mask, active_below], 96) else {
            return;
        };
        let want = expected_bpermute(96, 0, xor_mask, active_below);
        assert_eq!(got, want, "partial tail [96]: xor={xor_mask}, active<{active_below}");
    }
}

#[test]
fn bpermute_in_lane_branch_widens_and_lds_overflow_refuses() {
    // (a) `ds_bpermute_b32` inside a per-lane branch triggers `needs_every_lane`
    // and succeeds once the branch is widened with `structure_wide`.
    let prog = bpermute_program(64, 0, true, 0);
    let p = ir::build(&prog.code, prog.wave64, prog.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&prog, &p, &HashSet::new());
    let err = match per_lane {
        Err(ShaderError::Unsupported { what, .. }) => what,
        other => panic!("expected unsupported refusal, got {:?}", other.err()),
    };
    assert_eq!(err, "a lane permute where not every lane runs");
    assert!(lower::needs_every_lane(err));
    let st = structure::structure(&p).expect("structure");
    assert_eq!(
        lower::branches_around(&st, &lower::take_needs()),
        Some(vec![branch])
    );

    let shader = lowered(&prog, &p, &HashSet::from([branch])).expect("widened lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(got) = on_the_card_uav(&shader, [32, 40], 64) {
        let want = expected_bpermute(64, 0, 32, 40);
        assert_eq!(got, want, "widened lane-branch bpermute");
    }

    // (b) Pre-allocated 32 KiB LDS leaves no room for the 64-dword/wave scratch.
    let big_lds = bpermute_program(128, 0, false, 32768);
    let p_big = ir::build(&big_lds.code, big_lds.wave64, big_lds.vgprs).expect("ir::build");
    let overflow = lowered(&big_lds, &p_big, &HashSet::new());
    assert!(
        matches!(
            &overflow,
            Err(ShaderError::Unsupported { what, .. })
                if *what == "a lane permute across two host waves (not enough LDS)"
        ),
        "expected LDS overflow refusal, got {:?}",
        overflow.err()
    );
}

/// Wave64 compute groups whose size is not a multiple of 32 (e.g. 48 = 32 + 16,
/// and 16 < 32) lowered to 32-lane host waves: every console thread index
/// `0..threads` must execute exactly once per group, with no skipped threads,
/// duplicate wraps, or out-of-bounds indices.
#[test]
fn partial_group_wave64_on_wave32_assigns_every_thread_once() {
    let words = [
        // ds_swizzle_b32 v2, v0 offset:0x80E4 (triggers console_lanes -> halves)
        (0x36 << 26) | (0x35 << 18) | 0x80E4,
        2 << 24,
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0 (byte offset = thread_id.x * 4)
        0x7E06_0281, // v_mov_b32 v3, 1
        // buffer_atomic_add v3, v1, s[8:11], 0 offen
        (0x38 << 26) | (0x32 << 18) | (1 << 12),
        (0x80 << 24) | (2 << 16) | (3 << 8) | 1,
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for threads in [16u32, 48u32] {
        let native = native_of(&words, true, threads);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("DXIL validation failed for {threads} threads: {e}"));
        }
        let Ok(host) = Host::new(false) else {
            continue;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            continue;
        };
        let groups = 64u32;
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [groups, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed ({threads} threads): {e}"));
        for slot in 0..threads as usize {
            assert_eq!(
                out[0][slot], groups,
                "threads={threads}: slot {slot} got {} (expected {groups})",
                out[0][slot]
            );
        }
        for slot in threads as usize..64 {
            assert_eq!(
                out[0][slot], 0,
                "threads={threads}: out-of-range slot {slot} was written ({})",
                out[0][slot]
            );
        }
    }
}

/// A 64-lane `v_add_f64` whole-wave sum lowered on a 32-lane host:
/// - for a group of 32 threads (`e.1 == true`), one host wave holds the whole group so no
///   cross-half exchange is needed;
/// - inside a per-lane branch under `with_local_waves_any` (`self.local == true`), each
///   host wave takes its own 32-lane sum instead of failing with
///   `"a wave reduction where not every lane runs"`.
#[test]
fn wave64_f64_sum_on_wave32_host_respects_small_groups_and_local_waves() {
    let vop1 = |op: u16, vdst: u8, src0: u16| {
        (0x3Fu32 << 25) | (u32::from(vdst) << 17) | (u32::from(op) << 9) | u32::from(src0)
    };
    let vop3 = |op: u16, vdst: u8, src0: u16, src1: u16, src2: u16| {
        [
            (0x35u32 << 26) | (u32::from(op) << 16) | u32::from(vdst),
            u32::from(src0) | (u32::from(src1) << 9) | (u32::from(src2) << 18),
        ]
    };
    let v_mov_dpp8 = |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), u32::from(vsrc) | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            u32::from(vsrc) | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };
    let vgpr = |n: u8| 256u16 + u16::from(n);
    let sgpr = |n: u8| u16::from(n);

    // Whole-wave f64 sum of per-lane (s0, s1) inside `if (v0 < s2)`.
    let mut words = vec![
        vop1(0x001, 6, sgpr(0)), // v_mov_b32 v6, s0
        vop1(0x001, 7, sgpr(1)), // v_mov_b32 v7, s1
        0x7D88_0002,             // v_cmp_gt_u32 vcc, s2, v0
        0xBE92_246A,             // s_and_saveexec_b64 s[18:19], vcc
    ];
    let skip_at = words.len();
    words.push(0xBF88_0000); // s_cbranch_execz (patched below)
    let body_start = words.len();
    words.push(0xBE8C_25C1); // s_or_saveexec_b64 s[12:13], -1
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12))); // v_cndmask_b32 v2, 0, v6, s[12:13]
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12))); // v_cndmask_b32 v3, 0, v7, s[12:13]
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, vgpr(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, vgpr(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, vgpr(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.extend_from_slice(&vop3(0x164, 2, sgpr(14), vgpr(2), 0)); // v_add_f64 v[2:3], s[14:15], v[2:3]
    words.extend_from_slice(&vop3(0x360, 16, vgpr(2), 0xA0, 0)); // v_readlane_b32 s16, v2, 32
    words.extend_from_slice(&vop3(0x360, 17, vgpr(3), 0xA0, 0)); // v_readlane_b32 s17, v3, 32
    words.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    words.push(vop1(0x00F, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&[0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16]); // buffer_store_dword v0, off, s[8:11], 0
    let body_words = (words.len() - body_start) as u32;
    words[skip_at] = 0xBF88_0000 | body_words;
    words.push(0xBEFE_0412); // s_mov_b64 exec, s[18:19]

    // 1. 64-thread group: fails without local waves, succeeds under `with_local_waves_any`.
    let n64 = native_of(&words, true, 64);
    let p64 = ir::build(&n64.code, n64.wave64, n64.vgprs).expect("ir::build");
    let err = lowered(&n64, &p64, &HashSet::new());
    assert!(
        matches!(&err, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "expected divergent reduction error without local waves, got {:?}",
        err.err(),
    );
    let shader64 = lower::with_local_waves_any(|| lowered(&n64, &p64, &HashSet::new())).expect("local waves f64 sum");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader64.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed (64-thread local)");
    }

    // 2. 32-thread group (wave64 on wave32 host, `e.1 == true`): no other half to exchange with.
    let n32 = native_of(&words, true, 32);
    let p32 = ir::build(&n32.code, n32.wave64, n32.vgprs).expect("ir::build");
    let shader32 = lowered(&n32, &p32, &HashSet::new()).expect("32-thread wave64 f64 sum");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader32.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed (32-thread)");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    let Ok((root, pso)) = host.compute_pipeline(&shader32) else { return };
    let b = 1.25f64.to_bits();
    let inputs = [b as u32, (b >> 32) as u32, 4u32, 0u32];
    let mut ud = vec![0u32; shader32.user_data_dwords().max(1) as usize];
    for u in &shader32.user_data {
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
        .expect("dispatch");
    assert_eq!(out[0][0], 5.0f32.to_bits(), "32-thread wave64 f64 sum of 4 lanes * 1.25");
}

/// Issue #111: `v_writelane_b32` writes its target lane whenever any lane of
/// the console wave is in EXEC, even if the target lane itself is outside EXEC;
/// if no lane is in EXEC (the wave skipped), the write must not happen.
#[test]
fn a_lane_write_reaches_its_lane_outside_exec() {
    // One console wave of 64: v9 = 7, s6 = 5, if (t > s0) { v_writelane_b32 v9, s6, 0 },
    // then every thread stores v_readlane_b32(v9, 0). v9 also has an ordinary use (v_mov_b32)
    // so ir::unspill keeps v_writelane_b32 / v_readlane_b32.
    let words = [
        0x7E12_0287, // v_mov_b32 v9, 7
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7D82_0000, // v_cmp_lt_u32 vcc, s0, v0  (t > s0)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0002, // s_cbranch_execz +2 (over the 2-dword v_writelane_b32)
        0xD761_0009,
        0x0001_0006, // v_writelane_b32 v9, s6, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0xD760_0007,
        0x0001_0109, // v_readlane_b32 s7, v9, 0
        0x7E02_0207, // v_mov_b32 v1, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let io = io::raise(&p, &native).expect("io");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("res");
    let empty_bytes = crate::dxbc::build(&[]);
    let empty = crate::Container::parse(&empty_bytes).unwrap();

    // On a 32-lane host, per-lane lowering refuses the skipped write so emit()
    // switches to the without_skips form where all 64 lanes reach the write.
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane write where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    let shader = lower::TEST_HOST_WAVE.with(|cell| {
        let was = cell.replace(Some(32));
        let r = super::emit(&empty, &native, &p, &st, &io, &res);
        cell.set(was);
        r
    })
    .expect("emit on 32-lane host");
    // Also check native 64-lane lowering (without_skips form).
    let no_skips = ir::without_skips(&p, false).expect("without_skips");
    let st_ns = structure::structure(&no_skips).expect("structure");
    let io_ns = io::raise(&no_skips, &native).expect("io");
    let res_ns = resource::track(&no_skips, &native.layout, native.stage, Some(&st_ns)).expect("res");
    let shader64 = lower::TEST_HOST_WAVE.with(|cell| {
        let was = cell.replace(Some(64));
        let r = super::emit(&empty, &native, &no_skips, &st_ns, &io_ns, &res_ns);
        cell.set(was);
        r
    })
    .expect("emit on 64-lane host");

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("32-lane DXIL validation failed");
        v.validate(&mut shader64.dxil.clone()).expect("64-lane DXIL validation failed");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (s0, expected) in [(10u32, 5u32), (62, 5), (63, 7)] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        assert_eq!(got, expected, "s0 = {s0}");
    }
    eprintln!("ran on the hardware device: v_writelane_b32 reaches lane 0 outside EXEC");
}

/// Runs a 64-thread shader on the hardware device and returns all 64 output UAV dwords.
fn on_the_card_64(shader: &HostShader, constants: [u32; 2]) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

/// Lowers `native` for a 32-lane host and returns both the host shader and
/// `(wave_size, half_exchanges)` diagnostics.
fn lowered_with_stats(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
) -> Result<(HostShader, Option<u32>, u32), ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    let stats = (lowered.wave_size, lowered.half_exchanges);
    Ok((
        HostShader {
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
        },
        stats.0,
        stats.1,
    ))
}

/// A 64-lane console wave whose only cross-lane instruction is `v_readfirstlane_b32`:
/// on a 32-lane host, it must still set up half-wave exchange so lanes 32..63 read
/// the console wave's first active lane (not lane 32's half).
#[test]
fn wave64_readfirstlane_exchanges_across_halves_on_32_lane_host() {
    let words = [
        // v5 = 100 + v0 (literal 100)
        0x4A0A_00FF,
        100, // v_add_nc_u32 v5, 100, v0
        0x7D86_0000, // v_cmp_le_u32 vcc, s0, v0 (active iff v0 >= s0)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0x7E0E_0505, // v_readfirstlane_b32 s7, v5
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E0C_0207, // v_mov_b32 v6, s7
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0xE070_1000,
        0x8002_0607, // buffer_store_dword v6, v7, s[8:11], 0 offen
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let (shader, wave_size, exchanges) =
        lowered_with_stats(&native, &p, &HashSet::new()).expect("lower");
    assert_eq!(wave_size, Some(32), "expected 32-lane split wave size");
    assert!(
        exchanges >= 2,
        "expected guarded wave64 v_readfirstlane_b32 to emit half-wave exchanges, got {exchanges}",
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // s0 = 0 -> lane 0 (100); s0 = 17 -> lane 17 in lower half (117);
    // s0 = 40 -> lane 40 in upper half (140); s0 = 64 -> no active lane -> lane 0 (100).
    for (first, expected) in [(0u32, 100u32), (17, 117), (40, 140), (64, 100)] {
        let Some(got) = on_the_card_64(&shader, [first, 0]) else {
            return;
        };
        assert!(
            got.iter().all(|&x| x == expected),
            "first={first}: expected all 64 lanes to read {expected}, got lane0={} lane32={} lane63={}",
            got[0],
            got[32],
            got[63],
        );
    }
}

/// Implicit VGPR/VCC reads (`v_fmac_f32` accumulator and VOP2 `v_cndmask_b32` VCC)
/// must prevent `lane_uniform` from marking a per-lane result uniform across a
/// split 64-lane wave.
#[test]
fn wave64_implicit_reads_are_not_marked_lane_uniform() {
    // Case 1: v5 = float(v0); v6 = 2.0 (uniform); v_fmac_f32 v5, 1.0, v6 (uniform
    // explicit sources, per-lane implicit accumulator v5); v_readfirstlane_b32 s7, v5.
    let fmac_words = [
        0x7E0A_0D00, // v_cvt_f32_u32 v5, v0
        0x7E0C_02F4, // v_mov_b32 v6, 2.0 (inline const 244)
        0x560A_0CF2, // v_fmac_f32 v5, 1.0 (inline const 242), v6
        0x7E0E_0505, // v_readfirstlane_b32 s7, v5
        0x7E0C_0207, // v_mov_b32 v6, s7
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0xE070_1000,
        0x8002_0607, // buffer_store_dword v6, v7, s[8:11], 0 offen
    ];
    // Case 2: v6 = 20 (uniform); vcc = (32 <= v0); v_cndmask_b32 v5, 10, v6, vcc
    // (uniform explicit sources, per-lane implicit VCC); v_readfirstlane_b32 s7, v5.
    let cndmask_words = [
        0x7E0C_0294, // v_mov_b32 v6, 20 (0x80 + 20)
        0x7D86_00A0, // v_cmp_le_u32 vcc, 32 (0x80 + 32), v0
        0x020A_0C8A, // v_cndmask_b32 v5, 10 (0x80 + 10), v6, vcc
        0x7E0E_0505, // v_readfirstlane_b32 s7, v5
        0x7E0C_0207, // v_mov_b32 v6, s7
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0xE070_1000,
        0x8002_0607, // buffer_store_dword v6, v7, s[8:11], 0 offen
    ];
    // Control case: truly uniform `v_mov_b32 v5, 42` still skips half-wave exchange.
    let uniform_words = [
        0x7E0A_02AA, // v_mov_b32 v5, 42 (0x80 + 42)
        0x7E0E_0505, // v_readfirstlane_b32 s7, v5
        0x7E0C_0207, // v_mov_b32 v6, s7
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0xE070_1000,
        0x8002_0607, // buffer_store_dword v6, v7, s[8:11], 0 offen
    ];
    let u_native = native(&uniform_words);
    let u_p = ir::build(&u_native.code, u_native.wave64, u_native.vgprs).expect("ir::build");
    let (u_shader, u_wave, u_exchanges) =
        lowered_with_stats(&u_native, &u_p, &HashSet::new()).expect("lower");
    assert_eq!(u_wave, Some(32));
    assert_eq!(
        u_exchanges, 0,
        "truly uniform v_readfirstlane_b32 should not emit half-wave exchange",
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = u_shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, words, expected) in [
        ("v_fmac_f32", &fmac_words[..], 2.0f32.to_bits()),
        ("v_cndmask_b32", &cndmask_words[..], 10u32),
    ] {
        let native = native(words);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let (shader, wave_size, exchanges) =
            lowered_with_stats(&native, &p, &HashSet::new()).expect("lower");
        assert_eq!(wave_size, Some(32), "{label}: expected 32-lane split wave size");
        assert!(
            exchanges >= 1,
            "{label}: expected half-wave exchange (must not be marked lane_uniform), got {exchanges}",
        );
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Some(got) = on_the_card_64(&shader, [0, 0]) else {
            return;
        };
        assert!(
            got.iter().all(|&x| x == expected),
            "{label}: expected all 64 lanes to read {expected:#x}, got lane0={:#x} lane32={:#x} lane63={:#x}",
            got[0],
            got[32],
            got[63],
        );
    }
}

/// One console wave of 64 threads with 512 bytes of LDS: thread 0's lane zeroes the flag at
/// LDS 0, the threads below s0 raise it (`ds_write_b32` of 1) and read it back inside their
/// branch, and every thread reads it after the branch and stores it to the UAV. `second`: a
/// second write in the branch after the read, of that constant.
fn flag_program(second: Option<u32>) -> NativeProgram {
    let mut body = vec![
        0x7E12_0280, // v_mov_b32 v9, 0
    ];
    body.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 9)); // ds_write_b32 v9, v9
    let mut branch = vec![0x7E04_0281]; // v_mov_b32 v2, 1
    branch.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 2)); // ds_write_b32 v9, v2
    // (a scalar exit from the branch makes it blocks of their own, which only its lanes run)
    let mut rest = ds(DS_READ_B32, 3, 9, 0).to_vec(); // ds_read_b32 v3, v9
    if let Some(k) = second {
        rest.push(0x7E0C_02FF); // v_mov_b32 v6, lit
        rest.push(k);
        rest.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 6)); // ds_write_b32 v9, v6
    }
    branch.extend_from_slice(&[
        0xBF07_8001,                       // s_cmp_lg_u32 s1, 0
        0xBF85_0000 | rest.len() as u32, // s_cbranch_scc1 past the rest
    ]);
    branch.extend_from_slice(&rest);
    body.extend_from_slice(&[
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | branch.len() as u32, // s_cbranch_execz past the branch
    ]);
    body.extend_from_slice(&branch);
    body.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    body.extend_from_slice(&ds(DS_READ_B32, 5, 9, 0)); // ds_read_b32 v5, v9
    body.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ]);
    let mut native = native(&body);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [0x36 << 26 | (op as u32) << 18, (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_READ_B32: u8 = 0x36;

/// The wave size `native` lowers to on a host of 32-lane waves.
fn flag_wave(native: &NativeProgram) -> Option<u32> {
    let p = crate::spill::build(native).expect("build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    lowered.expect("lower").wave_size
}

#[test]
fn a_branch_that_only_raises_an_lds_flag_runs_in_waves_of_32() {
    // a read that misses the other host wave's write changes nothing the branch leaves behind
    assert_eq!(flag_wave(&flag_program(None)), Some(32));
    // after the read the branch writes the flag again, with the same constant
    assert_eq!(flag_wave(&flag_program(Some(1))), Some(32));
}

#[test]
fn a_branch_that_writes_lds_two_different_values_keeps_64_lanes() {
    // a second value after the read: what the flag ends up holding depends on what was read
    assert_eq!(flag_wave(&flag_program(Some(2))), Some(64));
}

/// A 64-lane console compute shader on a native wave64 host (`HOST_MAX_WAVE = 64`):
/// a 16-bit float wave sum (`half_sum`) after an early-out per-lane guard. On wave32
/// hosts `emit()` already retried without early-outs; native wave64 hosts must take the
/// same barrier-free retry path so `half_sum` sees all 64 lanes active.
#[test]
fn native_wave64_half_sum_inside_early_out_translates_and_runs_exact() {
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0017, // s_cbranch_execz +23 (past the 20-word reduction + 3-word store to s_endpgm)
    ];
    words.extend(
        super::wave::WAVE_OR_V7_TO_S7
            .iter()
            .map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }),
    );
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = crate::dxbc::Container::parse(&bytes).expect("empty container");
    let was = lower::TEST_HOST_WAVE.replace(Some(64));
    let shader = super::emit(&c, &native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let shader = shader.expect("native wave64 emit with early-out retry");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0f32
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0u16;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let expected_for = |below: u32| -> u32 {
        let mut tree: Vec<u16> = (0..64u16)
            .map(|t| if u32::from(t) < below { 0x3C00 + 37 * t } else { 0 })
            .collect();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        u32::from(tree[0])
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for below in [64u32, 37, 13, 1] {
        let Some(got) = on_the_card(&shader, [below, 0]) else { return };
        assert_eq!(got, expected_for(below), "native wave64 half_sum for {below} active lanes");
    }
    eprintln!("ran on the hardware device: native wave64 half_sum inside early-out matches console tree");
}

/// A 64-lane console compute shader on a native wave64 host (`HOST_MAX_WAVE = 64`):
/// inside a per-lane branch, `s_not_b64` builds a lane mask holding lanes outside `EXEC`
/// and `s_bcnt1_i32_b64` reads it as data. Native wave64 computes full 64-lane waves, so
/// retrying with every lane running the block produces the exact count of inactive lanes
/// with no group barriers.
#[test]
fn native_wave64_mask_outside_exec_inside_branch_translates_and_runs_exact() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0005, // s_cbranch_execz +5 (past body to restore)
        0xBE8E_087E, // s_not_b64 s[14:15], exec
        0xBE87_100E, // s_bcnt1_i32_b64 s7, s[14:15]
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = crate::dxbc::Container::parse(&bytes).expect("empty container");
    let was = lower::TEST_HOST_WAVE.replace(Some(64));
    let shader = super::emit(&c, &native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let shader = shader.expect("native wave64 emit with mask outside EXEC");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, expected) in [(64u32, 0u32), (37, 27), (13, 51), (1, 63)] {
        let Some(got) = on_the_card(&shader, [below, 0]) else { return };
        assert_eq!(got, expected, "native wave64 ~exec popcount for {below} active lanes");
    }
    eprintln!("ran on the hardware device: native wave64 mask outside EXEC matches console popcount");
}

/// One console wave of 64. `v_writelane_b32` writes its lane whatever EXEC is: inside
/// `if (t > s0) { ... }`, lane 0 (never inside) takes the scalar as soon as any thread is.
/// The compiler parks scalars this way in a VGPR that has other uses too (`unspill` leaves
/// those alone). Every thread stores what lane 0 holds after the join: 5 once the body
/// ran, 7 when the wave skipped it. As it is the program is refused (lane 0 is not in the
/// block); the forms without the skip have every lane there.
#[test]
fn a_lane_write_reaches_its_lane_outside_exec_in_every_form() {
    let words = [
        0x7E12_0287, // v_mov_b32 v9, 7
        0x7D82_0000, // v_cmp_lt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0003, // s_cbranch_execz to "join"
        0xBE86_0385, // s_mov_b32 s6, 5
        0xD761_0009,
        0x0001_0006, // v_writelane_b32 v9, s6, 0
        0xBEFE_040C, // join: s_mov_b64 exec, s[12:13]
        0xD760_0007,
        0x0001_0109, // v_readlane_b32 s7, v9, 0
        0x7E02_0207, // v_mov_b32 v1, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        eprintln!("  {:#06x} {}", i.machine.offset, i.name());
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let forms = [
        ("as is", Some(p.clone())),
        ("without skips", ir::without_skips(&p, false)),
        ("without skips, scalars followed", ir::without_skips(&p, true)),
    ];
    let mut wrong = Vec::new();
    for (name, q) in &forms {
        let Some(q) = q else {
            eprintln!("{name}: the skip is kept");
            continue;
        };
        lower::take_needs();
        let shader = match lowered(&native, q, &HashSet::new()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{name}: {e}");
                continue;
            }
        };
        // (threads above s0 inside) -> lane 0 after the join
        for (s0, expected) in [(10u32, 5u32), (62, 5), (63, 7)] {
            let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
            eprintln!("{name}: threads above {s0} inside -> {got} (the console: {expected})");
            if got != expected {
                wrong.push(format!("{name}: s0 = {s0}: {got}, not {expected}"));
            }
        }
    }
    // The same with lane 0 inside (`t < s0`): the write lands too.
    let mut inside = words;
    inside[1] = 0x7D88_0000; // v_cmp_gt_u32 vcc, s0, v0
    let native = native_of(&inside, true, 64);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let q = ir::without_skips(&p, false).expect("the skip is left out");
    lower::take_needs();
    let shader = lowered(&native, &q, &HashSet::new()).expect("lower");
    for (s0, expected) in [(10u32, 5u32), (0, 7)] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        eprintln!("lane 0 inside, without skips: threads below {s0} inside -> {got} (the console: {expected})");
        assert_eq!(got, expected, "lane 0 inside, s0 = {s0}");
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A 32-lane console wave whose waves are the host's, as the title's trace programs: they
/// keep a small array of scalars in the lanes of a VGPR, written by constant lane and read
/// by a computed one (`shadowmask:CS_TraceShadowRays_*`, `rtdirectionalshadows:*`).
/// `if (t > s0) { v_writelane_b32 v9, 5, 0 }`, then every thread stores lane s1 (= 0) of v9.
/// Lane 0 is never inside: per lane the program is refused, and with the skip taken by the
/// wave lane 0 is there and takes the scalar once some lane is inside.
#[test]
fn a_lane_array_is_written_and_read_with_the_skip_wave_wide() {
    let words = [
        0x7E12_0287, // v_mov_b32 v9, 7
        0x7D82_0000, // v_cmp_lt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0003, // s_cbranch_execz to "join"
        0xBE86_0385, // s_mov_b32 s6, 5
        0xD761_0009,
        0x0001_0006, // v_writelane_b32 v9, s6, 0
        0xBEFE_030D, // join: s_mov_b32 exec_lo, s13
        0xD760_0007,
        0x0000_0309, // v_readlane_b32 s7, v9, s1
        0x7E02_0207, // v_mov_b32 v1, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == lower::LANE_ELSEWHERE),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (threads above s0 inside) -> lane 0 after the join
    for (s0, expected) in [(10u32, 5u32), (30, 5), (31, 7)] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        assert_eq!(got, expected, "threads above {s0} inside");
    }
    // The last resort, among the lanes that are there: it lowers, and the write is lost.
    let lax = lower::with_lanes_as_they_stand(|| lowered(&native, &p, &HashSet::new())).expect("lower");
    let Some(got) = on_the_card(&lax, [10, 0]) else { return };
    assert_eq!(got, 7, "as before: lane 0 is not inside");
    eprintln!("ran on the hardware device: lane 0 takes the scalar once some lane is inside");
}


/// A console wave32 program with console lanes (`v_mbcnt`) in a 32 x 2 group: thread (x, y)
/// stores 1 at dword 32y + x. Every dword must be written once: the host's relabelled thread
/// ids are a bijection of the group. A 32-lane host need not give a host wave a 32-aligned run
/// of flat ids (2-D groups are filled from 16 x 2 tiles on NVIDIA and RDNA), so the first flat
/// id of a wave rounded down gave two waves the same ids (issue #1243); the waves now take
/// their threads from a group counter. With `VICTORY_THREAD_TICKET=0` (the old rounding) the
/// count of dwords written is printed, not checked: it depends on the card.
#[test]
fn the_console_threads_of_a_two_dimensional_group_are_each_given_once() {
    let words = [
        0xD765_0003, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0x0001_007E,
        0x3408_0285, // v_lshlrev_b32 v4, 5, v1
        0x4A08_0104, // v_add_nc_u32 v4, v4, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4, // buffer_store_dword v5, v4, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, false, 64);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: 5 << 1 | 1 << 11, // 5 user SGPRs, thread ids x and y in v0, v1
        threads: [32, 2, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lower_with = |ticket: bool| {
        // SAFETY: tests that read this switch hold ONE_AT_A_TIME
        unsafe { std::env::set_var("VICTORY_THREAD_TICKET", if ticket { "1" } else { "0" }) };
        let shader = lowered_on(&native, &p, &HashSet::new(), 32);
        unsafe { std::env::remove_var("VICTORY_THREAD_TICKET") };
        shader.expect("lower")
    };
    let shader = lower_with(true);
    let old = lower_with(false);
    assert_ne!(shader.dxil, old.dxil, "the counter is used for a group of two host waves");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let run = |shader: &HostShader| -> Option<Vec<u32>> {
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("the hardware device does not take this program: {e}; not run");
                return None;
            }
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        Some(out[0][..64].to_vec())
    };
    let Some(got) = run(&shader) else { return };
    let missing: Vec<usize> = (0..64).filter(|&i| got[i] != 1).collect();
    assert!(missing.is_empty(), "dwords not written once: {missing:?}");
    if let Some(before) = run(&old) {
        let written = before.iter().filter(|&&d| d == 1).count();
        eprintln!("the old rounding on this device: {written} of 64 dwords written");
    }
    eprintln!("ran on the hardware device: 64 of 64 console threads, each once");
}
