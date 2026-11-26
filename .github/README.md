# VIctory — integration fork (RX 7900 XTX / Windows 10)

This fork tracks [fxceless/victory-fork](https://github.com/fxceless/victory-fork) (itself a fork of
[Marvisak/VIctory](https://github.com/Marvisak/VIctory), GPL-3.0-or-later) and **merges into one tree the
open pull requests of the other contributors** that were tested on an AMD card under Windows 10.
The project's own documentation is in [`README.md`](../README.md) (fxceless's text, kept as is); this
page describes only what is different here.

**No game files, code or assets are in this repository**, and none are needed to build it. It is not
affiliated with or endorsed by the game's developer or publisher.

## Branches

| Branch | What it is |
|--------|------------|
| `main` | fxceless `main` + every pull request below + our CI fixes. The CI of this repository runs here |
| `integration` | the same tree as `main` (the working branch of the merge script) |

fxceless's new commits are merged in about every two hours; a pull request is taken (or taken again when
its author pushes) only when the test suites show no new failure on the test machine.

## Status (2026-11-27)

- **Base:** fxceless `main` `4c60466` (2026-11-26 evening), plus 47 pull requests and the fixes below.
- **CI of this fork:** green (Clippy, Clippy with diagnostics, workspace tests, `victory-gpu` tests).
  fxceless `main` is red since `58de9bf` only because two source files passed the asset policy's
  512 KiB limit; here the limit is 2 MiB (run logs 64 MiB, gzipped 32 MiB).
- **Test machine:** Radeon RX 7900 XTX (RDNA 3), driver 32.0.32015.7029, Windows 10 Pro 22H2 (19045),
  i7-13700KF, 96 GB RAM, Rust 1.98.1 MSVC.
- **Tests on that machine** (whole merged tree on fxceless `4c60466`, 2026-11-27): `victory-shader` 298 passed,
  0 failed; `victory-gpu` 229 passed, 0 failed (one test left out, see below). The same suites on fxceless `main` alone: 27 failures
  (26 shader tests that need shader model 6.6, which Windows 10's own D3D12 runtime does not give —
  fixed by the Agility SDK of #402 — and one `victory-gpu` test). No test that passes on fxceless
  `main` fails here.
- **The game on that machine** (own dump, Series X profile): all 25,636 pipelines of the title's
  shader archive are created by the AMD driver, none refused, no compiler crash. Play sessions of
  up to 51 minutes in the open world with no device removal; about 7 to 12 frames a second in the world
  at 4K, 30 in menus. Those sessions ran on a tree close to this one, not yet on this exact commit.
- **Known on Windows 10 only:** `victory-loader`'s `crash::tests::unwinds_like_the_system_walk` fails
  locally (also on fxceless `main`); it passes on the CI runner.
- **Crashes on the RX 7900 XTX, also on fxceless `main`:** `victory-gpu`'s
  `tiles::tests::card_wrote_failure_does_not_poison_card_mask` ends the test process with an access
  violation (3 of 3 runs) since `VICTORY_TILES_SMALL_WHEN_FULL_MB` defaults to 512 (fxceless `30c14669`);
  the other 229 `victory-gpu` tests pass. It passes on the CI runner (WARP).

## Pull requests merged here

All by their authors, unchanged except where a merge conflict had to be resolved (noted below).

| PR | Title | Author |
|----|-------|--------|
| [#402](https://github.com/fxceless/victory-fork/pull/402) | Windows 10: load the D3D12 Agility SDK runtime for shader model 6.6 | domints |
| [#875](https://github.com/fxceless/victory-fork/pull/875) | GPU tests: one test on the software adapter at a time (#850) | eazyves |
| [#849](https://github.com/fxceless/victory-fork/pull/849) | Indirect test: the two groups charge the budget in either order (the test failed 5 runs in 6) | MamaLeVrai |
| [#692](https://github.com/fxceless/victory-fork/pull/692) | Remove bindless dependency from the depth-bounds test | TokyoMkt |
| [#129](https://github.com/fxceless/victory-fork/pull/129) | Shaders: query untyped buffer dimensions through a raw handle | vicennpacheco |
| [#615](https://github.com/fxceless/victory-fork/pull/615) | Shaders: support cross-half ds_bpermute_b32 and fix group <= 32 OOB lanes on 32-lane GPUs | orazal12 |
| [#592](https://github.com/fxceless/victory-fork/pull/592) | Shaders: a lane-branch region that only raises an LDS flag runs as two host waves (one more wave64 compute on NVIDIA) | FrancescoGelosoEterea |
| [#561](https://github.com/fxceless/victory-fork/pull/561) | Shaders/lower: pin partial tail wave in wave64-on-wave32 wave_base() | AscilCH |
| [#461](https://github.com/fxceless/victory-fork/pull/461) | Shaders: skip cross-half exchange in active_f64_sum for local waves and groups <= 32 | Fimpos |
| [#277](https://github.com/fxceless/victory-fork/pull/277) | Shaders: keep literal select mask bits when lowering local half-waves | fluffyflattershy-afk |
| [#146](https://github.com/fxceless/victory-fork/pull/146) | Shaders: lower image_store with any non-zero channel mask (including sparse) | davi713albano-coder |
| [#137](https://github.com/fxceless/victory-fork/pull/137) | Shaders: lower SV_RenderTargetArrayIndex and SV_ViewportArrayIndex exports | davi713albano-coder |
| [#657](https://github.com/fxceless/victory-fork/pull/657) | Shaders: lower pixel shader SV_Coverage input via dx.op.coverage | davi713albano-coder |
| [#233](https://github.com/fxceless/victory-fork/pull/233) | Shaders: lower the console's scalar bit searches (s_flbit_i32_b64, s_ff0_i32_b32/b64) | pablogonz12 |
| [#1002](https://github.com/fxceless/victory-fork/pull/1002) | Shaders: the tiled cascade shadow programs translate exactly on 32-lane hosts | brunofgmag |
| [#558](https://github.com/fxceless/victory-fork/pull/558) | Shaders: retry barrier-free all-lane forms on native wave64 hosts | tulerws |
| [#119](https://github.com/fxceless/victory-fork/pull/119) | Shaders: write v_writelane_b32 under wave-level EXEC (#111) | adritoyalser |
| [#135](https://github.com/fxceless/victory-fork/pull/135) | Shaders: 64-bit integer compares on their dwords (AMD orders a shared-dword i64 compare wrong) | alexdeliu |
| [#694](https://github.com/fxceless/victory-fork/pull/694) | Reuse the D3D12 queue in shader tests | TokyoMkt |
| [#377](https://github.com/fxceless/victory-fork/pull/377) | Shaders: lane masks read through a join need the console's wave size; tests skip devices without 32-lane waves (#356) | RenzoWit35 |
| [#354](https://github.com/fxceless/victory-fork/pull/354) | Shaders: include v_readfirstlane_b32 in wave64 exchange and check implicit reads in lane_uniform | AscilCH |
| [#219](https://github.com/fxceless/victory-fork/pull/219) | Shaders: a lane write or read finds its lane where not every lane runs (#111) | Alexr03 |
| [#260](https://github.com/fxceless/victory-fork/pull/260) | Shaders: CalculateLOD gets only the spatial coordinates | Ortomanic |
| [#278](https://github.com/fxceless/victory-fork/pull/278) | Tests: add VALU hardware tests for integer min3/max3/med3, class_f32 and s_bfe_u32 | towiice |
| [#319](https://github.com/fxceless/victory-fork/pull/319) | Shader tests: special values of v_rcp/v_rsq/v_sqrt/v_log/v_exp_f32 | vladbondarenko97 |
| [#320](https://github.com/fxceless/victory-fork/pull/320) | Shader tests: f32/int conversions saturate, take NaN as 0, round to even | vladbondarenko97 |
| [#635](https://github.com/fxceless/victory-fork/pull/635) | Shaders: lower v_mad_i64_i32; test the 64-bit integer vector operations against their definitions | Gruzzberry |
| [#636](https://github.com/fxceless/victory-fork/pull/636) | Shaders: lower six integer operations; test the integer vector and scalar ops and the scalar compares against their definitions | Gruzzberry |
| [#637](https://github.com/fxceless/victory-fork/pull/637) | Shader tests: vector compares, carries, conversions and float arithmetic against the RDNA2 definitions | Gruzzberry |
| [#638](https://github.com/fxceless/victory-fork/pull/638) | Shaders: ds_sub_u32 adds the negated value (DXIL has no atomic subtraction); test the LDS atomics against their definitions | Gruzzberry |
| [#639](https://github.com/fxceless/victory-fork/pull/639) | Shaders: lower s_brev_b64; test the scalar 64-bit operations against their definitions | Gruzzberry |
| [#658](https://github.com/fxceless/victory-fork/pull/658) | Shaders: lower v_cmp_{f,t}_{i64,u64} constant 64-bit compares | davi713albano-coder |
| [#726](https://github.com/fxceless/victory-fork/pull/726) | Shaders: lower 64-bit shifts without INT64 | scoresby13 |
| [#741](https://github.com/fxceless/victory-fork/pull/741) | Shaders: lower mul_hi without INT64 | scoresby13 |
| [#948](https://github.com/fxceless/victory-fork/pull/948) | Shaders: honor clamp on unsigned VOP3 add and subtract (port of upstream VIctory #1398) | Moon0755 |
| [#951](https://github.com/fxceless/victory-fork/pull/951) | Shaders: compute signed overflow SCC for s_add_i32, s_addk_i32, and s_sub_i32 (port of upstream VIctory #1095) | Moon0755 |
| [#957](https://github.com/fxceless/victory-fork/pull/957) | Shaders: return zero for unused image query extents (port of upstream VIctory #1313) | Moon0755 |
| [#1019](https://github.com/fxceless/victory-fork/pull/1019) | Shaders: preserve signaling NaNs in typed buffer loads (port of upstream VIctory #1642) | Moon0755 |
| [#1030](https://github.com/fxceless/victory-fork/pull/1030) | Shaders: return zero for inactive compute swizzle sources (port of upstream VIctory #1416) | Moon0755 |
| [#1036](https://github.com/fxceless/victory-fork/pull/1036) | Shaders: ignore op_sel on 16-bit comparisons (port of upstream VIctory #926) | Moon0755 |
| [#1040](https://github.com/fxceless/victory-fork/pull/1040) | Shaders: lower v_div_fixup_f16 (port of upstream VIctory #379) | Moon0755 |
| [#1042](https://github.com/fxceless/victory-fork/pull/1042) | Shaders: support saturating clamp on packed 16-bit integer ops (port of upstream VIctory #1142) | Moon0755 |
| [#808](https://github.com/fxceless/victory-fork/pull/808) | shader: lower packed dot products to DXIL dot4AddPacked and fmad | UnkwnDev |
| [#1165](https://github.com/fxceless/victory-fork/pull/1165) | Skip ray-box children whose title-memory pages are missing | BradRoland |
| [#1191](https://github.com/fxceless/victory-fork/pull/1191) | Trim unused trailing dwords from grouped scalar buffer reads | BradRoland |
| [#1261](https://github.com/fxceless/victory-fork/pull/1261) | Shaders: console threads of 2-D groups on 32-lane hosts come from a group counter (fixes #1243) | Garu667 |
| [#1166](https://github.com/fxceless/victory-fork/pull/1166) | Fix Windows shader warm-up, wave permutes, and BVH residency | Lkxex |

Conflicts resolved by hand, against fxceless `main` or against each other: #849, #129, #592 (since
taken by fxceless itself), #558, #135, #219, #639, #1165, #1261. Of #1166 only the shader warm-up part
is here (its BVH and lane-permute changes are covered by fxceless `main` and #1165).

**Not merged:** #1026 (already in fxceless `main`); #1140 (replaces the LDS analysis on an older base
and would drop #592: waiting for its author to rebase).

## Credits

Everything here is the work of the [VIctory](https://github.com/Marvisak/VIctory) authors, of fxceless and
of the authors of the pull requests above. This fork only merges, tests and resolves conflicts; that work
was AI-assisted (Claude Code).
