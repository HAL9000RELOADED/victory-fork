//! The D3D12 Agility SDK in this crate's unit-test executable.
//!
//! Windows 10's in-box D3D12 stops at shader model 6.5 and refuses the root
//! signatures translated shaders use, so the tests that make host pipelines
//! from them failed there. D3D12 looks for these two exports on the
//! process's executable, as for `victory.exe` (`apps/victory/src/agility.rs`);
//! they are exported (below) only when the runtime is in
//! `<target>/<profile>/D3D12/` at build time (`build.rs`). Unexported, they
//! are inert and the in-box runtime is used.
#![allow(non_upper_case_globals)]

/// The runtime's SDK version (`Microsoft.Direct3D.D3D12` 1.620.2). Keep it
/// in step with `victory.exe`'s.
#[unsafe(no_mangle)]
pub static D3D12SDKVersion: u32 = 620;

/// A C string pointer D3D12 reads once at device creation.
#[repr(transparent)]
pub struct SdkPath(*const std::ffi::c_char);

// SAFETY: points at a static, immutable C string.
unsafe impl Sync for SdkPath {}

/// Where `D3D12Core.dll` is, relative to the executable: test executables
/// are in `<profile>/deps/`, the runtime in `<profile>/D3D12/`.
#[unsafe(no_mangle)]
pub static D3D12SDKPath: SdkPath = SdkPath(c"..\\D3D12\\".as_ptr());

/// The linker directive that exports the two statics above, when the runtime
/// is staged (`build.rs`). It lives in this test-only module, so only the
/// crate's unit-test executable carries it: the integration tests link the
/// library without this module, and an export of a symbol they lack would
/// not link.
#[cfg(agility_runtime)]
#[used]
#[unsafe(link_section = ".drectve")]
static EXPORTS: [u8; 56] = *b" /EXPORT:D3D12SDKVersion,DATA /EXPORT:D3D12SDKPath,DATA ";
