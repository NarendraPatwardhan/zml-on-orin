use std::ffi::{c_char, c_int};

#[repr(C)]
pub struct Engine {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn engine_init() -> *mut Engine;
    pub fn engine_deinit(ctx: *mut Engine);

    pub fn engine_matmul(
        ctx: *mut Engine,
        m: u32,
        k: u32,
        n: u32,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
    ) -> c_int;

    pub fn engine_last_error() -> *const c_char;
    pub fn engine_platform_name() -> *const c_char;
    pub fn engine_set_verbose(verbose: bool);
}
