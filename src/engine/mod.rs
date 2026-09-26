mod ffi;

pub struct Engine {
    ptr: *mut ffi::Engine,
}

pub fn set_verbose(verbose: bool) {
    unsafe { ffi::engine_set_verbose(verbose) };
}

fn last_error() -> String {
    let ptr = unsafe { ffi::engine_last_error() };
    if ptr.is_null() {
        return "engine failed".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

impl Engine {
    /// The calling thread owns the platform for the life of this value.
    pub fn new() -> Result<Self, String> {
        let ptr = unsafe { ffi::engine_init() };
        if ptr.is_null() {
            return Err(last_error());
        }
        Ok(Self { ptr })
    }

    pub fn platform_name(&self) -> String {
        let ptr = unsafe { ffi::engine_platform_name() };
        if ptr.is_null() {
            return "unknown".to_string();
        }
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    /// Row-major `(m, k) × (k, n) → (m, n)`.
    pub fn matmul(&self, m: u32, k: u32, n: u32, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let a_len = (m as usize)
            .checked_mul(k as usize)
            .ok_or_else(|| "shape overflow".to_string())?;
        let b_len = (k as usize)
            .checked_mul(n as usize)
            .ok_or_else(|| "shape overflow".to_string())?;
        let out_len = (m as usize)
            .checked_mul(n as usize)
            .ok_or_else(|| "shape overflow".to_string())?;
        if a.len() != a_len || b.len() != b_len {
            return Err(format!(
                "expected a[{a_len}] and b[{b_len}], got a[{}] and b[{}]",
                a.len(),
                b.len()
            ));
        }
        let mut out = vec![0.0; out_len];
        let status = unsafe {
            ffi::engine_matmul(
                self.ptr,
                m,
                k,
                n,
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error());
        }
        Ok(out)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::engine_deinit(self.ptr) };
    }
}
