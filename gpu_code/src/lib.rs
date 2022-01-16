#![feature(abi_ptx, core_intrinsics)]
#![no_std]
#![feature(asm_experimental_arch)]

use core::arch::asm;

#[inline(always)]
pub fn clock() -> u32{
    let mut result: u32;
    unsafe {
        asm!(
            "mov.u32 {output}, %clock;",
            output = out(reg32) result,
        );
    }
    result
}

#[inline(always)]
pub fn thread_idx_x() -> u32{
    let mut result: u32;
    unsafe {
        asm!(
            "mov.u32 {r}, %tid.x;",
            r = out(reg32) result,
        );
    }
    result
}

#[inline(always)]
pub fn block_idx_x() -> u32{
    let mut result: u32;
    unsafe {
        asm!(
            "mov.u32 {r}, %ctaid.x;",
            r = out(reg32) result,
        );
    }
    result
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn my_kernel(
    _input: *mut u32,
    _input_len: usize,
    output: *mut u32,
    _output_len: usize,
) {
    let idx = (block_idx_x() * 1024 + thread_idx_x()) as isize;

    let start = clock();
    let end = clock();

    *output.offset(idx) = end - start;
}


#[panic_handler]
pub unsafe fn breakpoint_panic_handler(_: &::core::panic::PanicInfo) -> ! {
    core::intrinsics::breakpoint();
    core::hint::unreachable_unchecked();
}   
