use cuda_driver_sys::*;
use cuda_runtime_sys::*;
use std::ffi::{CString, c_void};

const PTX_PATH: &str = "../gpu_code/target/nvptx64-nvidia-cuda/release/gpu_code.ptx";
                        
fn main() {
    unsafe{unsafe_main()}
}

unsafe fn allocate<T>(size: usize) -> CUdeviceptr {
    let mut dptr: CUdeviceptr = 0;
    let error = cuMemAlloc_v2(
        &mut dptr as *mut CUdeviceptr, 
        size * core::mem::size_of::<T>()
    );
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);
    dptr
}

unsafe fn unsafe_main() {
    // Init the cuda library
    cuInit(0);

    // Get the first available device
    let mut device: CUdevice = 0;
    let error = cuDeviceGet(&mut device as *mut CUdevice, 0);    
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);

    // create a context
    let mut context: CUcontext = core::ptr::null_mut();
    let error = cuCtxCreate_v2(
        &mut context as *mut CUcontext, 
        cudaDeviceScheduleAuto, 
        device
    );
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);

    // Load the PTX file
    let mut module: CUmodule = core::ptr::null_mut();
    let file_name = CString::new(PTX_PATH).unwrap();
    let error = cuModuleLoad(
        &mut module as *mut CUmodule,
        file_name.as_ptr(),
    );
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);

    // Create a stream
    let mut stream = core::mem::MaybeUninit::uninit().assume_init();
    let error = cuStreamCreate(
        &mut stream as *mut CUstream, 
        0,
    );
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);

    // allocate the results buffer in the device
    let mut input_len = 4096 / core::mem::size_of::<u32>();
    let mut inputs = allocate::<u32>(input_len); 
    let mut output_len = 1024 * 1024;
    let mut outputs = allocate::<u32>(output_len); 

    // get the kernel function to call
    let func_name = CString::new("kernel").unwrap();
    let mut func: CUfunction = core::ptr::null_mut();
    let error = cuModuleGetFunction(
        &mut func as *mut CUfunction, 
        module, 
        func_name.as_ptr(),
    );
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);

    // Run the kernel
    let mut args = vec![
        &mut inputs     as *mut _ as *mut c_void, 
        &mut input_len  as *mut _ as *mut c_void,
        &mut outputs    as *mut _ as *mut c_void, 
        &mut output_len as *mut _ as *mut c_void,
    ];
    let error = cuLaunchKernel(
        func, 
        1024,
        1,
        1,
        1024,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        core::ptr::null_mut(),
    );
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);

    // wait for the gpu to finish
    let error = cuStreamSynchronize(stream);
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);
    
    let mut result_buffer = vec![0_u32; output_len];
    let error = cuMemcpyDtoH_v2(
        result_buffer.as_mut_ptr() as _,
        outputs,
        output_len * core::mem::size_of::<u32>(),
    );
    assert_eq!(error, cudaError_enum::CUDA_SUCCESS);

    println!("{:?}", result_buffer);
}
