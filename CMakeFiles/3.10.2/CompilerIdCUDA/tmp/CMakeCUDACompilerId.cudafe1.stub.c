#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "CMakeCUDACompilerId.fatbin.c"
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_27_CMakeCUDACompilerId_cpp1_ii_bd57c623(void) __attribute__((__constructor__));
static void __nv_cudaEntityRegisterCallback(void **__T20){__nv_dummy_param_ref(__T20);__nv_save_fatbinhandle_for_managed_rt(__T20);}
static void __sti____cudaRegisterAll_27_CMakeCUDACompilerId_cpp1_ii_bd57c623(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}

#pragma GCC diagnostic pop
