
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xaxpy kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

#define USE_CL_MAD 0

#if USE_CL_MAD == 1
  #define MultiplyAdd(c, a, b) c = mad(a, b, c)
#else
  #define MultiplyAdd(c, a, b) c += a * b
#endif

// Full version of the kernel with offsets and strided accesses
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void func( const int                      n,
           const float                    arg_alpha,
           const __global float* restrict xgm,
           const int                      x_offset,
           const int                      x_inc,
           __global float*                ygm,
           const int                      y_offset,
           const int                      y_inc
         )
{
  const float alpha = arg_alpha;

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  #pragma unroll
  for (int id = get_global_id(0); id<n; id += get_global_size(0)) {
    float xvalue = xgm[id*x_inc + x_offset];
    MultiplyAdd(ygm[id*y_inc + y_offset], alpha, xvalue);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
