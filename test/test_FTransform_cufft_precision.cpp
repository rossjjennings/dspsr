#include <iostream>
#include <complex>
#include <string>
#include <sstream>

#include "catch.hpp"
#include "cufft.h"

#include "FTransformAgent.h"
#include "FTransform.h"
#include "dsp/OptimalFFT.h"

#include "util/TestConfig.hpp"

static test::util::TestConfig test_config;

void throw_on_error (const cudaError& error, const std::string& msg="")
{
  if (error != 0) {
    std::ostringstream oss;
    oss << msg << " Error: " << error ;
    throw oss.str();
  }
}

TEST_CASE ("Consecutive FFTW and CUFFT calls produce numerically similar results",
          "[cuda][no_file][cufft_precision]")
{


  if (test::util::config::verbose) {
    std::cerr << "test_FTransform_cufft_precision" << std::endl;
  }

  std::vector<float> tols = test_config.get_thresh();
  int nchan = test_config.get_field<int>("test_FTransform_cufft_precision.nchan");
  int forward_fft_size = test_config.get_field<int>("test_FTransform_cufft_precision.fft_size");
  int backward_fft_size = nchan * forward_fft_size;
  // float scale = test_config.get_field<float>("test_FTransform_cufft_precision.scale");

  float atol = tols[0];
  float rtol = tols[1];

  auto random_gen = test::util::random<float>();

  typedef std::complex<float> complex_f;

  std::vector<complex_f> in_data_cpu (backward_fft_size);
  std::vector<complex_f> out_data_cpu0 (backward_fft_size);
  std::vector<complex_f> out_data_cpu (backward_fft_size);
  std::vector<complex_f> out_data_gpu (backward_fft_size);

  for (int idx=0; idx<backward_fft_size; idx++) {
    in_data_cpu[idx] = complex_f(random_gen(), random_gen());
  }

  FTransform::Plan* forward = FTransform::Agent::current->get_plan(
    forward_fft_size, FTransform::fcc);
  FTransform::Plan* backward = FTransform::Agent::current->get_plan(
    backward_fft_size, FTransform::bcc);

  float* in_ptr = (float*) in_data_cpu.data();
  float* out_ptr = (float*) out_data_cpu0.data();
  for (int ichan=0; ichan<nchan; ichan++)
  {
    forward->fcc1d(forward_fft_size, out_ptr, in_ptr);
    in_ptr += 2*forward_fft_size;
    out_ptr += 2*forward_fft_size;
  }
  backward->bcc1d(
    backward_fft_size,
    (float*) out_data_cpu.data(),
    (float*) out_data_cpu0.data()
  );

  size_t fft_size_bytes = backward_fft_size * sizeof(cufftComplex);

  cufftComplex* in_data_gpu_d;
  cufftComplex* out_data_gpu_d0;
  cufftComplex* out_data_gpu_d;

  cudaError error;
  cufftResult result;

  error = cudaMalloc((void**) &in_data_gpu_d, fft_size_bytes);
  throw_on_error(error, "cudaMalloc");

  error = cudaMalloc((void**) &out_data_gpu_d0, fft_size_bytes);
  throw_on_error(error, "cudaMalloc");

  error = cudaMalloc((void**) &out_data_gpu_d, fft_size_bytes);
  throw_on_error(error, "cudaMalloc");

  error = cudaMemcpy(
    in_data_gpu_d,
    (cufftComplex*) in_data_cpu.data(),
    fft_size_bytes,
    cudaMemcpyHostToDevice
  );
  throw_on_error(error, "cudaMemcpy");

  cufftHandle cufft_forward;
  cufftHandle cufft_backward;

  int rank = 1; // 1D transform
  int n[] = {forward_fft_size}; /* 1d transforms of length _input_fft_length */
  int howmany = nchan;
  int idist = forward_fft_size;
  int odist = forward_fft_size;
  int istride = 1;
  int ostride = 1;
  int *inembed = n, *onembed = n;

  result = cufftPlanMany(
    &cufft_forward, rank, n,
    inembed, istride, idist,
    onembed, ostride, odist,
    CUFFT_C2C, howmany);

  // result = cufftPlan1d (&cufft_forward, forward_fft_size, CUFFT_C2C, 1);
  result = cufftPlan1d (&cufft_backward, backward_fft_size, CUFFT_C2C, 1);

  result = cufftExecC2C(
    cufft_forward, in_data_gpu_d, out_data_gpu_d0, CUFFT_FORWARD
  );

  // cufftComplex* in_ptr_d = in_data_gpu_d;
  // cufftComplex* out_ptr_d = out_data_gpu_d0;
  // for (int ichan=0; ichan<nchan; ichan++)
  // {
  //   result = cufftExecC2C(
  //     cufft_forward,
  //     in_ptr_d,
  //     out_ptr_d,
  //     CUFFT_FORWARD
  //   );
  //   in_ptr_d += forward_fft_size;
  //   out_ptr_d += forward_fft_size;
  // }

  result = cufftExecC2C(
    cufft_backward,
    out_data_gpu_d0,
    out_data_gpu_d,
    CUFFT_INVERSE
  );
  result = cufftDestroy(cufft_forward);
  result = cufftDestroy(cufft_backward);

  error = cudaMemcpy(
    (cufftComplex*) out_data_gpu.data(),
    out_data_gpu_d,
    fft_size_bytes,
    cudaMemcpyDeviceToHost
  );
  throw_on_error(error, "cudaMemcpy");

  // for (unsigned idx=0; idx<out_data_cpu.size(); idx++)
  // {
  //   std::cerr << out_data_cpu[idx] << ", " << out_data_gpu[idx] << std::endl;
  // }

  // now compare the results
  unsigned nclose = test::util::nclose(out_data_cpu, out_data_gpu, atol, rtol);
  if (test::util::config::verbose)
  {
    std::cerr << "test_FTransform_cufft_precision: "
      << nclose << "/" << backward_fft_size << " ("
      << 100 * (float) nclose / backward_fft_size << "%)"
      << std::endl;
    std::cerr << "test_FTransform_cufft_precision: "
      << " max cpu=" << test::util::max<float>((float*) out_data_cpu.data(), backward_fft_size * 2)
      << ", max gpu=" << test::util::max<float>((float*) out_data_gpu.data(), backward_fft_size * 2)
      << std::endl;

  }
  REQUIRE(nclose == out_data_cpu.size());
}
