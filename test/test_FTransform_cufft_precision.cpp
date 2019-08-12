#include <iostream>
#include <complex>
#include <string>
#include <sstream>

#include "catch.hpp"
#include "cufft.h"

#include "FTransformAgent.h"
#include "FTransform.h"
#include "dsp/OptimalFFT.h"

#include "TestConfig.hpp"

static util::TestConfig test_config;

void throw_on_error (const cudaError& error, const std::string& msg="")
{
  if (error != 0) {
    std::ostringstream oss;
    oss << msg << " Error: " << error ;
    throw oss.str();
  }
}

TEST_CASE ("FFTW and CUFFT produce numerically similar results")
{

  std::vector<float> tols = test_config.get_thresh();
  int fft_size = test_config.get_field<int>("test_FTransform_cufft_precision.fft_size");
  float scale = test_config.get_field<float>("test_FTransform_cufft_precision.scale");

  float atol = tols[0];
  float rtol = tols[1];

  auto random_gen = util::random<float>();

  typedef std::complex<float> complex_f;

  size_t fft_size_bytes = fft_size * sizeof(cufftComplex);
  std::vector<complex_f> in_data_cpu (fft_size);
  std::vector<complex_f> out_data_cpu (fft_size);
  std::vector<complex_f> out_data_gpu (fft_size);
  for (int idx=0; idx<fft_size; idx++) {
    in_data_cpu[idx] = complex_f(scale*random_gen(), scale*random_gen());
  }

  FTransform::Plan* forward = FTransform::Agent::current->get_plan(fft_size, FTransform::fcc);
  forward->frc1d(fft_size, (float*) out_data_cpu.data(), (float*) in_data_cpu.data());

  // fftwf_plan plan_fftw = fftwf_plan_dft_1d(
  //   fft_size,
  //   (fftwf_complex*) in_data_cpu.data(),
  //   (fftwf_complex*) out_data_cpu.data(),
  //   FFTW_FORWARD, FFTW_ESTIMATE);
  //
  // fftwf_execute(plan_fftw);
  // fftwf_destroy_plan(plan_fftw);

  cufftComplex* in_data_gpu_d;
  cufftComplex* out_data_gpu_d;
  cudaError error;
  cufftResult result;
  error = cudaMalloc((void**) &in_data_gpu_d, fft_size_bytes);
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

  cufftHandle plan_cufft;
  result = cufftPlan1d (&plan_cufft, fft_size, CUFFT_C2C, 1);
  cufftExecC2C(
    plan_cufft,
    in_data_gpu_d,
    out_data_gpu_d,
    CUFFT_FORWARD
  );
  result = cufftDestroy(plan_cufft);

  error = cudaMemcpy(
    (cufftComplex*) out_data_gpu.data(),
    out_data_gpu_d,
    fft_size_bytes,
    cudaMemcpyDeviceToHost
  );
  throw_on_error(error, "cudaMemcpy");

  // now compare the results
  unsigned nclose = util::nclose(out_data_cpu, out_data_gpu, atol, rtol);
  if (util::config::verbose)
  {
    std::cerr << "test_FTransform_cufft_precision: "
      << nclose << "/" << fft_size << " ("
      << 100 * (float) nclose / fft_size << "%)"
      << std::endl;
    std::cerr << "test_FTransform_cufft_precision: "
      << " max cpu=" << util::max<float>((float*) out_data_cpu.data(), fft_size * 2)
      << ", max gpu=" << util::max<float>((float*) out_data_gpu.data(), fft_size * 2)
      << std::endl;

  }
  REQUIRE(nclose == out_data_cpu.size());
}
