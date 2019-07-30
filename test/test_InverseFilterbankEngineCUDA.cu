#include <complex>
#include <vector>
#include <iostream>

#include "catch.hpp"

#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "Rational.h"

#include "util.hpp"
#include "util.cuda.hpp"
#include "InverseFilterbank_test_config.h"

void check_error (const char*);

TEST_CASE ("InverseFilterbankEngineCUDA", "") {
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);

  auto pred_setup_fft_plan = [] (cufftResult r) -> bool { return r == CUFFT_SUCCESS; };

  SECTION ("can create forward cufft plan")
  {

    std::vector<cufftResult> results = engine.setup_forward_fft_plan(
      1024, 8, CUFFT_R2C);
    REQUIRE(all_of(results.begin(), results.end(), pred_setup_fft_plan) == true);

    results = engine.setup_forward_fft_plan(1024, 8, CUFFT_C2C);
    REQUIRE(all_of(results.begin(), results.end(), pred_setup_fft_plan) == true);
  }

  SECTION ("can create backward cufft plan")
  {
    std::vector<cufftResult> results = engine.setup_backward_fft_plan(1024, 8);
    REQUIRE(all_of(results.begin(), results.end(), pred_setup_fft_plan) == true);
  }
}

TEST_CASE ("cufft kernels can operate on data", "")
{
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);

  unsigned fft_length = 1024;
  unsigned nchan = 8;
  unsigned ndat = fft_length * nchan;

  SECTION (
    "forward FFT kernel can operate on data"
  )
  {
    std::vector<std::complex<float>> in_c (ndat);
    std::vector<float> in_r (ndat);
    std::vector<std::complex<float>> out (ndat);

    engine.setup_forward_fft_plan(fft_length, nchan, CUFFT_R2C);
    engine.apply_cufft_forward<CUFFT_R2C>(in_r, out);
    engine.setup_forward_fft_plan(fft_length, nchan, CUFFT_C2C);
    engine.apply_cufft_forward<CUFFT_C2C>(in_c, out);
  }

  SECTION (
    "backward FFT kernel can operate on data"
  )
  {
    std::vector<std::complex<float>> in (ndat);
    std::vector<std::complex<float>> out (ndat);
    engine.setup_backward_fft_plan(fft_length, nchan);
    engine.apply_cufft_backward(in, out);
  }
}

TEST_CASE (
  "InverseFilterbankEngineCUDA can operate on data",
	""
)
{

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);

	util::IntegrationTestConfiguration<dsp::InverseFilterbank> config;
	config.filterbank->set_device(device_memory);

	int idx = 2;
	test_config::TestShape test_shape = test_config::test_shapes[idx];

	Rational os_factor (4, 3);
	unsigned npart = test_shape.npart;

	config.setup (
    os_factor, npart, test_shape.npol,
    test_shape.nchan, test_shape.output_nchan,
    test_shape.ndat, test_shape.overlap
  );

	config.transfer2GPU(
		cuda_stream, device_memory
	);

  config.filterbank->set_pfb_dc_chan(true);
  config.filterbank->set_pfb_all_chan(true);


  SECTION ("can call setup method")
  {
    engine.setup(config.filterbank);
  }

  SECTION ("can call perform method")
  {
		engine.setup(config.filterbank);
		std::vector<float *> scratch = config.allocate_scratch<CUDA::DeviceMemory>(device_memory);
		engine.set_scratch(scratch[0]);
		engine.setup(config.filterbank);
    engine.perform(
      config.input, config.output, npart
    );
		engine.finish();
		cudaFree(scratch[0]);
  }
}