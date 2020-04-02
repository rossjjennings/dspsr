#include <complex>
#include <vector>
#include <iostream>

#include "catch.hpp"

#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "Rational.h"

#include "util/util.hpp"
#include "InverseFilterbankTestConfig.hpp"

static test::util::InverseFilterbank::InverseFilterbankTestConfig test_config;

void check_error (const char*);

TEST_CASE (
  "InverseFilterbankEngineCUDA",
  "[unit][cuda][no_file][InverseFilterbankEngineCUDA]"
)
{
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

TEST_CASE (
  "cufft kernels can operate on data",
  "[cuda][no_file][InverseFilterbankEngineCUDA][component]"
)
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
  "[cuda][no_file][InverseFilterbankEngineCUDA][component]"
)
{
  std::vector<test::util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));

	test::util::TestShape test_shape = test_shapes[idx];

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);
  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;

  Rational os_factor (4, 3);

  unsigned npart = test_shape.npart;
  test::util::InverseFilterbank::InverseFilterbankProxy proxy(
    os_factor, npart, test_shape.input_npol,
    test_shape.input_nchan, test_shape.output_nchan,
    test_shape.input_ndat, test_shape.overlap_pos
  );
  proxy.filterbank->set_device(device_memory);
  proxy.filterbank->set_pfb_dc_chan(true);
  proxy.filterbank->set_pfb_all_chan(true);

  proxy.setup(in, out, false, false);

  auto transfer = test::util::transferTimeSeries(cuda_stream, device_memory);

  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  SECTION ("can call setup method")
  {
    engine.setup(proxy.filterbank);
  }

  SECTION ("can call perform method")
  {
    engine.setup(proxy.filterbank);
    proxy.set_memory<CUDA::DeviceMemory>(device_memory);
    float* scratch = proxy.allocate_scratch(engine.get_total_scratch_needed());
    engine.set_scratch(scratch);
    engine.perform(
      in_gpu, out_gpu, npart
    );
    engine.finish();
  }
}
