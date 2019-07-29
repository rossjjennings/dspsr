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

const float thresh = 1e-5;

TEST_CASE ("InverseFilterbankEngineCUDA", "") {
	util::set_verbose(true);
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
	int idx = 2;

	test_config::TestShape test_shape = test_config::test_shapes[idx];

  util::set_verbose(true);
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);

  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);
	Reference::To<dsp::InverseFilterbank> filterbank = new dsp::InverseFilterbank;


	Rational os_factor (4, 3);

	unsigned npart = test_shape.npart;
  unsigned npol = test_shape.npol;
  unsigned input_nchan = test_shape.nchan;
  unsigned output_nchan = test_shape.output_nchan;

  auto os_in2out = [input_nchan, output_nchan, os_factor] (unsigned n) -> unsigned {
    return os_factor.normalize(n) * input_nchan / output_nchan;
  };

  unsigned input_fft_length = test_shape.ndat;
  unsigned output_fft_length = os_in2out(input_fft_length);
  unsigned input_overlap = test_shape.overlap;
  unsigned output_overlap = os_in2out(input_overlap);

  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  Reference::To<dsp::TimeSeries> input_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output_gpu = new dsp::TimeSeries;
  input_gpu->set_memory(device_memory);
  output_gpu->set_memory(device_memory);


  std::vector<unsigned> in_dim = {
    input_nchan, npol, input_fft_length*npart};
  std::vector<unsigned> out_dim = {
    output_nchan, npol, output_fft_length*npart};

  unsigned in_size = util::product(in_dim);
  unsigned out_size = util::product(out_dim);

  std::vector<std::complex<float>> in_vec(in_size);
  std::vector<std::complex<float>> out_vec(out_size);

  for (unsigned idx=0; idx<in_size; idx++) {
    in_vec[idx] = std::complex<float>(idx, idx);
  }

  util::loadTimeSeries<std::complex<float>>(in_vec, input, in_dim);
  util::loadTimeSeries<std::complex<float>>(out_vec, output, out_dim);

  dsp::TransferCUDA transfer(cuda_stream);

  transfer.set_input(input);
  transfer.set_output(input_gpu);
  transfer.prepare();
  transfer.operate();

  transfer.set_input(output);
  transfer.set_output(output_gpu);
  transfer.prepare();
  transfer.operate();

	// now load up InverseFilterbank object
	// input_gpu->set_oversampling_factor(os_factor);
	filterbank->set_input(input_gpu);
	filterbank->set_output(output_gpu);

	filterbank->set_oversampling_factor(os_factor);
  filterbank->set_input_fft_length(input_fft_length);
  filterbank->set_output_fft_length(output_fft_length);
  filterbank->set_input_discard_pos(input_overlap);
  filterbank->set_input_discard_neg(input_overlap);
  filterbank->set_output_discard_pos(output_overlap);
  filterbank->set_output_discard_neg(output_overlap);
  filterbank->set_pfb_dc_chan(true);
  filterbank->set_pfb_all_chan(true);

  SECTION ("can call setup method")
  {
    engine.setup(filterbank);
  }

  SECTION ("can call perform method")
  {
    engine.setup(filterbank);
    engine.perform(
      input_gpu, output_gpu, npart
    );
		engine.finish();
  }
}
