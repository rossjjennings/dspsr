#include <vector>

#include "catch.hpp"

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/MemoryCUDA.h"


#include "util.hpp"
#include "InverseFilterbank_test_config.h"

TEST_CASE (
  "InverseFilterbankEngineCPU and InverseFilterbankEngineCUDA produce same output",
	"[InverseFilterbankEngineCPU]"
)
{
	int idx = 2;

	test_config::TestShape test_shape = test_config::test_shapes[idx];

  util::set_verbose(true);

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);

  CUDA::InverseFilterbankEngineCUDA engine_cuda(cuda_stream);

  dsp::InverseFilterbankEngineCPU engine_cpu;
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

	// now load up InverseFilterbank object
  // input->set_oversampling_factor(os_factor);
	filterbank->set_input(input);
	filterbank->set_output(output);

  filterbank->set_oversampling_factor(os_factor);
  filterbank->set_input_fft_length(input_fft_length);
  filterbank->set_output_fft_length(output_fft_length);
  filterbank->set_input_discard_pos(input_overlap);
  filterbank->set_input_discard_neg(input_overlap);
  filterbank->set_output_discard_pos(output_overlap);
  filterbank->set_output_discard_neg(output_overlap);
  filterbank->set_pfb_dc_chan(true);
  filterbank->set_pfb_all_chan(true);

  // engine.setup(filterbank);
  // engine.perform(
  //   input, output, npart
  // );
  // engine.finish();
}
