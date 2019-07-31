#include <complex>
#include <vector>
#include <iostream>

#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "Rational.h"

#include "util.hpp"
#include "InverseFilterbank_test_config.h"

void check_error (const char*);

int main ()
{
  int idx = 2;
  test_config::TestShape test_shape = test_config::test_shapes[idx];

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
  util::IntegrationTestConfiguration<dsp::InverseFilterbank> config(
    os_factor, npart, test_shape.npol,
    test_shape.nchan, test_shape.output_nchan,
    test_shape.ndat, test_shape.overlap
  );
  config.filterbank->set_device(device_memory);
  config.filterbank->set_pfb_dc_chan(true);
  config.filterbank->set_pfb_all_chan(true);

  config.setup(in, out);

  auto transfer = util::transferTimeSeries(cuda_stream, device_memory);

  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  engine.setup(config.filterbank);
  std::vector<float *> scratch = config.allocate_scratch<CUDA::DeviceMemory>(device_memory);
  engine.set_scratch(scratch[0]);
  engine.perform(
    in_gpu, out_gpu, npart
  );
  engine.finish();
}
