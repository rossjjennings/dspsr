#include <vector>

#include "catch.hpp"

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "util.hpp"
#include "InverseFilterbank_test_config.h"


void check_error (const char*);


TEST_CASE (
  "InverseFilterbankEngineCPU and InverseFilterbankEngineCUDA produce same output",
  "[InverseFilterbankEngineCPU]"
)
{
    void* stream = 0;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
    CUDA::InverseFilterbankEngineCUDA engine_cuda(cuda_stream);
    dsp::InverseFilterbankEngineCPU engine_cpu;

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
    config.filterbank->set_pfb_dc_chan(true);
    config.filterbank->set_pfb_all_chan(true);

    std::cerr << "setting up CPU engine" << std::endl;
    engine_cpu.setup(config.filterbank);
    std::vector<float *> scratch_cpu = config.allocate_scratch<dsp::Memory> ();
    engine_cpu.set_scratch(scratch_cpu[0]);
    engine_cpu.perform(
      config.input, config.output, npart
    );
    engine_cpu.finish();
    std::cerr << "CPU engine finished" << std::endl;

    config.transfer2GPU(
      cuda_stream, device_memory
    );
    check_error("IntegrationTestConfiguration::transfer2GPU");

    std::cerr << "setting up CUDA engine" << std::endl;
    engine_cuda.setup(config.filterbank);
    std::vector<float *> scratch_cuda = config.allocate_scratch<CUDA::DeviceMemory>(device_memory);
    engine_cuda.set_scratch(scratch_cuda[0]);
    engine_cuda.perform(
      config.input, config.output, npart
    );
    check_error("InverseFilterbankEngineCUDA::perform");
    engine_cuda.finish();
    std::cerr << "CUDA engine finished" << std::endl;

    cudaFree(scratch_cuda[0]);
}
