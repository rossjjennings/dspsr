#include <vector>

#include "catch.hpp"

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngine.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "util.hpp"
#include "InverseFilterbankTestConfig.hpp"

static util::InverseFilterbank::InverseFilterbankTestConfig test_config;

void check_error (const char*);

class Reporter : public dsp::InverseFilterbank::Engine::Reporter {
public:

  Reporter (bool _iscuda = false): iscuda(_iscuda) {}

  Reporter (cudaStream_t _stream): stream(_stream) { iscuda = true; }

  void operator() (float* arr, unsigned nchan, unsigned npol, unsigned ndat, unsigned ndim)
  {
    unsigned total_size = nchan * npol * ndat * ndim;
    if (util::config::verbose)
    {
      std::cerr << "Reporter::operator() ("
        << arr << ", "
        << nchan << ", "
        << npol << ", "
        << ndat << ", "
        << ndim << ")"
        << std::endl;
      std::cerr << "Reporter::operator() total_size=" << total_size << std::endl;
    }
    std::vector<float> data (total_size);
    if (iscuda) {
      float* data_ptr = data.data();
      size_t total_size_bytes = total_size * sizeof(float);
      cudaError error;
      if (stream) {
        error = cudaMemcpyAsync(data_ptr, arr, total_size_bytes, cudaMemcpyDeviceToHost, stream);
        if (error != 0) {
          throw "cudaMemcpyAsync error";
        }
        error = cudaStreamSynchronize(stream);
      } else {
        error = cudaMemcpy((void*) data_ptr, (void*) arr, total_size_bytes, cudaMemcpyDeviceToHost);
        if (error != 0) {
          throw "cudaMemcpy error";
        }
        error = cudaThreadSynchronize();
      }
      check_error("Reporter::operator()");
    } else {
      if (util::config::verbose) {
        std::cerr << "Reporter::operator() assigning vector contents" << std::endl;
      }
      data.assign(arr, arr + total_size);
    }
    data_vectors.push_back(data);
  }

  cudaStream_t stream;
  bool iscuda;
  std::vector<std::vector<float>> data_vectors;

};



TEST_CASE (
  "InverseFilterbankEngineCPU and InverseFilterbankEngineCUDA produce same output",
  "[InverseFilterbankEngineCPU]"
)
{

  std::vector<float> thresh = test_config.get_thresh();
  std::vector<util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));
  util::TestShape test_shape = test_shapes[idx];


  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::InverseFilterbankEngineCUDA engine_cuda(cuda_stream);
  dsp::InverseFilterbankEngineCPU engine_cpu;

  Reporter reporter_cpu;
  Reporter reporter_cuda(cuda_stream);
  engine_cpu.reporter.on("data", &reporter_cpu);
  engine_cuda.reporter.on("data", &reporter_cuda);


  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_cuda = new dsp::TimeSeries;

  Rational os_factor (4, 3);
  unsigned npart = test_shape.npart;

  util::IntegrationTestConfiguration<dsp::InverseFilterbank> config(
    os_factor, npart, test_shape.input_npol,
    test_shape.input_nchan, test_shape.output_nchan,
    test_shape.input_ndat, test_shape.overlap_pos
  );
  config.filterbank->set_pfb_dc_chan(true);
  config.filterbank->set_pfb_all_chan(true);

  config.setup (in, out);

  engine_cpu.setup(config.filterbank);
  std::vector<float *> scratch_cpu = config.allocate_scratch<dsp::Memory> ();
  engine_cpu.set_scratch(scratch_cpu[0]);
  engine_cpu.perform(
    in, out, npart
  );
  engine_cpu.finish();
  auto transfer = util::transferTimeSeries(cuda_stream, device_memory);
  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  // config.filterbank->set_device(device_memory);
  engine_cuda.setup(config.filterbank);
  std::vector<float *> scratch_cuda = config.allocate_scratch<CUDA::DeviceMemory>(device_memory);
  engine_cuda.set_scratch(scratch_cuda[0]);
  engine_cuda.perform(
    in_gpu, out_gpu, npart
  );
  engine_cuda.finish();
  // now lets compare the two time series
  transfer(out_gpu, out_cuda, cudaMemcpyDeviceToHost);

  // for (unsigned i=0; i<reporter_cpu.data_vectors.size(); i++)
  // {
  //   REQUIRE(util::allclose<float>(
  //     reporter_cpu.data_vectors[i],
  //     reporter_cuda.data_vectors[i],
  //     thresh[0], thresh[1]
  //   ));
  // }



  REQUIRE(util::allclose(out_cuda, out, thresh[0], thresh[1]));


}
