#include <vector>
#include <fstream>
#include <string>

#include "catch.hpp"

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngine.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "util/util.hpp"
#include "InverseFilterbankTestConfig.hpp"

static test::util::InverseFilterbank::InverseFilterbankTestConfig test_config;

void check_error (const char*);

class Reporter : public dsp::InverseFilterbank::Engine::Reporter {
public:

  Reporter (bool _iscuda = false): iscuda(_iscuda) {}

  Reporter (cudaStream_t _stream): stream(_stream) { iscuda = true; }

  void operator() (float* arr, unsigned nchan, unsigned npol, unsigned ndat, unsigned ndim)
  {
    unsigned total_size = nchan * npol * ndat * ndim;
    if (test::util::config::verbose)
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
        error = cudaMemcpy(data_ptr, arr, total_size_bytes, cudaMemcpyDeviceToHost);
        if (error != 0) {
          throw "cudaMemcpy error";
        }
        error = cudaThreadSynchronize();
      }
      check_error("Reporter::operator()");
    } else {
      // if (test::util::config::verbose) {
      //   std::cerr << "Reporter::operator() assigning vector contents" << std::endl;
      // }
      data.assign(arr, arr + total_size);
    }
    // if (test::util::config::verbose) {
    //   std::cerr << "Reporter::operator() data_vectors.size()=" << data_vectors.size() << std::endl;
    // }
    data_vectors.push_back(data);
  }


  void concatenate_data_vectors (std::vector<float>& result)
  {

    unsigned n_vec = data_vectors.size();
    unsigned size_each = data_vectors[0].size();
    unsigned total_size = size_each * n_vec;
    // if (test::util::config::verbose) {
    //   std::cerr << "Reporter::concatenate_data_vectors: n_vec=" << n_vec
    //     << " size_each=" << size_each << " total_size=" << total_size << std::endl;
    // }
    result.resize(total_size);

    for (unsigned ipart=0; ipart<n_vec; ipart++) {
      for (unsigned idat=0; idat<size_each; idat++) {
        result[ipart * size_each + idat] = data_vectors[ipart][idat];
      }
    }

  }

  cudaStream_t stream;
  bool iscuda;
  std::vector<std::vector<float>> data_vectors;

};



TEST_CASE (
  "InverseFilterbankEngineCPU and InverseFilterbankEngineCUDA produce same output",
  "[cuda][InverseFilterbankEngineCUDA][InverseFilterbankEngineCPU][integration]"
)
{

  std::vector<float> thresh = test_config.get_thresh();
  std::vector<test::util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  bool do_fft_window = test_config.get_field<bool>(
    "InverseFilterbank.test_InverseFilterbankEngine_integration.do_fft_window");
  bool do_response = test_config.get_field<bool>(
    "InverseFilterbank.test_InverseFilterbankEngine_integration.do_response");

  auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));
  if (test::util::config::verbose) {
    std::cerr << "test_InverseFilterbankEngine_integration: idx=" << idx << std::endl;
  }

  test::util::TestShape test_shape = test_shapes[idx];


  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::InverseFilterbankEngineCUDA engine_cuda(cuda_stream);
  dsp::InverseFilterbankEngineCPU engine_cpu;

  engine_cuda.set_report(true);
  engine_cpu.set_report(true);

  std::vector<std::string> reporter_names = {
    // "fft_window",
    // "fft",
    // "response_stitch",
    // "ifft"
  };

  std::vector<Reporter> reporters = {
    Reporter(),
    Reporter(cuda_stream),
    Reporter(),
    Reporter(cuda_stream),
    Reporter(),
    Reporter(cuda_stream),
    Reporter(),
    Reporter(cuda_stream)
  };

  for (unsigned idx=0; idx<reporter_names.size(); idx++) {
    engine_cpu.reporter.on(reporter_names[idx], &reporters[2*idx]);
    engine_cuda.reporter.on(reporter_names[idx], &reporters[2*idx + 1]);
  }

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_cuda = new dsp::TimeSeries;

  Rational os_factor (4, 3);
  unsigned npart = test_shape.npart;
  unsigned npol = test_shape.input_npol;

  test::util::InverseFilterbank::InverseFilterbankProxy proxy(
    os_factor, npart, npol,
    test_shape.input_nchan, test_shape.output_nchan,
    test_shape.input_ndat, test_shape.overlap_pos
  );
  proxy.filterbank->set_pfb_dc_chan(true);
  proxy.filterbank->set_pfb_all_chan(true);

  proxy.setup (in, out, do_fft_window, do_response);

  engine_cpu.setup(proxy.filterbank);
  float* scratch_cpu = proxy.allocate_scratch(engine_cpu.get_total_scratch_needed());
  engine_cpu.set_scratch(scratch_cpu);
  engine_cpu.perform(
    in, out, npart
  );
  engine_cpu.finish();
  auto transfer = test::util::transferTimeSeries(cuda_stream, device_memory);
  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  engine_cuda.setup(proxy.filterbank);
  proxy.set_memory<CUDA::DeviceMemory>(device_memory);
  float * scratch_cuda = proxy.allocate_scratch(engine_cuda.get_total_scratch_needed());
  engine_cuda.set_scratch(scratch_cuda);
  engine_cuda.perform(
    in_gpu, out_gpu, npart
  );
  engine_cuda.finish();
  transfer(out_gpu, out_cuda, cudaMemcpyDeviceToHost);

  // if (npol == 2) {
  //   reporters = {
  //     reporter_cpu_response_stitch,
  //     reporter_cuda_response_stitch,
  //     reporter_cpu_ifft,
  //     reporter_cuda_ifft
  //   };
  //
  //   reporter_names = {
  //     "response_stitch",
  //     "ifft"
  //   };
  // }
  //
  // unsigned nclose;
  // unsigned size;
  // std::vector<float> cpu_vector;
  // std::vector<float> cuda_vector;
  //
  // for (unsigned r_idx=0; r_idx<reporter_names.size(); r_idx++)
  // {
  //   reporters[r_idx*2].concatenate_data_vectors(cpu_vector);
  //   reporters[r_idx*2 + 1].concatenate_data_vectors(cuda_vector);
  //
  //   size = cpu_vector.size();
  //
  //   nclose = test::util::nclose(
  //     cpu_vector,
  //     cuda_vector,
  //     thresh[0], thresh[1]
  //   );
  //   // std::ofstream cpu_file(
  //   //   reporter_names[r_idx] + ".cpu.dat", std::ios::out | std::ios::binary);
  //   // std::ofstream cuda_file(
  //   //   reporter_names[r_idx] + ".cuda.dat", std::ios::out | std::ios::binary);
  //   //
  //   // cpu_file.write(
  //   //   reinterpret_cast<const char*>(cpu_vector.data()),
  //   //   size * sizeof(float)
  //   // );
  //   // cpu_file.close();
  //   //
  //   // cuda_file.write(
  //   //   reinterpret_cast<const char*>(cuda_vector.data()),
  //   //   size * sizeof(float)
  //   // );
  //   // cuda_file.close();
  //
  //   if (test::util::config::verbose)
  //   {
  //     std::cerr << "test_InverseFilterbankEngine_integration: "
  //       << reporter_names[r_idx] << " " << nclose << "/" << size
  //       << " (" << 100 * (float) nclose / size << "%)" << std::endl;
  //     std::cerr << "test_InverseFilterbankEngine_integration: "
  //       << reporter_names[r_idx]
  //       << " max cpu=" << test::util::max(cpu_vector)
  //       << ", max gpu=" << test::util::max(cuda_vector)
  //       << std::endl;
  //   }
  //
  //   CHECK (nclose == size);
  //
  // }

  REQUIRE(test::util::allclose(out_cuda, out, thresh[0], thresh[1]));
}
