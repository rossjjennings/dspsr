#include <vector>
#include <fstream>
#include <string>

#include "catch.hpp"

#include "dsp/SpectralKurtosis.h"
#include "dsp/SpectralKurtosisCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "util.hpp"
#include "TestConfig.hpp"
#include "TransformationProxy.hpp"

static util::TestConfig test_config;


void check_error (const char*);


class Reporter : public dsp::SpectralKurtosis::Reporter {
// class Reporter  {
public:

  Reporter (bool _iscuda = false): iscuda(_iscuda) {}

  Reporter (cudaStream_t _stream): stream(_stream) { iscuda = true; }

  void operator() (
    float* arr, unsigned nchan, unsigned npol, unsigned ndat, unsigned ndim)
  {
    unsigned total_size = nchan * npol * ndat * ndim;
    // if (util::config::verbose)
    // {
    //   std::cerr << "Reporter::operator() ("
    //     << arr << ", "
    //     << nchan << ", "
    //     << npol << ", "
    //     << ndat << ", "
    //     << ndim << ")"
    //     << std::endl;
    //   std::cerr << "Reporter::operator() total_size=" << total_size << std::endl;
    // }
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
      // if (util::config::verbose) {
      //   std::cerr << "Reporter::operator() assigning vector contents" << std::endl;
      // }
      data.assign(arr, arr + total_size);
    }
    // if (util::config::verbose) {
    //   std::cerr << "Reporter::operator() data_vectors.size()=" << data_vectors.size() << std::endl;
    // }
    data_vectors.push_back(data);
  }


  void concatenate_data_vectors (std::vector<float>& result)
  {
    unsigned n_vec = data_vectors.size();
    unsigned size_each = data_vectors[0].size();
    unsigned total_size = size_each * n_vec;
    // if (util::config::verbose) {
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
  "SpectralKurtosis CPU and CUDA implementations produce same output",
  "[SpectralKurtosis]"
)
{
  std::vector<float> thresh = test_config.get_thresh();
  // std::vector<util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  //
  // auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));
  // if (util::config::verbose) {
  //   std::cerr << "test_SpectralKurtosis_integration: idx=" << idx << std::endl;
  // }
  //
  util::TransformationProxy sk_proxy(
    32, 32, 2, 2, 2, 2, 1024, 1024);

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::SpectralKurtosisEngine engine_cuda(device_memory);

  dsp::SpectralKurtosis sk_cpu;
  sk_cpu.set_buffering_policy(nullptr);
  dsp::SpectralKurtosis sk_cuda;
  sk_cuda.set_buffering_policy(nullptr);
  sk_cuda.set_engine(&engine_cuda);

  Reporter reporter_cpu;
  Reporter reporter_cuda(cuda_stream);
  sk_cpu.reporter.on("compute", &reporter_cpu);
  sk_cuda.reporter.on("compute", &reporter_cuda);

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_cuda = new dsp::TimeSeries;

  sk_proxy.setup (in, out);
  sk_cpu.set_input(in);
  sk_cpu.set_output(out);

  sk_cpu.prepare();
  sk_cpu.operate();

  // sk_cpu.finish();

  auto transfer = util::transferTimeSeries(cuda_stream, device_memory);
  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  sk_cuda.set_input(in_gpu);
  sk_cuda.set_output(out_gpu);

  sk_cuda.prepare();
  sk_cuda.operate();

  transfer(out_gpu, out_cuda, cudaMemcpyDeviceToHost);

  REQUIRE(util::allclose(out_cuda, out, thresh[0], thresh[1]));

  // Reference::To<dsp::TimeSeries> out_cuda = new dsp::TimeSeries;
  //
  // Rational os_factor (4, 3);
  // unsigned npart = test_shape.npart;
  // unsigned npol = test_shape.input_npol;
  //
  // util::InverseFilterbank::InverseFilterbankProxy proxy(
  //   os_factor, npart, npol,
  //   test_shape.input_nchan, test_shape.output_nchan,
  //   test_shape.input_ndat, test_shape.overlap_pos
  // );
  // proxy.filterbank->set_pfb_dc_chan(true);
  // proxy.filterbank->set_pfb_all_chan(true);
  //
  // proxy.setup (in, out, do_fft_window, do_response);
  //
  // engine_cpu.setup(proxy.filterbank);
  // float* scratch_cpu = proxy.allocate_scratch(engine_cpu.get_total_scratch_needed());
  // engine_cpu.set_scratch(scratch_cpu);
  // engine_cpu.perform(
  //   in, out, npart
  // );
  // engine_cpu.finish();
  // auto transfer = util::transferTimeSeries(cuda_stream, device_memory);
  // transfer(in, in_gpu, cudaMemcpyHostToDevice);
  // transfer(out, out_gpu, cudaMemcpyHostToDevice);
  //
  // // proxy.filterbank->set_device(device_memory);
  // engine_cuda.setup(proxy.filterbank);
  // proxy.set_memory<CUDA::DeviceMemory>(device_memory);
  // float * scratch_cuda = proxy.allocate_scratch(engine_cuda.get_total_scratch_needed());
  // engine_cuda.set_scratch(scratch_cuda);
  // engine_cuda.perform(
  //   in_gpu, out_gpu, npart
  // );
  // engine_cuda.finish();
  // // now lets compare the two time series
  // transfer(out_gpu, out_cuda, cudaMemcpyDeviceToHost);

  // std::vector<Reporter> reporters = {
  //   reporter_cpu_fft_window,
  //   reporter_cuda_fft_window,
  //   reporter_cpu_fft,
  //   reporter_cuda_fft,
  //   reporter_cpu_response_stitch,
  //   reporter_cuda_response_stitch,
  //   reporter_cpu_ifft,
  //   reporter_cuda_ifft
  // };
  //
  // std::vector<std::string> reporter_names = {
  //   "fft_window",
  //   "fft",
  //   "response_stitch",
  //   "ifft"
  // };
  //
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
  //   nclose = util::nclose(
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
  //   if (util::config::verbose)
  //   {
  //     std::cerr << "test_InverseFilterbankEngine_integration: "
  //       << reporter_names[r_idx] << " " << nclose << "/" << size
  //       << " (" << 100 * (float) nclose / size << "%)" << std::endl;
  //     std::cerr << "test_InverseFilterbankEngine_integration: "
  //       << reporter_names[r_idx]
  //       << " max cpu=" << util::max(cpu_vector)
  //       << ", max gpu=" << util::max(cuda_vector)
  //       << std::endl;
  //   }
  //
  //   CHECK (nclose == size);
  //
  // }

  // REQUIRE(util::allclose(out_cuda, out, thresh[0], thresh[1]));
}
