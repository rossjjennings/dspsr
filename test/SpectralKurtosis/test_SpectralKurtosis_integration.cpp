#include <vector>
#include <fstream>
#include <string>

#include "catch.hpp"

#include "dsp/SpectralKurtosis.h"
#include "dsp/SpectralKurtosisCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "util/util.hpp"
#include "util/TestConfig.hpp"
#include "util/TestReporter.hpp"
#include "util/TransformationProxy.hpp"

static util::TestConfig test_config;

TEST_CASE (
  "SpectralKurtosis CPU and CUDA implementations produce same output",
  "[SpectralKurtosis]"
)
{
  typedef util::TestReporter<
    dsp::SpectralKurtosis::Reporter<float>, float
  > FloatSpectralKurtosisReporter;
  typedef util::TestReporter<
    dsp::SpectralKurtosis::Reporter<unsigned char>, unsigned char
  > CharSpectralKurtosisReporter;

  std::string file_name = "1644-4559.pre_Convolution.4.dump";
  std::string file_path = util::get_test_data_dir() + "/" + file_name;
  std::vector<float> thresh = test_config.get_thresh();

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::SpectralKurtosisEngine engine_cuda(device_memory);

  int tscrunch = 64;
  int nparts = 2;
  int block_size = tscrunch * nparts;

  dsp::SpectralKurtosis sk_cpu;
  sk_cpu.set_buffering_policy(nullptr);
  sk_cpu.set_thresholds(tscrunch, 4);

  dsp::SpectralKurtosis sk_cuda;
  sk_cuda.set_buffering_policy(nullptr);
  sk_cuda.set_thresholds(tscrunch, 4);
  sk_cuda.set_engine(&engine_cuda);

  std::vector<std::string> float_reporter_names = {
    "estimates",
    "estimates_tscr",
    "output"
  };

  std::vector<std::string> char_reporter_names = {
    "zapmask_tscr",
    "zapmask_skfb",
    "zapmask_fscr",
    "zapmask"
  };

  std::vector<FloatSpectralKurtosisReporter> float_reporters = {
    FloatSpectralKurtosisReporter(),
    FloatSpectralKurtosisReporter(cuda_stream),
    FloatSpectralKurtosisReporter(),
    FloatSpectralKurtosisReporter(cuda_stream),
    FloatSpectralKurtosisReporter(),
    FloatSpectralKurtosisReporter(cuda_stream)
  };

  std::vector<CharSpectralKurtosisReporter> char_reporters = {
    CharSpectralKurtosisReporter(),
    CharSpectralKurtosisReporter(cuda_stream),
    CharSpectralKurtosisReporter(),
    CharSpectralKurtosisReporter(cuda_stream),
    CharSpectralKurtosisReporter(),
    CharSpectralKurtosisReporter(cuda_stream),
    CharSpectralKurtosisReporter(),
    CharSpectralKurtosisReporter(cuda_stream)
  };


  for (unsigned idx=0; idx<float_reporter_names.size(); idx++) {
    sk_cpu.float_reporter.on(
      float_reporter_names[idx], &float_reporters[2*idx]);
    sk_cuda.float_reporter.on(
      float_reporter_names[idx], &float_reporters[2*idx + 1]);
  }

  for (unsigned idx=0; idx<char_reporter_names.size(); idx++) {
    sk_cpu.char_reporter.on(
      char_reporter_names[idx], &char_reporters[2*idx]);
    sk_cuda.char_reporter.on(
      char_reporter_names[idx], &char_reporters[2*idx + 1]);
  }

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_cuda = new dsp::TimeSeries;

  dsp::IOManager manager;
  manager.open(file_path);

  util::load_psr_data(manager, block_size, in, 1);

  sk_cpu.set_input(in);
  sk_cpu.set_output(out);

  sk_cpu.prepare();
  sk_cpu.operate();

  auto transfer = util::transferTimeSeries(cuda_stream, device_memory);
  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  sk_cuda.set_input(in_gpu);
  sk_cuda.set_output(out_gpu);

  sk_cuda.prepare();
  sk_cuda.operate();

  transfer(out_gpu, out_cuda, cudaMemcpyDeviceToHost);

  unsigned nclose;
  unsigned size;
  std::vector<float> float_cpu_vector;
  std::vector<float> float_cuda_vector;
  std::vector<unsigned char> char_cpu_vector;
  std::vector<unsigned char> char_cuda_vector;

  for (unsigned r_idx=0; r_idx<float_reporter_names.size(); r_idx++)
  {
    float_reporters[r_idx*2].concatenate_data_vectors(float_cpu_vector);
    float_reporters[r_idx*2 + 1].concatenate_data_vectors(float_cuda_vector);

    size = float_cpu_vector.size();

    nclose = util::nclose(
      float_cpu_vector,
      float_cuda_vector,
      thresh[0], thresh[1]
    );

    // for (unsigned idx=0; idx<float_cpu_vector.size(); idx++)
    // {
    //   // if (util::isclose(float_cpu_vector[idx], float_cuda_vector[idx], thresh[0], thresh[1])) {
    //   //   std::cerr << "idx=" << idx << std::endl;
    //   // }
    //   std::cerr << "(" << float_cpu_vector[idx] << ", " << float_cuda_vector[idx] << ") ";
    // }
    // std::cerr << std::endl;

    if (util::config::verbose)
    {
      std::cerr << "test_SpectralKurtosis_integration: "
        << float_reporter_names[r_idx] << " " << nclose << "/" << size
        << " (" << 100 * (float) nclose / size << "%)" << std::endl;
    }

    CHECK (nclose == size);
  }


  for (unsigned r_idx=0; r_idx<char_reporter_names.size(); r_idx++)
  {
    char_reporters[r_idx*2].concatenate_data_vectors(char_cpu_vector);
    char_reporters[r_idx*2 + 1].concatenate_data_vectors(char_cuda_vector);

    size = char_cpu_vector.size();
    nclose = 0;

    for (unsigned idx=0; idx<size; idx++) {
      if (char_cpu_vector[idx] == char_cuda_vector[idx]) {
        nclose += 1;
      }
    }

    // for (unsigned idx=0; idx<char_cpu_vector.size(); idx++)
    // {
    //   // if (util::isclose(char_cpu_vector[idx], char_cuda_vector[idx], thresh[0], thresh[1])) {
    //   //   std::cerr << "idx=" << idx << std::endl;
    //   // }
    //   std::cerr << "(" << char_cpu_vector[idx] << ", " << char_cuda_vector[idx] << ") ";
    // }
    // std::cerr << std::endl;

    if (util::config::verbose)
    {
      std::cerr << "test_SpectralKurtosis_integration: "
        << char_reporter_names[r_idx] << " " << nclose << "/" << size
        << " (" << 100 * (float) nclose / size << "%)" << std::endl;
    }

    CHECK (nclose == size);
  }




  REQUIRE(util::allclose(out_cuda, out, thresh[0], thresh[1]));
}
