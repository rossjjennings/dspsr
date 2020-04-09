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

static test::util::TestConfig test_config;

TEST_CASE (
  "SpectralKurtosis CPU and CUDA implementations produce same output",
  "[cuda][SpectralKurtosis]"
)
{
  typedef test::util::TestReporter<
    dsp::SpectralKurtosis::Reporter<float>, float
  > FloatSpectralKurtosisReporter;
  typedef test::util::TestReporter<
    dsp::SpectralKurtosis::Reporter<unsigned char>, unsigned char
  > CharSpectralKurtosisReporter;


  // std::string file_name = "1644-4559.pre_Convolution.4.dump"; // four channel for easy testing
  std::string file_name = test_config.get_field<std::string>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.file_name"); // 128 channel for more realistic testing
  std::string file_path = test::util::get_test_data_dir() + "/" + file_name;
  std::vector<float> thresh = test_config.get_thresh();

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::SpectralKurtosisEngine engine_cuda(device_memory);

  unsigned tscrunch = test_config.get_field<unsigned>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.tscrunch");
  unsigned std_devs = test_config.get_field<unsigned>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.std_devs");
  unsigned nparts = test_config.get_field<unsigned>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.nparts");
  unsigned block_size = test_config.get_field<unsigned>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.block_size");

  unsigned schan = test_config.get_field<unsigned>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.s_chan");
  unsigned echan = test_config.get_field<unsigned>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.e_chan");

  bool disable_fscr = test_config.get_field<bool>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.disable_fscr");
  bool disable_tscr = test_config.get_field<bool>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.disable_tscr");;
  bool disable_ft = test_config.get_field<bool>(
    "SpectralKurtosis.test_SpectralKurtosis_integration.disable_ft");;

  if (test::util::config::verbose) {
    std::cerr << "test_SpectralKurtosis_integration: tscrunch=" << tscrunch << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: std_devs=" << std_devs << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: nparts=" << nparts << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: block_size=" << block_size << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: schan=" << schan << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: echan=" << echan << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: disable_fscr=" << disable_fscr << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: disable_tscr=" << disable_tscr << std::endl;
    std::cerr << "test_SpectralKurtosis_integration: disable_ft=" << disable_ft << std::endl;
  }


  dsp::SpectralKurtosis sk_cpu;
  sk_cpu.set_report(true);
  sk_cpu.set_M(tscrunch);
  sk_cpu.set_thresholds(std_devs);
  sk_cpu.set_options(disable_fscr, disable_tscr, disable_ft);
  sk_cpu.set_channel_range(schan, echan);

  dsp::SpectralKurtosis sk_cuda;
  sk_cuda.set_report(true);
  sk_cuda.set_M(tscrunch);
  sk_cuda.set_thresholds(std_devs);
  sk_cuda.set_engine(&engine_cuda);
  sk_cuda.set_options(disable_fscr, disable_tscr, disable_ft);
  sk_cuda.set_channel_range(schan, echan);

  std::vector<std::string> float_reporter_names = {
    "input",
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

  auto transfer = test::util::transferTimeSeries(cuda_stream, device_memory);

  dsp::IOManager manager;

  manager.set_output(in);

  sk_cpu.set_input(in);
  sk_cpu.set_output(out);

  sk_cuda.set_input(in_gpu);
  sk_cuda.set_output(out_gpu);

  manager.open(file_path);
  manager.set_block_size(block_size);

  for (unsigned ipart=0; ipart<nparts; ipart++) {
    if (test::util::config::verbose) {
      std::cerr << "test_SpectralKurtosis_integration: ipart=" << ipart << std::endl;
    }
    manager.operate();
    sk_cpu.operate();
    transfer(in, in_gpu, cudaMemcpyHostToDevice);
    transfer(out, out_gpu, cudaMemcpyHostToDevice); // for some reason this is necessary
    sk_cuda.operate();
  }

  unsigned nclose;
  unsigned size;
  std::vector<float> float_cpu_vector;
  std::vector<float> float_cuda_vector;
  std::vector<unsigned char> char_cpu_vector;
  std::vector<unsigned char> char_cuda_vector;

  for (unsigned r_idx=0; r_idx<float_reporter_names.size(); r_idx++)
  {
    if (test::util::config::verbose) {
      std::cerr << "test_SpectralKurtosis_integration: Checking "
        << float_reporter_names[r_idx]   << std::endl;
    }
    float_reporters[r_idx*2].concatenate_data_vectors(float_cpu_vector);
    float_reporters[r_idx*2 + 1].concatenate_data_vectors(float_cuda_vector);

    size = float_cpu_vector.size();

    nclose = test::util::nclose(
      float_cpu_vector,
      float_cuda_vector,
      thresh[0], thresh[1]
    );
    //
    // for (unsigned idx=0; idx<float_cpu_vector.size(); idx++)
    // {
    //   // if (! test::util::isclose(float_cpu_vector[idx], float_cuda_vector[idx], thresh[0], thresh[1])) {
    //   //   std::cerr << "idx=" << idx;
    //   //   std::cerr << " (" << float_cpu_vector[idx] << ", " << float_cuda_vector[idx] << ") ";
    //   //
    //   // }
    //   // std::cerr << " (" << float_cpu_vector[idx] << ", " << float_cuda_vector[idx] << ") ";
    //
    // }
    // std::cerr << std::endl;

    if (test::util::config::verbose)
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

    if (char_cpu_vector.size() != char_cuda_vector.size()) {
      throw "Vectors are not the same size!";
    }

    size = char_cpu_vector.size();
    nclose = 0;


    for (unsigned idx=0; idx<size; idx++) {
      // std::cerr << "(" << (unsigned) char_cpu_vector[idx] << ", "
      //   << (unsigned) char_cuda_vector[idx] << ") ";
      if (char_cpu_vector[idx] == char_cuda_vector[idx]) {
        nclose += 1;
      }
    }
    // std::cerr << std::endl;

    if (test::util::config::verbose)
    {
      std::cerr << "test_SpectralKurtosis_integration: "
        << char_reporter_names[r_idx] << " " << nclose << "/" << size
        << " (" << 100 * (float) nclose / size << "%)" << std::endl;
    }

    CHECK (nclose == size);
  }

}
