#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include <sstream>

#include "catch.hpp"

#include "Rational.h"
#include "dsp/MultiFile.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldConfig.h"
#include "dsp/ConvolutionConfig.h"
#include "dsp/InverseFilterbankConfig.h"

#include "util.hpp"

const std::string file_path = "/home/SWIN/dshaff/ska/test_data/channelized.simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump";
const double dm = 2.64476;
const double folding_period = 0.00575745;

// class InverseFilterbankPipelineConfiguration {
//
// };


void build_loadtofold (dsp::LoadToFold& loader, dsp::Input* input, dsp::LoadToFold::Config* config);
void run_loadtofold (dsp::LoadToFold& loader);
void change_inverse_filterbank_config (dsp::LoadToFold::Config* config, const std::string& config_str);
void setup_config (dsp::LoadToFold::Config* config);
void setup_input (dsp::Input* input, std::string file_path);

void change_inverse_filterbank_config (dsp::LoadToFold::Config* config, const std::string& config_str)
{
  // set up inverse filterbank configuration
  std::istringstream iss = std::istringstream(config_str);
  iss >> config->inverse_filterbank;
}

void setup_config (dsp::LoadToFold::Config* config)
{
  config->dispersion_measure = dm;
  config->folding_period = folding_period;
  config->coherent_dedispersion = true;
  config->do_deripple = true;

  // set up inverse filterbank configuration
  dsp::InverseFilterbank::Config if_config;
  config->inverse_filterbank = if_config;
  config->inverse_filterbank_fft_window = "tukey";
}

void run_loadtofold (dsp::LoadToFold& loader)
{
  loader.prepare();
  loader.run();
  loader.finish();
}

void build_loadtofold (dsp::LoadToFold& loader, dsp::Input* input, dsp::LoadToFold::Config* config)
{
  loader.set_configuration(config);
  loader.set_input(input);
  loader.construct();
}

void setup_input (dsp::Input* input, std::string file_path) {
  Reference::To<dsp::File> file;
  file = dsp::File::create(file_path);
  input = file.release();
}


TEST_CASE("InverseFilterbank works in no dedispersion case", "[.]")
{
  util::set_verbose(true);
  std::cerr << "no dedispersion 0" << std::endl;
  Reference::To<dsp::LoadToFold::Config> config = new dsp::LoadToFold::Config;
  dsp::Input* input;
  std::cerr << "no dedispersion 1" << std::endl;

  setup_config(config);
  setup_input(input, file_path);
  std::cerr << "no dedispersion 2" << std::endl;

  dsp::LoadToFold loader;
  change_inverse_filterbank_config(config, "1:1024:128");
  std::cerr << "no dedispersion 3" << std::endl;
  REQUIRE(config->inverse_filterbank.get_convolve_when() == dsp::Convolution::Config::After);
  REQUIRE(config->inverse_filterbank.get_freq_res() == 1024);
  std::cerr << "no dedispersion 4" << std::endl;
  config->coherent_dedispersion = false;
  config->archive_filename = "InverseFilterbank.test_no_dedispersion";
  build_loadtofold(loader, input, config);
  std::cerr << "no dedispersion 5" << std::endl;
  std::string operation_names[] = {
      "IOManager:DADA", "InverseFilterbank", "Detection", "Fold"
  };
  std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
  for (unsigned iop=0; iop < operations.size(); iop++){
    REQUIRE(operations[iop]->get_name() == operation_names[iop]);
  }
  run_loadtofold(loader);
  std::cerr << "no dedispersion 3" << std::endl;
}
//
// TEST_CASE("InverseFilterbank workds in during dedispersion case")
// {
//   std::cerr << "TestInverseFilterbankPipeline::test_during_dedispersion " << std::endl;
//   dsp::LoadToFold loader;
//   change_inverse_filterbank_config("1:D:128");
//   REQUIRE(config->inverse_filterbank.get_convolve_when() == dsp::Convolution::Config::During);
//   REQUIRE(config->inverse_filterbank.get_freq_res() == 0);
//   config->coherent_dedispersion = true;
//   config->archive_filename = "TestInverseFilterbankPipeline.test_during_dedispersion";
//   build_loadtofold(loader);
//   std::string operation_names[] = {
//       "IOManager:DADA", "InverseFilterbank", "Detection", "Fold"
//   };
//   std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
//   for (unsigned iop=0; iop < operations.size(); iop++){
//     REQUIRE(operations[iop]->get_name() == operation_names[iop]);
//   }
//   run_loadtofold(loader);
// }
//
// TEST_CASE("InverseFilterbank workds in after dedispersion case")
// {
//   dsp::LoadToFold loader;
//   change_inverse_filterbank_config("1:1024:128");
//   REQUIRE(config->inverse_filterbank.get_convolve_when() == dsp::Convolution::Config::After);
//   REQUIRE(config->inverse_filterbank.get_freq_res() == 1024);
//   config->coherent_dedispersion = true;
//   config->archive_filename = "TestInverseFilterbankPipeline.test_after_dedispersion";
//   build_loadtofold(loader);
//   std::string operation_names[] = {
//       "IOManager:DADA", "InverseFilterbank", "Convolution", "Detection", "Fold"
//   };
//   std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
//   for (unsigned iop=0; iop < operations.size(); iop++){
//     REQUIRE(operations[iop]->get_name() == operation_names[iop]);
//   }
//   run_loadtofold(loader);
// }
//
// TEST_CASE("InverseFilterbank workds in before dedispersion case")
// {
//
// }
