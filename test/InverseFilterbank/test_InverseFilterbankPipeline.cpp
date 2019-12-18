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
#include "dsp/InverseFilterbankConfig.h"

#include "util/util.hpp"
#include "util/TestConfig.hpp"

static test::util::TestConfig test_config;


class PipelineConfig {

  public:

    PipelineConfig ();
    void build_loadtofold (dsp::LoadToFold& loader);
    void run_loadtofold (dsp::LoadToFold& loader);
    void change_inverse_filterbank_config (const std::string& config_str);
    void setup_config (const double dm, const double period);
    void setup_input (std::string file_path);

    Reference::To<dsp::LoadToFold::Config> config;
    dsp::Input* input;
};

PipelineConfig::PipelineConfig () {
  config = new dsp::LoadToFold::Config;
}

void PipelineConfig::change_inverse_filterbank_config (
    const std::string& config_str)
{
  // set up inverse filterbank configuration
  std::istringstream iss (config_str);
  iss >> config->inverse_filterbank;
}

void PipelineConfig::setup_config (const double dm, const double period)
{
  config->dispersion_measure = dm;
  config->folding_period = period;
  config->coherent_dedispersion = true;
  config->do_deripple = true;

  // set up inverse filterbank configuration
  dsp::InverseFilterbank::Config if_config;
  config->inverse_filterbank = if_config;
  config->inverse_filterbank_fft_window = "tukey";
}

void PipelineConfig::run_loadtofold (dsp::LoadToFold& loader)
{
  loader.prepare();
  loader.run();
  loader.finish();
}

void PipelineConfig::build_loadtofold (dsp::LoadToFold& loader)
{
  loader.set_configuration(config);
  loader.set_input(input);
  loader.construct();
}

void PipelineConfig::setup_input (std::string file_path) {
  Reference::To<dsp::File> file;
  file = dsp::File::create(file_path);
  input = file.release();
}


TEST_CASE(
  "InverseFilterbank works in larger LoadToFold context",
  "[InverseFilterbank][integration]"
)
{

  const double dm = test_config.get_field<double>(
    "InverseFilterbank.test_InverseFilterbankPipeline.dm");
  const double folding_period = test_config.get_field<double>(
    "InverseFilterbank.test_InverseFilterbankPipeline.period");

  const std::string file_name = test_config.get_field<std::string>(
    "InverseFilterbank.test_InverseFilterbankPipeline.file_name");

  const std::string file_path = (
    test::util::get_test_data_dir() + "/" + file_name);

  // test::util::set_verbose(true);
  PipelineConfig pipeline_config;
  pipeline_config.setup_config(dm, folding_period);
  pipeline_config.setup_input(file_path);

  SECTION("InverseFilterbank works in no dedispersion case")
  {
    dsp::LoadToFold loader;

    pipeline_config.change_inverse_filterbank_config("1:1024:128");

    REQUIRE(pipeline_config.config->inverse_filterbank.get_convolve_when() == dsp::Filterbank::Config::After);
    REQUIRE(pipeline_config.config->inverse_filterbank.get_freq_res() == 1024);

    pipeline_config.config->coherent_dedispersion = false;
    pipeline_config.config->archive_filename = "InverseFilterbank.test_no_dedispersion";

    pipeline_config.build_loadtofold(loader);

    std::string operation_names[] = {
        "IOManager:DADA", "InverseFilterbank", "Detection", "Fold"
    };
    std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
    for (unsigned iop=0; iop < operations.size(); iop++){
      REQUIRE(operations[iop]->get_name() == operation_names[iop]);
    }

    pipeline_config.run_loadtofold(loader);
  }

  SECTION("InverseFilterbank works in during dedispersion case")
  {
    dsp::LoadToFold loader;

    pipeline_config.change_inverse_filterbank_config("1:D:128");

    REQUIRE(pipeline_config.config->inverse_filterbank.get_convolve_when() == dsp::Filterbank::Config::During);
    REQUIRE(pipeline_config.config->inverse_filterbank.get_freq_res() == 0);

    pipeline_config.config->coherent_dedispersion = true;
    pipeline_config.config->archive_filename = "InverseFilterbank.test_during_dedispersion";
    pipeline_config.build_loadtofold(loader);

    std::string operation_names[] = {
        "IOManager:DADA", "InverseFilterbank", "Detection", "Fold"
    };
    std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
    for (unsigned iop=0; iop < operations.size(); iop++){
      REQUIRE(operations[iop]->get_name() == operation_names[iop]);
    }

    pipeline_config.run_loadtofold(loader);
  }

  SECTION("InverseFilterbank works in after dedispersion case")
  {
    dsp::LoadToFold loader;

    pipeline_config.change_inverse_filterbank_config("1:1024:128");

    REQUIRE(pipeline_config.config->inverse_filterbank.get_convolve_when() == dsp::Filterbank::Config::After);
    REQUIRE(pipeline_config.config->inverse_filterbank.get_freq_res() == 1024);

    pipeline_config.config->coherent_dedispersion = true;
    pipeline_config.config->archive_filename = "InverseFilterbank.test_after_dedispersion";
    pipeline_config.build_loadtofold(loader);

    std::string operation_names[] = {
        "IOManager:DADA", "InverseFilterbank", "Convolution", "Detection", "Fold"
    };
    std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
    for (unsigned iop=0; iop < operations.size(); iop++){
      REQUIRE(operations[iop]->get_name() == operation_names[iop]);
    }

    pipeline_config.run_loadtofold(loader);
  }
}
