#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <sstream>


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

std::string file_path = "/home/SWIN/dshaff/ska/test_data/channelized.simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump";
double dm = 2.64476;
double folding_period = 0.00575745;

class TestInverseFilterbankPipeline : public util::TestDataLoader {

  public:
    void setup_class ();
    void setup ();
    void teardown ();

    void test_no_dedispersion ();
    void test_during_dedispersion ();
    void test_after_dedispersion ();
    void test_before_dedispersion ();

  private:

    void change_inverse_filterbank_config (const std::string& config_str);
    void build_loadtofold (dsp::LoadToFold& loader);
    void run_loadtofold (dsp::LoadToFold& loader);
    Reference::To<dsp::LoadToFold::Config> config;
    dsp::Input* input;
};

void TestInverseFilterbankPipeline::teardown () {
  config->inverse_filterbank.set_freq_res(0);
  config->inverse_filterbank.set_input_overlap(0);
  config->inverse_filterbank.set_nchan(0);
}

void TestInverseFilterbankPipeline::change_inverse_filterbank_config (const std::string& config_str)
{
  // set up inverse filterbank configuration
  std::istringstream iss = std::istringstream(config_str);
  iss >> config->inverse_filterbank;
}

void TestInverseFilterbankPipeline::run_loadtofold (dsp::LoadToFold& loader)
{
  loader.prepare();
  loader.run();
  loader.finish();
}

void TestInverseFilterbankPipeline::build_loadtofold (dsp::LoadToFold& loader)
{
  loader.set_configuration(config);
  loader.set_input(input);
  loader.construct();
}


void TestInverseFilterbankPipeline::setup_class () {
  std::cerr << "TestInverseFilterbankPipeline::setup_class " << std::endl;
  config = new dsp::LoadToFold::Config;
  config->dispersion_measure = dm;
  config->folding_period = folding_period;
  config->coherent_dedispersion = true;
  config->do_deripple = true;

  // set up inverse filterbank configuration
  dsp::InverseFilterbank::Config if_config;
  config->inverse_filterbank = if_config;
  config->inverse_filterbank_fft_window = "tukey";
}

void TestInverseFilterbankPipeline::setup () {
  std::cerr << "TestInverseFilterbankPipeline::setup " << std::endl;

  Reference::To<dsp::File> file;
  file = dsp::File::create(test_data_file_path);
  input = file.release();

}


void TestInverseFilterbankPipeline::test_no_dedispersion ()
{
  std::cerr << "TestInverseFilterbankPipeline::test_no_dedispersion " << std::endl;
  dsp::LoadToFold loader;
  change_inverse_filterbank_config("1:1024:128");
  assert(config->inverse_filterbank.get_convolve_when() == dsp::Convolution::Config::After);
  assert(config->inverse_filterbank.get_freq_res() == 1024);
  config->coherent_dedispersion = false;
  config->archive_filename = "TestInverseFilterbankPipeline.test_no_dedispersion";
  build_loadtofold(loader);
  std::string operation_names[] = {
      "IOManager:DADA", "InverseFilterbank", "Detection", "Fold"
  };
  std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
  for (unsigned iop=0; iop < operations.size(); iop++){
    assert(operations[iop]->get_name() == operation_names[iop]);
  }
  run_loadtofold(loader);

}

void TestInverseFilterbankPipeline::test_during_dedispersion ()
{
  std::cerr << "TestInverseFilterbankPipeline::test_during_dedispersion " << std::endl;
  dsp::LoadToFold loader;
  change_inverse_filterbank_config("1:D:128");
  assert(config->inverse_filterbank.get_convolve_when() == dsp::Convolution::Config::During);
  assert(config->inverse_filterbank.get_freq_res() == 0);
  config->coherent_dedispersion = true;
  config->archive_filename = "TestInverseFilterbankPipeline.test_during_dedispersion";
  build_loadtofold(loader);
  std::string operation_names[] = {
      "IOManager:DADA", "InverseFilterbank", "Detection", "Fold"
  };
  std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
  for (unsigned iop=0; iop < operations.size(); iop++){
    assert(operations[iop]->get_name() == operation_names[iop]);
  }
  run_loadtofold(loader);

}

void TestInverseFilterbankPipeline::test_after_dedispersion ()
{
  std::cerr << "TestInverseFilterbankPipeline::test_after_dedispersion " << std::endl;
  dsp::LoadToFold loader;
  change_inverse_filterbank_config("1:1024:128");
  assert(config->inverse_filterbank.get_convolve_when() == dsp::Convolution::Config::After);
  assert(config->inverse_filterbank.get_freq_res() == 1024);
  config->coherent_dedispersion = true;
  config->archive_filename = "TestInverseFilterbankPipeline.test_after_dedispersion";
  build_loadtofold(loader);
  std::string operation_names[] = {
      "IOManager:DADA", "InverseFilterbank", "Convolution", "Detection", "Fold"
  };
  std::vector< Reference::To<dsp::Operation>> operations = loader.get_operations();
  for (unsigned iop=0; iop < operations.size(); iop++){
    assert(operations[iop]->get_name() == operation_names[iop]);
  }
  run_loadtofold(loader);
}

int main () {
  util::TestRunner<TestInverseFilterbankPipeline> runner;
	TestInverseFilterbankPipeline tester;
  tester.set_test_data_file_path(file_path);
  tester.set_verbose(true);
  // runner.register_test_method(
  //   &TestInverseFilterbankPipeline::test_no_dedispersion);
  runner.register_test_method(
    &TestInverseFilterbankPipeline::test_during_dedispersion);
  // runner.register_test_method(
  //   &TestInverseFilterbankPipeline::test_after_dedispersion);

  runner.run(tester);

	return 0;
}
