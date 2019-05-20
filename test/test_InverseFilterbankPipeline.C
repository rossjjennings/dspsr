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
#include "dsp/InverseFilterbankConfig.h"

#include "util.hpp"

std::string file_path = "/home/SWIN/dshaff/ska/test_data/channelized.simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump";
double dm = 2.64476;
double folding_period = 0.00575745;

class TestInverseFilterbankPipeline : public util::TestDataLoader {

  public:
    void setup_class ();
    void setup ();

    void test_no_dedispersion ();
    void test_during_dedispersion ();
    void test_after_dedispersion ();
    void test_before_dedispersion ();

  private:

    void change_inverse_filterbank_config (std::string config_str);

    Reference::To<dsp::LoadToFold::Config> config;
    dsp::Input* input;
};

void TestInverseFilterbankPipeline::change_inverse_filterbank_config (std::string config_str)
{
  // set up inverse filterbank configuration
  std::istringstream iss = std::istringstream(config_str);
  iss >> config->inverse_filterbank;
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
  change_inverse_filterbank_config("1:16384:128");
  config->coherent_dedispersion = false;
  loader.set_configuration(config);
  loader.set_input(input);
  loader.construct();
}

void TestInverseFilterbankPipeline::test_during_dedispersion ()
{
  std::cerr << "TestInverseFilterbankPipeline::test_during_dedispersion " << std::endl;
  dsp::LoadToFold loader;
  change_inverse_filterbank_config("1:D");
  config->coherent_dedispersion = true;
  loader.set_configuration(config);
  loader.set_input(input);
  loader.construct();
  loader.prepare();
}

int main () {
  util::TestRunner<TestInverseFilterbankPipeline> runner;
	TestInverseFilterbankPipeline tester;
  tester.set_test_data_file_path(file_path);
  tester.set_verbose(true);
  runner.register_test_method(&TestInverseFilterbankPipeline::test_no_dedispersion);
  runner.register_test_method(&TestInverseFilterbankPipeline::test_during_dedispersion);
  runner.run(tester);

	return 0;
}
