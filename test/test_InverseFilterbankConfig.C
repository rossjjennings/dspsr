#include <string>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <math.h>

#include "dsp/InverseFilterbankConfig.h"
#include "dsp/ConvolutionConfig.h"


class TestInverseFilterbankConfig {

public:
	void setup ();
	void test_ctor ();
  void test_istream ();

};

void TestInverseFilterbankConfig::setup ()
{

}

void TestInverseFilterbankConfig::test_ctor ()
{
	std::cerr << "TestInverseFilterbankConfig::test_ctor " << std::endl;
  dsp::InverseFilterbank::Config config;
}



void TestInverseFilterbankConfig::test_istream ()
{
  std::cerr << "TestInverseFilterbankConfig::test_istream" << std::endl;
  dsp::InverseFilterbank::Config config;
	std::string stringvalues = "1:D";
  std::istringstream iss (stringvalues);

  iss >> config;
	assert(config.get_convolve_when() == dsp::Convolution::Config::During);
	assert(config.get_nchan() == 1);

	stringvalues = "1:16384";
	iss = std::istringstream(stringvalues);

	iss >> config;
	assert(config.get_convolve_when() == dsp::Convolution::Config::After);
	assert(config.get_nchan() == 1);
	assert(config.get_freq_res() == 16384);

	stringvalues = "1:16384:128";
	iss = std::istringstream(stringvalues);

	iss >> config;
	assert(config.get_convolve_when() == dsp::Convolution::Config::After);
	assert(config.get_nchan() == 1);
	assert(config.get_freq_res() == 16384);
	assert(config.get_input_overlap() == 128);


}

int main () {
	// dsp::Input::verbose = true;
	// dsp::Operation::verbose = true;
	// dsp::Shape::verbose = true;

	TestInverseFilterbankConfig tester;

  tester.setup();
  tester.test_ctor();
  tester.test_istream();

	return 0;
}
