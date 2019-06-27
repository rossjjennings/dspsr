#include <string>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <math.h>

#include "catch.hpp"

#include "dsp/InverseFilterbankConfig.h"
#include "dsp/ConvolutionConfig.h"

TEST_CASE("InverseFilterbankConfig can intake arguments", "[InverseFilterbankConfig]")
{
  dsp::InverseFilterbank::Config config;

  SECTION ("istream method produces correct configuration")
  {
    dsp::InverseFilterbank::Config config;
    std::string stringvalues = "1:D";
    std::istringstream iss (stringvalues);

    iss >> config;
    REQUIRE(config.get_convolve_when() == dsp::Convolution::Config::During);
    REQUIRE(config.get_nchan() == 1);

    stringvalues = "1:16384";
    iss = std::istringstream(stringvalues);

    iss >> config;
    REQUIRE(config.get_convolve_when() == dsp::Convolution::Config::After);
    REQUIRE(config.get_nchan() == 1);
    REQUIRE(config.get_freq_res() == 16384);

    stringvalues = "1:16384:128";
    iss = std::istringstream(stringvalues);

    iss >> config;
    REQUIRE(config.get_convolve_when() == dsp::Convolution::Config::After);
    REQUIRE(config.get_nchan() == 1);
    REQUIRE(config.get_freq_res() == 16384);
    REQUIRE(config.get_input_overlap() == 128);
  }
}
