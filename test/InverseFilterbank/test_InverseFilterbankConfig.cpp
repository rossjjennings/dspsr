#include <string>
#include <sstream>
#include <iostream>

#include "catch.hpp"

#include "dsp/InverseFilterbankConfig.h"
#include "dsp/FilterbankConfig.h"

TEST_CASE(
  "InverseFilterbankConfig can intake arguments",
  "[unit][no_file][InverseFilterbankConfig]"
)
{

  SECTION ("istream method produces correct configuration")
  {
    dsp::InverseFilterbank::Config config;
    std::string stringvalues = "1:D";
    std::istringstream iss (stringvalues);

    iss >> config;
    REQUIRE(config.get_convolve_when() == dsp::Filterbank::Config::During);
    REQUIRE(config.get_nchan() == 1);

    stringvalues = "1:16384";
    iss.clear();
    iss.str(stringvalues);

    iss >> config;
    REQUIRE(config.get_convolve_when() == dsp::Filterbank::Config::After);
    REQUIRE(config.get_nchan() == 1);
    REQUIRE(config.get_freq_res() == 16384);

    stringvalues = "1:16384:128";
    iss.clear();
    iss.str(stringvalues);

    iss >> config;
    REQUIRE(config.get_convolve_when() == dsp::Filterbank::Config::After);
    REQUIRE(config.get_nchan() == 1);
    REQUIRE(config.get_freq_res() == 16384);
    REQUIRE(config.get_input_overlap() == 128);
  }
}

TEST_CASE(
  "InverseFilterbankConfig can stream correct internal representation",
  "[unit][no_file][InverseFilterbankConfig]"
)
{
  dsp::InverseFilterbank::Config config;
  // std::string stringvalues = "1:D";
  std::ostringstream oss ;

  SECTION ("can output string indicating convolution happens during operation")
  {
    config.set_convolve_when(dsp::Filterbank::Config::During);
    config.set_nchan(1);
    oss << config;
    REQUIRE(oss.str() == "1:D");
  }

  SECTION ("can output string indicating convolution happens after operation")
  {
    config.set_convolve_when(dsp::Filterbank::Config::After);
    config.set_nchan(1);
    config.set_freq_res(16384);
    oss << config;
    REQUIRE(oss.str() == "1:16384");
  }

  SECTION ("can output string indicating convolution happens after operation, with input overlap information")
  {
    config.set_convolve_when(dsp::Filterbank::Config::After);
    config.set_nchan(1);
    config.set_freq_res(16384);
    config.set_input_overlap(128);
    oss << config;
    REQUIRE(oss.str() == "1:16384:128");
  }
}
