#include <string>
#include <sstream>
#include <iostream>

#include "catch.hpp"

#include "dsp/FilterbankConfig.h"

TEST_CASE(
  "FilterbankConfig can intake arguments",
  "[unit][no_file][FilterbankConfig]"
)
{

  SECTION ("istream method produces correct configuration")
  {
    dsp::Filterbank::Config config;
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
  }
}

TEST_CASE(
  "FilterbankConfig can stream correct internal representation",
  "[unit][no_file][FilterbankConfig]"
)
{
  dsp::Filterbank::Config config;
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
}
