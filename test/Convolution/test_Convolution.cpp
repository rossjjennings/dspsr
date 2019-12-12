#include <string>
#include <iostream>
#include <assert.h>

#include "catch.hpp"

#include "dsp/Convolution.h"

TEST_CASE (
  "Convolution",
  "[unit][no_file][Convolution]"
)
{
  dsp::Convolution convolution;

  SECTION("Can get and set zero_DM flag")
  {
    REQUIRE(convolution.get_zero_DM() == false);
    convolution.set_zero_DM(true);
    REQUIRE(convolution.get_zero_DM() == true);
  }

  SECTION("Can get and set zero_DM output TimeSeries")
  {
    Reference::To<dsp::TimeSeries> zero_DM_output = new dsp::TimeSeries;

    REQUIRE(convolution.has_zero_DM_output() == false);

    convolution.set_zero_DM_output(zero_DM_output);

    REQUIRE(convolution.get_zero_DM() == true);
    REQUIRE(convolution.has_zero_DM_output() == true);
  }
}
