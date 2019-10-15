#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>

#include "catch.hpp"

#include "Rational.h"
#include "dsp/FIRFilter.h"

TEST_CASE("FIRFilter filter taps and coefficients can be manipulated", "[unit][no_file][FIRFilter]")
{
  dsp::FIRFilter filter;
  std::vector<float> coeff = {2.0, 1.0};
  dsp::FIRFilter filter1(coeff, Rational(1, 1), 8);

  SECTION("can get and set number of filter taps"){
    dsp::FIRFilter filter;
    filter.set_ntaps(400);
    REQUIRE(filter.get_ntaps() == 400);
    REQUIRE(filter1.get_ntaps() == 2);
  }


  SECTION("can get and set filter coefficients with operator[]"){
    std::vector<float> coeff = {2.0, 1.0};
    dsp::FIRFilter filter(coeff, Rational(1, 1), 8);
    std::vector<float>* coeff_ref = filter.get_coeff();
    coeff_ref->at(0) = 1.0;

    bool allclose = true;
    for (unsigned i=0; i<filter.get_ntaps(); i++) {
      if (filter[i] != 1.0) {
        allclose = false;
      }
    }
    REQUIRE(allclose == true);

    allclose = true;

    filter[0] = 2.0;
    filter[1] = 2.0;

    for (unsigned i=0; i<filter.get_ntaps(); i++) {
      if (filter[i] != 2.0) {
        allclose = false;
      }
    }
    REQUIRE(allclose == true);

  }
}
