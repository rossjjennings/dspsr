#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>

#include "catch.hpp"

#include "Rational.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/FIRFilter.h"
#include "dsp/Apodization.h"

#include "util/util.hpp"

std::string tukey_file_path = "/home/SWIN/dshaff/ska/test_data/tukey_window.dat";
std::string tophat_file_path = "/home/SWIN/dshaff/ska/test_data/tophat_window.dat";


TEST_CASE("Apodization produces correct window functions") {
  dsp::Apodization window;

  SECTION("Tukey window produces correct output")
  {
    window.Tukey(1024, 0, 128, false);
    std::vector<float> expected_data;
    util::load_binary_data(tukey_file_path, expected_data);
    bool output_equal = util::allclose<float>(
      window.get_datptr(0, 0),
      expected_data.data(),
      expected_data.size());
    REQUIRE(output_equal == true);
  }

  SECTION("TopHat window produces correct output")
  {
    window.TopHat(1024, 128, false);
    std::vector<float> expected_data;
    util::load_binary_data(tophat_file_path, expected_data);
    bool output_equal = util::allclose<float>(
      window.get_datptr(0, 0),
      expected_data.data(),
      expected_data.size());
    REQUIRE(output_equal == true);

  }

  SECTION("None window produces correct output")
  {
    window.None(1024, false);
    std::vector<float> expected_data (1024, 1.0);
    bool output_equal = util::allclose<float>(
      window.get_datptr(0, 0),
      expected_data.data(),
      expected_data.size());
    REQUIRE(output_equal == true);

  }

  SECTION("Type map correctly looks up Apodization Types")
  {
    dsp::Apodization::Type t = dsp::Apodization::type_map["tukey"];
    REQUIRE(t == dsp::Apodization::tukey);
  }
}
