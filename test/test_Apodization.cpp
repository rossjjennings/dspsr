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
#include "util/TestConfig.hpp"

static test::util::TestConfig test_config;


TEST_CASE("Apodization produces correct window functions", "[Apodization]") {

  dsp::Apodization window;

  SECTION("Tukey window produces correct output")
  {
    const std::string tukey_file_name = test_config.get_field<std::string>(
      "test_Apodization.tukey_window_file_name");
    const std::string tukey_file_path = test::util::get_test_data_dir() + "/" + tukey_file_name;

    if (test::util::config::verbose) {
      std::cerr << "test_Apodization: tukey_file_path=" << tukey_file_path << std::endl;
    }

    window.Tukey(1024, 0, 128, false);
    std::vector<float> expected_data;
    test::util::load_binary_data(tukey_file_path, expected_data);
    if (test::util::config::verbose) {
      std::cerr << "test_Apodization: expected_data.size()=" << expected_data.size() << std::endl;
    }
    REQUIRE(expected_data.size() == 1024);
    bool output_equal = test::util::allclose<float>(
      window.get_datptr(0, 0),
      expected_data.data(),
      expected_data.size());
    REQUIRE(output_equal == true);
  }

  SECTION("TopHat window produces correct output")
  {
    const std::string tophat_file_name = test_config.get_field<std::string>(
      "test_Apodization.tophat_window_file_name");
    const std::string tophat_file_path = test::util::get_test_data_dir() + "/" + tophat_file_name;

    if (test::util::config::verbose) {
      std::cerr << "test_Apodization: tophat_file_path=" << tophat_file_path << std::endl;
    }

    window.TopHat(1024, 128, false);
    std::vector<float> expected_data;
    test::util::load_binary_data(tophat_file_path, expected_data);
    if (test::util::config::verbose) {
      std::cerr << "test_Apodization: expected_data.size()=" << expected_data.size() << std::endl;
    }
    REQUIRE(expected_data.size() == 1024);
    bool output_equal = test::util::allclose<float>(
      window.get_datptr(0, 0),
      expected_data.data(),
      expected_data.size());
    REQUIRE(output_equal == true);

  }

  SECTION("None window produces correct output")
  {
    window.None(1024, false);
    std::vector<float> expected_data (1024, 1.0);
    bool output_equal = test::util::allclose<float>(
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
