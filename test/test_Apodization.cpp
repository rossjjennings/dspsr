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


TEST_CASE("Apodization produces correct window functions", "[Apodization]")
{
  dsp::Apodization window;

  SECTION("Tukey window produces correct output")
  {
    const std::string tukey_file_name = test_config.get_field<std::string>(
      "test_Apodization.tukey_window_file_name");
    const std::string tukey_file_path = test::util::get_test_data_dir() + "/" + tukey_file_name;

    if (test::util::config::verbose) {
      std::cerr << "test_Apodization: tukey_file_path=" << tukey_file_path << std::endl;
    }

    window.set_type( "tukey" );
    window.set_size (1024);
    window.set_transition (128);
    window.set_analytic (false);
    window.build ();

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

  SECTION("None window produces correct output")
  {
    window.set_type( "none" );
    window.set_size (1024);
    window.build ();

    std::vector<float> expected_data (1024, 1.0);
    bool output_equal = test::util::allclose<float>(
      window.get_datptr(0, 0),
      expected_data.data(),
      expected_data.size());
    REQUIRE(output_equal == true);

  }

  SECTION("Apodization::set_type fails on incorrect type strings")
  {
    try {
      window.set_type( "junk" );
      REQUIRE(false);
    }
    catch (Error& error)
    {
      REQUIRE(true);
    }
  }
}

