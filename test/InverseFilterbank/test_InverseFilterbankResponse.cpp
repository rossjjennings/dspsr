#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <fstream>

#include "catch.hpp"

#include "Rational.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/FIRFilter.h"
#include "dsp/InverseFilterbankResponse.h"

#include "util/util.hpp"
#include "util/TestConfig.hpp"

static test::util::TestConfig test_config;

const unsigned ntaps_expected = 81;
const unsigned input_fft_length = 128;
const Rational os_factor = Rational(4, 3);
const unsigned nchan = 8;
const unsigned freq_response_size = nchan * os_factor.normalize(input_fft_length);

TEST_CASE("InverseFilterbankResponse attributes can be manipulated",
          "[unit][no_file][InverseFilterbankResponse]")
{
  dsp::InverseFilterbankResponse deripple_response;

  SECTION ("can get and set ndat")
  {
    unsigned new_ndat = 128;
    deripple_response.set_ndat(new_ndat);
    REQUIRE(deripple_response.get_ndat() == new_ndat);
  }

  SECTION ("can get and set nchan")
  {
    unsigned new_nchan = 3;
    deripple_response.set_nchan(new_nchan);
    REQUIRE(deripple_response.get_nchan() == new_nchan);
  }
}

// TEST_CASE ("InverseFilterbankResponse roll should produce expected result",
//            "[InverseFilterbankResponse]")
// {
//
// }


TEST_CASE("InverseFilterbankResponse produces correct derippling response",
          "[InverseFilterbankResponse][component]")
{
  using IFResponse = dsp::InverseFilterbankResponse;
  const std::string file_name = test_config.get_field<std::string>(
    "InverseFilterbank.test_InverseFilterbankResponse.fir_file_name");

  const std::string file_path = test::util::get_test_data_dir() + "/" + file_name;

  // if (test::util::config::verbose) {
  //   std::cerr << "test_InverseFilterbankResponse: FIR file path=" << file_path << std::endl;
  // }

  // test::util::set_verbose(true);
  dsp::IOManager manager;
  manager.open(file_path);
  dsp::Observation* info = manager.get_input()->get_info();

  int block_size = 2*freq_response_size;
  dsp::TimeSeries* freq_response_expected = new dsp::TimeSeries;
  test::util::load_psr_data(manager, block_size, freq_response_expected);

  const std::vector<dsp::FIRFilter> filters = info->get_deripple();
  IFResponse deripple_response;

  deripple_response.set_fir_filter(filters[0]);
  deripple_response.set_pfb_dc_chan(false);
  deripple_response.set_apply_deripple(true);
  deripple_response.set_ndat(freq_response_size);
  deripple_response.set_input_nchan(nchan);
  deripple_response.set_oversampling_factor(os_factor);
  deripple_response.resize(1, 1, freq_response_size, 2); // have to explicitly call this
  deripple_response.build();


  float expected_val;
  float test_val;


  SECTION ("copy operator works as expected") {
    IFResponse new_deripple_response(deripple_response);

    float* deripple_response_buffer = deripple_response.get_datptr(0, 0);
    float* new_deripple_response_buffer = new_deripple_response.get_datptr(0, 0);

    unsigned nclose = test::util::nclose(
      deripple_response_buffer,
      new_deripple_response_buffer,
      block_size
    );

    REQUIRE(nclose == block_size);
    if (test::util::config::verbose) {
      std::cerr << "test_InverseFilterbankResponse: "
        << nclose << "/" << block_size << " (" << ((float) nclose / block_size) * 100
        << "%)" << std::endl;
    }
  }

  SECTION ("copy operator works with pointers") {
    Reference::To<
      IFResponse
    > ref_deripple_response = new IFResponse;
    ref_deripple_response->resize(1, 1, freq_response_size, 2); // have to explicitly call this

    Reference::To<
      IFResponse
    > new_ref_deripple_response = new IFResponse(*ref_deripple_response);
  }


  SECTION ("build produces correct frequency response") {
    float* freq_response_expected_buffer = freq_response_expected->get_datptr(0, 0);

    float* freq_response_test_buffer = deripple_response.get_datptr(0, 0);
    // deripple_response.roll<float>(
    //     freq_response_test_buffer,
    //     freq_response_size*2,
    //     freq_response_size/nchan);

    test::util::write_binary_data<float>(
        "freq_response_test.dat", freq_response_test_buffer, 2*freq_response_size);
    test::util::write_binary_data<float>(
        "freq_response_expected.dat", freq_response_expected_buffer, 2*freq_response_size);

    bool isclose;
    bool allclose = true;

    for (int i=0; i<block_size; i++) {
      test_val = freq_response_test_buffer[i];
      expected_val = freq_response_expected_buffer[i];
      if (expected_val != 0) {
        expected_val = 1.0 / expected_val;
      }
      isclose = test::util::isclose<float>(test_val, expected_val, 1e-7, 1e-5);
      if (! isclose) {
        allclose = false;
      }
    }
    REQUIRE(allclose == true);
  }
}
