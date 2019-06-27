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

#include "util.hpp"

const std::string file_path = "/home/SWIN/dshaff/ska/test_data/fir.768.dada";

const unsigned ntaps_expected = 81;
const unsigned input_fft_length = 128;
const Rational os_factor = Rational(4, 3);
const unsigned nchan = 8;
const unsigned freq_response_size = nchan * os_factor.normalize(input_fft_length);

TEST_CASE("InverseFilterbankResponse attributes can be manipulated", "[InverseFilterbankResponse]")
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

TEST_CASE ("InverseFilterbankResponse roll should produce expected result", "[InverseFilterbankResponse]")
{
  
}


TEST_CASE("InverseFilterbankResponse produces correct derippling response", "[InverseFilterbankResponse]")
{
  // util::set_verbose(true);
  dsp::IOManager manager;
  manager.open(file_path);
  dsp::Observation* info = manager.get_input()->get_info();

  int block_size = 2*freq_response_size;
  dsp::TimeSeries* freq_response_expected = new dsp::TimeSeries;
  util::load_psr_data(manager, block_size, freq_response_expected);

  SECTION ("build produces correct frequency response") {
    const std::vector<dsp::FIRFilter> filters = info->get_deripple();
    dsp::InverseFilterbankResponse deripple_response;

    deripple_response.set_fir_filter(filters[0]);
    deripple_response.set_pfb_dc_chan(false);
    deripple_response.set_apply_deripple(true);
    deripple_response.set_ndat(freq_response_size);
    deripple_response.set_input_nchan(nchan);
    deripple_response.set_oversampling_factor(os_factor);
    deripple_response.resize(1, 1, freq_response_size, 2); // have to explicitly call this
    deripple_response.build();

    float* freq_response_expected_buffer = freq_response_expected->get_datptr(0, 0);

    float* freq_response_test_buffer = deripple_response.get_datptr(0, 0);
    // deripple_response.roll<float>(
    //     freq_response_test_buffer,
    //     freq_response_size*2,
    //     freq_response_size/nchan);

    util::write_binary_data<float>(
        "freq_response_test.dat", freq_response_test_buffer, 2*freq_response_size);
    util::write_binary_data<float>(
        "freq_response_expected.dat", freq_response_expected_buffer, 2*freq_response_size);

    float expected_val;
    float test_val;

    bool isclose;
    bool allclose = true;

    for (int i=0; i<block_size; i++) {
      test_val = freq_response_test_buffer[i];
      expected_val = freq_response_expected_buffer[i];
      if (expected_val != 0) {
        expected_val = 1.0 / expected_val;
      }
      isclose = util::isclose<float>(test_val, expected_val, 1e-7, 1e-5);
      if (! isclose) {
        allclose = false;
      }
    }
    REQUIRE(allclose == true);
  }
}
