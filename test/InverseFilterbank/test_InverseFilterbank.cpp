#include <string>
#include <iostream>
#include <assert.h>

#include "catch.hpp"

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/ResponseProduct.h"
#include "dsp/Dedispersion.h"
#include "dsp/FIRFilter.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankResponse.h"

#include "util/util.hpp"
#include "util/TestConfig.hpp"


static test::util::TestConfig test_config;


TEST_CASE (
  "InverseFilterbank",
  "[unit][no_file][InverseFilterbank]"
) {
  dsp::InverseFilterbank filterbank;

  SECTION ("InverseFilterbank engine can be set")
  {
    dsp::InverseFilterbankEngineCPU* filterbank_engine = new dsp::InverseFilterbankEngineCPU;
    filterbank.set_engine(filterbank_engine);
  }

  SECTION ("setting input also sets oversampling factor")
  {
    Rational os_factor_old (8, 7);
    Rational os_factor_new (4, 3);

    filterbank.set_oversampling_factor(os_factor_old);
    REQUIRE (filterbank.get_oversampling_factor() == os_factor_old);

    Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
    input->set_oversampling_factor(os_factor_new);
    filterbank.set_input (input);
    REQUIRE (filterbank.get_oversampling_factor() == os_factor_new);
  }
}

TEST_CASE (
  "InverseFilterbank runs on channelized data",
  "[InverseFilterbank][component]")
{

  const std::string file_name = test_config.get_field<std::string>(
    "InverseFilterbank.test_InverseFilterbank.file_name");
  const std::string file_path = test::util::get_test_data_dir() + "/" + file_name;

  const unsigned block_size = test_config.get_field<unsigned>(
    "InverseFilterbank.test_InverseFilterbank.block_size");
  const double dm = test_config.get_field<double>(
    "InverseFilterbank.test_InverseFilterbank.dm");
  const unsigned freq_res = test_config.get_field<unsigned>(
    "InverseFilterbank.test_InverseFilterbank.freq_res");

  dsp::IOManager manager;
  manager.open(file_path);

  Reference::To<dsp::TimeSeries> unpacked = new dsp::TimeSeries;
  test::util::load_psr_data(manager, block_size, unpacked);

  dsp::Observation* info = manager.get_input()->get_info();
  info->set_dispersion_measure(dm);

  const Rational os_factor = info->get_oversampling_factor();
  unsigned input_nchan = info->get_nchan();
  unsigned input_npol = info->get_npol();

  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  dsp::InverseFilterbank filterbank;
  filterbank.set_buffering_policy(NULL);
  filterbank.set_output_nchan(1);
  filterbank.set_input(unpacked);
  filterbank.set_output(output);
  filterbank.set_pfb_dc_chan(true);
  filterbank.set_fft_window_str("no_window");

  unsigned ndat = input_nchan*os_factor.normalize(freq_res);

  dsp::InverseFilterbankEngineCPU* filterbank_engine = new dsp::InverseFilterbankEngineCPU;
  Reference::To<dsp::InverseFilterbankResponse> deripple = new dsp::InverseFilterbankResponse;
  deripple->set_fir_filter(info->get_deripple()[0]);
  deripple->set_apply_deripple(false);
  deripple->set_ndat(freq_res);
  deripple->resize(1, 1, freq_res, 2);

  filterbank.set_engine(filterbank_engine);
  filterbank.set_response(deripple);

  SECTION ("runs in zero DM case") {

    Reference::To<
      dsp::InverseFilterbankResponse
    > zero_DM_response = new dsp::InverseFilterbankResponse;
    *zero_DM_response = *deripple;

    Reference::To<dsp::TimeSeries> zero_DM_output = new dsp::TimeSeries;

    filterbank.set_zero_DM(true);
    filterbank.set_zero_DM_output(zero_DM_output);
    filterbank.set_zero_DM_response(zero_DM_response);

    filterbank.prepare();
    filterbank.operate();

    REQUIRE(zero_DM_response->get_ndat() == filterbank.get_response()->get_ndat());
  }

  SECTION ("runs in non zero DM case") {
    filterbank.prepare();
    filterbank.operate();
  }

}
