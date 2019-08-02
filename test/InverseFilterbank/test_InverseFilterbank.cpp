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

#include "util.hpp"

const std::string file_path = util::get_test_data_dir() + "/channelized.simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump";
const unsigned block_size = 699048; // this is taken from dspsr logs
const double dm = 2.64476;
const unsigned freq_res = 1024;

TEST_CASE ("InverseFilterbank") {
  dsp::InverseFilterbank filterbank;

  SECTION ("InverseFilterbank prepare method runs")
  {
    // filterbank.prepare();
  }

  SECTION ("InverseFilterbank reserve method runs")
  {
    // filterbank.reserve();
  }

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

TEST_CASE ("InverseFilterbank runs on channelized data", "")
{
  // util::set_verbose(true);

  dsp::IOManager manager;
  manager.open(file_path);

  dsp::TimeSeries* unpacked = new dsp::TimeSeries;
  util::load_psr_data(manager, block_size, unpacked);

  dsp::Observation* info = manager.get_input()->get_info();
  info->set_dispersion_measure(dm);

  const Rational os_factor = info->get_oversampling_factor();
  unsigned input_nchan = info->get_nchan();
  unsigned input_npol = info->get_npol();

  dsp::TimeSeries* output = new dsp::TimeSeries;
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
  // deripple->set_nchan(1);
  // deripple->resize(input_npol, 1, ndat, 2);

  filterbank.set_engine(filterbank_engine);
  filterbank.set_response(deripple);

  filterbank.prepare();
  filterbank.operate();
}
