#include "catch.hpp"

#include "dsp/SpectralKurtosis.h"

#include "util/util.hpp"

TEST_CASE(
  "SpectralKurtosis",
  "[unit][no_file][SpectralKurtosis]"
)
{
  SECTION ("Can construct SpectralKurtosis object") {
    dsp::SpectralKurtosis sk;
  }
}

TEST_CASE(
  "SpectralKurtosis::prepare",
  "[component][no_file][SpectralKurtosis]"
)
{
  dsp::SpectralKurtosis sk;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> zero_DM_input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  std::vector<unsigned> shape {
      8, 2, 100
  };

  input->set_state(Signal::Analytic);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_ndim(2);
  input->resize(shape[2]);
  zero_DM_input->copy_configuration(input);
  zero_DM_input->resize(shape[2]);

  sk.set_input(input);
  sk.set_output(output);

  SECTION("Can call prepare with no zero_DM_input")
  {
    sk.prepare();
  }

  SECTION("Can call prepare with zero_DM_input")
  {
    sk.set_zero_DM_input(zero_DM_input);
    sk.prepare();
  }
}

TEST_CASE(
  "SpectralKurtosis::operate",
  "[component][no_file][SpectralKurtosis]"
)
{
  dsp::SpectralKurtosis sk;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> zero_DM_input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  std::vector<unsigned> shape {
      8, 2, 100
  };

  input->set_state(Signal::Analytic);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_ndim(2);
  input->resize(shape[2]);
  zero_DM_input->copy_configuration(input);
  zero_DM_input->resize(shape[2]);

  auto filler = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    if (idim == 0) {
      return (float) (ichan*shape[1]*shape[2] + ipol*shape[2] + idat);
    } else {
      return 0.0;
    }
  };

  test::util::init_TimeSeries(input, filler);
  test::util::init_TimeSeries(zero_DM_input, filler);
  sk.set_input(input);
  sk.set_output(output);

  SECTION("Can call operate with no zero_DM_input")
  {
    input->set_input_sample(0);
    sk.operate();
  }

  SECTION("Can call operate with zero_DM_input")
  {
    input->set_input_sample(0);
    zero_DM_input->set_input_sample(0);
    sk.set_zero_DM_input(zero_DM_input);
    sk.operate();
  }
}
