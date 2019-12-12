#include <string>
#include <iostream>
#include <assert.h>

#include "catch.hpp"

#include "dsp/Convolution.h"

#include "util/util.hpp"

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

TEST_CASE(
  "Convolution::prepare",
  "[component][no_file][Convolution]"
)
{
  dsp::Convolution convolution;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;
  Reference::To<dsp::Response> response = new dsp::Response;

  std::vector<unsigned> shape {
      8, 2, 100
  };

  unsigned freq_res = 16;

  input->set_state(Signal::Analytic);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_ndim(2);
  input->resize(shape[2]);

  response->resize(1, shape[0], freq_res, 2);

  auto filler = [] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    if (idim == 0) {
      return 1.0;
    } else {
      return 0.0;
    }
  };

  test::util::init_TimeSeries(input, filler);
  convolution.set_input(input);
  convolution.set_output(output);
  convolution.set_response(response);

  SECTION("prepare method throws runtime error if input data are detected")
  {
    input->set_state(Signal::Stokes);
    REQUIRE_THROWS(convolution.prepare());
  }

  SECTION ("prepare method throws runtime error without response")
  {
    convolution.set_response(nullptr);
    REQUIRE_THROWS(convolution.prepare());
  }

  SECTION ("prepare method does not set up zero_DM_output if zero_DM is false")
  {
    convolution.set_zero_DM(false);
    convolution.prepare();
  }

  SECTION ("prepare method sets up zero_DM_output if zero_DM is true")
  {
    convolution.set_zero_DM(true);
    convolution.prepare();
    dsp::TimeSeries* zero_DM_output = convolution.get_zero_DM_output();

    CHECK(output->get_nchan() == zero_DM_output->get_nchan());
    CHECK(output->get_npol() == zero_DM_output->get_npol());
    CHECK(output->get_ndat() == zero_DM_output->get_ndat());
    CHECK(output->get_state() == zero_DM_output->get_state());
    CHECK(output->get_ndim() == zero_DM_output->get_ndim());
  }

}
