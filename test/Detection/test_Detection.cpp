#include "catch.hpp"

#include "dsp/Detection.h"

#include "util/util.hpp"

#include <iostream>
#include <functional>

using std::cerr;
using std::endl;

TEST_CASE(
  "Detection",
  "[unit][no_file][Detection]"
)
{
  SECTION ("Can construct Detection object") {
    dsp::Detection det;
  }
}

TEST_CASE(
  "Detection::prepare",
  "[component][no_file][Detection]"
)
{
  dsp::Detection det;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
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

  det.set_input(input);
  det.set_output(output);

  SECTION("Can call prepare")
  {
    det.prepare();
  }
}

TEST_CASE(
  "Detection::operate pol=1",
  "[component][no_file][Detection]"
)
{
  dsp::Detection det;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  // nchan, npol, ndat, ndim
  std::vector<unsigned> shape {
      8, 1, 100, 1
  };

  auto filler = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (ichan*shape[1]*shape[2] + ipol*shape[2] + idat);
  };

  auto intensity = [&shape, &filler] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    if (shape[3] == 2) 
      return (filler(ichan, 0, idat, 0) * filler(ichan, 0, idat, 0)) +
             (filler(ichan, 0, idat, 1) * filler(ichan, 0, idat, 1));
    else
      return filler(ichan, 0, idat, 0) * filler(ichan, 0, idat, 0);
  };

  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);

  det.set_output_state (Signal::Intensity);
  det.set_input(input);
  det.set_output(output);

  std::vector<Signal::State> input_states = { Signal::Analytic, Signal::Nyquist };

  for (std::vector<Signal::State>::iterator it = input_states.begin(); it != input_states.end(); ++it)
  {
    input->set_state(*it);
    shape[3] = (*it == Signal::Analytic) ? 2 : 1;
    input->resize(shape[2]);

    test::util::init_TimeSeries(input, filler);

    det.prepare();

    SECTION("Check Detection::operate on single polarisation with input_state {0}", Signal::state_string(*it));
    {
      input->set_input_sample(0);
      det.operate();

      uint64_t nerrors = test::util::check_TimeSeries(output, intensity);
      CHECK(nerrors == 0);
    }

  }
}


TEST_CASE(
  "Detection::operate pol=2",
  "[component][no_file][Detection]"
)
{
  dsp::Detection det;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  // nchan, npol, ndat, ndim
  std::vector<unsigned> shape {
      8, 2, 100, 1
  };

  auto filler = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (ichan*shape[1]*shape[2] + ipol*shape[2] + idat);
  };

  auto intensity = [&shape, &filler] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    // Signal::Analytic
    if (shape[3] == 2)
    {
      return (filler(ichan, 0, idat, 0) * filler(ichan, 0, idat, 0)) +
             (filler(ichan, 0, idat, 1) * filler(ichan, 0, idat, 1)) + 
             (filler(ichan, 1, idat, 0) * filler(ichan, 1, idat, 0)) +
             (filler(ichan, 1, idat, 1) * filler(ichan, 1, idat, 1));
    }
    // Signal::Nyquist
    else 
    {
      return (filler(ichan, 0, idat, 0) * filler(ichan, 0, idat, 0)) +
             (filler(ichan, 1, idat, 0) * filler(ichan, 1, idat, 0));
    }
  };

  auto ppqq = [&shape, &filler] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    // Signal::Analytic
    if (shape[3] == 2)
    {
      return (filler(ichan, ipol, idat, 0) * filler(ichan, ipol, idat, 0)) +
             (filler(ichan, ipol, idat, 1) * filler(ichan, ipol, idat, 1));
    }
    // Signal::Nyquist
    else
    {
      return filler(ichan, ipol, idat, 0) * filler(ichan, ipol, idat, 0);
    }
  };

  auto pp_state = [&shape, &filler] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    // Signal::Analytic
    if (shape[3] == 2)
    { 
      return (filler(ichan, 0, idat, 0) * filler(ichan, 0, idat, 0)) +
             (filler(ichan, 0, idat, 1) * filler(ichan, 0, idat, 1));
    }
    // Signal::Nyquist
    else
    {
      return filler(ichan, 0, idat, 0) * filler(ichan, 0, idat, 0);
    }
  };

  auto qq_state = [&shape, &filler] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    // Signal::Analytic
    if (shape[3] == 2)
    {
      return (filler(ichan, 1, idat, 0) * filler(ichan, 1, idat, 0)) +
             (filler(ichan, 1, idat, 1) * filler(ichan, 1, idat, 1));
    }
    // Signal::Nyquist
    else
    {
      return filler(ichan, 1, idat, 0) * filler(ichan, 1, idat, 0);
    }
  };

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map;

  map[Signal::Intensity] = intensity;
  map[Signal::PPQQ] = ppqq;
  map[Signal::PP_State] = pp_state;
  map[Signal::QQ_State] = qq_state;

  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);

  det.set_input(input);
  det.set_output(output);

  input->set_input_sample(0);

  std::vector<Signal::State> input_states = { Signal::Analytic, Signal::Nyquist };
  std::string section;

  section = std::string("Check Detection::operate for inputs and outputs");
  SECTION(section);
  {
    for (std::vector<Signal::State>::iterator it = input_states.begin(); it != input_states.end(); ++it)
    {
      input->set_state(*it);
      shape[3] = (*it == Signal::Analytic) ? 2 : 1;
      input->resize(shape[2]);

      test::util::init_TimeSeries(input, filler);

      for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map.begin(); ot != map.end(); ++ot)
      {
        det.set_output_state (ot->first);
        det.prepare();

        det.operate();
        uint64_t nerrors = test::util::check_TimeSeries(output, ot->second);
        CHECK(nerrors == 0);
      }
    }
  }
}


