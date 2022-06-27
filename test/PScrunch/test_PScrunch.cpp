
#include "catch.hpp"

#include "dsp/PScrunch.h"

#include "util/util.hpp"

#include <iostream>
#include <functional>

using std::cerr;
using std::endl;

TEST_CASE(
  "PScrunch",
  "[unit][no_file][PScrunch]"
)
{
  SECTION ("Can construct PScrunch object") {
    dsp::PScrunch pscrunch;
  }
}

TEST_CASE(
  "PScrunch::prepare",
  "[component][no_file][PScrunch]"
)
{
  dsp::PScrunch pscrunch;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  // nchan, npol, ndat
  std::vector<unsigned> shape {
      8, 2, 100
  };

  input->set_state(Signal::Coherence);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_ndim(1);
  input->resize(shape[2]);

  pscrunch.set_input(input);
  pscrunch.set_output(output);
  pscrunch.set_output_state(Signal::Intensity);

  SECTION("Can call prepare")
  {
    pscrunch.prepare();
  }
}

TEST_CASE(
  "PScrunch::operate input npol=2",
  "[component][no_file][PScrunch]"
)
{
  dsp::PScrunch pscrunch;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  // nchan, npol, ndat
  std::vector<unsigned> shape {
      8, 2, 100 
  };

  auto filler_fpt = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (ichan*shape[1]*shape[2] + ipol*shape[2] + idat);
  };

  auto filler_tfp = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (idat*shape[0]*shape[1] + ichan*shape[1] + ipol);
  };

  auto intensity_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_fpt(ichan, 0, idat, 0) + filler_fpt(ichan, 1, idat, 0));
  };

  auto intensity_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_tfp(ichan, 0, idat, 0) + filler_tfp(ichan, 1, idat, 0));
  };

  auto pp_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 0, idat, 0);
  };

  auto pp_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 0, idat, 0);
  };

  auto qq_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 1, idat, 0);
  };

  auto qq_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 1, idat, 0);
  };

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_fpt;
  map_fpt[Signal::Intensity] = intensity_fpt;
  map_fpt[Signal::PP_State] = pp_state_fpt;
  map_fpt[Signal::QQ_State] = qq_state_fpt;

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_tfp;
  map_tfp[Signal::Intensity] = intensity_tfp;
  map_tfp[Signal::PP_State] = pp_state_tfp;
  map_tfp[Signal::QQ_State] = qq_state_tfp;

  input->set_ndim(1);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_state(Signal::PPQQ);
  input->set_order(dsp::TimeSeries::Order::OrderFPT);
  input->resize(shape[2]);

  pscrunch.set_output_state(Signal::Intensity);
  pscrunch.set_input(input);
  pscrunch.set_output(output);

  std::vector<Signal::State> output_states = { Signal::PP_State, Signal::QQ_State, Signal::Intensity };

  test::util::init_TimeSeries(input, filler_fpt);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_fpt.begin(); ot != map_fpt.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    uint64_t nerrors = test::util::check_TimeSeries(output, ot->second);
    CHECK(nerrors == 0);
  }

  input->set_order(dsp::TimeSeries::Order::OrderTFP);
  input->resize(shape[2]);

  test::util::init_TimeSeries_TFP(input, filler_tfp);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_tfp.begin(); ot != map_tfp.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    uint64_t nerrors = test::util::check_TimeSeries_TFP(output, ot->second);
    CHECK(nerrors == 0);
  }

}


TEST_CASE(
  "PScrunch::operate input npol=4",
  "[component][no_file][PScrunch]"
)
{
  dsp::PScrunch pscrunch;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  // nchan, npol, ndat
  std::vector<unsigned> shape {
      8, 4, 100
  };

  auto filler_fpt = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (ichan*shape[1]*shape[2] + ipol*shape[2] + idat);
  };

  auto filler_tfp = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (idat*shape[0]*shape[1] + ichan*shape[1] + ipol);
  };

  auto ppqq_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, ipol, idat, 0);
  };

  auto ppqq_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, ipol, idat, 0);
  };

  auto intensity_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_fpt(ichan, 0, idat, 0) + filler_fpt(ichan, 1, idat, 0));
  };

  auto intensity_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_tfp(ichan, 0, idat, 0) + filler_tfp(ichan, 1, idat, 0));
  };

  auto pp_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 0, idat, 0);
  };

  auto pp_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 0, idat, 0);
  };

  auto qq_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 1, idat, 0);
  };

  auto qq_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 1, idat, 0);
  };

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_fpt;
  map_fpt[Signal::PPQQ] = ppqq_fpt;
  map_fpt[Signal::Intensity] = intensity_fpt;
  map_fpt[Signal::PP_State] = pp_state_fpt;
  map_fpt[Signal::QQ_State] = qq_state_fpt;

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_tfp;
  map_tfp[Signal::PPQQ] = ppqq_tfp;
  map_tfp[Signal::Intensity] = intensity_tfp;
  map_tfp[Signal::PP_State] = pp_state_tfp;
  map_tfp[Signal::QQ_State] = qq_state_tfp;

  input->set_ndim(1);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_state(Signal::Coherence);
  input->set_order(dsp::TimeSeries::Order::OrderFPT);
  input->resize(shape[2]);

  pscrunch.set_output_state(Signal::Intensity);
  pscrunch.set_input(input);
  pscrunch.set_output(output);

  std::vector<Signal::State> output_states = { Signal::PP_State, Signal::QQ_State, Signal::Intensity };

  test::util::init_TimeSeries(input, filler_fpt);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_fpt.begin(); ot != map_fpt.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    uint64_t nerrors = test::util::check_TimeSeries(output, ot->second);
    CHECK(nerrors == 0);
  }

  input->set_order(dsp::TimeSeries::Order::OrderTFP);
  input->resize(shape[2]);

  test::util::init_TimeSeries_TFP(input, filler_tfp);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_tfp.begin(); ot != map_tfp.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    uint64_t nerrors = test::util::check_TimeSeries_TFP(output, ot->second);
    CHECK(nerrors == 0);
  }
}
