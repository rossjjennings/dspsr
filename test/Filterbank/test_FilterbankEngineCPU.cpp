#include "catch.hpp"

#include "dsp/FilterbankEngineCPU.h"


TEST_CASE ("can create instance of FilterbankEngineCPU", "[unit][no_file][FilterbankEngineCPU]")
{
  dsp::FilterbankEngineCPU engine;
}

// TEST_CASE ("FilterbankEngineCPU can operate on data", "[FilterbankEngineCPU]")
// {
//   dsp::FilterbankEngineCPU engine;
//   Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
//   Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
//
//
//   // SECTION ("can do bare setup")
//   // {
//   //   engine.setup(filterbank);
//   // }
// }
