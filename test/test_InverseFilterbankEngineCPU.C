#include "catch.hpp"

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"

#include "util.hpp"

TEST_CASE ("InverseFilterbankEngineCPU::ctor", "[InverseFilterbankEngineCPU]")
{
  dsp::InverseFilterbankEngineCPU engine;
}


// TEST_CASE ("InverseFilterbankEngineCPU::perform", "[InverseFilterbankEngineCPU]")
// {
//   dsp::InverseFilterbankEngineCPU engine;
// }
