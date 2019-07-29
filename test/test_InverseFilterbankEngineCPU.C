#include <vector>

#include "catch.hpp"

#include "dsp/Memory.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"

#include "util.hpp"
#include "InverseFilterbank_test_config.h"

TEST_CASE ("InverseFilterbankEngineCPU", "[InverseFilterbankEngineCPU]")
{
  dsp::InverseFilterbankEngineCPU engine;
}


TEST_CASE (
  "InverseFilterbankEngineCPU can operate on data",
	"[InverseFilterbankEngineCPU]"
)
{
  util::set_verbose(true);
  dsp::InverseFilterbankEngineCPU engine;
  util::IntegrationTestConfiguration<dsp::InverseFilterbank> config;

  Rational os_factor (4, 3);

  int idx = 2;
	test_config::TestShape test_shape = test_config::test_shapes[idx];
  unsigned npart = test_shape.npart;

  config.setup (
    os_factor, npart, test_shape.npol,
    test_shape.nchan, test_shape.output_nchan,
    test_shape.ndat, test_shape.overlap
  );

  config.filterbank->set_pfb_dc_chan(true);
  config.filterbank->set_pfb_all_chan(true);

  SECTION ("can call setup method")
  {
    engine.setup(config.filterbank);
  }

  SECTION ("can call perform method")
  {
    engine.setup(config.filterbank);
    std::vector<float*> scratch = config.allocate_scratch<dsp::Memory>();
    engine.set_scratch(scratch[0]);
    engine.perform(
      config.input, config.output, npart
    );
    engine.finish();
  }
}
