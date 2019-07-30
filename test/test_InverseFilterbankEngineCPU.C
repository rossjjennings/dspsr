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
  dsp::InverseFilterbankEngineCPU engine;
  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;

  Rational os_factor (4, 3);

  int idx = 2;
	test_config::TestShape test_shape = test_config::test_shapes[idx];
  unsigned npart = test_shape.npart;

  util::IntegrationTestConfiguration<dsp::InverseFilterbank> config (
    os_factor, npart, test_shape.npol,
    test_shape.nchan, test_shape.output_nchan,
    test_shape.ndat, test_shape.overlap
  );

  config.filterbank->set_pfb_dc_chan(true);
  config.filterbank->set_pfb_all_chan(true);
  config.setup (in, out);

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
      in, out, npart
    );
    engine.finish();
  }
}
