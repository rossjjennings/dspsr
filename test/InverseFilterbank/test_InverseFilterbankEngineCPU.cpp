#include <vector>

#include "catch.hpp"

#include "dsp/Memory.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"

#include "util/util.hpp"
#include "InverseFilterbankTestConfig.hpp"

static test::util::InverseFilterbank::InverseFilterbankTestConfig test_config;

TEST_CASE (
  "InverseFilterbankEngineCPU",
  "[unit][no_file][InverseFilterbankEngineCPU]")
{
  if (test::util::config::verbose) {
    std::cerr << "test_InverseFilterbankEngineCPU: [no_file][unit]" << std::endl;
  }
  dsp::InverseFilterbankEngineCPU engine;
}

TEST_CASE (
  "InverseFilterbankEngineCPU can operate on data",
	"[no_file][InverseFilterbankEngineCPU][component]"
)
{
  if (test::util::config::verbose) {
    std::cerr << "test_InverseFilterbankEngineCPU: [no_file]" << std::endl;
  }
  dsp::InverseFilterbankEngineCPU engine;
  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;

  Rational os_factor (4, 3);

  int idx = 0;
  std::vector<test::util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  test::util::TestShape test_shape = test_shapes[idx];
  unsigned npart = test_shape.npart;

  test::util::InverseFilterbank::InverseFilterbankProxy proxy (
    os_factor, npart, test_shape.input_npol,
    test_shape.input_nchan, test_shape.output_nchan,
    test_shape.input_ndat, test_shape.overlap_pos
  );
  proxy.filterbank->set_pfb_dc_chan(true);
  proxy.filterbank->set_pfb_all_chan(true);

  proxy.setup (in, out, false, false);

  SECTION ("can call setup method")
  {
    engine.setup(proxy.filterbank);
  }

  SECTION ("can call perform method")
  {
    engine.setup(proxy.filterbank);
    float* scratch = proxy.allocate_scratch(engine.get_total_scratch_needed());
    engine.set_scratch(scratch);
    engine.perform(
      in, out, npart
    );
    engine.finish();
  }
}
