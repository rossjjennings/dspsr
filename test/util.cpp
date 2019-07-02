#include <cstdlib>
#include <string>

#include "dsp/Shape.h"

#include "util.hpp"


void util::load_psr_data (dsp::IOManager manager, int block_size, dsp::TimeSeries* ts)
{
  dsp::Input* input = manager.get_input();
  input->set_block_size(block_size);
  while (! manager.get_input()->eod()) {
    manager.load(ts);
  }
}

void util::set_verbose (bool val)
{
  dsp::Input::verbose = val;
  dsp::Operation::verbose = val;
  dsp::Shape::verbose = val;
}


std::string util::get_test_data_dir ()
{
  const char* env_data_dir = std::getenv("DSPSR_TEST_DATA_DIR");
  if (env_data_dir) {
    return std::string(env_data_dir);
  } else {
    return ".";
  }
}
