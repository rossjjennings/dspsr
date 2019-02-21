#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>

#include "Rational.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/FIRFilter.h"
#include "dsp/DerippleResponse.h"

std::string file_path = "/home/SWIN/dshaff/ska/pfb-channelizer/fir.1024.dada";

const unsigned ntaps_expected = 321;
const unsigned freq_response_size = 1024;

void test_DerippleResponse_ctor () {
	std::cerr << "test_DerippleResponse_ctor" << std::endl;
  dsp::DerippleResponse deripple;
}


void test_DerippleResponse_calc_freq_response () {

	std::cerr << "test_DerippleResponse_calc_freq_response" << std::endl;

	int block_size = 2*freq_response_size;
	dsp::TimeSeries freq_response_expected;

	dsp::IOManager manager;
	manager.open(file_path);
	dsp::Input* input = manager.get_input();
	dsp::Observation* info = input->get_info();
	input->set_block_size(block_size);

	while (! manager.get_input()->eod()) {
		manager.load(&freq_response_expected);
	}

	const std::vector<dsp::FIRFilter> filters = info->get_deripple();

  dsp::DerippleResponse deripple_response;

  deripple_response.set_fir_filter(filters[0]);
  deripple_response.calc_freq_response(freq_response_size);

	float* freq_response_expected_buffer = freq_response_expected.get_datptr(0, 0);

	const std::vector<float> freq_response = deripple_response.get_freq_response();

	const float thresh = 1e-5;

	for (unsigned i=0; i<freq_response.size(); i++) {
		assert(fabs(freq_response[i] - freq_response_expected_buffer[i]) < thresh);
	}

}

int main () {
	// dsp::Input::verbose = true;
	// dsp::Operation::verbose = true;
	// dsp::Shape::verbose = true;

  test_DerippleResponse_ctor();
	test_DerippleResponse_calc_freq_response();

	return 0;
}
