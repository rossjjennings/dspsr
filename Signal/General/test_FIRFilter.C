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

std::string file_path = "/home/SWIN/dshaff/ska/pfb-channelizer/fir.1024.dada";

const unsigned ntaps_expected = 321;
const unsigned freq_response_size = 1024;

void test_FIRFilter_ctor () {
	std::cerr << "test_FIRFilter_ctor" << std::endl;
  dsp::FIRFilter filter;
	std::vector<float> coeff = {2.0, 1.0};
	dsp::FIRFilter filter1(coeff, Rational(1, 1), 8);
}

void test_FIRFilter_ntaps () {
	std::cerr << "test_FIRFilter_ntaps" << std::endl;
	dsp::FIRFilter filter;
	filter.set_ntaps(400);
	assert(filter.get_ntaps() == 400);
	std::vector<float> coeff = {2.0, 1.0};
	dsp::FIRFilter filter0(coeff, Rational(1, 1), 8);
	assert(filter0.get_ntaps() == 2);
}

/**
 * Test we can set filter coefficient elements
 * @method test_FIRFilter_set_coeff_elem
 */
void test_FIRFilter_set_coeff_elem () {
	std::cerr << "test_FIRFilter_set_coeff_elem" << std::endl;
	std::vector<float> coeff = {2.0, 1.0};
	dsp::FIRFilter filter(coeff, Rational(1, 1), 8);
	std::vector<float>* coeff_ref = filter.get_coeff();
	coeff_ref->at(0) = 1.0;

	for (int i=0; i<filter.get_ntaps(); i++) {
		assert(filter[i] == 1.0);
	}

	filter[0] = 2.0;
	filter[1] = 2.0;

	for (int i=0; i<filter.get_ntaps(); i++) {
		assert(filter[i] == 2.0);
	}
}

void test_FIRFilter_load_from_observation () {

	std::cerr << "test_FIRFilter_load_from_observation" << std::endl;

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

	const std::vector<dsp::FIRFilter> deripple = info->get_deripple();

	assert(deripple[0].get_ntaps() == ntaps_expected);
	// std::cerr << "test_FIRFilter_freq_response: filter taps: " << fir_filter[0].ntaps << std::endl;

	// dsp::FIRFilter filter;
	// filter.set_ntaps(fir_filter[0].ntaps);
	// float* filter_buffer = filter.get_coeff()->data();

	// for (int i=0; i<fir_filter[0].coeff.size(); i++) {
	// 	filter_buffer[i] = fir_filter[0].coeff[i];
	// }

	// float* freq_response_expected_buffer = freq_response_expected.get_datptr(0, 0);
	//
	// filter.calc_freq_response(1024);
	//
	// std::vector<float> freq_response = filter.get_freq_response();
	//
	// const float thresh = 1e-5;
	//
	// for (unsigned i=0; i<freq_response.size(); i++) {
	// 	assert(fabs(freq_response[i] - freq_response_expected_buffer[i]) < thresh);
	// }

}

int main () {
	// dsp::Input::verbose = true;
	// dsp::Operation::verbose = true;
	// dsp::Shape::verbose = true;



  test_FIRFilter_ctor();
	test_FIRFilter_ntaps();
	test_FIRFilter_set_coeff_elem();
	test_FIRFilter_load_from_observation();
	// test_FIRFilter_calc_freq_response(info, freq_response_expected);

	return 0;
}
