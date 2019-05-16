#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <fstream>

#include "Rational.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/FIRFilter.h"
#include "dsp/InverseFilterbankResponse.h"

#include "util.hpp"

std::string file_path = "/home/SWIN/dshaff/ska/test_data/fir.768.dada";

const unsigned ntaps_expected = 81;
const unsigned input_fft_length = 128;
const Rational os_factor = Rational(4, 3);
const unsigned nchan = 8;
const unsigned freq_response_size = nchan * os_factor.normalize(input_fft_length);

class TestInverseFilterbankResponse {

public:
	void setup ();
	void test_ctor ();
	void test_build ();
	void test_calc_freq_response ();
	void test_set_ndat ();
	void test_set_nchan ();

private:

	void load_data (dsp::TimeSeries* ts, int block_size);

	dsp::IOManager manager ;
	dsp::Input* input;
	dsp::Observation* info;

};

void TestInverseFilterbankResponse::load_data (dsp::TimeSeries* ts, int block_size)
{
	// std::cerr << "TestInverseFilterbankResponse::load_data" << std::endl;
	input->set_block_size(block_size);
	while (! manager.get_input()->eod()) {
		manager.load(ts);
	}
}

void TestInverseFilterbankResponse::setup ()
{
	// std::cerr << "TestInverseFilterbankResponse::setup" << std::endl;
	dsp::IOManager _manager;
	_manager.open(file_path);
	dsp::Input* _input = _manager.get_input();
	dsp::Observation* _info = _input->get_info();

	manager = _manager;
	input = _input;
	info = _info;
}

void TestInverseFilterbankResponse::test_ctor ()
{
	std::cerr << "TestInverseFilterbankResponse::test_ctor " << std::endl;
  dsp::InverseFilterbankResponse deripple_response;
}

void TestInverseFilterbankResponse::test_set_ndat ()
{
	std::cerr << "TestInverseFilterbankResponse::test_set_ndat " << std::endl;
	dsp::InverseFilterbankResponse deripple_response;
	unsigned new_ndat = 128;
	deripple_response.set_ndat(new_ndat);
	assert(deripple_response.get_ndat() == new_ndat);
}

void TestInverseFilterbankResponse::test_set_nchan ()
{
	std::cerr << "TestInverseFilterbankResponse::test_set_nchan " << std::endl;
	dsp::InverseFilterbankResponse deripple_response;
	unsigned new_nchan = 3;
	deripple_response.set_nchan(new_nchan);
	assert(deripple_response.get_nchan() == new_nchan);
}

void TestInverseFilterbankResponse::test_build () {

	std::cerr << "TestInverseFilterbankResponse::test_build " << std::endl;

	int block_size = 2*freq_response_size;
	dsp::TimeSeries* freq_response_expected = new dsp::TimeSeries;
	load_data(freq_response_expected, block_size);

	const std::vector<dsp::FIRFilter> filters = info->get_deripple();

  dsp::InverseFilterbankResponse deripple_response;

  deripple_response.set_fir_filter(filters[0]);
	deripple_response.set_pfb_dc_chan(true);
	deripple_response.set_apply_deripple(true);
	deripple_response.set_ndat(input_fft_length);
	deripple_response.set_input_nchan(nchan);
	deripple_response.set_oversampling_factor(os_factor);
	deripple_response.build();

	float* freq_response_expected_buffer = freq_response_expected->get_datptr(0, 0);
	float* freq_response_test_buffer = deripple_response.get_datptr(0, 0);

	const float thresh = 1e-5;
	float expected_val;
	float test_val;
	bool isclose;

	for (unsigned i=0; i<block_size; i++) {
		test_val = freq_response_test_buffer[i];
		expected_val = freq_response_expected_buffer[i];
		if (expected_val != 0) {
			expected_val = 1.0 / expected_val;
		}
		isclose = util::isclose<float>(test_val, expected_val, 1e-7, 1e-5);
		assert(isclose);
		// if (! util::isclose<float>(test_val, expected_val, 1e-7, 1e-5)) {
		// 	std::cout << "(" << freq_response_test_buffer[i] << ", "
		// 		<< expected_val << ")" << std::endl;
		// }
	}
	// std::ofstream freq_response_file_after("freq_response.buffer.dat", std::ios::out | std::ios::binary);
  // freq_response_file_after.write(
  //     reinterpret_cast<const char*>(freq_response_test_buffer),
  //     block_size*sizeof(float)
  // );
  // freq_response_file_after.close();
	//
	// std::ofstream freq_response_file("freq_response_expected.buffer.dat", std::ios::out | std::ios::binary);
  // freq_response_file.write(
  //     reinterpret_cast<const char*>(freq_response_expected_buffer),
  //     block_size*sizeof(float)
  // );
  // freq_response_file.close();


}

void TestInverseFilterbankResponse::test_calc_freq_response ()
{ }

int main () {
	// dsp::Input::verbose = true;
	// dsp::Operation::verbose = true;
	// dsp::Shape::verbose = true;

	TestInverseFilterbankResponse tester;

	tester.setup();
	tester.test_ctor();
	tester.test_set_ndat();
	tester.test_set_nchan();
	tester.test_build();

	return 0;
}
