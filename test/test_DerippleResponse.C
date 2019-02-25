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

class TestDerippleResponse {

public:
	void setup ();
	void test_ctor ();
	void test_calc_freq_response ();
	void test_set_ndat();
	void test_set_nchan();

private:

	void load_data (dsp::TimeSeries* ts, int block_size);

	dsp::IOManager manager ;
	dsp::Input* input;
	dsp::Observation* info;

};

void TestDerippleResponse::load_data (dsp::TimeSeries* ts, int block_size)
{
	// std::cerr << "TestDerippleResponse::load_data" << std::endl;
	input->set_block_size(block_size);
	while (! manager.get_input()->eod()) {
		manager.load(ts);
	}
}

void TestDerippleResponse::setup ()
{
	// std::cerr << "TestDerippleResponse::setup" << std::endl;
	dsp::IOManager _manager;
	_manager.open(file_path);
	dsp::Input* _input = _manager.get_input();
	dsp::Observation* _info = _input->get_info();

	manager = _manager;
	input = _input;
	info = _info;
}

void TestDerippleResponse::test_ctor ()
{
	std::cerr << "TestDerippleResponse::test_ctor " << std::endl;
  dsp::DerippleResponse deripple_response;
}

void TestDerippleResponse::test_set_ndat ()
{
	std::cerr << "TestDerippleResponse::test_set_ndat " << std::endl;
	dsp::DerippleResponse deripple_response;
	unsigned new_ndat = 128;
	deripple_response.set_ndat(new_ndat);
	assert(deripple_response.get_ndat() == new_ndat);
}

void TestDerippleResponse::test_set_nchan ()
{
	std::cerr << "TestDerippleResponse::test_set_nchan " << std::endl;
	dsp::DerippleResponse deripple_response;
	unsigned new_nchan = 3;
	deripple_response.set_nchan(new_nchan);
	assert(deripple_response.get_nchan() == new_nchan);
}

void TestDerippleResponse::test_calc_freq_response () {

	std::cerr << "TestDerippleResponse::test_calc_freq_response " << std::endl;

	int block_size = 2*freq_response_size;
	dsp::TimeSeries* freq_response_expected = new dsp::TimeSeries;
	load_data(freq_response_expected, block_size);

	const std::vector<dsp::FIRFilter> filters = info->get_deripple();

	std::vector<float> freq_response;
  dsp::DerippleResponse deripple_response;

  deripple_response.set_fir_filter(filters[0]);
  deripple_response.calc_freq_response(freq_response, freq_response_size);

	float* freq_response_expected_buffer = freq_response_expected->get_datptr(0, 0);

	const float thresh = 1e-5;

	for (unsigned i=0; i<freq_response.size(); i++) {
		assert(fabs(freq_response[i] - freq_response_expected_buffer[i]) < thresh);
	}

}

int main () {
	// dsp::Input::verbose = true;
	// dsp::Operation::verbose = true;
	// dsp::Shape::verbose = true;

	TestDerippleResponse tester;

	tester.setup();
	tester.test_ctor();
	tester.test_set_ndat();
	tester.test_set_nchan();
	tester.test_calc_freq_response();

	return 0;
}
