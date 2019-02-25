#include <string>
#include <iostream>
#include <assert.h>

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/OptimalFFT.h"
#include "dsp/Dedispersion.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"

const std::string file_path = "/home/SWIN/dshaff/mnt/ozstar/projects/PST_Matlab_pulsar_signal_processing_model_CDR/data/py_channelized.simulated_pulsar.noise_0.0.nseries_10.ndim_2.os.dump";
const unsigned block_size = 699048; // this is taken from dspsr logs
// const unsigned block_size = 400000;
const double dm = 2.64476;

class TestInverseFilterbank {

public:
	void setup ();
	void test_ctor ();
  void test_prepare ();
	void test_reserve ();
	void test_set_engine ();
	// test prepare and operate methods on real data.
	void test_pipeline ();

private:

	void load_data (dsp::TimeSeries* ts, int block_size);

	dsp::IOManager manager ;
	dsp::Input* input;
	dsp::TimeSeries* unpacked;
	dsp::Observation* info;

};

void TestInverseFilterbank::load_data (dsp::TimeSeries* ts, int block_size)
{
	std::cerr << "TestInverseFilterbank::load_data" << std::endl;
	input->set_block_size(block_size);
	while (! manager.get_input()->eod()) {
		manager.load(ts);
	}
}

void TestInverseFilterbank::setup ()
{
	std::cerr << "TestInverseFilterbank::setup: creating IOManager instance" << std::endl;
  dsp::IOManager _manager;
	manager = _manager;
	manager.open(file_path);
	std::cerr << "TestInverseFilterbank::setup: creating Input instance" << std::endl;
	input = manager.get_input();
	std::cerr << "TestInverseFilterbank::setup: creating Observation instance" << std::endl;
	info = input->get_info();
	info->set_dispersion_measure(dm);
	unpacked = new dsp::TimeSeries;
	load_data (unpacked, block_size);
}

void TestInverseFilterbank::test_ctor ()
{
	std::cerr << "TestInverseFilterbank::test_ctor" << std::endl;
  dsp::InverseFilterbank filterbank;
}

void TestInverseFilterbank::test_prepare ()
{
	std::cerr << "TestInverseFilterbank::test_prepare" << std::endl;
}

void TestInverseFilterbank::test_set_engine ()
{
	std::cerr << "TestInverseFilterbank::test_set_engine" << std::endl;
}

void TestInverseFilterbank::test_reserve ()
{
	std::cerr << "TestInverseFilterbank::test_reserve" << std::endl;
}

void TestInverseFilterbank::test_pipeline ()
{
	std::cerr << "TestInverseFilterbank::test_pipeline" << std::endl;
	dsp::TimeSeries* output = new dsp::TimeSeries;
	dsp::InverseFilterbank filterbank;
	dsp::InverseFilterbankEngineCPU* filterbank_engine = new dsp::InverseFilterbankEngineCPU;
	Reference::To<dsp::Dedispersion> kernel = new dsp::Dedispersion;
	kernel->set_dispersion_measure(dm);
	// kernel->set_optimal_fft (new dsp::OptimalFFT);
	dsp::Response* response = kernel.ptr();
	filterbank.set_output_nchan(1);
	filterbank.set_input(unpacked);
	filterbank.set_output(output);
	filterbank.set_engine(filterbank_engine);
	filterbank.set_response(response);
	filterbank.prepare();
	filterbank.operate();
}



int main () {
	dsp::Input::verbose = true;
	dsp::Operation::verbose = true;
	dsp::Shape::verbose = true;


  TestInverseFilterbank tester;
  tester.setup();
	tester.test_ctor();
	tester.test_prepare();
	tester.test_reserve();
	tester.test_set_engine();
	tester.test_pipeline();
}
