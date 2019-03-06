#include <string>
#include <iostream>
#include <assert.h>

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/ResponseProduct.h"
#include "dsp/Dedispersion.h"
#include "dsp/FIRFilter.h"
#include "dsp/DerippleResponse.h"
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
	filterbank.set_buffering_policy(NULL);
	filterbank.set_output_nchan(1);
	filterbank.set_input(unpacked);
	filterbank.set_output(output);

	dsp::InverseFilterbankEngineCPU* filterbank_engine = new dsp::InverseFilterbankEngineCPU;
	Reference::To<dsp::Dedispersion> kernel = new dsp::Dedispersion;
	kernel->set_dispersion_measure(dm);
	Reference::To<dsp::DerippleResponse> deripple = new dsp::DerippleResponse;
	deripple->set_fir_filter(info->get_deripple()[0]);

	// kernel->match(filterbank.get_input(), filterbank.get_output_nchan());
	//
	// int output_discard_pos = kernel->get_impulse_pos();
	// int output_discard_neg = kernel->get_impulse_neg();
	// int input_discard_neg = 0;
	// int input_discard_pos = 0;
	//
	// int input_fft_length = 0;
	// int output_fft_length = kernel->get_ndat();


// if (verbose) {
// 	cerr << "dsp::InverseFilterbank::make_preparations:"
// 			<< " output_discard_neg(before)=" << output_discard_neg
// 			<< " output_discard_pos(before)=" << output_discard_pos
// 			<< " output_fft_length(before)=" << output_fft_length
// 			<< endl;
// }

	// filterbank.optimize_discard_region(
	// 	&input_discard_neg, &input_discard_pos,
	// 	&output_discard_neg, &output_discard_pos
	// );
	// filterbank.optimize_fft_length(
	// 	&input_fft_length, &output_fft_length);

// if (verbose) {
// 	cerr << "dsp::InverseFilterbank::make_preparations:"
// 			<< " input_discard_neg(after)=" << input_discard_neg
// 			<< " input_discard_pos(after)=" << input_discard_pos
// 			<< " input_fft_length(after)=" << input_fft_length
// 			<< " output_discard_neg(after)=" << output_discard_neg
// 			<< " output_discard_pos(after)=" << output_discard_pos
// 			<< " output_fft_length(after)=" << output_fft_length
// 			<< endl;
// }
	// unsigned input_nchan = info->get_nchan();
	// filterbank.set_input_discard_neg(input_discard_neg);
	// filterbank.set_input_discard_pos(input_discard_pos);
	// filterbank.set_output_discard_neg(output_discard_neg);
	// filterbank.set_output_discard_pos(output_discard_pos);
	// filterbank.set_input_fft_length(input_fft_length);
	// filterbank.set_output_fft_length(output_fft_length);
	//
	// kernel->set_impulse_pos(output_discard_pos);
	// kernel->set_impulse_neg(output_discard_neg);
	// kernel->set_frequency_resolution(output_fft_length);
	// kernel->build();
	//
	// deripple->set_nchan(input_nchan);
	// deripple->set_ndat(kernel->get_ndat() / input_nchan);
	// deripple->build();
	//
	// dsp::Response* response = kernel.ptr();

	// dsp::ResponseProduct* response_product = new dsp::ResponseProduct;
	// response_product->add_response(kernel);
	// response_product->add_response(deripple);
	// dsp::Response* response = response_product;
	// response->match(filterbank.get_input(), filterbank.get_output_nchan());
	std::cerr << "TestInverseFilterbank::test_pipeline: kernel->ndim=" << kernel->get_ndim() << std::endl;
	std::cerr << "TestInverseFilterbank::test_pipeline: kernel->nchan=" << kernel->get_nchan() << std::endl;
	std::cerr << "TestInverseFilterbank::test_pipeline: kernel->ndat=" << kernel->get_ndat() << std::endl;

	filterbank.set_engine(filterbank_engine);
	filterbank.set_response(kernel);
	filterbank.set_deripple(deripple);
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
