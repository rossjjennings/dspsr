//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/TimeSeries.h"
#include "dsp/Scratch.h"
#include "dsp/Apodization.h"
#include "dsp/OptimalFFT.h"

#include <iostream>
#include <assert.h>
#include <cstring>

#include "FTransform.h"

using namespace std;

dsp::InverseFilterbankEngineCPU::InverseFilterbankEngineCPU ()
{
  input_fft_length = 0;
  fft_plans_setup = false;
  response = nullptr;
}

dsp::InverseFilterbankEngineCPU::~InverseFilterbankEngineCPU ()
{
}

double dsp::InverseFilterbankEngineCPU::setup_fft_plans (
  dsp::InverseFilterbank* filterbank )
{

  const TimeSeries* input = filterbank->get_input();
  TimeSeries* output = filterbank->get_output();

  if (filterbank->has_response()) {
    response = filterbank->get_response();
  }
  real_to_complex = (input->get_state() == Signal::Nyquist);


  input_fft_length = filterbank->get_input_fft_length();
  output_fft_length = filterbank->get_output_fft_length();

  OptimalFFT* optimal = 0;
  if (response && response->has_optimal_fft()) {
    optimal = response->get_optimal_fft();
    if (optimal) {
      FTransform::set_library(optimal->get_library(input_fft_length));
    }
  }
  forward = FTransform::Agent::current->get_plan(
      input_fft_length,
      real_to_complex ? FTransform::frc: FTransform::fcc);

  if (optimal) {
    FTransform::set_library(optimal->get_library(output_fft_length));
  }
  backward = FTransform::Agent::current->get_plan(output_fft_length, FTransform::bcc);
  if (verbose) {
    cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: done setting up FFT plans" << endl;
  }

  // Compute FFT scale factors
  if (FTransform::get_norm() == FTransform::unnormalized) {
    scalefac = pow(double(output_fft_length), 2);
    scalefac *= pow(filterbank->get_oversampling_factor().doubleValue(), 2);
  }
  fft_plans_setup = true;
  if (verbose) {
    cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: scalefac=" << scalefac << endl;
  }

  return scalefac;

}

void dsp::InverseFilterbankEngineCPU::setup (
  dsp::InverseFilterbank* filterbank)
{

  if (! fft_plans_setup) {
    setup_fft_plans(filterbank);
  }

  verbose = filterbank->verbose;
  const TimeSeries* input = filterbank->get_input();
  TimeSeries* output = filterbank->get_output();

  n_per_sample = real_to_complex ? 2: 1;

  input_nchan = input->get_nchan();
  output_nchan = output->get_nchan();

  input_fft_length = filterbank->get_input_fft_length();
  output_fft_length = filterbank->get_output_fft_length();

  input_discard_pos = filterbank->get_input_discard_pos();
  input_discard_neg = filterbank->get_input_discard_neg();
  output_discard_pos = filterbank->get_output_discard_pos();
  output_discard_neg = filterbank->get_output_discard_neg();

  input_discard_total = n_per_sample*(input_discard_neg + input_discard_pos);
  input_sample_step = input_fft_length - input_discard_total;

  output_discard_total = n_per_sample*(output_discard_neg + output_discard_pos);
  output_sample_step = output_fft_length - output_discard_total;

  // compute oversampling keep/discard region
  input_os_keep = filterbank->get_oversampling_factor().normalize(input_fft_length);
  input_os_discard = input_fft_length - input_os_keep;

  // setup scratch space
  int in_npol = input->get_npol();
  int input_fft_points = in_npol*input_fft_length;
  int output_fft_points = 2*output_fft_length; // always return complex result
  int response_stitch_points = in_npol*input_fft_length*input_nchan;
  int fft_shift_points = 2*output_fft_length;
  int stitch_points = 2*output_fft_length;

  if (verbose) {
    cerr << "dsp::InverseFilterbankEngineCPU::setup"
          << " input_fft_points=" << input_fft_points
          << " output_fft_points=" << output_fft_points
          << " response_stitch_points=" << response_stitch_points
          << " fft_shift_points=" << fft_shift_points
          << " stitch_points=" << stitch_points
          << endl;
  }

  dsp::Scratch* scratch = new Scratch;
	input_fft_scratch = scratch->space<float>
		(input_fft_points + output_fft_points + response_stitch_points + fft_shift_points + stitch_points);

  output_fft_scratch = input_fft_scratch + input_fft_points;
  response_stitch_scratch = output_fft_scratch + output_fft_points;
  fft_shift_scratch = response_stitch_scratch + response_stitch_points;
  stitch_scratch = fft_shift_scratch + fft_shift_points;
}


void dsp::InverseFilterbankEngineCPU::set_scratch (float *)
{
}

void dsp::InverseFilterbankEngineCPU::perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
              uint64_t npart, uint64_t in_step, uint64_t out_step)
{
  static const int floats_per_complex = 2;
	static const int sizeof_complex = sizeof(float) * floats_per_complex;
	const unsigned n_dims = in->get_ndim();
	const unsigned n_pol = in->get_npol();

	unsigned input_os_keep_2 = input_os_keep / 2;
  unsigned input_os_discard_2 = input_os_discard / 2;

	unsigned response_offset;
	unsigned response_offset_pos;
	unsigned response_offset_neg;

	unsigned stitched_offset;
	unsigned stitched_offset_pos;
	unsigned stitched_offset_neg;

	int circ_shift_size = 1;

  float* freq_dom_ptr;
  float* time_dom_ptr;

	cerr << "dsp::InverseFilterbankEngineCPU::perform: writing " << npart << " chunks" << endl;

	for(unsigned output_ichan = 0; output_ichan < output_nchan; output_ichan++) {
		for(uint64_t ipart = 0; ipart < npart; ipart++) {
			for(unsigned ipol = 0; ipol < n_pol; ipol++) {
				for(unsigned input_ichan = 0; input_ichan < input_nchan; input_ichan++) {
					// cerr << "dsp::InverseFilterbankEngineCPU::perform: forward FFT, pol: " << ipol << ", input channel: " << input_ichan << ", loop: " << ipart << "/" << npart << endl;
					// cerr << "dsp::InverseFilterbankEngineCPU::perform: ipart=" << ipart << " input_ichan=" << input_ichan << " getting freq_dom_ptr" << endl;
					freq_dom_ptr = input_fft_scratch;
					// cerr << "dsp::InverseFilterbankEngineCPU::perform: ipart=" << ipart << " input_ichan=" << input_ichan << " freq_dom_ptr=" << freq_dom_ptr << endl;
					// cerr << "dsp::InverseFilterbankEngineCPU::perform: ipart=" << ipart << " input_ichan=" << input_ichan << " getting time_dom_ptr" << endl;
					time_dom_ptr = const_cast<float*>(in->get_datptr(input_ichan, ipol));
					// cerr << "dsp::InverseFilterbankEngineCPU::perform: ipart=" << ipart << " input_ichan=" << input_ichan << " time_dom_ptr=" << time_dom_ptr << endl;
					time_dom_ptr += n_dims*ipart*(input_fft_length - input_discard_total);
					// cerr << "dsp::InverseFilterbankEngineCPU::perform: ipart=" << ipart << " input_ichan=" << input_ichan << " time_dom_ptr (after increment)=" << time_dom_ptr << endl;
					// time_dom_ptr += n_dims*ipart*(input_fft_length - _input_discard.neg);
					// time_dom_ptr += n_dims*ipart*(input_fft_length);
					// perform forward FFT to convert time domain data to the frequency domain
					if (real_to_complex) {
						forward->frc1d(input_fft_length, freq_dom_ptr, time_dom_ptr);
					} else {
						// fcc1d(number_of_points, destinationPtr, sourcePtr);
						forward->fcc1d(input_fft_length, freq_dom_ptr, time_dom_ptr);
					}
					response_offset = n_dims*input_ichan*input_fft_length;

					memcpy(
						response_stitch_scratch + response_offset,
						freq_dom_ptr,
						input_fft_length*sizeof_complex
					);
				} // end of for input_nchan

        // assemble spectrum
				for (uint64_t input_ichan = 0; input_ichan < input_nchan; input_ichan++) {
					stitched_offset_neg = n_dims*input_os_keep*input_ichan;
					stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;

					response_offset_neg = n_dims*(input_fft_length*(input_ichan + 1) - input_os_keep_2);
					response_offset_pos = n_dims*input_fft_length*input_ichan;

					// response_offset_neg = n_dims*(input_fft_length*input_ichan + input_os_discard);
					// response_offset_pos = n_dims*(n_dims*input_fft_length*input_ichan + input_fft_length/2);

					memcpy(
						stitch_scratch + stitched_offset_neg,
						response_stitch_scratch + response_offset_neg,
						input_os_keep_2 * sizeof_complex
					);

					memcpy(
						stitch_scratch + stitched_offset_pos,
						response_stitch_scratch + response_offset_pos,
						input_os_keep_2 * sizeof_complex
					);
				}

				// for (int i=0; i<2*input_fft_length*input_nchan; i++) {
				// 	*(response_stitch_scratch + i) = 0.0;
				// }

				// do circular shift
				// first copy all of stitch_scratch to fft_shift_scratch
				memcpy(
					fft_shift_scratch,
					stitch_scratch,
					sizeof_complex * (input_os_keep * input_nchan)
				);
				// copy offset chunk to start of stitch_scratch
				memcpy(
					stitch_scratch,
					fft_shift_scratch + (circ_shift_size*n_dims*input_os_keep_2),
					sizeof_complex*((input_nchan*2) - circ_shift_size)*input_os_keep_2
				);
				// copy bit we stored in fft_shift_scratch to back of stitch_scratch
				memcpy(
					stitch_scratch + ((input_nchan*2) - circ_shift_size)*n_dims*input_os_keep_2,
					fft_shift_scratch,
					sizeof_complex * (input_os_keep_2 * circ_shift_size)
				);

				if (response != nullptr) {
					response->operate(stitch_scratch, ipol, 0, 1);
				}


				if (out != nullptr) {

          if (verbose)
  					cerr << "dsp::InverseFilterbankEngineCPU::perform: pol=" << ipol <<" loop=" << ipart+1 << "/" << npart << " doing inverse FFT" << endl; //. output_fft_length: "<< output_fft_length << endl;

					backward->bcc1d(output_fft_length, output_fft_scratch, stitch_scratch);

          if (verbose)
  					cerr << "dsp::InverseFilterbankEngineCPU::perform: backward FFT complete." << endl;

					// Output is in FPT order.
					void* sourcePtr = (void*)(output_fft_scratch + output_discard_pos*floats_per_complex);
					void* destinationPtr = (void *)(out->get_datptr(0, ipol) + ipart*output_sample_step*floats_per_complex);
					memcpy(destinationPtr, sourcePtr, output_sample_step*sizeof_complex);
				} // end of if(out!=nullptr)
			} // end of n_pol
		} // end of ipart
	} // end of output_ichan
  if (verbose) {
    cerr << "dsp::InverseFilterbankEngineCPU::perform: finish" << endl;
  }

}

void dsp::InverseFilterbankEngineCPU::finish ()
{
  if (verbose) {
    cerr << "dsp::InverseFilterbankEngineCPU::finish" << endl;
  }
}
