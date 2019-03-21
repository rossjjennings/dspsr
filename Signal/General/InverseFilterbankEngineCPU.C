//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "FTransform.h"

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/TimeSeries.h"
#include "dsp/Scratch.h"
#include "dsp/Apodization.h"
#include "dsp/OptimalFFT.h"

#include <fstream>
#include <iostream>
#include <assert.h>
#include <cstring>


dsp::InverseFilterbankEngineCPU::InverseFilterbankEngineCPU ()
{
  input_fft_length = 0;
  fft_plans_setup = false;
  response = nullptr;

  pfb_dc_chan = 0;
  pfb_all_chan = 0;
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
    if (verbose) {
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: setting response" << std::endl;
    }
    response = filterbank->get_response();
  }

  // if (filterbank->has_deripple()) {
  //   if (verbose) {
  //     std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: setting deripple" << std::endl;
  //   }
  //   deripple = filterbank->get_deripple();
  //   if (verbose) {
  //     std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: deripple nchan=" << deripple->get_nchan() << std::endl;
  //     std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: deripple ndat=" << deripple->get_ndat() << std::endl;
  //     std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: deripple ndim=" << deripple->get_ndim() << std::endl;
  //     std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: deripple npol=" << deripple->get_npol() << std::endl;
  //   }
  // }

  real_to_complex = (input->get_state() == Signal::Nyquist);


  input_fft_length = filterbank->get_input_fft_length();
  output_fft_length = filterbank->get_output_fft_length();

  // if (verbose) {
  //   std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: input_fft_length="
  //             << input_fft_length << ", output_fft_length=" << output_fft_length << std::endl;
  // }

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
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: done setting up FFT plans" << std::endl;
  }

  // Compute FFT scale factors
  if (FTransform::get_norm() == FTransform::unnormalized) {
    scalefac = pow(double(output_fft_length), 2);
    scalefac *= pow(filterbank->get_oversampling_factor().doubleValue(), 2);
  }
  fft_plans_setup = true;
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: scalefac=" << scalefac << std::endl;
  }

  return scalefac;

}

void dsp::InverseFilterbankEngineCPU::setup (
  dsp::InverseFilterbank* filterbank)
{
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup" << std::endl;
  }
  if (! fft_plans_setup) {
    setup_fft_plans(filterbank);
  }

  verbose = filterbank->verbose;
  const TimeSeries* input = filterbank->get_input();
  TimeSeries* output = filterbank->get_output();

  n_per_sample = real_to_complex ? 2: 1;

  pfb_dc_chan = filterbank->get_pfb_dc_chan();
  pfb_all_chan = filterbank->get_pfb_all_chan();

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

  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup"
      << " input_nchan=" << input_nchan
      << " output_nchan=" << output_nchan
      << " input_fft_length=" << input_fft_length
      << " output_fft_length=" << output_fft_length
      << " input_discard_pos=" << input_discard_pos
      << " input_discard_neg=" << input_discard_neg
      << " output_discard_pos=" << output_discard_pos
      << " output_discard_neg=" << output_discard_neg
      << std::endl;
  }

  // compute oversampling keep/discard region
  input_os_keep = filterbank->get_oversampling_factor().normalize(input_fft_length);
  input_os_discard = input_fft_length - input_os_keep;

  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup"
      << " input_os_keep=" << input_os_keep
      << " input_os_discard=" << input_os_discard
      << std::endl;
  }

  // setup scratch space
  unsigned in_ndim = input->get_ndim();
  unsigned input_fft_points = in_ndim*input_fft_length;
  unsigned output_fft_points = 2*output_fft_length; // always return complex result
  unsigned stitch_points = 2*output_fft_length;

  dsp::Scratch* scratch = new Scratch;
  input_fft_scratch = scratch->space<float>
    (input_fft_points + output_fft_points  + stitch_points);

  output_fft_scratch = input_fft_scratch + input_fft_points;
  stitch_scratch = output_fft_scratch + output_fft_points;

  // // setup scratch space
  // int in_ndim = input->get_ndim();
  // int input_fft_points = in_ndim*input_fft_length;
  // int output_fft_points = 2*output_fft_length; // always return complex result
  // int response_stitch_points = in_ndim*input_fft_length*input_nchan;
  // int fft_shift_points = 2*output_fft_length;
  // int stitch_points = 2*output_fft_length;
  //
  // if (verbose) {
  //   std::cerr << "dsp::InverseFilterbankEngineCPU::setup"
  //         << " oversampling factor=" << filterbank->get_oversampling_factor()
  //         << std::endl;
  //   std::cerr << "dsp::InverseFilterbankEngineCPU::setup"
  //         << " input_fft_points=" << input_fft_points
  //         << " output_fft_points=" << output_fft_points
  //         << " response_stitch_points=" << response_stitch_points
  //         << " fft_shift_points=" << fft_shift_points
  //         << " stitch_points=" << stitch_points
  //         << std::endl;
  //   std::cerr << "dsp::InverseFilterbankEngineCPU::setup"
  //         << " input_os_keep=" << input_os_keep
  //         << " input_os_discard=" << input_os_discard
  //         << " input_discard_total=" << input_discard_total
  //         << " input_sample_step=" << input_sample_step
  //         << " output_discard_total=" << output_discard_total
  //         << " output_sample_step=" << output_sample_step
  //         << std::endl;
  // }
  //
  // dsp::Scratch* scratch = new Scratch;
	// input_fft_scratch = scratch->space<float>
	// 	(input_fft_points + output_fft_points + response_stitch_points + fft_shift_points + stitch_points);
  //
  // output_fft_scratch = input_fft_scratch + input_fft_points;
  // response_stitch_scratch = output_fft_scratch + output_fft_points;
  // fft_shift_scratch = response_stitch_scratch + response_stitch_points;
  // stitch_scratch = fft_shift_scratch + fft_shift_points;
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
  // unsigned input_os_discard_2 = input_os_discard / 2;

	// unsigned response_offset;
	// unsigned response_offset_pos;
	// unsigned response_offset_neg;
  //
	// unsigned stitched_offset;
	unsigned stitched_offset_pos;
	unsigned stitched_offset_neg;

  float* freq_dom_ptr;
  float* time_dom_ptr;



  if (verbose) {
    if (pfb_dc_chan) {
      if (pfb_all_chan) {
        std::cerr << "dsp::InverseFilterbankEngineCPU::perform: "
          << "moving first half channel to end of assembled spectrum" << std::endl;
      } else {
        std::cerr << "dsp::InverseFilterbankEngineCPU::perform: "
          << "moving first half channel to end of assembled spectrum and zeroing" << std::endl;
      }
    } else {
      std::cerr << "dsp::InverseFilterbankEngineCPU::perform: "
        << "leaving assembled spectrum as is" << std::endl;
    }
    std::cerr << "dsp::InverseFilterbankEngineCPU::perform: writing " << npart << " chunks" << std::endl;
  }

  // std::ofstream deripple_before_file("deripple.before.dat", std::ios::out | std::ios::binary);
  // std::ofstream deripple_after_file("deripple.after.dat", std::ios::out | std::ios::binary);

	for(unsigned output_ichan = 0; output_ichan < output_nchan; output_ichan++) {
		for(uint64_t ipart = 0; ipart < npart; ipart++) {
			for(unsigned ipol = 0; ipol < n_pol; ipol++) {
				for(unsigned input_ichan = 0; input_ichan < input_nchan; input_ichan++) {
					freq_dom_ptr = input_fft_scratch;
					time_dom_ptr = const_cast<float*>(in->get_datptr(input_ichan, ipol));
					time_dom_ptr += n_dims*ipart*(input_fft_length - input_discard_total);
					if (real_to_complex) {
						forward->frc1d(input_fft_length, freq_dom_ptr, time_dom_ptr);
					} else {
						// fcc1d(number_of_points, destinationPtr, sourcePtr);
						forward->fcc1d(input_fft_length, freq_dom_ptr, time_dom_ptr);
					}
          // discard oversampled regions and do circular shift

          if (pfb_dc_chan) {
            if (input_ichan == 0) {
              stitched_offset_neg = n_dims*(input_fft_length - input_os_keep_2);
              stitched_offset_pos = 0;
            } else {
              stitched_offset_neg = n_dims*(input_os_keep*input_ichan - input_os_keep_2);
              stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;
            }
          } else {
            stitched_offset_neg = n_dims*input_os_keep*input_ichan;
            stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;
          }

          // discard oversampled regions
          // stitched_offset_neg = n_dims*(input_os_keep*input_ichan);
          // stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;

          // response_offset_neg = n_dims*(input_fft_length*(input_ichan + 1) - input_os_keep_2);
          // response_offset_pos = n_dims*input_fft_length*input_ichan;

          // response_offset_neg = n_dims*(input_fft_length*input_ichan + input_os_discard);
          // response_offset_pos = n_dims*(n_dims*input_fft_length*input_ichan + input_fft_length/2);

          std::memcpy(
            stitch_scratch + stitched_offset_neg,
            freq_dom_ptr + n_dims*(input_fft_length - input_os_keep_2),
            input_os_keep_2 * sizeof_complex
          );

          std::memcpy(
            stitch_scratch + stitched_offset_pos,
            freq_dom_ptr,
            input_os_keep_2 * sizeof_complex
          );

          // response_offset = n_dims*input_ichan*input_fft_length;
        	// std::memcpy(
					// 	response_stitch_scratch + response_offset,
					// 	freq_dom_ptr,
					// 	input_fft_length*sizeof_complex
					// );
				} // end of for input_nchan

        // If we have the zeroth PFB channel and we don't have all the PFB channels,
        // then we zero the last half channel.
        if (! pfb_all_chan && pfb_dc_chan) {
          int offset = n_dims*(output_fft_length - input_os_keep_2);
          for (int i=0; i<n_dims*input_os_keep_2; i++) {
            stitch_scratch[offset + i] = 0.0;
          }
        }

        // assemble spectrum
				// for (uint64_t input_ichan = 0; input_ichan < input_nchan; input_ichan++) {
				// 	stitched_offset_neg = n_dims*input_os_keep*input_ichan;
				// 	stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;
        //
				// 	response_offset_neg = n_dims*(input_fft_length*(input_ichan + 1) - input_os_keep_2);
				// 	response_offset_pos = n_dims*input_fft_length*input_ichan;
        //
				// 	// response_offset_neg = n_dims*(input_fft_length*input_ichan + input_os_discard);
				// 	// response_offset_pos = n_dims*(n_dims*input_fft_length*input_ichan + input_fft_length/2);
        //
				// 	std::memcpy(
				// 		stitch_scratch + stitched_offset_neg,
				// 		response_stitch_scratch + response_offset_neg,
				// 		input_os_keep_2 * sizeof_complex
				// 	);
        //
				// 	std::memcpy(
				// 		stitch_scratch + stitched_offset_pos,
				// 		response_stitch_scratch + response_offset_pos,
				// 		input_os_keep_2 * sizeof_complex
				// 	);
				// }

        // deripple_before_file.write(
        //   reinterpret_cast<const char*>(stitch_scratch),
        //   output_fft_length*sizeof_complex
        // );
        //! Deripple correction is setup to be done before circular shift
        // if (deripple != nullptr) {
        //   deripple->operate(stitch_scratch, 0, 0, input_nchan);
        // }

        // deripple_after_file.write(
        //   reinterpret_cast<const char*>(stitch_scratch),
        //   output_fft_length*sizeof_complex
        // );


				// do circular shift
				// first copy all of stitch_scratch to fft_shift_scratch
				// std::memcpy(
				// 	fft_shift_scratch,
				// 	stitch_scratch,
				// 	sizeof_complex * (input_os_keep * input_nchan)
				// );
				// // copy offset chunk to start of stitch_scratch
				// std::memcpy(
				// 	stitch_scratch,
				// 	fft_shift_scratch + (circ_shift_size*n_dims*input_os_keep_2),
				// 	sizeof_complex*((input_nchan*2) - circ_shift_size)*input_os_keep_2
				// );
				// // copy bit we stored in fft_shift_scratch to back of stitch_scratch
				// std::memcpy(
				// 	stitch_scratch + ((input_nchan*2) - circ_shift_size)*n_dims*input_os_keep_2,
				// 	fft_shift_scratch,
				// 	sizeof_complex * (input_os_keep_2 * circ_shift_size)
				// );

        // if (deripple != nullptr) {
        //   deripple->operate(stitch_scratch, ipol, 0, 1);
        // }

				if (response != nullptr) {
          if (verbose) {
            std::cerr << "dsp::InverseFilterbankEngineCPU::perform: applying response" << std::endl;
          }
          response->operate(stitch_scratch, ipol, 0, 1);
				} else {
          if (verbose) {
            std::cerr << "dsp::InverseFilterbankEngineCPU::perform: NOT applying response" << std::endl;
          }
        }

				if (out != nullptr) {

          if (verbose) {
  					std::cerr << "dsp::InverseFilterbankEngineCPU::perform: pol=" << ipol <<" loop=" << ipart+1 << "/" << npart << " doing inverse FFT" << std::endl; //. output_fft_length: "<< output_fft_length << std::endl;
          }
					backward->bcc1d(output_fft_length, output_fft_scratch, stitch_scratch);

          if (verbose)
  					std::cerr << "dsp::InverseFilterbankEngineCPU::perform: backward FFT complete." << std::endl;

					// Output is in FPT order.
					void* sourcePtr = (void*)(output_fft_scratch + output_discard_pos*floats_per_complex);
					void* destinationPtr = (void *)(out->get_datptr(0, ipol) + ipart*output_sample_step*floats_per_complex);
					std::memcpy(destinationPtr, sourcePtr, output_sample_step*sizeof_complex);
				} // end of if(out!=nullptr)
			} // end of n_pol
		} // end of ipart
	} // end of output_ichan
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::perform: finish" << std::endl;
  }
  // deripple_before_file.close();
  // deripple_after_file.close();
}

void dsp::InverseFilterbankEngineCPU::finish ()
{
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::finish" << std::endl;
  }
}
