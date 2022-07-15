//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten, Andrew Jameson and Dean Shaff
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "FTransform.h"

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/TimeSeries.h"
#include "dsp/Scratch.h"
#include "dsp/Apodization.h"
#include "dsp/OptimalFFT.h"
#include "dsp/Observation.h"

#include <fstream>
#include <iostream>
#include <assert.h>
#include <cstring>
#include <complex>

using namespace std;

dsp::InverseFilterbankEngineCPU::InverseFilterbankEngineCPU ()
{
  input_fft_length = 0;
  fft_plans_setup = false;
  response = nullptr;
  zero_DM_response = nullptr;

  temporal_apodization = nullptr;
  spectral_apodization = nullptr;

  pfb_dc_chan = 0;
  pfb_all_chan = 0;
  verbose = Observation::verbose;
  report = false;
}

dsp::InverseFilterbankEngineCPU::~InverseFilterbankEngineCPU ()
{
}

void dsp::InverseFilterbankEngineCPU::setup (dsp::InverseFilterbank* filterbank)
{
  if (verbose) {
    cerr << "dsp::InverseFilterbankEngineCPU::setup" << endl;
  }

  verbose = filterbank->verbose;
  const TimeSeries* input = filterbank->get_input();
  TimeSeries* output = filterbank->get_output();

  real_to_complex = (input->get_state() == Signal::Nyquist);

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

  input_discard_total = input_discard_neg + input_discard_pos;
  input_sample_step = input_fft_length - input_discard_total;

  output_discard_total = output_discard_neg + output_discard_pos;
  output_sample_step = output_fft_length - output_discard_total;

  if (filterbank->has_response())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup setting response" << endl;
 
    response = filterbank->get_response();
  }

  if (filterbank->has_temporal_apodization())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal apodization" << endl;

    temporal_apodization = filterbank->get_temporal_apodization();
    if (verbose)
    {
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal_apodization.get_type() "
	   << temporal_apodization->get_type() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal_apodization.get_ndim() "
	   << temporal_apodization->get_ndim() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal_apodization.get_ndat() "
	   << temporal_apodization->get_ndat() << endl;
    }
  }

  if (filterbank->has_spectral_apodization())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral apodization" << endl;

    spectral_apodization = filterbank->get_spectral_apodization();
    if (verbose)
    {
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral_apodization.get_type() "
           << spectral_apodization->get_type() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral_apodization.get_ndim() "
           << spectral_apodization->get_ndim() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral_apodization.get_ndat() "
           << spectral_apodization->get_ndat() << endl;
    }
  }

  if (filterbank->get_zero_DM())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup setting zero_DM_response" << endl;

    zero_DM_response = filterbank->get_zero_DM_response();
  }

  OptimalFFT* optimal = 0;
  if (response && response->has_optimal_fft())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup getting OptimalFFT object" << endl;

    optimal = response->get_optimal_fft();
  }

  if (optimal)
    FTransform::set_library( optimal->get_library(input_fft_length) );

  forward = FTransform::Agent::current->get_plan(
      input_fft_length,
      real_to_complex ? FTransform::frc: FTransform::fcc);

  if (optimal)
    FTransform::set_library( optimal->get_library(output_fft_length) );

  backward = FTransform::Agent::current->get_plan (output_fft_length,
						   FTransform::bcc);
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::setup FFT plans created" << endl;
  
  fft_plans_setup = true;

  if (verbose)
  {
    cerr << "dsp::InverseFilterbankEngineCPU::setup"
      << " input_nchan=" << input_nchan
      << " output_nchan=" << output_nchan
      << " input_fft_length=" << input_fft_length
      << " output_fft_length=" << output_fft_length
      << " input_discard_pos=" << input_discard_pos
      << " input_discard_neg=" << input_discard_neg
      << " output_discard_pos=" << output_discard_pos
      << " output_discard_neg=" << output_discard_neg
      << endl;
  }

  // compute oversampling keep/discard region
  input_os_keep = filterbank->get_oversampling_factor().normalize(input_fft_length);
  input_os_discard = input_fft_length - input_os_keep;

  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::setup"
      << " input_os_keep=" << input_os_keep
      << " input_os_discard=" << input_os_discard
      << endl;

  if (input_os_discard % 2 != 0)
    throw Error (InvalidState, "dsp::InverseFilterbankEngineCPU::setup",
		 "input_os_discard=%u must be divisible by two",
		 input_os_discard);

  // setup scratch space
  input_fft_scratch_nfloat = input->get_ndim()*input_fft_length;
  input_time_scratch_nfloat = input_fft_scratch_nfloat;
  output_fft_scratch_nfloat = 2*output_fft_length; // always return complex result
  stitch_scratch_nfloat = 2*output_nchan*output_fft_length;

  total_scratch_needed = input_fft_scratch_nfloat +
                         input_time_scratch_nfloat +
                         output_fft_scratch_nfloat +
                         stitch_scratch_nfloat;

  if (zero_DM_response != nullptr)
    total_scratch_needed += stitch_scratch_nfloat;

  // dsp::Scratch* scratch = new Scratch;
  // input_fft_scratch = scratch->space<float>
  //   (input_time_points + input_fft_points + output_fft_points  + stitch_points);
  // input_time_scratch = input_fft_scratch + input_fft_points;
  // output_fft_scratch = input_time_scratch + input_time_points;
  // stitch_scratch = output_fft_scratch + output_fft_points;
}


void dsp::InverseFilterbankEngineCPU::set_scratch (float * _scratch)
{
  scratch = _scratch;
  input_fft_scratch = scratch;
  input_time_scratch = input_fft_scratch + input_fft_scratch_nfloat;
  output_fft_scratch = input_time_scratch + input_time_scratch_nfloat;
  stitch_scratch = output_fft_scratch + output_fft_scratch_nfloat;
  if (zero_DM_response != nullptr)
    stitch_scratch_zero_DM = stitch_scratch + stitch_scratch_nfloat;
}

void dsp::InverseFilterbankEngineCPU::perform (
  const dsp::TimeSeries* in,
  dsp::TimeSeries* out,
  dsp::TimeSeries* zero_DM_out,
  uint64_t npart,
  uint64_t in_step,
  uint64_t out_step
)
{
  static const int floats_per_complex = 2;
  static const int sizeof_complex = sizeof(float) * floats_per_complex;
  const unsigned n_dims = in->get_ndim();
  const unsigned n_pol = in->get_npol();

  unsigned input_os_keep_2 = input_os_keep / 2;
  unsigned stitched_offset_pos;
  unsigned stitched_offset_neg;

  float* freq_dom_ptr;
  float* time_dom_ptr;
  float* output_freq_dom_ptr;

  if (verbose)
  {
    if (pfb_dc_chan)
    {
      if (pfb_all_chan)
        cerr << "dsp::InverseFilterbankEngineCPU::perform "
          << "moving first half channel to end of assembled spectrum" << endl;
       else 
        cerr << "dsp::InverseFilterbankEngineCPU::perform "
          << "moving first half channel to end of assembled spectrum and zeroing" << endl;
      
    }
    else
    {
      cerr << "dsp::InverseFilterbankEngineCPU::perform "
        << "leaving assembled spectrum as is" << endl;
    }
    cerr << "dsp::InverseFilterbankEngineCPU::perform writing " << npart << " chunks" << endl;
  }
  // ofstream stitched_file("/home/SWIN/dshaff/stitched.dat", ios::out | ios::binary);
  // ofstream ifft_file("/home/SWIN/dshaff/ifft.dat", ios::out | ios::binary);
  // ofstream out_file("/home/SWIN/dshaff/out.dat", ios::out | ios::binary);
  // ofstream deripple_before_file("deripple.before.dat", ios::out | ios::binary);
  // ofstream deripple_before_file("deripple.before.dat", ios::out | ios::binary);
  // ofstream deripple_after_file("deripple.after.dat", ios::out | ios::binary);

  if (verbose)
  {
    if (temporal_apodization)
    {
      cerr << "dsp::InverseFilterbankEngineCPU::perform temporal_apodization.get_type() "
        << temporal_apodization->get_type() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::perform temporal_apodization.get_ndim() "
        << temporal_apodization->get_ndim() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::perform temporal_apodization.get_ndat() "
        << temporal_apodization->get_ndat() << endl;
    }
    cerr << "dsp::InverseFilterbankEngineCPU::perform output_nchan="
      << output_nchan << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform"
      << " out dim=(" << out->get_nchan()
      << ", " << out->get_npol()
      << ", " << out->get_ndat() << ")" << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform"
      << " in dim=(" << in->get_nchan()
      << ", " << in->get_npol()
      << ", " << in->get_ndat() << ")" << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform input_nchan="
      << input_nchan << endl;
  }

  void* src_ptr;
  void* dest_ptr;

  for (uint64_t ipart = 0; ipart < npart; ipart++)
  {
    for (unsigned ipol = 0; ipol < n_pol; ipol++)
    {
      for (unsigned input_ichan = 0; input_ichan < input_nchan; input_ichan++)
      {
        freq_dom_ptr = input_fft_scratch;
        time_dom_ptr = const_cast<float*>(in->get_datptr(input_ichan, ipol));
        time_dom_ptr += n_dims*ipart*(input_fft_length - input_discard_total);

        memcpy(
          input_time_scratch,
          time_dom_ptr,
          sizeof_complex*input_fft_length
        );

        if (temporal_apodization)
	{
          // if (verbose && input_ichan == 0) {
          //   cerr << "dsp::InverseFilterbankEngineCPU::perform applying temporal_apodization" << endl;
          // }
          temporal_apodization->operate(input_time_scratch);
        }

        if (report)
          reporter.emit("temporal_apodization", input_time_scratch, 1, 1, input_fft_length, 2);

        if (real_to_complex)
          forward->frc1d(input_fft_length, freq_dom_ptr, input_time_scratch);
        else
          // fcc1d(number_of_points, dest_ptr, src_ptr);
          forward->fcc1d(input_fft_length, freq_dom_ptr, input_time_scratch);

        // discard oversampled regions and do circular shift
        if (report)
          reporter.emit("fft", freq_dom_ptr, 1, 1, input_fft_length, 2);

        if (pfb_dc_chan)
	{
          if (input_ichan == 0)
	  {
            stitched_offset_neg = n_dims*(output_nchan*output_fft_length - input_os_keep_2);
            stitched_offset_pos = 0;
          }
	  else
	  {
            stitched_offset_neg = n_dims*(input_os_keep*input_ichan - input_os_keep_2);
            stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;
          }
        }
	else
	{
          stitched_offset_neg = n_dims*input_os_keep*input_ichan;
          stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;
        }

        memcpy(
          stitch_scratch + stitched_offset_neg,
          freq_dom_ptr + n_dims*(input_fft_length - input_os_keep_2),
          input_os_keep_2 * sizeof_complex
        );

        memcpy(
          stitch_scratch + stitched_offset_pos,
          freq_dom_ptr,
          input_os_keep_2 * sizeof_complex
        );

      } // end of for input_nchan

      // If we have the zeroth PFB channel and we don't have all the
      // PFB channels, then we zero the last half channel.
      if (! pfb_all_chan && pfb_dc_chan)
      {
        // if (verbose) {
        //   cerr << "dsp::InverseFilterbankEngineCPU::perform zeroing last half channel" << endl;
        // }
        int offset = n_dims*(output_nchan*output_fft_length - input_os_keep_2);
        for (unsigned i=0; i<n_dims*input_os_keep_2; i++)
          stitch_scratch[offset + i] = 0.0;
      }


      if (zero_DM_response != nullptr)
      {
        // copy data from stitch_scratch into stitch_scratch_zero_DM
        if (verbose)
          cerr << "dsp::InverseFilterbankEngineCPU::perform"
	    " applying zero_DM_response" << endl;

        memcpy (stitch_scratch_zero_DM, stitch_scratch,
		output_nchan*output_fft_length*sizeof_complex);
	
        zero_DM_response->operate(stitch_scratch_zero_DM, ipol, 0, output_nchan);
      }
      else if (verbose)
	cerr << "dsp::InverseFilterbankEngineCPU::perform"
	  " NOT applying zero_DM_response" << endl;

      if (response != nullptr)
      {
        // if (verbose) {
        //   cerr << "dsp::InverseFilterbankEngineCPU::perform applying response" << endl;
        // }
        response->operate(stitch_scratch, ipol, 0, output_nchan);
      }
      else
      {
        // if (verbose) {
        //   cerr << "dsp::InverseFilterbankEngineCPU::perform NOT applying response" << endl;
        // }
      }

      if (report)
        reporter.emit("response_stitch", stitch_scratch, 1, 1, output_fft_length*output_nchan, 2);

      // complex<float>* stitch_scratch_complex = reinterpret_cast<complex<float>*>(stitch_scratch);
      // for (int idat=0; idat<output_fft_length*output_nchan; idat++)
      // {
      //   cerr << stitch_scratch_complex[idat] << " ";
      // }
      // cerr << endl;

      if (out != nullptr)
      {
        // if (verbose) {
        //   cerr << "dsp::InverseFilterbankEngineCPU::perform pol=" << ipol <<" loop=" << ipart+1 << "/" << npart << " doing inverse FFT" << endl; //. output_fft_length: "<< output_fft_length << endl;
        // }
        output_freq_dom_ptr = stitch_scratch;
        for (unsigned output_ichan=0; output_ichan<output_nchan; output_ichan++)
	{
          // if (verbose) {
          //   cerr << "dsp::InverseFilterbankEngineCPU::perform output_ichan=" << output_ichan << endl;
          // }

          if (spectral_apodization != nullptr)
          {
            spectral_apodization -> operate (output_freq_dom_ptr);
          }

          // cerr << "dsp::InverseFilterbankEngineCPU::perform before fft" << endl;
          backward->bcc1d(output_fft_length, output_fft_scratch, output_freq_dom_ptr);
          if (report)
            reporter.emit("ifft", output_fft_scratch, 1, 1, output_fft_length, 2);

          // cerr << "dsp::InverseFilterbankEngineCPU::perform after fft" << endl;
          // ifft_file.write(
          //   reinterpret_cast<const char*>(output_fft_scratch),
          //   output_fft_length*sizeof_complex
          // );

          // Output is in FPT order.
          src_ptr = (void *)(
              output_fft_scratch + output_discard_pos*floats_per_complex);
          dest_ptr = (void *)(
              out->get_datptr(output_ichan, ipol) +
              ipart*output_sample_step*floats_per_complex);
          memcpy(dest_ptr, src_ptr, output_sample_step*sizeof_complex);
          output_freq_dom_ptr += output_fft_length*n_dims;

          if (zero_DM_response != nullptr)
	  {
            backward->bcc1d(output_fft_length, output_fft_scratch, stitch_scratch_zero_DM + output_ichan*output_fft_length*n_dims);
            src_ptr = (void *)(
                output_fft_scratch + output_discard_pos*floats_per_complex);
            dest_ptr = (void *)(
                zero_DM_out->get_datptr(output_ichan, ipol) +
                ipart*output_sample_step*floats_per_complex);
            memcpy(dest_ptr, src_ptr, output_sample_step*sizeof_complex);
          }

          // cerr << "dsp::InverseFilterbankEngineCPU::perform after copy" << endl;
          // out_file.write(
          //   reinterpret_cast<const char*>(dest_ptr),
          //   output_sample_step*sizeof_complex
          // );
        }
        // if (verbose) {
        //   cerr << "dsp::InverseFilterbankEngineCPU::perform backward FFTs complete." << endl;
        // }
      }
    } // end of n_pol
  } // end of ipart

  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::perform finish" << endl;

  // stitched_file.close();
  // ifft_file.close();
  // out_file.close();
  // deripple_before_file.close();
  // deripple_after_file.close();
}


void dsp::InverseFilterbankEngineCPU::perform (
  const dsp::TimeSeries* in,
  dsp::TimeSeries* out,
  uint64_t npart,
  uint64_t in_step,
  uint64_t out_step
)
{
  perform(in, out, nullptr, npart, in_step, out_step);
}

void dsp::InverseFilterbankEngineCPU::finish ()
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::finish" << endl;
}
