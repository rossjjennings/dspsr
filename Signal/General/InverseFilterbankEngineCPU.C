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


dsp::InverseFilterbankEngineCPU::InverseFilterbankEngineCPU ()
{
  input_fft_length = 0;
  fft_plans_setup = false;
  response = nullptr;
  zero_DM_response = nullptr;
  fft_window = nullptr;

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
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup" << std::endl;
  }

  verbose = filterbank->verbose;
  const TimeSeries* input = filterbank->get_input();
  TimeSeries* output = filterbank->get_output();
  real_to_complex = (input->get_state() == Signal::Nyquist);
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

  if (filterbank->has_response()) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup: setting response" << std::endl;
    }
    response = filterbank->get_response();
  }

  if (filterbank->has_apodization()) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup: setting fft_window" << std::endl;
    }
    fft_window = filterbank->get_apodization();
    if (verbose) {
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup: fft_window.get_type() "
      << fft_window->get_type() << std::endl;
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup: fft_window.get_ndim() "
      << fft_window->get_ndim() << std::endl;
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup: fft_window.get_ndat() "
      << fft_window->get_ndat() << std::endl;
    }
  }

  if (filterbank->get_zero_DM()) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup: setting zero_DM_response" << std::endl;
    }
    zero_DM_response = filterbank->get_zero_DM_response();
  }

  OptimalFFT* optimal = 0;
  if (response && response->has_optimal_fft()) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbankEngineCPU::setup: getting OptimalFFT object" << std::endl;
    }
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
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup: done setting up FFT plans" << std::endl;
  }
  fft_plans_setup = true;

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

  if (input_os_discard % 2 != 0)
  {
    throw "dsp::InverseFilterbankEngineCPU::setup: input_os_discard must be divisible by two";
  }

  // setup scratch space
  input_fft_scratch_samples = input->get_ndim()*input_fft_length;
  input_time_scratch_samples = input_fft_scratch_samples;
  output_fft_scratch_samples = 2*output_fft_length; // always return complex result
  stitch_scratch_samples = 2*output_nchan*output_fft_length;

  total_scratch_needed = input_fft_scratch_samples +
                         input_time_scratch_samples +
                         output_fft_scratch_samples +
                         stitch_scratch_samples;

  if (zero_DM_response != nullptr) {
    total_scratch_needed += stitch_scratch_samples;
  }
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
  input_time_scratch = input_fft_scratch + input_fft_scratch_samples;
  output_fft_scratch = input_time_scratch + input_time_scratch_samples;
  stitch_scratch = output_fft_scratch + output_fft_scratch_samples;
  if (zero_DM_response != nullptr) {
    stitch_scratch_zero_DM = stitch_scratch + stitch_scratch_samples;
  }

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
  // std::ofstream stitched_file("/home/SWIN/dshaff/stitched.dat", std::ios::out | std::ios::binary);
  // std::ofstream ifft_file("/home/SWIN/dshaff/ifft.dat", std::ios::out | std::ios::binary);
  // std::ofstream out_file("/home/SWIN/dshaff/out.dat", std::ios::out | std::ios::binary);
  // std::ofstream deripple_before_file("deripple.before.dat", std::ios::out | std::ios::binary);
  // std::ofstream deripple_before_file("deripple.before.dat", std::ios::out | std::ios::binary);
  // std::ofstream deripple_after_file("deripple.after.dat", std::ios::out | std::ios::binary);

  if (verbose) {
    if (fft_window) {
      std::cerr << "dsp::InverseFilterbankEngineCPU::perform: fft_window.get_type() "
        << fft_window->get_type() << std::endl;
      std::cerr << "dsp::InverseFilterbankEngineCPU::perform: fft_window.get_ndim() "
        << fft_window->get_ndim() << std::endl;
      std::cerr << "dsp::InverseFilterbankEngineCPU::perform: fft_window.get_ndat() "
        << fft_window->get_ndat() << std::endl;
    }
    std::cerr << "dsp::InverseFilterbankEngineCPU::perform: output_nchan="
      << output_nchan << std::endl;
    std::cerr << "dsp::InverseFilterbankEngineCPU::perform:"
      << " out dim=(" << out->get_nchan()
      << ", " << out->get_npol()
      << ", " << out->get_ndat() << ")" << std::endl;
    std::cerr << "dsp::InverseFilterbankEngineCPU::perform:"
      << " in dim=(" << in->get_nchan()
      << ", " << in->get_npol()
      << ", " << in->get_ndat() << ")" << std::endl;
    std::cerr << "dsp::InverseFilterbankEngineCPU::perform: input_nchan="
      << input_nchan << std::endl;
  }

  void* src_ptr;
  void* dest_ptr;

  for (uint64_t ipart = 0; ipart < npart; ipart++) {
    for (unsigned ipol = 0; ipol < n_pol; ipol++) {
      for (unsigned input_ichan = 0; input_ichan < input_nchan; input_ichan++) {
        freq_dom_ptr = input_fft_scratch;
        time_dom_ptr = const_cast<float*>(in->get_datptr(input_ichan, ipol));
        time_dom_ptr += n_dims*ipart*(input_fft_length - input_discard_total);

        memcpy(
          input_time_scratch,
          time_dom_ptr,
          sizeof_complex*input_fft_length
        );

        if (fft_window) {
          // if (verbose && input_ichan == 0) {
          //   std::cerr << "dsp::InverseFilterbankEngineCPU::perform: applying fft_window" << std::endl;
          // }
          fft_window->operate(input_time_scratch);
        }

        if (report) {
          reporter.emit("fft_window", input_time_scratch, 1, 1, input_fft_length, 2);
        }

        if (real_to_complex) {
          forward->frc1d(input_fft_length, freq_dom_ptr, input_time_scratch);
        } else {
          // fcc1d(number_of_points, dest_ptr, src_ptr);
          forward->fcc1d(input_fft_length, freq_dom_ptr, input_time_scratch);
        }
        // discard oversampled regions and do circular shift
        if (report) {
          reporter.emit("fft", freq_dom_ptr, 1, 1, input_fft_length, 2);
        }

        if (pfb_dc_chan) {
          if (input_ichan == 0) {
            stitched_offset_neg = n_dims*(output_nchan*output_fft_length - input_os_keep_2);
            stitched_offset_pos = 0;
          } else {
            stitched_offset_neg = n_dims*(input_os_keep*input_ichan - input_os_keep_2);
            stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;
          }
        } else {
          stitched_offset_neg = n_dims*input_os_keep*input_ichan;
          stitched_offset_pos = stitched_offset_neg + n_dims*input_os_keep_2;
        }



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

      } // end of for input_nchan

      // If we have the zeroth PFB channel and we don't have all the PFB channels,
      // then we zero the last half channel.
      if (! pfb_all_chan && pfb_dc_chan) {
        // if (verbose) {
        //   std::cerr << "dsp::InverseFilterbankEngineCPU::perform: zeroing last half channel" << std::endl;
        // }
        int offset = n_dims*(output_nchan*output_fft_length - input_os_keep_2);
        for (unsigned i=0; i<n_dims*input_os_keep_2; i++) {
          stitch_scratch[offset + i] = 0.0;
        }
      }


      if (zero_DM_response != nullptr) {
        // copy data from stitch_scratch into stitch_scratch_zero_DM
        if (verbose) {
          std::cerr << "dsp::InverseFilterbankEngineCPU::perform: applying zero_DM_response" << std::endl;
        }
        memcpy(stitch_scratch_zero_DM, stitch_scratch, output_nchan*output_fft_length*sizeof_complex);
        zero_DM_response->operate(stitch_scratch_zero_DM, ipol, 0, output_nchan);
      } else {
        if (verbose) {
          std::cerr << "dsp::InverseFilterbankEngineCPU::perform: NOT applying zero_DM_response" << std::endl;
        }
      }

      if (response != nullptr) {
        // if (verbose) {
        //   std::cerr << "dsp::InverseFilterbankEngineCPU::perform: applying response" << std::endl;
        // }
        response->operate(stitch_scratch, ipol, 0, output_nchan);
      } else {
        // if (verbose) {
        //   std::cerr << "dsp::InverseFilterbankEngineCPU::perform: NOT applying response" << std::endl;
        // }
      }

      if (report) {
        reporter.emit("response_stitch", stitch_scratch, 1, 1, output_fft_length*output_nchan, 2);
      }
      // std::complex<float>* stitch_scratch_complex = reinterpret_cast<std::complex<float>*>(stitch_scratch);
      // for (int idat=0; idat<output_fft_length*output_nchan; idat++)
      // {
      //   std::cerr << stitch_scratch_complex[idat] << " ";
      // }
      // std::cerr << std::endl;

      if (out != nullptr) {
        // if (verbose) {
        //   std::cerr << "dsp::InverseFilterbankEngineCPU::perform: pol=" << ipol <<" loop=" << ipart+1 << "/" << npart << " doing inverse FFT" << std::endl; //. output_fft_length: "<< output_fft_length << std::endl;
        // }
        output_freq_dom_ptr = stitch_scratch;
        for (unsigned output_ichan=0; output_ichan<output_nchan; output_ichan++) {
          // if (verbose) {
          //   std::cerr << "dsp::InverseFilterbankEngineCPU::perform output_ichan=" << output_ichan << std::endl;
          // }
          // std::cerr << "dsp::InverseFilterbankEngineCPU::perform: before fft" << std::endl;
          backward->bcc1d(output_fft_length, output_fft_scratch, output_freq_dom_ptr);
          if (report) {
            reporter.emit("ifft", output_fft_scratch, 1, 1, output_fft_length, 2);
          }
          // std::cerr << "dsp::InverseFilterbankEngineCPU::perform: after fft" << std::endl;
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
          std::memcpy(dest_ptr, src_ptr, output_sample_step*sizeof_complex);
          output_freq_dom_ptr += output_fft_length*n_dims;

          if (zero_DM_response != nullptr) {
            backward->bcc1d(output_fft_length, output_fft_scratch, stitch_scratch_zero_DM + output_ichan*output_fft_length*n_dims);
            src_ptr = (void *)(
                output_fft_scratch + output_discard_pos*floats_per_complex);
            dest_ptr = (void *)(
                zero_DM_out->get_datptr(output_ichan, ipol) +
                ipart*output_sample_step*floats_per_complex);
            std::memcpy(dest_ptr, src_ptr, output_sample_step*sizeof_complex);
          }


          // std::cerr << "dsp::InverseFilterbankEngineCPU::perform: after copy" << std::endl;
          // out_file.write(
          //   reinterpret_cast<const char*>(dest_ptr),
          //   output_sample_step*sizeof_complex
          // );
        }
        // if (verbose) {
        //   std::cerr << "dsp::InverseFilterbankEngineCPU::perform: backward FFTs complete." << std::endl;
        // }
      }
    } // end of n_pol
  } // end of ipart

  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::perform: finish" << std::endl;
  }
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
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::finish" << std::endl;
  }
}
