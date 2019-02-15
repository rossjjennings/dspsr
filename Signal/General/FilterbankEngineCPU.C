//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#define _DEBUG 1

#include "dsp/FilterbankEngineCPU.h"
#include "dsp/TimeSeries.h"
#include "dsp/Scratch.h"
#include "dsp/Apodization.h"
#include "dsp/OptimalFFT.h"

#include <iostream>
#include <assert.h>

#include "FTransform.h"

using namespace std;

dsp::FilterbankEngineCPU::FilterbankEngineCPU ()
{
  real_to_complex = false;

  nfilt_pos = 0;
  forward = 0;
  backward = 0;
  verbose = false;

  response = nullptr;
  passband = nullptr;
  apodization = nullptr;
}

dsp::FilterbankEngineCPU::~FilterbankEngineCPU ()
{
}

void dsp::FilterbankEngineCPU::set_passband (dsp::Response* _passband) {
  passband = _passband;
}

const dsp::Response* dsp::FilterbankEngineCPU::get_passband () const {
  return passband;
}

void dsp::FilterbankEngineCPU::setup (dsp::Filterbank* filterbank)
{
  verbose = filterbank->verbose;
  if (verbose) {
    cerr << "dsp::FilterbankEngineCPU::setup" << endl;
  }

  const dsp::TimeSeries* input = filterbank->get_input();

  if (filterbank->has_response()) {
    response = filterbank->get_response();
  }
  if (filterbank->has_apodization()) {
    apodization = filterbank->get_apodization();
  }

  matrix_convolution = filterbank->get_matrix_convolution();
  freq_res = filterbank->get_frequency_resolution();
  nchan_subband = filterbank->get_nchan_subband();
  nsamp_fft = filterbank->get_minimum_samples();
  nfilt_pos = filterbank->get_nfilt_pos();
  nfilt_neg = filterbank->get_nfilt_neg();
  nkeep = freq_res - (nfilt_pos + nfilt_neg);

  unsigned n_fft = nchan_subband * freq_res;
  //! configure passband
  if (passband)
  {
    if (response)
      passband -> match (response);

    unsigned passband_npol = input->get_npol();
    if (matrix_convolution)
      passband_npol = 4;

    passband->resize (passband_npol, input->get_nchan(), n_fft, 1);

    if (!response)
      passband->match (input);
  }

  // setup fourier transform plans
  forward = filterbank->get_forward();
  backward = filterbank->get_backward();

  // using namespace FTransform;
  //
  // OptimalFFT* optimal = 0;
  // if (response && response->has_optimal_fft()){
  //   optimal = response->get_optimal_fft();
  // }
  //
  // if (optimal){
  //   FTransform::set_library( optimal->get_library( nsamp_fft ) );
  // }
  //
  // if (input->get_state() == Signal::Nyquist){
  //   forward = FTransform::Agent::current->get_plan (nsamp_fft, FTransform::frc);
  // } else {
  //   forward = FTransform::Agent::current->get_plan (nsamp_fft, FTransform::fcc);
  // }
  //
  // if (optimal){
  //   FTransform::set_library( optimal->get_library( freq_res ) );
  // }
  //
  // if (freq_res > 1){
  //   backward = FTransform::Agent::current->get_plan (freq_res, FTransform::bcc);
  // }

  // setup scratch space
  unsigned bigfftsize = nchan_subband * freq_res * 2;
  if (input->get_state() == Signal::Nyquist)
    bigfftsize += 256;

  // also need space to hold backward FFTs
  unsigned scratch_needed = bigfftsize + 2 * freq_res;

  if (apodization) {
    if (verbose) {
      cerr << "dsp::FilterbankEngineCPU::setup: adding space for apodization" << endl;
    }
    scratch_needed += bigfftsize;
  }

  if (matrix_convolution){
    if (verbose) {
      cerr << "dsp::FilterbankEngineCPU::setup: adding space for matrix convolution" << endl;
    }
    scratch_needed += bigfftsize;
  }

  // divide up the scratch space
  dsp::Scratch* scratch = new Scratch;
  freq_domain_scratch[0] = scratch->space<float> (scratch_needed);
  freq_domain_scratch[1] = freq_domain_scratch[0];
  if (matrix_convolution) {
    freq_domain_scratch[1] += bigfftsize;
  }
  time_domain_scratch = freq_domain_scratch[1] + bigfftsize;
  windowed_time_domain_scratch = time_domain_scratch + 2 * freq_res;




  if (verbose) {
    cerr << "dsp::FilterbankEngineCPU::setup: freq_res=" << freq_res <<
    " nchan_subband=" << nchan_subband <<
    " nsamp_fft=" << nsamp_fft <<
    " nfilt_pos=" << nfilt_pos <<
    " nfilt_neg=" << nfilt_neg <<
    " forward=" << forward <<
    " backward=" << backward <<
    " nkeep=" << nkeep << endl;
    cerr << "dsp::FilterbankEngineCPU::setup: finished" << endl;
  }
}

void dsp::FilterbankEngineCPU::set_scratch (float * _scratch)
{
  scratch = _scratch;
}

void dsp::FilterbankEngineCPU::finish ()
{
  if (verbose) {
    cerr << "dsp::FilterbankEngineCPU::finish" << endl;
  }
}


void dsp::FilterbankEngineCPU::perform (
    const dsp::TimeSeries * in,
    dsp::TimeSeries * out,
    uint64_t npart,
    const uint64_t in_step,
    const uint64_t out_step
)
{
  if (verbose) {
    cerr << "dsp::FilterbankEngineCPU::perform" <<
    " npart=" << npart << " in_step=" << in_step << " out_step=" << out_step
    << endl;
  }

  unsigned cross_pol = 1;
  if (matrix_convolution) {
    cross_pol = 2;
  }

  // counters
  unsigned ipt, ipol, jpol, ichan;
  uint64_t ipart;

  const unsigned npol = in->get_npol();

  // offsets into input and output
  uint64_t in_offset, out_offset;

  // some temporary pointers
  float* time_dom_ptr = NULL;
  float* freq_dom_ptr = NULL;

  // do a 64-bit copy
  uint64_t* data_into = NULL;
  uint64_t* data_from = NULL;

  if (verbose) {
    cerr << "dsp::FilterbankEngineCPU::perform: matrix_convolution= " << matrix_convolution << " pol=" << npol << " cross_pol=" << cross_pol << endl;
    cerr << "dsp::FilterbankEngineCPU::perform: starting main loop" << endl;
  }

  for (unsigned input_ichan=0; input_ichan<in->get_nchan(); input_ichan++)
  {

    for (ipart=0; ipart<npart; ipart++)
    {
  #ifdef _DEBUG
      cerr << "dsp::FilterbankEngineCPU::perform: ipart=" << ipart << endl;
  #endif
      in_offset = ipart * in_step;
      out_offset = ipart * out_step;

      for (ipol=0; ipol < npol; ipol++)
      {
        for (jpol=0; jpol<cross_pol; jpol++)
        {
          if (matrix_convolution) {
            ipol = jpol;
          }
          #ifdef _DEBUG
            cerr << "dsp::FilterbankEngineCPU::perform: ipol=" << ipol << " jpol=" << jpol << " input_ichan=" << input_ichan << endl;
          #endif
          time_dom_ptr = const_cast<float*>(in->get_datptr (input_ichan, ipol));
          time_dom_ptr += in_offset;
          #ifdef _DEBUG
            cerr << "dsp::FilterbankEngineCPU::perform: time_dom_ptr: " << time_dom_ptr << endl;
          #endif

          if (apodization != nullptr)
          {
            apodization -> operate (time_dom_ptr, windowed_time_domain_scratch);
            time_dom_ptr = windowed_time_domain_scratch;
          }

          if (in->get_state() == Signal::Nyquist){
            forward->frc1d (nsamp_fft, freq_domain_scratch[ipol], time_dom_ptr);
          } else {
            forward->fcc1d (nsamp_fft, freq_domain_scratch[ipol], time_dom_ptr);
          }
        }

        if (matrix_convolution)
        {
          if (passband){
            passband->integrate (freq_domain_scratch[0], freq_domain_scratch[1], input_ichan);
          }
          // cross filt can be set only if there is a response
          response->operate (freq_domain_scratch[0], freq_domain_scratch[1]);
        }
        else
        {
          if (passband){
            passband->integrate (freq_domain_scratch[ipol], ipol, input_ichan);
          }

          if (response){
            response->operate (freq_domain_scratch[ipol], ipol,
                               input_ichan*nchan_subband, nchan_subband);
          }
        }

        for (jpol=0; jpol<cross_pol; jpol++)
        {
          if (matrix_convolution)
            ipol = jpol;

          if (freq_res == 1)
          {
            data_from = (uint64_t*)( freq_domain_scratch[ipol] );
            for (ichan=0; ichan < nchan_subband; ichan++)
            {
              data_into = (uint64_t*)( out->get_datptr (input_ichan*nchan_subband+ichan, ipol) + out_offset );

              *data_into = data_from[ichan];
            }
            continue;
          }


          // freq_res > 1 requires a backward fft into the time domain
          // for each channel

          unsigned jchan = input_ichan * nchan_subband;
          freq_dom_ptr = freq_domain_scratch[ipol];

          for (ichan=0; ichan < nchan_subband; ichan++)
          {
            backward->bcc1d (freq_res, time_domain_scratch, freq_dom_ptr);

            freq_dom_ptr += freq_res*2;

            data_into = (uint64_t*)( out->get_datptr (jchan+ichan, ipol) + out_offset);
            data_from = (uint64_t*)( time_domain_scratch + nfilt_pos*2 );  // complex nos.

            for (ipt=0; ipt < nkeep; ipt++){
              data_into[ipt] = data_from[ipt];
            }
          } // for each out channel

        } // for each cross poln

      } // for each polarization

    } // for each big fft (ipart)

  } // for each input channel

  if (verbose) {
    cerr << "dsp:FilterbankEngineCPU::finish" << endl;
  }
}
