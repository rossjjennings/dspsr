/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "FTransform.h"

#include "dsp/InverseFilterbankResponse.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

dsp::InverseFilterbankResponse::InverseFilterbankResponse ()
{
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::InverseFilterbankResponse::ctor" << std::endl;
  }
  ndim = 2;
  nchan = 1;
  input_nchan = 1;
  npol = 1;
  ndat = 1;
  input_overlap = 0;

  impulse_pos = 0;
  impulse_neg = 0;

  pfb_dc_chan = false;

  built = false;
}

dsp::InverseFilterbankResponse::~InverseFilterbankResponse ()
{
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::InverseFilterbankResponse::dtor" << std::endl;
  }
}

dsp::InverseFilterbankResponse::InverseFilterbankResponse (const dsp::InverseFilterbankResponse& response) {
  operator = (response);
}

const dsp::InverseFilterbankResponse& dsp::InverseFilterbankResponse::operator= (const dsp::InverseFilterbankResponse& response)
{
  if (this == &response)
    return *this;

  dsp::Response::operator= (response);

  fir_filter = response.fir_filter;
  forward = response.forward;
  built = response.built;
  pfb_dc_chan = response.pfb_dc_chan;
  apply_deripple = response.apply_deripple;
  oversampling_factor = response.oversampling_factor;
  input_overlap = response.input_overlap;

  return *this;
}



void dsp::InverseFilterbankResponse::build ()
{
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::InverseFilterbankResponse::build" << std::endl;
  }
  if (built) {
    return;
  }

  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::build: input_overlap=" << input_overlap << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: apply_deripple=" << apply_deripple << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: impulse_pos=" << impulse_pos << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: impulse_neg=" << impulse_neg << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: ndat=" << ndat << std::endl;
  }

  if (! apply_deripple) {
    std::complex<float>* phasors = reinterpret_cast< std::complex<float>* > ( buffer );
    uint64_t npt = ndat * nchan;

    for (unsigned ipt=0; ipt<npt; ipt++) {
      phasors[ipt] = std::complex<float> (1.0, 0.0);
    }

    built = true;
    return;
  }


  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::build: bufsize=" << bufsize << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: whole_swapped=" << whole_swapped << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: swap_divisions=" << swap_divisions << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: pfb_dc_chan=" << pfb_dc_chan << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::build: npol=" << npol
      << " input_nchan=" << input_nchan
      << " nchan=" << nchan
      << " ndat=" << ndat
      << " ndim=" << ndim
      << std::endl;
  }
  unsigned half_chan_shift = 0 ;
  if (pfb_dc_chan) {
    half_chan_shift = 1;
  }

  unsigned total_ndat = ndat * nchan;

  std::vector<float> freq_response;
  unsigned ndat_per_chan = total_ndat / input_nchan;
  // calc_freq_response(freq_response, total_ndat*nchan/2);
  calc_freq_response(freq_response, total_ndat/2);
  // roll the array by the appropriate number of bins
  int shift_bins = -1*static_cast<int>(ndim*half_chan_shift*ndat_per_chan/2);
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::build: shift_bins=" << shift_bins << std::endl;
  }

  std::complex<float>* freq_response_complex = reinterpret_cast<std::complex<float>*>(freq_response.data());

  std::complex<float>* phasors = reinterpret_cast<std::complex<float>*>(buffer);

  uint64_t npt = ndat_per_chan/2;

  int step = 0;
  for (int ichan=0; ichan < input_nchan; ichan++) {
    for (uint64_t ipt=0; ipt < npt; ipt++) {
      phasors[ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[ipt]), 0.0);
      phasors[npt + ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[npt - ipt]), 0.0);

      // phasors[ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[npt - ipt]), 0.0);
      // phasors[npt + ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[ipt]), 0.0);
    }
    step += ndat_per_chan;
  }

  if (shift_bins != 0) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbankResponse::build: rolling data" << std::endl;
    }
    int shift_bins_pos =  static_cast<int>(ndat) + shift_bins/static_cast<int>(ndim);
    roll<std::complex<float>>(phasors, ndat, shift_bins_pos);
  }
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::build: done" << std::endl;
  }


  built = true;

}

//! Set the dimensions of the data and update the built attribute
void dsp::InverseFilterbankResponse::resize (unsigned _npol, unsigned _nchan,
        unsigned _ndat, unsigned _ndim)
{
  if (verbose)
    std::cerr << "dsp::InverseFilterbankResponse::resize(" << _npol << "," << _nchan
         << "," << _ndat << "," << _ndim << ")" << std::endl;
  if (npol != _npol || nchan != _nchan || ndat != _ndat || ndim != _ndim)
  {
    built = false;
  }
  Shape::resize (_npol, _nchan, _ndat, _ndim);
}


//! Set the length of the frequnecy response in each channel
void dsp::InverseFilterbankResponse::set_ndat (unsigned _ndat)
{
  if (ndat != _ndat)
    built = false;
  ndat = _ndat;
}

//! Set the number of input channels
void dsp::InverseFilterbankResponse::set_nchan (unsigned _nchan)
{
  if (nchan != _nchan)
    built = false;
  nchan = _nchan;
}

void dsp::InverseFilterbankResponse::calc_freq_response (
  std::vector<float>& freq_response, unsigned n_freq)
{
  // std::ofstream freq_response_file("freq_response.dat", std::ios::out | std::ios::binary);
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankResponse::calc_freq_response:"
      << " freq_response.size()=" << freq_response.size()
      << " n_freq=" << n_freq
      << std::endl;
  }
  freq_response.resize(2*n_freq);
  std::vector<float> filter_coeff_padded (2*n_freq);
  std::vector<float> freq_response_temp (4*n_freq);
  std::fill(filter_coeff_padded.begin(), filter_coeff_padded.end(), 0.0);

  for (unsigned i=0; i<fir_filter.get_ntaps(); i++) {
    filter_coeff_padded[i] = fir_filter[i];
  }

  // need the factor of two for real ("Nyquist") input signal
  forward = FTransform::Agent::current->get_plan (2*n_freq, FTransform::frc);

  forward->frc1d(
    2*n_freq,
    freq_response_temp.data(),
    filter_coeff_padded.data()
  );

  for (int i=0; i<2*n_freq; i++) {
    freq_response[i] = freq_response_temp[i];
  }

  // freq_response_file.write(
  //     reinterpret_cast<const char*>(freq_response.data()),
  //     2*n_freq*sizeof(float)
  // );
  // freq_response_file.close();
}

//! Create an Scalar Filter with nchan channels
void dsp::InverseFilterbankResponse::match (const Observation* obs, unsigned channels)
{
  if (verbose){
    std::cerr << "dsp::InverseFilterbankResponse::match(const Observation*, unsigned) channels=" << channels << std::endl;
  }
  set_nchan (channels);

  if (verbose){
    std::cerr << "dsp::InverseFilterbankResponse::match obs->get_nchan() " << obs->get_nchan() << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::match ndat=" << ndat << std::endl;
    std::cerr << "dsp::InverseFilterbankResponse::match input_overlap=" << input_overlap << std::endl;
  }
  input_nchan = obs->get_nchan();
  oversampling_factor = obs->get_oversampling_factor();

  if (!built && ndat>1) {
    impulse_pos = input_overlap;
    impulse_neg = input_overlap;

    ndat = (input_nchan/channels)*oversampling_factor.normalize(ndat);
    impulse_pos = (input_nchan/channels)*oversampling_factor.normalize(impulse_pos);
    impulse_neg = (input_nchan/channels)*oversampling_factor.normalize(impulse_neg);

    resize (npol, nchan, ndat, ndim);
    build();
  }
}

//! Create a InverseFilterbankResponse with the same number of channels as Response
void dsp::InverseFilterbankResponse::match (const Response* response)
{
  if (verbose)
    std::cerr << "dsp::InverseFilterbankResponse::match (const Response*) nchan=" << response->get_nchan()
         << " ndat=" << response->get_ndat() << std::endl;

  if ( get_nchan() == response->get_nchan() &&
       get_ndat() == response->get_ndat() )
  {

    if (verbose)
      std::cerr << "dsp::InverseFilterbankResponse::match Response already matched" << std::endl;
    return;
  }

  input_nchan = response->get_input_nchan();

  resize (npol, response->get_nchan(), response->get_ndat(), ndim);
  impulse_pos = input_nchan*oversampling_factor.normalize(input_overlap);
  impulse_neg = input_nchan*oversampling_factor.normalize(input_overlap);


  if (!built){
    build();
  }
}
