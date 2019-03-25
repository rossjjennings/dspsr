/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/DerippleResponse.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

dsp::DerippleResponse::DerippleResponse ()
{
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::DerippleResponse::ctor" << std::endl;
  }
  ndim = 2;
  nchan = 1;
  input_nchan = 1;
  npol = 1;
  ndat = 1;

  pfb_dc_chan = false;

  built = false;
}

dsp::DerippleResponse::~DerippleResponse ()
{
}

void dsp::DerippleResponse::build ()
{
  if (built) {
    return;
  }
  resize (npol, nchan, ndat, ndim);

  if (verbose) {
    std::cerr << "dsp::DerippleResponse::build: bufsize=" << bufsize << std::endl;
    std::cerr << "dsp::DerippleResponse::build: whole_swapped=" << whole_swapped << std::endl;
    std::cerr << "dsp::DerippleResponse::build: swap_divisions=" << swap_divisions << std::endl;
    std::cerr << "dsp::DerippleResponse::build: pfb_dc_chan=" << pfb_dc_chan << std::endl;
    std::cerr << "dsp::DerippleResponse::build: npol=" << npol
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

  std::vector<float> freq_response;
  unsigned ndat_per_chan = ndat / input_nchan;
  // calc_freq_response(freq_response, ndat*nchan/2);
  calc_freq_response(freq_response, ndat/2);
  // roll the array by the appropriate number of bins
  int shift_bins = -1*static_cast<int>(ndim*half_chan_shift*ndat_per_chan/2);
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::build: shift_bins=" << shift_bins << std::endl;
  }

  std::complex<float>* freq_response_complex = reinterpret_cast<std::complex<float>*>(freq_response.data());

  std::complex<float>* phasors = reinterpret_cast<std::complex<float>*>(buffer);

  uint64_t npt = ndat_per_chan/2;

  int step = 0;
  for (int ichan=0; ichan < input_nchan; ichan++) {
    for (uint64_t ipt=0; ipt < npt; ipt++) {
      // this is correct.. but why?
      phasors[ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[ipt]), 0.0);
      phasors[npt + ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[npt - ipt + 1]), 0.0);

      // phasors[ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[npt - ipt + 1]), 0.0);
      // phasors[npt + ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[ipt]), 0.0);
    }
    step += ndat_per_chan;
  }

  // std::ofstream freq_response_file_before("freq_response.buffer.before.dat", std::ios::out | std::ios::binary);
  //
  // freq_response_file_before.write(
  //     reinterpret_cast<const char*>(buffer),
  //     ndat*nchan*ndim*npol*sizeof(float)
  // );
  //
  // freq_response_file_before.close();
  if (shift_bins != 0) {
    roll<std::complex<float>>(phasors, ndat, static_cast<int>(ndat) + shift_bins/static_cast<int>(ndim));
  }
  // roll<std::complex<float>>(phasors, ndat, shift_bins/static_cast<int>(ndim));

  // std::ofstream freq_response_file_after("freq_response.buffer.after.dat", std::ios::out | std::ios::binary);
  //
  // freq_response_file_after.write(
  //     reinterpret_cast<const char*>(buffer),
  //     ndat*nchan*ndim*npol*sizeof(float)
  // );
  //
  // freq_response_file_after.close();

  built = true;

}

//! Set the dimensions of the data and update the built attribute
void dsp::DerippleResponse::resize (unsigned _npol, unsigned _nchan,
        unsigned _ndat, unsigned _ndim)
{
  if (verbose)
    std::cerr << "dsp::DerippleResponse::resize(" << _npol << "," << _nchan
         << "," << _ndat << "," << _ndim << ")" << std::endl;
  if (npol != _npol || nchan != _nchan || ndat != _ndat || ndim != _ndim)
  {
    built = false;
  }
  Shape::resize (_npol, _nchan, _ndat, _ndim);
}


//! Set the length of the frequnecy response in each channel
void dsp::DerippleResponse::set_ndat (unsigned _ndat)
{
  if (ndat != _ndat)
    built = false;
  ndat = _ndat;
}

//! Set the number of input channels
void dsp::DerippleResponse::set_nchan (unsigned _nchan)
{
  if (nchan != _nchan)
    built = false;
  nchan = _nchan;
}



void dsp::DerippleResponse::calc_freq_response (
  std::vector<float>& freq_response, unsigned n_freq)
{
  std::ofstream freq_response_file("freq_response.dat", std::ios::out | std::ios::binary);
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::calc_freq_response" << std::endl;
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

  freq_response_file.write(
      reinterpret_cast<const char*>(freq_response.data()),
      2*n_freq*sizeof(float)
  );
  freq_response_file.close();
}

//! Create an Scalar Filter with nchan channels
void dsp::DerippleResponse::match (const Observation* obs, unsigned channels)
{
  if (verbose){
    std::cerr << "dsp::DerippleResponse::match channels=" << channels << std::endl;
  }
  // if (!channels)
  //   channels = obs->get_nchan();

  if (verbose){
    std::cerr << "dsp::DerippleResponse::match set_nchan(" << channels << ")" << std::endl;
  }

  set_nchan (channels);

  input_nchan = obs->get_nchan();
  pfb_dc_chan = obs->get_pfb_dc_chan();

  if (!built && ndat > 1) {
    build();
  }
}

//! Create a DerippleResponse with the same number of channels as Response
void dsp::DerippleResponse::match (const Response* response)
{
  if (verbose)
    std::cerr << "dsp::DerippleResponse::match Response nchan=" << response->get_nchan()
         << " ndat=" << response->get_ndat() << std::endl;

  if ( get_nchan() == response->get_nchan() &&
       get_ndat() == response->get_ndat() )
  {

    if (verbose)
      std::cerr << "dsp::DerippleResponse::match Response already matched" << std::endl;
    return;
  }

  input_nchan = response->get_input_nchan();

  resize (npol, response->get_nchan(), response->get_ndat(), ndim);

  if (!built){
    build();
  }
}
