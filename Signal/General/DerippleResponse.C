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
#include <cstring>

dsp::DerippleResponse::DerippleResponse ()
{
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::DerippleResponse()" << std::endl;
  }
  ndim = 2;
  nchan = 1;
  input_nchan = 1;
  npol = 1;
  ndat = 1;

  half_chan_shift = 1;

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
  }

  std::vector<float> freq_response;
  unsigned ndat_per_chan = ndat / input_nchan;
  // calc_freq_response(freq_response, ndat*nchan/2);
  calc_freq_response(freq_response, ndat/2);
  // roll the array by the appropriate number of bins
  roll(freq_response, -ndim*half_chan_shift*ndat_per_chan/2);

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
  // std::ofstream freq_response_file("freq_response.buffer.dat", std::ios::out | std::ios::binary);
  //
  // freq_response_file.write(
  //     reinterpret_cast<const char*>(buffer),
  //     ndat*nchan*ndim*npol*sizeof(float)
  // );
  //
  // freq_response_file.close();
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
  if (verbose)
    std::cerr << "dsp::DerippleResponse::match channels=" << channels << std::endl;

  if (!channels)
    channels = obs->get_nchan();

  if (verbose)
    std::cerr << "dsp::DerippleResponse::match set_nchan(" << channels << ")" << std::endl;
  set_nchan (channels);

  if (!built)
  {
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

  resize (npol, response->get_nchan(), response->get_ndat(), ndim);

  if (!built)
    build();
}

//! "roll" `arr` by `shift`
void dsp::DerippleResponse::roll (std::vector<float>& arr, int shift) {
  if (shift == 0) {
    return;
  } else {
    unsigned original_size = arr.size();
    arr.resize(original_size + abs(shift));
    float* buffer = arr.data();
    if (shift > 0) {
      std::memcpy(buffer + shift, buffer, original_size*sizeof(float));
      std::memcpy(buffer, buffer + original_size, shift*sizeof(float));
    } else {
      std::memcpy(buffer + original_size, buffer, abs(shift)*sizeof(float));
      std::memcpy(buffer, buffer + abs(shift), (original_size+shift)*sizeof(float));
      std::memcpy(buffer + original_size+shift, buffer + original_size, abs(shift)*sizeof(float));
    }
    arr.resize(original_size);
  }
}
