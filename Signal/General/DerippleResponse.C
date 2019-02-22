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
#include <iostream>
#include <vector>

dsp::DerippleResponse::DerippleResponse ()
{
  built = false;
  frequency_resolution_set = false;
  times_minimum_nfft = 0;

  ndim = 2;
  nchan = 1;
  npol = 1;
  ndat = 0;
}

void dsp::DerippleResponse::prepare (
  const dsp::Observation* input, unsigned channels)
{

  if (ndat == 0) {
    // ndat = input->get_frequency_resolution();
    throw std::runtime_error ("Need to set ndat before calling DerippleResponse::prepare");
  }
  set_nchan (channels);
  if (! built) {
    set_optimal_ndat();
    resize (1, nchan, ndat, 2);
    build();
  }
}

void dsp::DerippleResponse::match (
  const dsp::Observation* input, unsigned channels)
{
  prepare(input, channels);

  Response::match (input, channels);
}

void dsp::DerippleResponse::build ()
{
  if (built) {
    return;
  }
  std::vector<float> freq_response;

  calc_freq_response(freq_response, ndat);

  uint64_t npt = ndat*2;

  for (uint64_t ipt=0; ipt<npt; ipt++) {
    buffer[ipt] = freq_response[ipt];
  }
  built = true;

}

void dsp::DerippleResponse::set_optimal_ndat ()
{
  // if (frequency_resolution_set) {
  //   set_frequency_resolution(get_minimum_ndat());
  // } else {
  //   if (optimal_fft) {
  //     optimal_fft->set_simultaneous (nchan > 1);
  //   }
  //   Response::set_optimal_ndat ();
  // }
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::set_optimal_ndat" << std::endl;
  }
  Rational os = fir_filter.get_oversampling_factor();
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::set_optimal_ndat: fir_filter oversampling factor=" << os << std::endl;
  }
  // need to incorporate the fir_filter.pfb_nchan attribute
  ndat = os.normalize(ndat) * nchan;
}

void dsp::DerippleResponse::set_nchan (unsigned _nchan)
{
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::set_nchan ("<<_nchan<<")"<< std::endl;
  }
  if (_nchan != nchan) {
    built = false;
  }
  nchan = _nchan;
}

void dsp::DerippleResponse::set_frequency_resolution (unsigned nfft)
{
  if (verbose)
    std::cerr << "dsp::DerippleResponse::set_frequency_resolution ("<<nfft<<")"<< std::endl;
  resize (npol, nchan, nfft, ndim);

  frequency_resolution_set = true;
}

void dsp::DerippleResponse::set_times_minimum_nfft (unsigned times)
{
  if (verbose)
    std::cerr << "dsp::DerippleResponse::set_times_minimum_nfft ("<<times<<")"<<std::endl;

  times_minimum_nfft = times;
  frequency_resolution_set = true;
}

void dsp::DerippleResponse::calc_freq_response (
  std::vector<float>& freq_response, unsigned n_freq)
{
  // std::ofstream freq_response_file("freq_response.dat", std::ios::out | std::ios::binary);
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

  // freq_response_file.write(
  //     reinterpret_cast<const char*>(freq_response.data()),
  //     2*n_freq*sizeof(float)
  // );
  // freq_response_file.close();
}
