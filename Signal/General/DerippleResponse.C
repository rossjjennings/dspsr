/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <vector>
#include <iostream>

#include "dsp/DerippleResponse.h"

void dsp::DerippleResponse::calc_freq_response (unsigned n_freq) {

  // std::ofstream freq_response_file("freq_response.dat", std::ios::out | std::ios::binary);

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
