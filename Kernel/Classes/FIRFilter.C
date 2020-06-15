/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FIRFilter.h"

dsp::FIRFilter::FIRFilter () {}

dsp::FIRFilter::FIRFilter (
    std::vector<float> _coeff,
    const Rational& _osf,
    unsigned _pfb_nchan
) {
  coeff = _coeff;
  oversamp = _osf;
  pfb_nchan = _pfb_nchan;
}

void dsp::FIRFilter::set_ntaps (unsigned ntaps) {
  coeff.resize(ntaps);
}

float dsp::FIRFilter::operator[] (int i) const {
  return coeff[i];
}

float& dsp::FIRFilter::operator[] (int i) {
  return coeff[i];
}
