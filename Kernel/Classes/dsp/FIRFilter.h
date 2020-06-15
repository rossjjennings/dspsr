//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FIRFilter.h

#ifndef __FIRFilter_h
#define __FIRFilter_h

#include <vector>
#include "Rational.h"

namespace dsp {
  class FIRFilter  {

  public:

    FIRFilter ();

    FIRFilter (std::vector<float> _coeff, const Rational& _osf, unsigned _pfb_nchan);

    void set_oversamp (const Rational& _osf) { oversamp = _osf; }
    void set_oversampling_factor (const Rational& _osf) { return set_oversamp(_osf); }

    const Rational& get_oversamp () const { return oversamp; }
    const Rational& get_oversampling_factor () const { return get_oversamp(); }

    void set_pfb_nchan (unsigned _pfb_nchan)  { pfb_nchan = _pfb_nchan; }

    unsigned get_pfb_nchan () const { return pfb_nchan; }

    unsigned get_ntaps () const { return coeff.size(); }

    void set_ntaps (unsigned ntaps);

    const std::vector<float> get_coeff () const { return coeff; }

    std::vector<float>* get_coeff () { return &coeff; }

    void set_coeff (std::vector<float> _coeff) { coeff = _coeff; }

    // struct Deref {
    //   FIRFilter& filter;
    //   int index;
    //   Deref(FIRFilter& filter, int index) : filter(filter), index(index) {}
    //
    //   operator float() {
    //     return filter.coeff[index];
    //   }
    //
    //   float& operator= (const float& new_val) {
    //     return filter.coeff[index] = new_val;
    //   }
    // }
    // Deref operator[] (int i);
    float operator[](int i) const;
    float& operator[](int i);


  protected:

    //! oversampling factor associated with the filter.
    Rational oversamp;

    // number of associated output channels for this later of filtering
    unsigned pfb_nchan;

    std::vector<float> coeff;

  };
}


#endif
