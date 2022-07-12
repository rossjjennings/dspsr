//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Apodization_h
#define __Apodization_h

#include <map>
#include <string>

#include "dsp/Shape.h"

namespace dsp {

  //! Various apodizing (window) functions
  /* The Apodization class implements apodizing functions that may be used
     in the time domain before performing an FFT, in order to improve
     the spectral leakage characteristics */
  class Apodization : public Shape
  {

  public:

    enum Type { none, hanning, welch, bartlett, tukey, top_hat };

    //! Null constructor
    Apodization();

    //! Set the type of window function by name
    void set_type (const std::string&);

    //! Set true when the data to be tapered are complex-valued
    void set_analytic (bool f = true) { analytic = f; }
    bool get_analytic () const { return analytic; }

    //! Set the number of samples in the window
    void set_size (unsigned);

    //! Number of samples in the transition region on the start
    void set_transition_start (unsigned n) { transition_start = n; }
    //! Number of samples in the transition region on the end
    void set_transition_end (unsigned n) { transition_end = n; }
    void set_transition (unsigned n) { transition_end = transition_start = n; }

    //! Build the specified window
    void build ();

    //! Create a Hanning window function
    void Hanning ();

    //! Create a Welch window function
    void Welch ();

    //! Create a Bartlett window function
    void Bartlett ();

    //! Create Tukey window function.
    /*! A Tukey window is a top hat window with a Hann transition band.
        In other words, instead of abruptly transitioning to zero, it uses
        a Hann window to transition to zero. */
    void Tukey ();

    //! Create top hat window function.
    void TopHat ();

    //! make the integrated total of the window equal to one
    void normalize();

    Type getType () { return type; };

    Type get_type () { return getType(); };

    //! Multiply indata by the window function
    void operate (float* indata, float* outdata = 0) const;

    //! Returns SUM i=1..N {window[i] * data[i]}
    double integrated_product (float* data, unsigned incr=1) const;

    //! Write the window to a text file
    /*! One row per value.  Two columns: index value */
    void dump (const std::string& filename);

  private:

    static std::map<std::string, Type> type_map;

    //! The shape of the window
    Type type;

    //! If true, operate assumes that input and output are complex valued
    bool analytic;

    //! Number of samples zeroed on the start
    unsigned zero_start;
    //! Number of samples zeroed on the end
    unsigned zero_end;

    //! Number of samples in the transition region on the start
    unsigned transition_start;
    //! Number of samples in the transition region on the end
    unsigned transition_end;

    static std::map<std::string, Type> init_type_map ()
    {
      static std::map<std::string, Type> _type_map;
      _type_map["no_window"] = none;
      _type_map["none"] = none;
      _type_map["bartlett"] = bartlett;
      _type_map["hanning"] = hanning;
      _type_map["tukey"] = tukey;
      _type_map["welch"] = welch;
      _type_map["top_hat"] = top_hat;
      return _type_map;
    }

  };
}

#endif
