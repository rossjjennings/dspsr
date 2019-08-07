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
  class Apodization : public Shape {

  public:
    enum Type { none, hanning, welch, parzen, tukey, top_hat };

    //! Null constructor
    Apodization();

    //! Create a Hanning window function
    void Hanning (int npts, bool analytic);

    //! Create a Welch window function
    void Welch   (int npts, bool analytic);

    //! Create a Parzen window function
    // Note that this is not actually a Parzen window
    void Parzen  (int npts, bool analytic);

    //! Create Tukey window function.
    //! A Tukey window is a top hat window with a Hann transition band.
    //! In other words, instead of abruptly transitioning to zero, it uses
    //! a Hann window to transition to zero.
    //! stop_band is the number of points *on each side* that are not part
    //! of the passband.
    //! transition_band is the number of points on each side that form the
    //! Hann transition area.
    void Tukey (int npts, int stop_band, int transition_band, bool analytic);

    //! Create top hat window function.
    void TopHat (int npts, int stop_band, bool analytic);

    //! Create empty window function
    void None (int npts, bool analytic);

    //! Create a window with the specified shape
    void set_shape (int npts, Type type, bool analytic, int stop_band = 0, int transition_band = 0);

    //! make the integrated total of the window equal to one
    void normalize();

    Type getType () { return type; };

    Type get_type () { return getType(); };

    //! Multiply indata by the window function
    void operate (float* indata, float* outdata = 0) const;

    //! Returns SUM i=1..N {window[i] * data[i]}
    double integrated_product (float* data, unsigned incr=1) const;

    static std::map<std::string, Type> type_map;

  protected:
    Type type;

  private:

    static std::map<std::string, Type> init_type_map ()
    {
      static std::map<std::string, Type> _type_map;
      _type_map["no_window"] = none;
      _type_map["none"] = none;
      _type_map["tukey"] = tukey;
      _type_map["hanning"] = hanning;
      _type_map["parzen"] = parzen;
      _type_map["welch"] = welch;
      _type_map["top_hat"] = top_hat;
      return _type_map;
    }

  };
}

#endif
