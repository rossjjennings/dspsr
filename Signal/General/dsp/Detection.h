//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Detection.h,v $
   $Revision: 1.8 $
   $Date: 2003/08/25 12:55:17 $
   $Author: wvanstra $ */


#ifndef __Detection_h
#define __Detection_h

class Detection;

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

namespace dsp {

  //! Detects phase-coherent TimeSeries data
  /*!  The Detection class may be used to perform simple square law
  detection or calculation of the Stokes parameters or the coherency
  matrix.  In the case of Stokes/Coherency formation, the components
  may be stored in polarization-major order or time-major order, or a
  mixture of the two by calling set_output_ndim() with an argument of
  4, 1, or 2, respectively.  The three methods require different
  amounts of RAM and therefore result in performance benefits that are
  largely cache-dependent. */

  //! It is recommended that user calls both set_output_ndim() and set_output_state() 

  class Detection : public Transformation <TimeSeries, TimeSeries> {

  public:
    
    //! Constructor
    Detection ();
    
    //! Set the state of the output data
    void set_output_state (Signal::State _state);
    //! Get the state of the output data
    Signal::State get_output_state () const { return state; }

    //! Set the dimension of the output data
    void set_output_ndim (int _ndim) { ndim = _ndim; }
    //! Get the dimension of the output data
    bool get_output_ndim () const { return ndim; }

  protected:

    //! Detect the input data
    virtual void transformation ();

    //! Signal::State of the output data
    Signal::State state;

    //! Dimension of the output data
    int ndim;

    //! Perform simple square-law detection
    void square_law ();

    //! Polarization detection (Stokes parameters or Coherency products)
    void polarimetry ();

    //! Set the state of the output TimeSeries
    void resize_output ();
  };

}

#endif // !defined(__Detection_h)
