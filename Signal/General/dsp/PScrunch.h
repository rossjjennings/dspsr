//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_PScrunch_h
#define __baseband_dsp_PScrunch_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp
{
  //! PScrunch all channels and polarizations
  class PScrunch : public Transformation<TimeSeries,TimeSeries>
  {

  public:

    //! Default constructor
    PScrunch ();

    //! set the output polarisation state for the Pscrunch operation
    void set_output_state(Signal::State state);

    //! prepare the transformation
    void prepare();

    //! prepare the output time-series for the transformation
    void prepare_output();

    //! combine the polaristionas in the input time series to form the output time series
    void transformation ();

    class Engine;

    //! set the alternate processing engine to be used
    void set_engine (Engine*);

  protected:

    //! alternate processing engine
    Reference::To<Engine> engine;

  private:

    //! factor by which the output is scaled
    float sfactor;

    //! number of polarisations in the output
    unsigned output_npol;

    //! output state for the transformation
    Signal::State state;

  };

  class PScrunch::Engine : public OwnStream
  {
  public:

    virtual void setup () = 0;

    virtual void fpt_pscrunch (const dsp::TimeSeries * in,
                               dsp::TimeSeries * out) = 0;

    virtual void tfp_pscrunch (const dsp::TimeSeries* in,
                               dsp::TimeSeries* out) = 0;

   };

}

#endif
