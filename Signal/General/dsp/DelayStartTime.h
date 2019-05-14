//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_DelayStartTime_h
#define __baseband_dsp_DelayStartTime_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include "MJD.h"

#include <vector>

namespace dsp {

  class DelayStartTime : public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    DelayStartTime ();

    //! Configure the delay to the start of data in seconds
    void set_start_time (MJD);

    //! Computes the total delay and prepares the input buffer
    void prepare ();

    //! Prepares the output data buffer
    void prepare_output ();

    //! Applies the delays to the input
    void transformation ();

    //! Engine used to perform application of delays
    class Engine;
    void set_engine (Engine*);

  protected:

    //! The desired start time
    MJD start_mjd;

    //! The number of samples to delay
    int64_t delay_samples;

    //! Flag set when start_delay has been applied
    bool delay_applied;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  };

  class DelayStartTime::Engine : public Reference::Able
  {
  public:

    virtual void delay (const TimeSeries* in, TimeSeries* out,
                        uint64_t output_ndat, int64_t delay) = 0;

  };

}

#endif
