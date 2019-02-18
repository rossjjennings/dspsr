//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// Recall that PSRFITS search mode data are in TPF order, which complicates
// packing into bytes.

#ifndef __FITSDigitizer_h
#define __FITSDigitizer_h

#include "dsp/Digitizer.h"

namespace dsp
{  
  //! Converts floating point values to N-bit PSRFITS search-mode format
  class FITSDigitizer: public Digitizer
  {
  public:

    //! Default constructor
    FITSDigitizer (unsigned _nbit);

    //! Default destructor
    ~FITSDigitizer ();

    unsigned get_nbit () const {return nbit;}

    //! Set the number of samples to rescale before digitization.
    //! The default is 0, i.e. rescaling must be done elsewhere.
    void set_rescale_samples (unsigned nsamp);

    //! Set the number of blocks to remember when computing scales.
    //! The default is 1, corresponding to no memory.
    void set_rescale_nblock (unsigned nsamp);

    //! If true, leave scales/offsets constant after first measurement.
    void set_rescale_constant (bool rconst);

    //! Set the channel ordering of the output
    void set_upper_sideband_output (bool usb) { upper_sideband_output = usb; };

    //virtual void transformation ();

    //! Pack the data
    void pack ();

    //! Return minimum samples
    // TODO -- is this needed?
    uint64_t get_minimum_samples () { return 2048; }

    void get_scales (std::vector<float>* dat_scl, std::vector<float>* dat_offs);

    Callback<FITSDigitizer*> update;

    class Engine;

    void set_engine (Engine*);

  protected:

    void set_nbit (unsigned);

    //! rescale input based on mean / variance
    void rescale_pack ();

    void init ();
    void measure_scale ();

    void set_digi_scales();

    //! keep track of first time through scale-measuring procedure
    unsigned rescale_nsamp;
    unsigned rescale_idx;
    unsigned rescale_nblock;
    unsigned rescale_counter;

    //! Keep scaling/offset constant after first estimate.
    bool rescale_constant;

    float digi_mean,digi_scale;
    int digi_min,digi_max;

    //! arrays for accumulating and storing scales
    double *freq_totalsq, *freq_total, *scale, *offset;

    Reference::To<Engine> engine;

  protected:

    bool upper_sideband_output;

  };

  class ChannelSort
  {
    const bool flip_band;
    const bool swap_band;
    const unsigned nchan;
    const unsigned half_chan;
    const dsp::Observation* input;

  public:

    ChannelSort (const dsp::Observation* input, bool upper_sideband) :
      flip_band ((upper_sideband && input->get_bandwidth() < 0) ||
                 (!upper_sideband && input->get_bandwidth() > 0)),
      swap_band ( input->get_swap() ),
      nchan ( input->get_nchan() ),
      half_chan ( nchan / 2 ),
      input ( input ) { }

    //! Return the mapping from output channel to input channel
    inline unsigned operator () (unsigned out_chan)
    {
      unsigned in_chan = out_chan;
      if (flip_band)
        in_chan = (nchan-in_chan-1);
      if (input->get_nsub_swap() > 1)
        in_chan = input->get_unswapped_ichan(out_chan);
      else if (swap_band)
        in_chan = (in_chan+half_chan)%nchan;
      return in_chan;
    }
  };

  class FITSDigitizer::Engine : public OwnStream
  {
  public:

    virtual void set_scratch (dsp::Scratch *) = 0;

    virtual void set_rescale_nblock (const dsp::TimeSeries * in, 
                                     unsigned rescale_nblock) = 0;

    virtual void set_mapping (const dsp::TimeSeries * in,
                              dsp::ChannelSort& channel) = 0;

    virtual void measure_scale (const dsp::TimeSeries * in,
                                unsigned rescale_nsamp) = 0;

    virtual void digitize (const dsp::TimeSeries * in,
                           dsp::BitSeries * out, uint64_t ndat, unsigned nbit,
                           float digi_mean, float digi_scale,
                           int digi_min, int digi_max) = 0;

    virtual void get_scale_offsets (double * scale, double * offset,
                                    unsigned nchan, unsigned npol) = 0;

  };

}

#endif
