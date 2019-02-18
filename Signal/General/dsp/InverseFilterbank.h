//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/InverseFilterbank.h

#ifndef __InverseFilterbank_h
#define __InverseFilterbank_h

#include "dsp/Convolution.h"
// #include "Rational.h"

namespace dsp {

  //! Performs a synthesis operation, combining multiple input frequency channels
  //! into a smaller number of output channels.
  //! Input can be critically sampled or over sampled.
  //! In the over sampled case, input spectra will be appropriately stitched
  //! together, discarding band edge overlap regions.

  class InverseFilterbank: public Convolution {

  public:

    //! Configuration options
    class Config;

    //! Null constructor
    InverseFilterbank (const char* name = "InverseFilterbank", Behaviour type = outofplace);

    //! Prepare all relevant attributes
    void prepare ();

    //! Reserve the maximum amount of output space required
    void reserve ();

    //! Get the minimum number of samples required for operation
    uint64_t get_minimum_samples () { return nsamp_fft; }

    //! Get the minimum number of samples lost
    uint64_t get_minimum_samples_lost () { return nsamp_overlap; }

    //! Set the number of channels into which the input will be divided
    // void set_nchan (unsigned _nchan) { nchan = _nchan; }

    //! set/get input_nchan
    void set_input_nchan (unsigned _input_nchan) { input_nchan = _input_nchan; }
    unsigned get_input_nchan () const { return input_nchan; }

    //! set/get output_nchan
    void set_output_nchan (unsigned _output_nchan) { output_nchan = _output_nchan; }
    unsigned get_output_nchan () const { return output_nchan; }

    unsigned get_nchan_subband () const {return nchan_subband; }

    //! Set the frequency resolution factor
    void set_freq_res (unsigned _freq_res) { freq_res = _freq_res; }
    void set_frequency_resolution (unsigned fres) { freq_res = fres; }

    //! Get the frequency resolution factor
    unsigned get_freq_res () const { return freq_res; }
    unsigned get_frequency_resolution () const { return freq_res; }

    void set_frequency_overlap (unsigned over) { overlap_ratio = over; }
    unsigned get_frequency_overlap () const { return (unsigned) overlap_ratio; }

    void set_oversampling_factor (const Rational& _osf) { oversampling_factor = _osf; }

    const Rational& get_oversampling_factor () {return input->get_oversampling_factor();}

    int get_input_discard_neg() const {return input_discard_neg;}

    int get_input_discard_pos() const {return input_discard_pos;}

    int get_output_discard_neg() const {return output_discard_neg;}

    int get_output_discard_pos() const {return output_discard_pos;}

    int get_input_fft_length() const {return input_fft_length;}

    int get_output_fft_length() const {return output_fft_length;}


    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Perform the filterbank step
    virtual void filterbank ();
    virtual void custom_prepare () {}

    //! number of input channels
    unsigned input_nchan;

    //! Number of channels into which the input will be synthesized
    //! This is the final number of channels in the output
    unsigned output_nchan;

    //! Frequency resolution factor
    unsigned freq_res;

    //! This is the number of input channels per output channel,
    //! or input_nchan / output_nchan
    unsigned nchan_subband;

    //! Frequency channel overlap ratio
    double overlap_ratio;

    //! Polyphase filterbank oversampling ratio.
    //! This will be 1/1 for critically sampled,
    //! and some number greater than 1 for over sampled case
    Rational oversampling_factor;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  private:

    void optimize_discard_region(
      int* _input_discard_pos,
      int* _input_discard_neg,
      int* _output_discard_neg,
      int* _output_discard_pos);

    void optimize_fft_length(
      int* _input_fft_length,
      int* _output_fft_length);

    div_t calc_lcf (int a, int b, Rational osf);

    void make_preparations ();
    void prepare_output (uint64_t ndat = 0, bool set_ndat = false);
    void resize_output (bool reserve_extra = false);

    int input_fft_length;
    int output_fft_length;

    int input_discard_total;
    int input_sample_step;

    int output_discard_total;
    int output_sample_step;

    int input_discard_pos;
    int input_discard_neg;
    int output_discard_neg;
    int output_discard_pos;


  };

}

#endif
