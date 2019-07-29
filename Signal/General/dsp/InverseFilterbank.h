//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson, Dean Shaff, and Willem van Straten,
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/InverseFilterbank.h

#ifndef __InverseFilterbank_h
#define __InverseFilterbank_h

#include <string>

#include "dsp/Convolution.h"
#include "dsp/Apodization.h"

namespace dsp {

  //! Performs the PFB inversion synthesis operation, combining multiple input
  //! frequency channels into a smaller number of output channels.
  //!
  //! The PFB inversion algorithm is fed a chunk of input data. The size of this
  //! chunk is determined by the upstream IOManager. For each input chunk, it does
  //! the following.
  //!   - Advance through the input data by `input_fft_length` minus
  //!   `input_discard_total`, or the number of samples skipped from the
  //!   start and end of each input step. If the `input_discard_total` is
  //!   greater than zero, this means that each successive step will be reading
  //!   from data that we've already operated on.
  //!   - For each channel in the input data step, apply an `input_fft_length` size
  //!   FFT. If there is an oversampling factor that is greater than one, discard
  //!   samples from either end of the resulting frequency domain data, before
  //!   stitching into a large array whose size will be `output_fft_length` long.
  //!   How these samples get stitched into this array is determined by the
  //!   precense of the zeroth PFB channel.
  //!   - Apply an inverse FFT to the stitched spectrum, before copying the
  //!   newly created upsampled time domain data into the output TimeSeries
  //!   buffer. In the same way that we read overlapped chunks of input data,
  //!   we also write out the output data in overlapped segments.
  //!
  //! The input TimeSeries can be critically sampled or over sampled.
  //! In the over sampled case, input spectra will be appropriately stitched
  //! together, discarding band edge overlap regions.
  //!
  //! The manner in which input channels are synthesized depends on the number
  //! of input channels that are present, and if the zeroth PFB channel is present.
  //! Here, the "reassembled" spectrum refers to the spectrum that gets assembled
  //! from forward FFTs operating on each input channel in the transformation
  //! step.
  //! There are three distinct scenarios:
  //!   - All PFB channels are present, and the zeroth, or DC PFB channel is present.
  //!   Here, the first half channel of the reassembled spectrum gets put at the end
  //!   of the spectrum; we "roll" the assembled spectrum by half a channel.
  //!   - The DC PFB channel is present, but we only have a subset of the channels
  //!   from upstream channelization. In this case, we discard the zeroth channel,
  //!   and append a half channel's worth of zeros to the end of the assembled
  //!   spectrum
  //!   - We have neither the DC PFB channel nor all the PFB channels. In this
  //!   case we leave the assembled spectrum as is.


  class InverseFilterbank: public Convolution {

  public:

    //! Configuration options
    class Config;

    //! Null constructor
    InverseFilterbank (const char* name = "InverseFilterbank", Behaviour type = outofplace);

    void set_input (const TimeSeries* input);

    //! Prepare all relevant attributes
    void prepare ();

    //! Reserve the maximum amount of output space required
    void reserve ();

    //! Get the minimum number of samples required for operation
    uint64_t get_minimum_samples () { return nsamp_fft; }

    //! Get the minimum number of samples lost
    uint64_t get_minimum_samples_lost () { return nsamp_overlap; }

    //! Set the number of input channels
    void set_input_nchan (unsigned _input_nchan) { input_nchan = _input_nchan; }

    //! Get the number of input channels
    unsigned get_input_nchan () const { return input_nchan; }

    //! Set the number of output channels.
    void set_output_nchan (unsigned _output_nchan) { output_nchan = _output_nchan; }

    //! Get the number of output channels
    unsigned get_output_nchan () const { return output_nchan; }

    //! Get the number of input channels per output channels
    unsigned get_nchan_subband () const {return nchan_subband; }

    //! Set the frequency resolution factor
    void set_freq_res (unsigned _freq_res) { freq_res = _freq_res; }

    //! Set the frequency resolution factor
    void set_frequency_resolution (unsigned _freq_res) { freq_res = _freq_res; }

    //! Get the frequency resolution factor
    unsigned get_freq_res () const { return freq_res; }

    //! Get the frequency resolution factor
    unsigned get_frequency_resolution () const { return freq_res; }

    //! Set the frequency overlap
    void set_frequency_overlap (unsigned over) { overlap_ratio = over; }

    //! Get the frequency overlap
    unsigned get_frequency_overlap () const { return (unsigned) overlap_ratio; }

    //! Set oversampling_factor
    void set_oversampling_factor (const Rational& _osf) { oversampling_factor = _osf; }

    //! Get oversampling_factor
    const Rational& get_oversampling_factor () { return oversampling_factor; }

    //! Get the `pfb_dc_chan` flag
    bool get_pfb_dc_chan () const { return pfb_dc_chan; }

    //! Set the `pfb_dc_chan` flag
    void set_pfb_dc_chan (bool _pfb_dc_chan) { pfb_dc_chan = _pfb_dc_chan; }

    //! Get the `pfb_all_chan` flag
    bool get_pfb_all_chan () const { return pfb_all_chan; }

    //! Set the `pfb_all_chan` flag
    void set_pfb_all_chan (bool _pfb_all_chan) { pfb_all_chan = _pfb_all_chan; }

    //! Get the number of samples discarded at the end of an input step
    int get_input_discard_neg() const {return input_discard_neg;}

    //! Set the number of samples discarded at the end of an input step
    void set_input_discard_neg(int _input_discard_neg) { input_discard_neg = _input_discard_neg;}

    //! Get the number of samples discarded at the start of an input step
    int get_input_discard_pos() const {return input_discard_pos;}

    //! Set the number of samples discarded at the start of an input step
    void set_input_discard_pos(int _input_discard_pos) { input_discard_pos = _input_discard_pos;}

    //! Get the number of samples discarded at the end of an output step
    int get_output_discard_neg() const {return output_discard_neg;}

    //! Set the number of samples discarded at the end of an output step
    void set_output_discard_neg(int _output_discard_neg) { output_discard_neg = _output_discard_neg;}

    //! Get the number of samples discarded at the start of an output step
    int get_output_discard_pos() const {return output_discard_pos;}

    //! Get the number of samples discarded at the start of an output step
    void set_output_discard_pos(int _output_discard_pos) { output_discard_pos = _output_discard_pos;}

    //! Get the size of the forward fft, in number of floats
    int get_input_fft_length() const {return input_fft_length;}

    //! Set the size of the forward fft, in number of floats
    void set_input_fft_length(int _input_fft_length) { input_fft_length = _input_fft_length;}

    //! Get the size of the backward fft, in number of floats
    int get_output_fft_length() const {return output_fft_length;}

    //! Set the size of the backward fft, in number of floats
    void set_output_fft_length(int _output_fft_length) { output_fft_length = _output_fft_length;}


    //! Engine used to perform discrete convolution step
    class Engine;
    void set_engine (Engine*);

    void set_fft_window_str (std::string _fft_window_str) { fft_window_str = _fft_window_str; }

    std::string get_fft_window_str () const { return fft_window_str ; }

    // void optimize_discard_region(
    //   int* _input_discard_pos,
    //   int* _input_discard_neg,
    //   int* _output_discard_neg,
    //   int* _output_discard_pos);
    //
    // void optimize_fft_length(
    //   int* _input_fft_length,
    //   int* _output_fft_length);

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Perform the filterbank step
    virtual void filterbank ();

    //! Override this in a child class to do any custom pre-transformation
    //! data preparation
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


    //! This flag indicates whether we have the DC, or zeroth PFB channel.
    bool pfb_dc_chan;

    //! This flag indicates whether we have all the channels from the last
    //! stage of upstream channelization.
    bool pfb_all_chan;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! string representing the fft window to be used
    std::string fft_window_str;

  private:

    // div_t calc_lcf (int a, int b, Rational osf);

    //! Set the input and output fft lengths and discard regions according
    //! to the response. If there is an associated buffering policy, use the
    //! discard regions to set the minimum number of samples for the policy.
    //! Setup the engine.
    void make_preparations ();

    //! prepare the output TimeSeries
    void prepare_output (uint64_t ndat = 0, bool set_ndat = false);

    //! Determine the number of steps in a given TimeSeries.
    void resize_output (bool reserve_extra = false);

    //! The size of the forward FFT used in the Engine, in number of floats.
    int input_fft_length;

    //! The size of the backward FFT used in the Engine, in number of floats.
    int output_fft_length;

    //! The total number of floats discarded in a given input TimeSeries step.
    int input_discard_total;

    //! The number samples in an input TimeSeries step, or segment.
    int input_sample_step;

    //! The number of floats discarded in a given output TimeSeries step.
    int output_discard_total;

    //! The number of samples in an input TimeSeries step, or segment.
    int output_sample_step;

    //! The number of samples discarded at the start of an input TimeSeries
    int input_discard_pos;
    //! The number of samples discarded at the end of an input TimeSeries
    int input_discard_neg;

    //! The number of samples discarded at the start of an output TimeSeries
    int output_discard_pos;
    //! The number of samples discarded at the end of an output TimeSeries
    int output_discard_neg;


  };

}

#endif
