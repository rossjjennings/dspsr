//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Convolution.h

#ifndef __Convolution_h
#define __Convolution_h

#include "dsp/Response.h"
#include "dsp/ResponseProduct.h"
#include "dsp/ScalarFilter.h"

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "FTransformAgent.h"

namespace dsp {

  class Apodization;

  //! Convolves a TimeSeries using a frequency response function
  /*! This class implements the overlap-save method of discrete
    convolution with a finite impulse response (FIR) function, as
    described in Chapter 13.1 of Numerical Recipes.

    The algorithm can perform both scalar and matrix convolution
    methods, and is highly suited to phase-coherent dispersion removal
    and phase-coherent polarimetric calibration.

    If g(t) is the finite impulse response function with which the
    data stream will be convolved, then the Convolution::response
    attribute represents G(w), the FFT of g(t).  Convolution::response
    may contain an array of filters, one for each frequency channel.

    In order to improve the spectral leakage characteristics, an
    apodization function may be applied to the data in the time domain
    by setting the Convolution::apodization attribute.

    Referring to Figure 13.1.3 in Numerical Recipes,
    \f$m_+\f$=response->get_impulse_pos() and
    \f$m_-\f$=response->get_impulse_neg(), so that the duration,
    M=\f$m_+ + m_-\f$, of g(t) corresponds to the number of complex
    time samples in the result of each backward FFT that are polluted
    by the cyclical convolution transformation.  Consequently,
    \f$m_+\f$ and \f$m_-\f$ complex samples are dropped from the
    beginning and end, respectively, of the result of each backward
    FFT; neighbouring FFTs will overlap by the appropriate number of
    points to make up for this loss.  */

  class Convolution: public Transformation <TimeSeries, TimeSeries> {

  public:

    class Config;

    //! Null constructor
    Convolution (const char* name = "Convolution", Behaviour type = outofplace);

    //! Destructor
    virtual ~Convolution ();

    //! Prepare all relevant attributes
    void prepare ();

    //! Reserve the maximum amount of output space required
    void reserve ();

    //! Get the minimum number of samples required for operation
    uint64_t get_minimum_samples () { return nsamp_fft; }

    //! Get the minimum number of samples lost
    uint64_t get_minimum_samples_lost () { return nsamp_overlap; }

    //! Return a descriptive string
    //virtual const string descriptor () const;

    //! Initialize from a descriptor string as output by above
    //virtual void initialize (const string& descriptor);

    //! Set the frequency response function
    virtual void set_response (Response* response);

    //! Set the apodization function
    virtual void set_apodization (Apodization* function);

    //! Set the passband integrator
    virtual void set_passband (Response* passband);

    //! Return true if the response attribute has been set
    bool has_response () const;

    //! Return a pointer to the frequency response function
    virtual const Response* get_response() const;
    virtual Response* get_response();

    //! Return true if the passband attribute has been set
    bool has_passband () const;

    //! Return a pointer to the integrated passband
    virtual const Response* get_passband() const;
    virtual Response* get_passband();

    //! Return true if the apodization attribute has been set
    bool has_apodization() const;

    //! Return a pointer to to the apodization object
    virtual const Apodization* get_apodization() const;
    virtual Apodization* get_apodization();

    //! get the matrix_convolution flag
    bool get_matrix_convolution () const { return matrix_convolution; };

    //! Set the memory allocator to be used
    void set_device (Memory *);

    //! Engine used to perform discrete convolution step
    class Engine;

    void set_engine (Engine*);

    Engine* get_engine();

    //! get the zero_DM flag
    bool get_zero_DM () const { return zero_DM; }

    //! set the zero_DM flag
    void set_zero_DM (bool _zero_DM) { zero_DM = _zero_DM; }

    //! Return true if the zero_DM_output attribute has been set
    bool has_zero_DM_output () const;

    //! Set the zero_DM_output TimeSeries object
    virtual void set_zero_DM_output (TimeSeries* zero_DM_output);

    //! Return a pointer to the zero_DM_output TimeSeries object
    virtual const TimeSeries* get_zero_DM_output() const;
    virtual TimeSeries* get_zero_DM_output();


    //! Return true if the zero DM response attribute has been set
    bool has_zero_DM_response () const;

    //! Return a pointer to the zero DM frequency response function
    virtual const Response* get_zero_DM_response() const;
    virtual Response* get_zero_DM_response();

    //! Set the zero DM frequency response function
    virtual void set_zero_DM_response (Response* response);


  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Scalar filter (normalizer)
    Reference::To<ScalarFilter> normalizer;

    //! Frequency response (convolution kernel)
    Reference::To<Response> response;

    //! Frequency response to use in zero DM case
    Reference::To<Response> zero_DM_response;

    //! Product of response and normaliser
    Reference::To<ResponseProduct> response_product;

    //! Product of response and normaliser
    Reference::To<ResponseProduct> zero_dm_response_product;

    //! Apodization function (time domain window)
    Reference::To<Apodization> apodization;

    //! Integrated passband
    Reference::To<Response> passband;

    //! Prepare the output TimeSeries
    void prepare_output ();

    //! zero DM flag -- this indicates whether to do a parallel transformation
    //! without any dedispersion
    bool zero_DM;

    //! zero DM output timeseries from convolution
    Reference::To<dsp::TimeSeries> zero_DM_output;

  private:

    friend class Filterbank;
    friend class InverseFilterbank;
    friend class TFPFilterbank;
    friend class SKFilterbank;

    Reference::To<Memory> memory;

    unsigned nfilt_tot;
    unsigned nfilt_pos;
    unsigned nfilt_neg;

    unsigned nsamp_overlap;
    unsigned nsamp_step;
    unsigned nsamp_fft;

    double scalefac;

    bool matrix_convolution;

    FTransform::Plan* forward;
    FTransform::Plan* backward;

    unsigned scratch_needed;
    uint64_t npart;
    unsigned n_fft;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;
  };
}

class dsp::Convolution::Engine : public Reference::Able
{
  public:

    virtual void set_scratch (void *) = 0;

    virtual void prepare (dsp::Convolution * convolution) = 0;

    virtual void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out, unsigned npart) = 0;

    virtual void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out, dsp::TimeSeries* zero_DM_out, unsigned npart) = 0;
};

#endif
