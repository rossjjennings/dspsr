//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
 // dspsr/Signal/General/dsp/DerippleResponse.h

#ifndef __DerippleResponse_h
#define __DerippleResponse_h

#include <vector>
#include <cstring>

#include "dsp/Response.h"
#include "dsp/FIRFilter.h"

namespace dsp {
  class DerippleResponse : public Response {

  public:

    //! Default constructor
    DerippleResponse ();

    //! Destructor
    ~DerippleResponse ();

    //! Set the dimensions of the data, updating built attribute
    void resize(unsigned _npol, unsigned _nchan,
                unsigned _ndat, unsigned _ndim);

    void match (const Observation* input, unsigned channels, const Rational& osf);

    void match (const Observation* input, unsigned channels);

    //! Create a DerippleResponse with the same number of channels as Response
    void match (const Response* response);

    //! Set the number of input channels
    void set_nchan (unsigned _nchan);

    //! Set the length of the frequency response for each input channel
    void set_ndat (unsigned _ndat);

    //! build the frequency response corresponding to the FIR filter
    void build ();

    //! Calculate the frequency response, filling up freq_response vector.
    void calc_freq_response (std::vector<float>& freq_response, unsigned n_freq);

    //! set the FIR filter coefficients
    void set_fir_filter (const FIRFilter& _fir_filter) { fir_filter = _fir_filter; }

    //! get the FIR filter coefficients
    const FIRFilter& get_fir_filter () const { return fir_filter; }

    //! get the pfb_dc_chan flag
    const bool get_pfb_dc_chan () const { return pfb_dc_chan; }

    //! set the pfb_dc_chan flag
    void set_pfb_dc_chan (bool _pfb_dc_chan) { pfb_dc_chan = _pfb_dc_chan; }

  protected:

    //! Roll array `arr` by `shift` number of points
    template<typename T>
    void roll (std::vector<T>& arr, int shift);

    template<typename T>
    void roll (T* arr, unsigned len, int shift);

    //! FIR filter that contains time domain filter coefficients
    FIRFilter fir_filter;

    //! FFT plan for computing frequency response of FIR filter
    FTransform::Plan* forward;

    //! flag indicating whether frequency response has been built
    bool built;

    //! flag indicating whether the DC, or zeroth PFB channel is present
    //! in the input data. If it is present, then the response must be
    //! rolled by a half channel.
    bool pfb_dc_chan;
  };
}

template<typename T>
void dsp::DerippleResponse::roll (T* buffer, unsigned len, int shift) {
  if (verbose) {
    std::cerr << "dsp::DerippleResponse::roll"
      << " buffer=" << buffer
      << " len=" << len
      << " shift=" << shift
      << std::endl;
  }
	unsigned abs_shift = static_cast<unsigned>(abs(shift));
  T* scratch = new T[abs_shift];

	if (shift == 0) {
		return;
	}
	if (shift > 0) {
		std::memcpy(scratch, buffer + len - abs_shift, abs_shift*sizeof(T));
		std::memcpy(buffer + abs_shift, buffer, (len - abs_shift)*sizeof(T));
		std::memcpy(buffer, scratch, abs_shift*sizeof(T));
	} else {
		std::memcpy(scratch, buffer, abs_shift*sizeof(T));
		std::memcpy(buffer, buffer + abs_shift, (len - abs_shift)*sizeof(T));
		std::memcpy(buffer + len - abs_shift, scratch, abs_shift*sizeof(T));
	}
	delete [] scratch;
}

template<typename T>
void dsp::DerippleResponse::roll (std::vector<T>& arr, int shift) {
	unsigned len = arr.size();
	T* buffer = arr.data();
	roll<T> (buffer, len, shift);
}

#endif
