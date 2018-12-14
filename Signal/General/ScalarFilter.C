/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/ScalarFilter.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Input.h"

using namespace std;

dsp::ScalarFilter::ScalarFilter ()
{
  scale_factor = 1.0;
  calculated = false;
}

dsp::ScalarFilter::~ScalarFilter ()
{
}

//! Create an Scalar filter with nchan channels
void dsp::ScalarFilter::match (const Observation* obs, unsigned nchan)
{
  unsigned npol = 1;
  unsigned ndat = 1;
  unsigned ndim = 1;

  resize (npol, nchan, ndat, ndim);
}

void dsp::ScalarFilter::calculate (Response* bp)
{
  unsigned nchan_bp = bp -> get_ndat();

  float* p0ptr = bp->get_datptr (0, 0);
  float* p1ptr = bp->get_datptr (0, 1);

  // form the total intensity bandpass
  vector<float> spectrum (nchan_bp);
  unsigned ichan=0;

  for (ichan=0; ichan < nchan_bp; ichan++)
    spectrum[ichan] = p0ptr[ichan]+p1ptr[ichan];

  fft::median_smooth (spectrum, median_window);

  double variance = 0.0;
  for (ichan=0; ichan < nchan_bp; ichan++)
  {
    spectrum[ichan] -= (p0ptr[ichan]+p1ptr[ichan]);
    spectrum[ichan] *= spectrum[ichan];
    // p0ptr[ichan] = spectrum[ichan];
    variance += spectrum[ichan];
  }

  variance /= nchan_bp;

  resize (1, 1, nchan_bp, 1);
  float* ptr = get_datptr(0,0);

  bool zapped = true;
  unsigned round = 1;
  unsigned total_zapped = 0;

  while (zapped)  {

    float cutoff = 16.0 * variance;
    cerr << "\tround " << round << " cutoff = " << cutoff << endl;

    zapped = false;
    round ++;

    for (ichan=0; ichan < nchan_bp; ichan++)
      if (spectrum[ichan] > cutoff ||
          (ichan && fabs(spectrum[ichan]-spectrum[ichan-1]) > 2*cutoff)) {
        variance -= spectrum[ichan]/nchan_bp;
        spectrum[ichan] = p0ptr[ichan] = p1ptr[ichan] = ptr[ichan] = 0.0;
	total_zapped ++;
        zapped = true; 
      }
      else
        ptr[ichan] = 1;

  }

  cerr << "\tzapped " << total_zapped << " channels" << endl;
  calculated = true;
}

//! Create an Scalar filter with the same number of channels as Response
void dsp::ScalarFilter::match (const Response* response)
{
  if (verbose)
    cerr << "dsp::ScalarFilter::match Response nchan=" << response->get_nchan()
	 << " ndat=" << response->get_ndat() << endl;

  if ( get_nchan() == response->get_nchan() &&
       get_ndat() == response->get_ndat() ) {

    if (verbose)
      cerr << "dsp::ScalarFilter::match Response already matched" << endl;
    return;

  }

  unsigned required = response->get_nchan() * response->get_ndat();
  unsigned expand = required / get_ndat();
  unsigned shrink = get_ndat() / required;

  vector< complex<float> > phasors (required, 1.0);

  float* data = get_datptr(0,0);

  if (expand)
    for (unsigned idat=0; idat < get_ndat(); idat++)
      for (unsigned ip=0; ip < expand; ip++)
        phasors[idat*expand+ip] = data[idat];
  else if (shrink)
    for (unsigned idat=0; idat < required; idat++)
      for (unsigned ip=0; ip < expand; ip++)
        if (data[idat*shrink+ip] == 0.0)
          phasors[idat] = 0.0;
  else
    throw Error (InvalidState, "dsp::ScalarFilter::match Response",
                 "not matched and not able to shrink or expand");

  set (phasors);

  resize (1, response->get_nchan(),
	  response->get_ndat(), 2);

}

//! Set the number of channels into which the band will be divided
void dsp::ScalarFilter::set_nchan (unsigned nchan)
{

}

//! Set the interval over which the RFI mask will be calculated
void dsp::ScalarFilter::set_update_interval (double seconds)
{

}

//! Set the fraction of the data used to calculate the RFI mask
void dsp::ScalarFilter::set_duty_cycle (float cycle)
{

}

//! Set the source of the data
void dsp::ScalarFilter::set_input (IOManager* _input)
{
  input = _input;
}

//! Set the buffer into which data will be read
void dsp::ScalarFilter::set_buffer (TimeSeries* _buffer)
{
  buffer = _buffer;
}

//! Set the buffer into which the spectra will be integrated [optional]
void dsp::ScalarFilter::set_data (Response* _data)
{
  data = _data;
}

//! Set the tool used to compute the spectra [optional]
void dsp::ScalarFilter::set_bandpass (Bandpass* _bandpass)
{
  bandpass = _bandpass;
}
