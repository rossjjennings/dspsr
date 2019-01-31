/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson and Willem van Staten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/ScalarFilter.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Input.h"

using namespace std;

dsp::ScalarFilter::ScalarFilter ()
{
  if (verbose)
    cerr << "dsp::ScalarFilter::ScalarFilter()" << endl;
  npol = 1;
  ndat = 1;
  ndim = 2;
  nchan = 0;
  scale_factor = 1.0;
  built = false;
}

dsp::ScalarFilter::~ScalarFilter ()
{
}

//! Set the dimensions of the data and update the built attribute
void dsp::ScalarFilter::resize (unsigned _npol, unsigned _nchan,
        unsigned _ndat, unsigned _ndim)
{
  if (verbose)
    cerr << "dsp::ScalarFilter::resize(" << _npol << "," << _nchan 
         << "," << _ndat << "," << _ndim << ")" << endl;
  if (npol != _npol || nchan != _nchan || ndat != _ndat || ndim != _ndim)
  {
    built = false;
  }
  Shape::resize (_npol, _nchan, _ndat, _ndim);
}

//! Set the length of the frequnecy response in each channel
void dsp::ScalarFilter::set_ndat (unsigned _ndat)
{
  if (ndat != _ndat)
    built = false;
  ndat = _ndat;
}

//! Set the number of input channels 
void dsp::ScalarFilter::set_nchan (unsigned _nchan)
{
  if (nchan != _nchan)
    built = false;
  nchan = _nchan;
}

//! Set the scale factor to be applied by the filter
void dsp::ScalarFilter::set_scale_factor (float _scale_factor)
{
  if (verbose)
    cerr << "dsp::ScalarFilter::set_scale_factor factor=" << _scale_factor << endl;
  if (scale_factor != _scale_factor)
    built = false;
  scale_factor = _scale_factor;
}

float dsp::ScalarFilter::get_scale_factor ()
{
  return scale_factor;
}

float dsp::ScalarFilter::get_scale_factor () const
{
  return scale_factor;
}


void dsp::ScalarFilter::build ()
{
  if (built)
    return;

  resize (npol, nchan, ndat, ndim);

  // calculate the complex response of the scalar
  complex<float>* phasors = reinterpret_cast< complex<float>* > ( buffer );
  uint64_t npt = ndat * nchan;

  if (verbose)
    cerr << "dsp::ScalarFilter::build scale_factor=" << scale_factor << endl;

  for (unsigned ipt=0; ipt<npt; ipt++)
    phasors[ipt] = complex<float> (scale_factor, 0.0);
  built = true;
}

//! Create an Scalar Filter with nchan channels
void dsp::ScalarFilter::match (const Observation* obs, unsigned channels)
{
  if (verbose)
    cerr << "dsp::ScalarFilter::match channels=" << channels << endl;

  if (!channels)
    channels = obs->get_nchan();
  
  if (verbose)
    cerr << "dsp::ScalarFilter::match set_nchan(" << channels << ")" << endl;
  set_nchan (channels);

  if (!built)
  {
    build();
  }
}

//! Create an Scalar Filter with the same number of channels as Response
void dsp::ScalarFilter::match (const Response* response)
{
  if (verbose)
    cerr << "dsp::ScalarFilter::match Response nchan=" << response->get_nchan()
         << " ndat=" << response->get_ndat() << endl;

  if ( get_nchan() == response->get_nchan() &&
       get_ndat() == response->get_ndat() )
  {

    if (verbose)
      cerr << "dsp::ScalarFilter::match Response already matched" << endl;
    return;
  }

  resize (npol, response->get_nchan(), response->get_ndat(), ndim);

  if (!built)
    build();
}
