/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <map>
#include <math.h>
#include <stdlib.h>

#include "dsp/Apodization.h"

dsp::Apodization::Apodization()
{
  type = none;
}

void dsp::Apodization::None (int npts, bool analytic)
{
  if (verbose) {
    std::cerr << "dsp::Apodization::None: npts=" << npts
      << " analytic=" << analytic
      << std::endl;
  }
  resize (1, 1, npts, (analytic)?2:1);
  float* datptr = buffer;

  for (unsigned idat=0; idat<ndat; idat++) {
    *datptr = 1.0; datptr++;
    if (analytic) {
      *datptr = 1.0; datptr++;
    }
  }
  type = none;
}

void dsp::Apodization::Hanning (int npts, bool analytic)
{
  resize (1, 1, npts, (analytic)?2:1);

  float* datptr = buffer;
  float  value = 0.0;

  double denom = npts - 1.0;

  for (unsigned idat=0; idat<ndat; idat++) {
    value = 0.5 * (1 - cos(2.0*M_PI*double(idat)/denom));
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = hanning;
}

void dsp::Apodization::set_shape (int npts, Type type, bool analytic, int stop_band, int transition_band)
{
  switch (type)  {
    case hanning:
      Hanning (npts, analytic);
      break;
    case welch:
      Welch (npts, analytic);
      break;
    case parzen:
      Parzen (npts, analytic);
      break;
    case tukey:
      Tukey (npts, transition_band, stop_band, analytic);
      break;
    case top_hat:
      TopHat (npts, stop_band, analytic);
      break;
    case none:
      None (npts, analytic);
      break;
    default:
      break;
  }
}

void dsp::Apodization::Welch (int npts, bool analytic)
{
  resize (1, 1, npts, (analytic)?2:1);

  float* datptr = buffer;
  float  value = 0.0;
  float  tosquare = 0.0;

  float numerator = 0.5 * (npts - 1);
  float denominator = 0.5 * (npts + 1);

  for (unsigned idat=0; idat<ndat; idat++) {
    tosquare = (float(idat)-numerator)/denominator;
    value = 1.0 - tosquare * tosquare;
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = welch;
}

void dsp::Apodization::Parzen (int npts, bool analytic)
{
  resize (1, 1, npts, (analytic)?2:1);

  float* datptr = buffer;
  float  value = 0.0;

  float numerator = 0.5 * (npts - 1);
  float denominator = 0.5 * (npts + 1);

  for (unsigned idat=0; idat<ndat; idat++) {
    // Note that this formula does not in fact define a Parzen window
    value = 1.0 - fabs ((float(idat)-numerator)/denominator);
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = parzen;
}


/**
 * \method Tukey
 * \param npts the size of the window.
 * \param stop_band the number of points *on each side* that are not part
 * of the passband.
 * \param transition_band the number of points *on each side* that form the
 * Hann transition area.
**/
void dsp::Apodization::Tukey (int npts, int stop_band, int transition_band, bool analytic)
{
  if (verbose) {
    std::cerr << "dsp::Apodization::Tukey: npts=" << npts
      << " stop_band=" << stop_band
      << " transition_band=" << transition_band
      << " analytic=" << analytic
      << std::endl;
  }
  unsigned _ndim = (analytic)?2:1;
  resize (1, 1, npts, _ndim);

  float denom = 2*transition_band - 1.0;
  float value = 0.0;
  float* datptr = buffer;

  unsigned itrans = 0;

  for (unsigned idat=0; idat<ndat; idat++) {
    if (idat < stop_band || idat > (npts - stop_band)) {
      value = 0.0;
    } else if (idat >= transition_band && idat < (npts - transition_band)) {
      value = 1.0;
    } else {
      value = 0.5 * (1 - cos(2.0*M_PI*float(itrans)/denom)); // the Hann component
      itrans++ ;
    }

    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }
  type = tukey;
}

/**
 * \method TopHat
 * \param npts the size of the window.
 * \param stop_band the number of points *on each side* that are not part
 * of the passband.
**/
void dsp::Apodization::TopHat (int npts, int stop_band, bool analytic)
{
  unsigned _ndim = (analytic)?2:1;
  resize (1, 1, npts, _ndim);

  float* datptr = buffer;
  float value = 0.0;

  for (int idat=0; idat<ndat; idat++) {
    if (idat >= stop_band && idat < (npts - stop_band)) {
      value = 1.0;
    } else {
      value = 0.0;
    }
    *datptr = value; datptr++;
    if (analytic) {
      *datptr = value; datptr++;
    }
  }

  type = top_hat;
}


void dsp::Apodization::operate (float* indata, float* outdata) const
{
  int npts = ndat * ndim;
  float* winptr = buffer;

  if (outdata == NULL)
    outdata = indata;

  for (int ipt=0; ipt<npts; ipt++) {
    *outdata = *indata * *winptr;
    outdata ++; indata++; winptr++;
  }
}

void dsp::Apodization::normalize()
{
  float* winptr = buffer;

  double total = 0.0;
  for (unsigned idat=0; idat<ndat; idat++) {
    total += *winptr;
    winptr += ndim;
  }

  winptr = buffer;
  unsigned npts = ndat * ndim;
  for (unsigned ipt=0; ipt<npts; ipt++) {
    *winptr /= total;
    winptr ++;
  }
}

double dsp::Apodization::integrated_product (float* data, unsigned incr) const
{
  double total = 0.0;
  unsigned cdat = 0;

  for (unsigned idat=0; idat<ndat; idat++) {
    total += buffer[idat] * data[cdat];
    cdat += incr;
  }

  return total;
}

std::map<std::string, dsp::Apodization::Type> dsp::Apodization::type_map = init_type_map();
