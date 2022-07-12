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

using namespace std;

dsp::Apodization::Apodization()
{
  type = none;
  analytic = false;

  zero_start = 0;
  zero_end = 0;

  transition_start = 0;
  transition_end = 0;
}

void dsp::Apodization::set_size (unsigned _ndat)
{
  resize (1, 1, _ndat, 1);
}

void dsp::Apodization::set_type (const string& name) try
{
  type = type_map.at(name);
}
catch (out_of_range error)
{
  throw Error (InvalidParam, "dsp::Apodization::set_type",
               "invalid type name=%s", name.c_str());
}

void dsp::Apodization::build ()
{
  if (zero_start || zero_end)
    throw Error (InvalidState, "Apodization::build",
                 "stop band feature not implemented");

  switch (type)
  {
    case hanning:
      Hanning ();
      break;
    case welch:
      Welch ();
      break;
    case bartlett:
      Bartlett ();
      break;
    case tukey:
      Tukey ();
      break;
    case top_hat:
      TopHat ();
      break;
    case none:
    default:
      throw Error (InvalidState, "Apodization::build",
                   "no function for specified type");
  }
}

void dsp::Apodization::TopHat ()
{
  if (verbose)
    cerr << "dsp::Apodization::TopHat ndat=" << ndat << endl;
  
  for (unsigned idat=0; idat<ndat; idat++)
    buffer[idat] = 1.0;

  type = top_hat;
}

void dsp::Apodization::Hanning ()
{
  double denom = ndat - 1.0;

  for (unsigned idat=0; idat<ndat; idat++)
    buffer[idat] = 0.5 * (1 - cos(2.0*M_PI*double(idat)/denom));

  type = hanning;
}

void dsp::Apodization::Welch ()
{
  float centre = 0.5 * (ndat - 1);
  float denominator = 0.5 * (ndat + 1);

  for (unsigned idat=0; idat<ndat; idat++)
  {
    float tosquare = (float(idat)-centre)/denominator;
    buffer[idat] = 1.0 - tosquare * tosquare;
  }
  type = welch;
}

void dsp::Apodization::Bartlett ()
{
  float centre = 0.5 * (ndat - 1);
  float denominator = 0.5 * (ndat - 1);

  for (unsigned idat=0; idat<ndat; idat++)
    buffer[idat] = 1.0 - fabs ((float(idat)-centre)/denominator);

  type = bartlett;
}


/**
 * \method Tukey
**/
void dsp::Apodization::Tukey ()
{
  if (verbose)
    cerr << "dsp::Apodization::Tukey: ndat=" << ndat
         << " transition_start=" << transition_start
         << " transition_end=" << transition_end << endl;

  float denom_start = 2*transition_start;
  float denom_end = 2*transition_end;

  for (unsigned idat=0; idat<ndat; idat++)
  {
    if (idat < transition_start)
      buffer[idat] = 0.5 * (1 - cos(2.0*M_PI*float(idat)/denom_start));

    else if (idat+1 > ndat - transition_end)
      buffer[idat] = 0.5 * (1 - cos(2.0*M_PI*float(ndat-idat-1)/denom_end));

    else
      buffer[idat] = 1.0;
  }

  type = tukey;
}

void dsp::Apodization::operate (float* indata, float* outdata) const
{
  if (outdata == NULL)
    outdata = indata;

  if (verbose)
    cerr << "Apodization::operate ndat=" << ndat << endl;

  if (analytic)
    for (int ipt=0; ipt<ndat; ipt++)
    {
      outdata[ipt*2] = indata[ipt*2] * buffer[ipt];
      outdata[ipt*2+1] = indata[ipt*2+1] * buffer[ipt];
    }
  else
    for (int ipt=0; ipt<ndat; ipt++)
    {
      outdata[ipt] = indata[ipt] * buffer[ipt];
    }
}

void dsp::Apodization::normalize()
{
  double total = 0.0;
  for (unsigned idat=0; idat<ndat; idat++)
    total += buffer [idat];

  for (unsigned idat=0; idat<ndat; idat++)
    buffer [idat] /= total;
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

map<string, dsp::Apodization::Type> dsp::Apodization::type_map = init_type_map();

#include <fstream>

void dsp::Apodization::dump (const std::string& filename)
{
  ofstream out ( filename.c_str() );

  for (unsigned idat=0; idat < ndat; idat++)
    out << idat << " " << buffer[idat] << endl;
}

