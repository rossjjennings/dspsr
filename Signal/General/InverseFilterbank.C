/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngine.h"

#include "dsp/WeightedTimeSeries.h"
#include "dsp/Response.h"
#include "dsp/Apodization.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"
#include "dsp/OptimalFFT.h"

using namespace std;

#define _DEBUG 1

dsp::InverseFilterbank::InverseFilterbank (const char* name, Behaviour behaviour)
  : Convolution (name, behaviour)
{
  set_buffering_policy (new InputBuffering (this));
}

void dsp::InverseFilterbank::prepare ()
{
  if (verbose) {
    cerr << "dsp::InverseFilterbank::prepare" << endl;
  }

  make_preparations ();
  prepared = true;
}

void dsp::InverseFilterbank::reserve ()
{

}

void dsp::InverseFilterbank::set_engine (Engine* _engine)
{
  engine = _engine;
}


void dsp::InverseFilterbank::transformation ()
{
}

void dsp::InverseFilterbank::filterbank()
{
}

void dsp::InverseFilterbank::make_preparations ()
{
}

void dsp::InverseFilterbank::prepare_output (uint64_t ndat, bool set_ndat)
{
}

void dsp::InverseFilterbank::resize_output (bool reserve_extra)
{
}
