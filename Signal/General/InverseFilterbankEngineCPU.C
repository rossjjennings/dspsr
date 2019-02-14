//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/TimeSeries.h"
#include "dsp/Scratch.h"
#include "dsp/Apodization.h"
#include "dsp/OptimalFFT.h"

#include <iostream>
#include <assert.h>

#include "FTransform.h"

using namespace std;

dsp::InverseFilterbankEngineCPU::InverseFilterbankEngineCPU ()
{

}


void dsp::InverseFilterbankEngineCPU::setup (dsp::InverseFilterbank*)
{
}
void dsp::InverseFilterbankEngineCPU::set_scratch (float *)
{
}

void dsp::InverseFilterbankEngineCPU::perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
              uint64_t npart, uint64_t in_step, uint64_t out_step)
{
}

void dsp::InverseFilterbankEngineCPU::finish ()
{
}
