//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten, Andrew Jameson and Dean Shaff
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbankEngineCUDA.h"

CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA ()
{ }

CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA ()
{ }

void CUDA::InverseFilterbankEngineCUDA::setup (dsp::InverseFilterbank* filterbank)
{

}

double CUDA::InverseFilterbankEngineCUDA::setup_fft_plans (dsp::InverseFilterbank* filterbank)
{
  return 0.0;
}

void CUDA::InverseFilterbankEngineCUDA::set_scratch (float* )
{

}

void CUDA::InverseFilterbankEngineCUDA::perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
              uint64_t npart, uint64_t in_step, uint64_t out_step)
{

}

void CUDA::InverseFilterbankEngineCUDA::finish () {}
