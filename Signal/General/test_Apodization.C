/***************************************************************************
 *
 *   Copyright (C) 2022 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Apodization.h"

using namespace std;
using namespace dsp;

int main(int argc, char ** argv) try
{
  Apodization::verbose = true;

  Apodization apod;

  vector<string> tests;
  tests.push_back("bartlett");
  tests.push_back("hanning");
  tests.push_back("welch");
  tests.push_back("tukey");
  tests.push_back("top_hat");

  unsigned ndat = 150;
  unsigned transition = 50;

  for (auto test: tests)
  {
    apod.set_type (test);
    apod.set_size (ndat);
    apod.set_transition (transition);
    apod.build ();

    apod.dump (test + ".dat");
  }

  return 0;
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

