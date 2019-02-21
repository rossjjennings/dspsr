#include "dsp/FIRFilter.h"

void test_FIRFilter () {
  dsp::FIRFilter filter;
  filter.freqz(1024);
}


int main () {
  test_FIRFilter();
  return 0;
}
