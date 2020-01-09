#include <vector>
#include <utility>

#include "catch.hpp"

#include "dsp/Response.h"


class ResponseMock : public dsp::Response
{
public:

  void test_calc_lcf (unsigned a, unsigned b, const Rational& osf, std::vector<unsigned>& result);

  template<class ... Types>
  void test_calc_oversampled_discard_region (Types ... args);

  template<class ... Types>
  void test_calc_oversampled_fft_length (Types ... args);
};

void ResponseMock::test_calc_lcf (unsigned a, unsigned b, const Rational& osf, std::vector<unsigned>& result) {
  calc_lcf(a, b, osf, result);
}

template<class ... Types>
void ResponseMock::test_calc_oversampled_discard_region (Types ... args) {
  calc_oversampled_discard_region(args...);
}

template<class ... Types>
void ResponseMock::test_calc_oversampled_fft_length (Types ... args) {
  calc_oversampled_fft_length(args...);
}


const Rational osf(4, 3);

TEST_CASE ("calc_lcf works as expected", "[Response]")
{
  ResponseMock response;
  std::vector<unsigned> result(2);
  response.test_calc_lcf(131072, 256, osf, result);

  REQUIRE(result[0] == 170);
  REQUIRE(result[1] == 512);
}

TEST_CASE ("calc_oversampled_discard_region works as expected", "[Response]")
{
  ResponseMock response;
  unsigned neg = 7616;
  unsigned pos = 6176;
  unsigned nchan = 256;
  response.test_calc_oversampled_discard_region(&neg, &pos, nchan, osf);
  REQUIRE(neg == 7680);
  REQUIRE(pos == 6912);

  neg = 2;
  pos = 2;
  nchan = 2;
  response.test_calc_oversampled_discard_region(&neg, &pos, nchan, osf);
  REQUIRE(neg == 6);
  REQUIRE(pos == 6);
}


TEST_CASE ("calc_oversampled_fft_length works as expected", "[Response]")
{
  ResponseMock response;
  unsigned fft_length = 131072;
  unsigned nchan = 256;
  response.test_calc_oversampled_fft_length(&fft_length, nchan, osf, -1);
  REQUIRE(fft_length == 98304);

  fft_length = 16;
  nchan = 2;
  response.test_calc_oversampled_fft_length(&fft_length, nchan, osf, 1);
  REQUIRE(fft_length == 24);
}
