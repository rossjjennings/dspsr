#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>

#include "Rational.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Unpacker.h"
#include "dsp/FIRFilter.h"
#include "dsp/Apodization.h"

std::string tukey_file_path = "/home/SWIN/dshaff/ska/test_data/tukey_window.dat";
std::string tophat_file_path = "/home/SWIN/dshaff/ska/test_data/tophat_window.dat";


class TestApodization {

  public:
  	void test_ctor ();
  	void test_Tukey ();
    void test_TopHat ();

  private:

    void load_test_data (std::string file_path, std::vector<float>& test_data);
    void compare_test_data (float* test_data, float* expected_data, int npts);

    template<typename T>
    bool allclose (T a, T b, T atol, T rtol);
};

template<typename T>
bool TestApodization::allclose (T a, T b, T atol, T rtol)
{
  return abs(a - b) <= (atol + rtol * abs(b));
}


void TestApodization::compare_test_data (float* test_data, float* expected_data, int npts)
{
  float atol = 1e-8;
  float rtol = 1e-5;
  bool val;
  for (int i=0; i<npts; i++) {
    val = allclose(test_data[i], expected_data[i], atol, rtol);
    if (! val) {
      std::cerr << "i=" << i << " ("<< test_data[i] << ", " << expected_data[i] << ")" << std::endl;
    }
    assert(val);
  }
  // std::cerr << std::endl;
}

void TestApodization::load_test_data (std::string file_path, std::vector<float>& test_data)
{
  std::streampos size;
  char * memblock;

  std::ifstream file (file_path, std::ios::in|std::ios::binary|std::ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    // std::cerr << "size=" << size << std::endl;
    memblock = new char [size];
    file.seekg (0, std::ios::beg);
    file.read (memblock, size);
    file.close();

    int float_size = (size / sizeof(float));
    // std::cerr << "float_size=" << float_size << std::endl;

    const float* data = reinterpret_cast<const float*>(memblock);
    test_data.assign(data, data + float_size);

    delete[] memblock;
  }
}


void TestApodization::test_ctor ()
{
	std::cerr << "TestApodization::test_ctor " << std::endl;
  dsp::Apodization window;
}


void TestApodization::test_Tukey ()
{
  std::cerr << "TestApodization::test_Tukey " << std::endl;
  dsp::Apodization window;
  window.Tukey(1024, 0, 128, false);
  std::vector<float> expected_data;
  load_test_data(tukey_file_path, expected_data);
  compare_test_data(
    window.get_datptr(0, 0),
    expected_data.data(),
    expected_data.size());
}

void TestApodization::test_TopHat ()
{
  std::cerr << "TestApodization::test_TopHat " << std::endl;
  dsp::Apodization window;
  window.TopHat(1024, 128, false);
  std::vector<float> expected_data;
  load_test_data(tophat_file_path, expected_data);
  compare_test_data(
    window.get_datptr(0, 0),
    expected_data.data(),
    expected_data.size());
}


int main () {
	// dsp::Input::verbose = true;
	// dsp::Operation::verbose = true;
	// dsp::Shape::verbose = true;

	TestApodization tester;

	tester.test_ctor();
	tester.test_TopHat();
	tester.test_Tukey();

	return 0;
}
