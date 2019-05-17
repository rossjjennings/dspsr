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

#include "util.hpp"

std::string tukey_file_path = "/home/SWIN/dshaff/ska/test_data/tukey_window.dat";
std::string tophat_file_path = "/home/SWIN/dshaff/ska/test_data/tophat_window.dat";


class TestApodization {

  public:
  	void test_ctor ();
  	void test_Tukey ();
    void test_TopHat ();
    void test_None ();
    void test_type_map ();

};


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
  util::load_binary_data(tukey_file_path, expected_data);
  util::compare_test_data<float>(
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
  util::load_binary_data(tophat_file_path, expected_data);
  util::compare_test_data<float>(
    window.get_datptr(0, 0),
    expected_data.data(),
    expected_data.size());
}

void TestApodization::test_None ()
{
  std::cerr << "TestApodization::test_None " << std::endl;
  dsp::Apodization window;
  window.None(1024, false);
  std::vector<float> expected_data (1024, 1.0);
  util::compare_test_data<float>(
    window.get_datptr(0, 0),
    expected_data.data(),
    expected_data.size());

}

void TestApodization::test_type_map () {
  std::cerr << "TestApodization::test_type_map " << std::endl;
  dsp::Apodization::Type t = dsp::Apodization::type_map["tukey"];
  assert(t == dsp::Apodization::tukey);
}



int main () {
	// dsp::Input::verbose = true;
	// dsp::Operation::verbose = true;
	// dsp::Shape::verbose = true;

	TestApodization tester;

	tester.test_ctor();
	tester.test_TopHat();
  tester.test_Tukey();
  tester.test_None();
  tester.test_type_map();

	return 0;
}
