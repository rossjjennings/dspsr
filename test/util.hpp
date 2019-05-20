#ifndef __util_hpp
#define __util_hpp

#include <fstream>
#include <string>
#include <vector>

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"

namespace util {

  template<typename T>
  bool isclose (T a, T b, T atol, T rtol);

  template<typename T>
  bool compare_test_data (T* a, T* b, int size, T atol=1e-7, T rtol=1e-5);

  template<typename T>
  void load_binary_data (std::string file_path, std::vector<T>& test_data);

  class TestDataLoader {
    public:
      void setup_class () {}
      void teardown_class () {}
      void setup ();
      void teardown () {}

      void set_verbose (bool verbosity) ;

      void set_test_data_file_path (std::string _test_data_file_path)
        { test_data_file_path = _test_data_file_path; }

      std::string get_test_data_file_path () { return test_data_file_path; }

    protected:

      void load_data (dsp::TimeSeries* ts, int block_size);

      dsp::IOManager manager ;
      dsp::Input* input;
      dsp::Observation* info;
      std::string test_data_file_path;

    };

    template<class T>
    class TestRunner {
      public:
        TestRunner () {}
        void register_test_method (void (T::*x)());
        void run (T& obj);
      private:
        std::vector<void (T::*)()> test_methods;
    };


}

template<typename T>
void util::load_binary_data (std::string file_path, std::vector<T>& test_data)
{
  std::streampos size;

  std::ifstream file (file_path, std::ios::in|std::ios::binary|std::ios::ate);
  if (file.is_open())
  {
    file.seekg(0, std::ios::end);
    size = file.tellg();
    file.seekg(0, std::ios::beg);

    // read the data:
    std::vector<char> file_bytes(size);
    file.read(&file_bytes[0], size);
    file.close();

    int T_size = (size / sizeof(T));
    // std::cerr << "T_size=" << T_size << std::endl;

    const T* data = reinterpret_cast<const T*>(file_bytes.data());
    test_data.assign(data, data + T_size);
  }
}


template<typename T>
bool util::compare_test_data (T* a, T* b, int size, T atol, T rtol)
{
  bool ret;
  bool val;
  for (int i=0; i<size; i++) {
    val = util::isclose(a[i], b[i], atol, rtol);
    if (! val) {
      ret = false;
      std::cerr << "i=" << i << " ("<< a[i] << ", " << b[i] << ")" << std::endl;
    }
  }
  return ret;
}



template<typename T>
bool util::isclose (T a, T b, T atol, T rtol)
{
  return abs(a - b) <= (atol + rtol * abs(b));
}



void util::TestDataLoader::load_data (dsp::TimeSeries* ts, int block_size)
{
	// std::cerr << "util::TestDataLoader::load_data" << std::endl;
	input->set_block_size(block_size);
	while (! manager.get_input()->eod()) {
		manager.load(ts);
	}
}


void util::TestDataLoader::setup () {
  dsp::IOManager _manager;
  _manager.open(test_data_file_path);
  dsp::Input* _input = _manager.get_input();
  dsp::Observation* _info = _input->get_info();

  manager = _manager;
  input = _input;
  info = _info;
}


void util::TestDataLoader::set_verbose (bool verbosity)
{
  dsp::Input::verbose = verbosity;
  dsp::Operation::verbose = verbosity;
  dsp::Shape::verbose = verbosity;
}

template<class T>
void util::TestRunner<T>::register_test_method (void (T::*x)())
{
  test_methods.push_back(x);
}

template<class T>
void util::TestRunner<T>::run (T& obj)
{
  obj.setup_class();
  for (unsigned i=0; i<test_methods.size(); i++) {
    obj.setup();
    (obj.*test_methods[i])();
    obj.teardown();
  }
  obj.teardown_class();
}


#endif
