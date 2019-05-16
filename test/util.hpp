#ifndef __util_hpp
#define __util_hpp

#include <fstream>
#include <string>
#include <vector>

namespace util {

  template<typename T>
  bool isclose (T a, T b, T atol, T rtol);

  template<typename T>
  bool compare_test_data (T* a, T* b, int size, T atol=1e-7, T rtol=1e-5);

  template<typename T>
  void load_binary_data (std::string file_path, std::vector<T>& test_data);

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
    val = util::isclose<T>(a[i], b[i], atol, rtol);
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

#endif
