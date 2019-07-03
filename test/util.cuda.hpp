#ifndef __util_cuda_hpp
#define __util_cuda_hpp

#include <cuda_runtime.h>

#include "util.hpp"


namespace util
{

  template<typename T>
  void print_array (std::vector<T>& arr, dim3 dim);

}

template<typename T>
void util::print_array (std::vector<T>& arr, dim3 dim)
{
  int* dim_ptr = reinterpret_cast<int*>(&dim);
  std::vector<int> vec(dim_ptr, dim_ptr + 3); // dim3 is always 3 integers in size
  util::print_array<T>(arr, vec);
}

#endif
