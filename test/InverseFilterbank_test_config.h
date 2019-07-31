// configuration for InverseFilterbank testing

#ifndef __InverseFilterbank_test_config_h
#define __InverseFilterbank_test_config_h

#include <vector>


namespace test_config {

  struct TestShape {
    unsigned npart;
    unsigned npol;
    unsigned nchan;
    unsigned output_nchan;
    unsigned ndat;
    unsigned overlap;
  };

	// TestShape small_single_pol;
	// TestShape small_double_pol;
	// TestShape single_pol;
	// TestShape double_pol;
	//
	// std::vector<TestShape> test_shapes;


	const TestShape small_single_pol{2, 1, 4, 2, 8, 1};
	const TestShape small_double_pol{2, 2, 4, 2, 8, 1};
	const TestShape single_pol{10, 1, 64, 8, 256, 32};
	const TestShape double_pol{10, 2, 64, 8, 256, 32};
  const TestShape integration_shape{10, 1, 16, 1, 32, 8};
  const TestShape integration_shape1{10, 1, 20, 1, 32, 8};
  const TestShape integration_shape2{10, 1, 1024, 1, 32, 8};

	const std::vector<TestShape> test_shapes = {
		small_single_pol,
		small_double_pol,
		single_pol,
		double_pol,
    integration_shape,
    integration_shape1,
    integration_shape2
	};

  const float thresh = 1e-5;

}


#endif
