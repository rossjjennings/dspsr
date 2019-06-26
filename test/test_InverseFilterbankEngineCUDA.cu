#include "util.hpp"

class TestInverseFilterbankEngineCUDA {

  public:
    void setup_class ();
    void setup ();
    void teardown ();

    void test_apodization_kernel ();

};

TestInverseFilterbankEngineCUDA::setup () {}
TestInverseFilterbankEngineCUDA::setup_class () {}
TestInverseFilterbankEngineCUDA::teardown () {}

TestInverseFilterbankEngineCUDA::test_apodization_kernel () {


}

int main () {
  util::TestRunner<TestInverseFilterbankEngineCUDA> runner;
	TestInverseFilterbankEngineCUDA tester;
  tester.set_test_data_file_path(file_path);
  tester.set_verbose(true);

  runner.register_test_method(
    &TestInverseFilterbankEngineCUDA::test_apodization_kernel);

  runner.run(tester);
	return 0;
}
