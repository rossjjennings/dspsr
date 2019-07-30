// #define CATCH_CONFIG_MAIN
//
// #include "catch.hpp"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "util.hpp"

int main( int argc, char* argv[] )
{
  util::set_verbose(false);
  Catch::Session session; // There must be exactly one instance

  int returnCode = session.applyCommandLine( argc, argv );
  if ( returnCode != 0 ) { // Indicates a command line error
    return returnCode;
  }

  int numFailed = session.run();
  return numFailed;
}
