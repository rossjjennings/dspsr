// #define CATCH_CONFIG_MAIN
//
// #include "catch.hpp"
#include <string.h>
#include <iostream>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "util/util.hpp"

int main( int argc, char* argv[] )
{

  util::set_verbose(false);

  for (int i=0; i<argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      util::config::verbose = true;
      if (i + 1 < argc) {
        if (strcmp(argv[i+1], "high") == 0) {
          util::set_verbose(true);
          argv[i+1] = (char *)"normal";
        }
      }
    }
  }


  Catch::Session session; // There must be exactly one instance

  int returnCode = session.applyCommandLine( argc, argv );
  if ( returnCode != 0 ) { // Indicates a command line error
    return returnCode;
  }

  int numFailed = session.run();
  return numFailed;
}
