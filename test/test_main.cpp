// #define CATCH_CONFIG_MAIN
//
// #include "catch.hpp"
#include <string.h>
#include <iostream>
#include <sstream>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "Error.h"

#include "util/util.hpp"

CATCH_TRANSLATE_EXCEPTION( Error& ex ) {
  std::stringstream strs;
  ex.report(strs);
  return strs.str();
}


int main( int argc, char* argv[] )
{

  test::util::set_verbose(false);

  for (int i=0; i<argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      test::util::config::verbose = true;
      if (i + 1 < argc) {
        if (strcmp(argv[i+1], "high") == 0) {
          test::util::set_verbose(true);
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
