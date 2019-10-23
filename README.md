## DSPSR

[![pipeline status](https://gitlab.com/ska-telescope/dspsr/badges/master/pipeline.svg)](https://gitlab.com/ska-telescope/dspsr/commits/master)

### Documentation

Documentation for DSPSR can be found online [here](http://dspsr.sourceforge.net/).

### Building

DSPSR is dependent on [PSRCHIVE](https://sourceforge.net/projects/psrchive/)  and [PSRDADA](https://sourceforge.net/projects/psrdada/). It has several other dependencies, but the build system will direct you to install them if it detects they aren't present.

With autotools installed, building can be as simple as the following:

```
./bootstrap
mkdir build && cd build
./../configure
make
make install
```
