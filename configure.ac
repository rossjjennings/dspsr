#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.57)
AC_INIT([DSPSR],[2023-01-25+],[dspsr-developers@lists.sourceforge.net])

AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([config])
AC_CONFIG_SRCDIR([Kernel/Makefile.am])

AC_ARG_VAR([PSRHOME],    [Standard pulsar home directory])
AC_ARG_VAR([LOGIN_ARCH], [Architecture-dependent sub-directory of PSRHOME])
AC_ARG_VAR([PACKAGES],   [Root directories of third-party packages])
AC_ARG_VAR([CUDA_NVCC_FLAGS], [CUDA nvcc flags (e.g. -arch, etc)])

AC_PREFIX_DEFAULT([${PSRHOME:-"/usr/local"}${PSRHOME:+"/$LOGIN_ARCH"}])

################################################################################
#
# Warn about pulsar group environment variables
#

if test x"$PSRHOME" = x; then
  AC_MSG_WARN([The PSRHOME environment variable is not set])
else
  if test x"$LOGIN_ARCH" = x; then
    AC_MSG_WARN([The LOGIN_ARCH environment variable is not set])
  fi
fi

################################################################################
#
# Set the @prefix@ variable in autoconf-generated files
#
if test "x$prefix" = xNONE; then
  prefix=$ac_default_prefix
fi

DSPSR_INSTALL=$prefix
DSPSR_HTTP="http://psrchive.sourceforge.net"

AC_SUBST(DSPSR_INSTALL)
AC_SUBST(DSPSR_HTTP)

#
#
#

# Check if the user has set compiler options
SWIN_OPTIONS_SET

# Enable convenience compiler selection
SWIN_COMPILER

AM_INIT_AUTOMAKE([foreign subdir-objects])

# Create static libraries by default
LT_INIT([disable-shared])

#
# Check for selected formats in backends.list
#

DSPSR_FORMATS


# Enable large-file support
AC_SYS_LARGEFILE

# Determine the machine endian
AC_C_BIGENDIAN

# Checks for programs.
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_CC
AC_PROG_CPP
AM_PROG_CC_C_O

AC_PROG_F77

AC_PROG_INSTALL
AC_PROG_LIBTOOL

# check for C++11 compatibility
AX_CXX_COMPILE_STDCXX_11([noext], [optional])
AM_CONDITIONAL([HAVE_CXX11], [test x"$HAVE_CXX11" == x1])

#
# Disable the debugging information option, -g
#
SWIN_DEBUG
SWIN_LOCAL

#
# Checks for essential libraries
#
AC_CHECK_LIB([m], [sin])
SWIN_LIB_PSRCHIVE([],AC_MSG_ERROR([PSRCHIVE Library not found.

Please see http://psrchive.sourceforge.net

]))

PSRCHIVE_ACLOCAL=`$psrchive_config --aclocal`
AC_SUBST(PSRCHIVE_ACLOCAL)

PSRCHIVE_INCLUDE=`$psrchive_config --cflags | sed s/-pthread//`
AC_SUBST(PSRCHIVE_INCLUDE)

#
# Set up to use any packages already installed in the target directory
#
AC_ARG_ENABLE([existing],
              AC_HELP_STRING([--disable-existing],
              [Don't look for dependencies in existing installation]))

if test -d $DSPSR_INSTALL -a x"$enable_existing" != x"no"; then
  CPPFLAGS="-I$DSPSR_INSTALL/include $CPPFLAGS"
  LDFLAGS="-L$DSPSR_INSTALL/lib $LDFLAGS"
fi

#
# enable POSIX C99 format macros
#
AC_DEFINE([__STDC_FORMAT_MACROS], [1], [enable POSIX C99 format macros])

#
# Checks for optional libraries
#
SWIN_LIB_PSRDADA
SWIN_LIB_CFITSIO
SWIN_LIB_MPI
SWIN_LIB_CUDA
SWIN_LIB_GUPPI_DAQ
SWIN_LIB_MARK5ACCESS
SWIN_LIB_TEMPO2
SWIN_LIB_YAMLCPP

AX_OPENMP
AC_SUBST(OPENMP_CFLAGS)

#
# Checks for graphics libraries
#
SWIN_LIB_PGPLOT

#
# For developers: preserve file modification times
#
INSTALL_DATA="${INSTALL} -m 644 -p"
install_sh="CPPROG='cp -p' ${install_sh}"

#
# Checks for header files.
#
AC_CHECK_HEADERS([malloc.h])

# Check for openssl
MJK_LIB_CRYPTO

# Check for psrxml io library
MJK_LIB_PSRXML

# find HDF5 (needed for LOFAR)
AX_HDF5

#
# Checks for library functions.
#
SWIN_FUNC_GETOPT_LONG
SWIN_FUNC_AFFINITY

#
# Generate python module if --enable-shared is used 
# but not if --disable-python is also used.
#
AC_ARG_ENABLE([python],
              AC_HELP_STRING([--disable-python],
              [Don't build python extensions]))

if test x"$enable_shared" == xyes -a x"$enable_python" != xno; then
  AM_PATH_PYTHON(,, [:])
  AC_PROG_SWIG
  SWIG_ENABLE_CXX
  SWIG_PYTHON
else
  PYTHON=":"
fi

AM_CONDITIONAL([HAVE_PYTHON], [test "$PYTHON" != :])
AM_CONDITIONAL([HAVE_SWIG], [test x"$SWIG" != x])

# echo "HAVE_PYTHON=$HAVE_PYTHON"
# echo "HAVE_CUDA=$HAVE_CUDA"
#
#
# Initialize variables used by Makefile.include
#
INCLUDE_CPPFLAGS="$PSRCHIVE_CPPFLAGS"
AC_SUBST(INCLUDE_CPPFLAGS)

INCLUDE_LDFLAGS="$PSRCHIVE_LIBS"
AC_SUBST(INCLUDE_LDFLAGS)

AM_CONFIG_HEADER([config.h])
AC_CONFIG_FILES([Makefile
		config/Makefile
		Kernel/Makefile
		Kernel/Classes/Makefile
		Kernel/Applications/Makefile
		Kernel/Formats/Makefile
		Kernel/Formats/apsr/Makefile
		Kernel/Formats/asp/Makefile
		Kernel/Formats/bcpm/Makefile
                Kernel/Formats/boreas/Makefile
		Kernel/Formats/bpsr/Makefile
		Kernel/Formats/caspsr/Makefile
		Kernel/Formats/cpsr/Makefile
		Kernel/Formats/cpsr2/Makefile
		Kernel/Formats/dada/Makefile
		Kernel/Formats/dummy/Makefile
		Kernel/Formats/emerlin/Makefile
		Kernel/Formats/fadc/Makefile
        	Kernel/Formats/fits/Makefile
		Kernel/Formats/gmrt/Makefile
		Kernel/Formats/guppi/Makefile
		Kernel/Formats/kat/Makefile
		Kernel/Formats/lbadr/Makefile
		Kernel/Formats/lbadr64/Makefile
		Kernel/Formats/lofar_dal/Makefile
		Kernel/Formats/lump/Makefile
		Kernel/Formats/lwa/Makefile
		Kernel/Formats/spda1k/Makefile
		Kernel/Formats/mark4/Makefile
		Kernel/Formats/mark5/Makefile
		Kernel/Formats/mark5b/Makefile
		Kernel/Formats/maxim/Makefile
		Kernel/Formats/mopsr/Makefile
		Kernel/Formats/mwa/Makefile
		Kernel/Formats/pmdaq/Makefile
		Kernel/Formats/pdev/Makefile
		Kernel/Formats/puma/Makefile
		Kernel/Formats/puma2/Makefile
		Kernel/Formats/s2/Makefile
		Kernel/Formats/sigproc/Makefile
		Kernel/Formats/ska1/Makefile
		Kernel/Formats/spigot/Makefile
		Kernel/Formats/uwb/Makefile
		Kernel/Formats/vdif/Makefile
		Kernel/Formats/wapp/Makefile
		Signal/Makefile
		Signal/Statistics/Makefile
		Signal/General/Makefile
		Signal/Pulsar/Makefile
		Management/Makefile
		Management/dspsr_ldflags
		Management/dspsr_cflags
		Management/release.csh
		More/Makefile
		More/Plotting/Makefile
		More/Applications/Makefile
		python/Makefile
		test/Makefile])

AC_OUTPUT

echo

if test x"$selected_formats" = x; then
  echo "WARNING: no file formats have been selected"
  echo "Please see http://dspsr.sourceforge.net/formats for details"
else
  echo "DSPSR will support these formats: $selected_formats"
fi

echo
echo "DSPSR is now ready to be compiled.  Please run 'make'"
echo
