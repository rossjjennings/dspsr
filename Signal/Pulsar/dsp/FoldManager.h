//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FoldManager_h_
#define __FoldManager_h_

/*!
  Manages the execution of one or more Fold instances
*/

#include "dsp/Operation.h"

namespace dsp {

  class Fold;
  class Observation;

  //! Manage multiple Fold operations

  /*! The original intent for this class was to implement a solution to a
      thread deadlock issue.  However, in the end, 1) this class currently has
      a bug in it (see https://sourceforge.net/p/dspsr/bugs/93/); and 2) the
      thread deadlock issue was solved in a different way.

      When a thread reaches the end of a sub-integration, there are two 
      possibilities:

      a. the sub-integration is complete and can be written to disk; or

      b. the sub-integration is incomplete because one or more threads 
         still have data to fold into it (and have yet to do so).

      In case b, there are a couple of options:

      1. Clone the data and put it in a place where the other threads will 
      find it when they get to the end of the same sub-integration on the 
      same pulsar; each thread can add to it and, when it is complete, write 
      it to disk.  The clone is necessary so that the original thread can 
      resume where it left off in the time series and start folding the next 
      sub-integration into a different array.

      2. Put the data in a place where the other threads will find it when 
      they get to the end of the same sub-integration on the same pulsar, 
      then go to sleep.  The other threads add to the data and wake up the 
      original thread after they have done so; when the sub-integration is 
      complete, the original thread writes it to disk

      Option 1 is not very friendly on RAM, especially if sub-integration 
      data are somewhat large (e.g. many channels) and there are many 
      pulsars to fold.  Therefore, dspsr implements option 2.   However, 
      when there are multiple folds happening in parallel, different threads 
      can go to sleep waiting for sub-integrations to be completed on 
      different pulsars, and in the case of two threads it is possible for 
      thread A to be waiting on pulsar X and thread B to be waiting on 
      pulsar Y (deadlock).

      It is simplest to use option 1, but this could lead to large numbers 
      of cloned sub-integrations waiting around to be completed.  I guess 
      that there would be at most nthread cloned sub-integrations, and 
      perhaps this is not terrible in most cases.

      Another approach is to stick to option 2 but link the different Fold 
      transformations with pointers to each other, such that a thread will 
      go and finish other Fold operations (and wake up any other sleeping 
      threads) before going to sleep on its current Fold.  

      I like option 2 best and started working on this class to manage 
      the links.  In the end, I went with option 1.  Before option 2 can
      be attempted using this class, the bug in this class must be fixed.
  */
  class FoldManager : public Operation 
  {
    public :

    //! Default constructor
    FoldManager ();

    //! Add Fold instance to managed array      
    void manage (Fold*);

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Prepare to fold the given Observation
    void prepare (const Observation* observation);

    //! If Operation is a FoldManager, integrate its Fold instances
    void combine (const Operation*);

    //! Reset the PhaseSeries
    void reset ();

    //! Perform any final operations
    void finish ();

    //! Set verbosity ostream
    void set_cerr (std::ostream& os) const;

  protected:

    //! Simply executes all of the Fold operations in order
    virtual void operation();

    //! The set of fold operations
    std::vector< Reference::To<Fold> > fold;

  };

}

#endif
