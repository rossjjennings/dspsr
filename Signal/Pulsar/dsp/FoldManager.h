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

  protected:

    //! Simply executes all of the Fold operations in order
    virtual void operation();

    //! The set of fold operations
    std::vector< Reference::To<Fold> > fold;

  };

}

#endif
