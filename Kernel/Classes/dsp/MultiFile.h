//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/MultiFile.h,v $
   $Revision: 1.12 $
   $Date: 2003/04/23 08:04:49 $
   $Author: wvanstra $ */


#ifndef __MultiFile_h
#define __MultiFile_h

#include <vector>

#include "dsp/Observation.h"
#include "dsp/Seekable.h"
#include "dsp/PseudoFile.h"

namespace dsp {

  //! Loads BitSeries data from multiple files
  class MultiFile : public Seekable {

  public:
    
    //! Constructor
    MultiFile ();
    
    //! Destructor
    virtual ~MultiFile () { }
    
    //! Open a number of files and treat them as one logical observation.
    /*! This method forms the union of the existing filenames and the 
      new ones, and sort them by start time.

      \post Resets the file pointers
    */
    virtual void open (const vector<string>& new_filenames);

    //! Makes sure only these filenames are open
    //! Resets the file pointers
    virtual void have_open (const vector<string>& filenames);

    //! Retrieve a pointer to the loader File instance
    File* get_loader(){ if(!loader) return NULL; return loader.get(); }

    //! Retrieve a pointer to the pseudofile
    Observation* get_file(unsigned ifile){ return &files[ifile]; }

    //! Inquire the number of files
    unsigned nfiles(){ return files.size(); }

    //! Erase the entire list of loadable files
    //! Resets the file pointers
    virtual void erase_files();

    //! Erase just some of the list of loadable files
    //! Resets the file pointers regardless
    virtual void erase_files (const vector<string>& erase_filenames);

    //! Find out which file is currently open;
    string get_current_filename() const { return current_filename; }

    //! Find out the index of current file is
    unsigned get_index() const { return current_index; }

  protected:
    
    //! Load bytes from file
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes);
    
    //! Adjust the file pointer
    virtual int64 seek_bytes (uint64 bytes);

    // List of files
    vector<PseudoFile> files;

    //! Loader
    Reference::To<File> loader;

    //! Name of the currently opened file
    string current_filename;

    //! Return the index of the file containing the offset_from_obs_start byte
    /*! offsets do no include header_bytes */
    int getindex (int64 offset_from_obs_start, int64& offset_in_file);

    //! initialize variables
    void init();

    //! Ensure that files are contiguous
    void ensure_contiguity();

  private:

    //! Index of the current PseudoFile in use
    unsigned current_index;

    //! Setup loader and ndat etc after a change to the list of files
    void setup ();

    //! Set the loader to the specified PseudoFile
    void set_loader (unsigned index);
  };

}

#endif // !defined(__MultiFile_h)
  
