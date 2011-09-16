/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/SingleThread.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/Scratch.h"
#include "dsp/File.h"

#include "dsp/ExcisionUnpacker.h"
#include "dsp/WeightedTimeSeries.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDA.h"
#endif

#include "dsp/ObservationChange.h"
#include "dsp/Dump.h"

#include "Error.h"
#include "stringtok.h"
#include "pad.h"

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>


using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::SingleThread::SingleThread ()
  : cerr( std::cerr.rdbuf() ), error (InvalidState, "")
{
  manager = new IOManager;
  scratch = new Scratch;
  log = 0;
  minimum_samples = 0;

  state = Idle;
  state_change = 0;
  thread_id = 0;
  colleague = 0;

  input_context = 0;
  gpu_stream = undefined_stream;
}

dsp::SingleThread::~SingleThread ()
{
}

void dsp::SingleThread::set_configuration (Config* configuration)
{
  config = configuration;
}

void dsp::SingleThread::take_ostream (std::ostream* newlog)
{
  if (newlog)
    this->cerr.rdbuf( newlog->rdbuf() );

  if (log)
    delete log;

  log = newlog;
}

//! Set the Input from which data will be read
void dsp::SingleThread::set_input (Input* input)
{
  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_input input=" << input << endl;

  manager->set_input (input);
}

dsp::Input* dsp::SingleThread::get_input ()
{
  return manager->get_input ();
}

void dsp::SingleThread::set_affinity (int core)
{
#if HAVE_SCHED_SETAFFINITY
  cpu_set_t set;
  CPU_ZERO (&set);
  CPU_SET (core, &set);

  pid_t tpid = syscall (SYS_gettid);

  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_affinity thread=" << thread_id
         << " tpid=" << tpid << " core=" << core << endl;

  if (sched_setaffinity(tpid, sizeof(cpu_set_t), &set) < 0)
    throw Error (FailedSys, "dsp::SingleThread::set_affinity",
                 "sched_setaffinity (%d)", core);
#endif
}

//! Share any necessary resources with the specified thread
void dsp::SingleThread::share (SingleThread* other)
{
  colleague = other;
}

dsp::TimeSeries* dsp::SingleThread::new_time_series ()
{
  config->buffers ++;

  if (config->weighted_time_series)
  {
    if (Operation::verbose)
      cerr << "Creating WeightedTimeSeries instance" << endl;
    return new WeightedTimeSeries;
  }
  else
  {
    if (Operation::verbose)
      cerr << "Creating TimeSeries instance" << endl;
    return  new TimeSeries;
  }
}

template<typename T>
unsigned count (const std::vector<T>& data, T element)
{
  unsigned c = 0;
  for (unsigned i=0; i<data.size(); i++)
    if (data[i] == element)
      c ++;
  return c;
}

void dsp::SingleThread::prepare () try
{
  TimeSeries::auto_delete = false;

  operations.resize (0);

  // each timeseries created will be counted in new_time_series
  config->buffers = 0;

  if (thread_id < config->affinity.size())
    set_affinity (config->affinity[thread_id]);

  // only the first thread should run input_prepare
  if (thread_id == 0 && config->input_prepare)
    config->input_prepare( manager->get_input() );

  if (!unpacked)
    unpacked = new_time_series();

  manager->set_output (unpacked);

  operations.push_back (manager.get());

#if HAVE_CUDA

  bool run_on_gpu = thread_id < config->get_cuda_ndevice();

  cudaStream_t stream = 0;

  if (run_on_gpu)
  {
    // disable input buffering when data must be copied between devices
    if (config->get_total_nthread() > 1)
      config->input_buffering = false;

    int device = config->cuda_device[thread_id];
    cerr << "dspsr: thread " << thread_id 
         << " using CUDA device " << device << endl;

    int ndevice = 0;
    cudaGetDeviceCount(&ndevice);

    if (device >= ndevice)
      throw Error (InvalidParam, "dsp::SingleThread::prepare",
                   "device=%d >= ndevice=%d", device, ndevice);

    cudaError err = cudaSetDevice (device);
    if (err != cudaSuccess)
      throw Error (InvalidState, "dsp::SingleThread::prepare",
                   "cudaMalloc failed: %s", cudaGetErrorString(err));

    unsigned nstream = count (config->cuda_device, (unsigned)device);

    if (nstream > 1)
    {
      cudaStreamCreate( &stream );
      cerr << "dspsr: thread " << thread_id << " on stream " << stream << endl;
    }

    gpu_stream = stream;

    device_memory = new CUDA::DeviceMemory (stream);

    Unpacker* unpacker = manager->get_unpacker ();
    if (unpacker->get_device_supported( device_memory ))
    {
      if (Operation::verbose)
        cerr << "SingleThread: unpack on GraphicsPU" << endl;

      unpacker->set_device( device_memory );
      unpacked->set_memory( device_memory );
        
      BitSeries* bits = new BitSeries;
      bits->set_memory (new CUDA::PinnedMemory);
      manager->set_output (bits);
    }
    else
    {
      if (Operation::verbose)
        cerr << "SingleThread: unpack on CPU" << endl;

      TransferCUDA* transfer = new TransferCUDA;
      transfer->set_kind( cudaMemcpyHostToDevice );
      transfer->set_input( unpacked );
        
      unpacked = new_time_series ();
      unpacked->set_memory (device_memory);
      transfer->set_output( unpacked );
      operations.push_back (transfer);
    }    
  }

#endif // HAVE_CUFFT

}
catch (Error& error)
{
  throw error += "dsp::SingleThread::prepare";
}

void dsp::SingleThread::insert_dump_point (const std::string& transform_name)
{
  typedef HasInput<TimeSeries> Xform;

  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (operations[iop]->get_name() == transform_name)
    {
      Xform* xform = dynamic_cast<Xform*>( operations[iop].get() );
      if (!xform)
	throw Error (InvalidParam, "dsp::SingleThread::insert_dump_point",
		     transform_name + " does not have TimeSeries input");

      string filename = "pre_" + transform_name;

      if (config->get_total_nthread() > 1)
        filename += "." + tostring (thread_id);

      filename += ".dump";

      cerr << "dspsr: dump output in " << filename << endl;

      Dump* dump = new Dump;
      dump->set_output( fopen(filename.c_str(), "w") );
      dump->set_input( xform->get_input() ) ;
      dump->set_output_binary (true);

      operations.push_back (dump);
      iop++;
    }
  }
}

uint64_t dsp::SingleThread::get_minimum_samples () const
{
  return minimum_samples;
}

//! Run through the data
void dsp::SingleThread::run () try
{
  if (Operation::verbose)
    cerr << "dsp::SingleThread::run this=" << this 
         << " nops=" << operations.size() << endl;

  if (log)
    scratch->set_cerr (*log);

  // ensure that all operations are using the local log and scratch space
  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (log)
    {
      cerr << "dsp::SingleThread::run " << operations[iop]->get_name() << endl;
      operations[iop] -> set_cerr (*log);
    }
    if (!operations[iop] -> scratch_was_set ())
      operations[iop] -> set_scratch (scratch);

    operations[iop] -> reserve ();
  }

  Input* input = manager->get_input();

  uint64_t block_size = input->get_block_size();

  if (block_size == 0)
    throw Error (InvalidState, "dsp::SingleThread::run", "block_size=0");

  uint64_t total_samples = input->get_total_samples();
  uint64_t nblocks_tot = total_samples/block_size;

  unsigned block=0;

  int64_t last_decisecond = -1;

  bool finished = false;

  while (!finished)
  {
    while (!input->eod())
    {
      for (unsigned iop=0; iop < operations.size(); iop++) try
      {
	if (Operation::verbose)
	  cerr << "dsp::SingleThread::run calling " 
	       << operations[iop]->get_name() << endl;
      
	operations[iop]->operate ();
      
	if (Operation::verbose)
	  cerr << "dsp::SingleThread::run "
	       << operations[iop]->get_name() << " done" << endl;
      
      }
      catch (Error& error)
      {
	if (error.get_code() == EndOfFile)
	  break;

	end_of_data ();

	throw error += "dsp::SingleThread::run";
      }
    
      block++;
    
      if (thread_id==0 && config->report_done) 
      {
	double seconds = input->tell_seconds();
	int64_t decisecond = int64_t( seconds * 10 );
      
	if (decisecond > last_decisecond)
	{
	  last_decisecond = decisecond;
	  cerr << "Finished " << decisecond/10.0 << " s";

	  if (nblocks_tot)
	    cerr << " (" 
		 << int (100.0*input->tell()/float(input->get_total_samples()))
		 << "%)";

	  cerr << "   \r";
	}
      }
    }

    finished = true;

    if (config->run_repeatedly)
    {
      ThreadContext::Lock context (input_context);

      if (config->repeated == 0 && input->tell() != 0)
      {
	// cerr << "dspsr: do it again" << endl;
	File* file = dynamic_cast<File*> (input);
	if (file)
	{
	  finished = false;
	  string filename = file->get_filename();
	  file->close();
	  // cerr << "file closed" << endl;
	  file->open(filename);
	  // cerr << "file opened" << endl;
	  config->repeated = 1;
	  
	  if (config->input_prepare)
	    config->input_prepare (file);

	}
      }
      else if (config->repeated)
      {
	config->repeated ++;
	finished = false;

	if (config->repeated == config->get_total_nthread())
	  config->repeated = 0;
      }
    }
  }

  if (Operation::verbose)
    cerr << "dsp::SingleThread::run end of data id=" << thread_id << endl;

  end_of_data ();

  if (Operation::verbose)
    cerr << "dsp::SingleThread::run exit" << endl;
}
catch (Error& error)
{
  throw error += "dsp::SingleThread::run";
}

bool same_name (const dsp::Operation* A, const dsp::Operation* B)
{
  return A->get_name() == B->get_name();
}

template<typename C>
unsigned find_name (const C& container, unsigned i, const dsp::Operation* B)
{
  while (i < container.size() && ! same_name(container[i], B))
    i++;
  return i;
}

void dsp::SingleThread::combine (const SingleThread* that)
{
  if (Operation::verbose)
    cerr << "dsp::SingleThread::combine"
         << " this size=" << operations.size() 
         << " ptr=" << &(this->operations)
         << " that size=" << that->operations.size()
         << " ptr=" << &(that->operations) << endl;

  unsigned ithis = 0;
  unsigned ithat = 0;

  while (ithis < operations.size() && ithat < that->operations.size())
  {
    if (! same_name(operations[ithis], that->operations[ithat]))
    {
      // search for that in this
      unsigned jthis = find_name (operations, ithis, that->operations[ithat]);
      if (jthis == operations.size())
      {
        if (Operation::verbose)
          cerr << "dsp::SingleThread::combine insert "
               << that->operations[ithat]->get_name() << endl;

        // that was not found in this ... insert it and skip it
        operations.insert( operations.begin()+ithis, that->operations[ithat] );
        ithis ++;
        ithat ++;
      }
      else
      {
        // that was found later in this ... skip to it
        ithis = jthis;
      }

      continue;

#if 0
      if (operations[ithis]->get_function() != Operation::Procedural)
      {
        ithis ++;
        continue;
      }

      if (that->operations[ithat]->get_function() != Operation::Procedural)
      {
        ithat ++;
        continue;
      }

      throw Error (InvalidState, "dsp::SingleThread::combine",
                   "operation names do not match "
                   "'"+ operations[ithis]->get_name()+"'"
                   " != '"+that->operations[ithat]->get_name()+"'");
#endif
    }

    if (Operation::verbose)
      cerr << "dsp::SingleThread::combine "
           << operations[ithis]->get_name() << endl;

    operations[ithis]->combine( that->operations[ithat] );

    ithis ++;
    ithat ++;
  }

  if (ithis != operations.size() || ithat != that->operations.size())
    throw Error (InvalidState, "dsp::SingleThread::combine",
                 "processes have different numbers of operations");
}

//! Run through the data
void dsp::SingleThread::finish () try
{
  if (Operation::record_time)
    for (unsigned iop=0; iop < operations.size(); iop++)
      operations[iop]->report();
}
catch (Error& error)
{
  throw error += "dsp::SingleThread::finish";
}

void dsp::SingleThread::end_of_data ()
{
  // do nothing by default
}

dsp::SingleThread::Config::Config ()
{
  // be a little bit verbose by default
  report_done = true;
  report_vitals = true;

  // process each file once
  run_repeatedly = false;

  // use weighted time series
  weighted_time_series = true;

  // use input buffering
  input_buffering = true;

  nthread = 0;
  buffers = 0;
  repeated = 0;
}

//! set the number of CPU threads to be used
void dsp::SingleThread::Config::set_nthread (unsigned cpu_nthread)
{
  nthread = cpu_nthread;
}

//! get the total number of threads
unsigned dsp::SingleThread::Config::get_total_nthread () const
{
  unsigned total_nthread = nthread + get_cuda_ndevice();

  if (total_nthread)
    return total_nthread;

  return 1;
}

// set the cuda devices to be used
void dsp::SingleThread::Config::set_cuda_device (string txt)
{
  while (txt != "")
  {
    string dev = stringtok (txt, ",");
    cuda_device.push_back( fromstring<unsigned>(dev) );
  }
}

// set the cpu on which threads will run
void dsp::SingleThread::Config::set_affinity (string txt)
{
  while (txt != "")
  {
    string cpu = stringtok (txt, ",");
    affinity.push_back( fromstring<unsigned>(cpu) );
  }
}