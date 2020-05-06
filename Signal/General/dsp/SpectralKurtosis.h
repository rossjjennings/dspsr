//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2016 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/Memory.h"
#include "EventEmitter.h"

#ifndef __SpectralKurtosis_h
#define __SpectralKurtosis_h

#define ZAP_ALL  0
#define ZAP_SKFB 1
#define ZAP_FSCR 2
#define ZAP_TSCR 3

namespace dsp {

  //! Perform Spectral Kurtosis on Input Timeseries, creating output Time Series
  /*! Output will be in time, frequency, polarization order */

  class SpectralKurtosis: public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    SpectralKurtosis ();

    //! Destructor
    ~SpectralKurtosis ();

    bool get_order_supported (TimeSeries::Order order) const;

    //! Load configuration from YAML filename
    void load_configuration (const std::string& filename);

    void set_M (unsigned _M) { resolution[0].set_M( _M ); }
    void set_M (const std::vector<unsigned>&);

    //! Set the number of overlapping regions per time sample
    void set_noverlap (unsigned _nover) { resolution[0].noverlap = _nover; }
    void set_noverlap (const std::vector<unsigned>&);

    //! Set the RFI thresholds with the specified factor
    void set_thresholds (float _std_devs);
    void set_thresholds (const std::vector<float>&);

    //! Set the channel range to conduct detection
    void set_channel_range (unsigned start, unsigned end);

    //! Set various options for Specral Kurtosis
    void set_options (bool _disable_fscr, bool _disable_tscr, bool _disable_ft);

    void reserve ();

    void prepare ();

    void prepare_output ();

    //! The number of time samples used to calculate the SK statistic
    unsigned get_M () const
    { return resolution[0].get_M(); }

    //! The excision threshold in number of standard deviations
    unsigned get_excision_threshold () const
    { return resolution[0].get_std_devs(); }

    //! Total SK statistic for each poln/channel, post filtering
    void get_filtered_sum (std::vector<float>& sum) const
    { sum = filtered_sum; }

    //! Hits on filtered average for each channel
    void get_filtered_hits (std::vector<uint64_t>& hits) const
    { hits = filtered_hits; }

    //! Total SK statistic for each poln/channel, before filtering
    void get_unfiltered_sum (std::vector<float>& sum) const
    { sum = unfiltered_sum; }

    //! Hits on unfiltered SK statistic, same for each channel
    uint64_t get_unfiltered_hits () const { return unfiltered_hits; }

    //! The arrays will be reset when count_zapped is next called
    void reset_count () { unfiltered_hits = 0; }


    //! Engine used to perform computations on device other than CPU
    class Engine;

    void set_engine (Engine*);

    template<class T>
    class Reporter {
    public:
      virtual void operator() (T*, unsigned, unsigned, unsigned, unsigned) {};
    };

    // A event emitter that takes a data array, and the nchan, npol, ndat and ndim
    // associated with the data array
    EventEmitter<Reporter<float> > float_reporter;

    // This is for reporting the state of the bit zapmask
    EventEmitter<Reporter<unsigned char> > char_reporter;

    bool get_report () const { return report; }

    void set_report (bool _report) { report = _report; }

    //! Return true if the zero_DM_input attribute has been set
    bool has_zero_DM_input () const;
    virtual void set_zero_DM_input (TimeSeries* zero_DM_input);
    virtual const TimeSeries* get_zero_DM_input() const;
    virtual TimeSeries* get_zero_DM_input();

    // bool has_zero_DM_input_container () const;
    // virtual void set_zero_DM_input_container (const HasInput<TimeSeries> zero_DM_input_container&);
    // virtual const HasInput<TimeSeries>& get_zero_DM_input_container() const;
    // virtual HasInput<TimeSeries>& get_zero_DM_input_container();

    virtual void set_zero_DM_buffering_policy (BufferingPolicy* policy)
    { zero_DM_buffering_policy = policy; }

    bool has_zero_DM_buffering_policy() const
    { return zero_DM_buffering_policy; }

    BufferingPolicy* get_zero_DM_buffering_policy () const
    { return zero_DM_buffering_policy; }

    //! get the zero_DM flag
    bool get_zero_DM () const { return zero_DM; }

    //! set the zero_DM flag
    void set_zero_DM (bool _zero_DM) { zero_DM = _zero_DM; }

  protected:

    //! Perform the transformation on the input time series
    void transformation ();

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;


  private:

    void compute ();

    void detect ();
    void detect_tscr ();
    void detect_skfb (unsigned ires);
    void detect_fscr (unsigned ires);
    void count_zapped ();

    void mask ();
    void reset_mask ();

    void insertsk ();

    unsigned debugd;

    class Resolution
    {
    private:
      
      //! frequency channels to be zapped
      mutable std::vector<bool> channels;

      //! lower and upper thresholds of excision limits
      mutable std::vector<float> thresholds;

      //! number of samples used in each SK estimate
      unsigned M;

      //! Standard deviation used to compute thresholds
      float std_devs;
      
      //! compute the min and max SK thresholds
      void set_thresholds (float _std_devs, bool verbose = false) const;


    public:

      Resolution ()
      { 
        M = overlap_offset = 128; noverlap = 1;
        npart = output_ndat = 0;
        std_devs = 3.0;
      }

      //! Add a range of frequency channels to be zapped
      /*! from first to second inclusive; e.g. (0,1023) = 1024 channels */
      void add_include (const std::pair<unsigned, unsigned>&);

      //! Add a range of frequency channels not to be zapped
      /*! from first to second inclusive; e.g. (0,1023) = 1024 channels */
      void add_exclude (const std::pair<unsigned, unsigned>&);

      //! Get the channels to be zapped
      const std::vector<bool>& get_channels (unsigned nchan) const;
      
      //! number of samples used in each SK estimate
      unsigned get_M () const { return M; }
      void set_M (unsigned);

      //! oversampling factor
      /* NZAPP-206 WvS: I called this "noverlap" instead of oversampling_factor
         to avoid confusion with polyphase filterbank oversampling */
      unsigned noverlap;

      //! sample offset to start of next overlapping M-sample block
      unsigned overlap_offset;

      //! number of SK estimates produced
      uint64_t npart;

      //! number of output time samples flagged
      uint64_t output_ndat;

      //! ensure that noverlap divides M and compute overlap_offset
      void prepare (uint64_t ndat = 0);

      //! ensure that this shares boundaries with that
      void compatible (Resolution& that);

      //! number of std devs used to calculate excision limits
      float get_std_devs () const { return std_devs; }
      void set_std_devs (float);

      //! lower and upper thresholds of excision limits
      const std::vector<float>& get_thresholds () const;
      
      //! ranges of frequency channels to be zapped
      std::vector< std::pair<unsigned,unsigned> > include;

      //! ranges of frequency channels not to be zapped
      std::vector< std::pair<unsigned,unsigned> > exclude;
    };

    std::vector<Resolution> resolution;

    void resize_resolution (unsigned);

    //! integrate the S1 and S2 sum to new M and noverlap
    void tscrunch_sums (Resolution& from, Resolution& to);

    // for sorting by M
    static bool by_M (const Resolution& A, const Resolution& B);

    unsigned nchan;

    unsigned npol;

    unsigned ndim;

    //! S1 and S2 sums
    Reference::To<TimeSeries> sums;

    //! Tscrunched SK Estimate for block
    Reference::To<TimeSeries> estimates_tscr;

    //! Zap mask
    Reference::To<BitSeries> zapmask;

    //! accumulation arrays for S1 and S2 in tscrunch
    std::vector <float> S1_tscr;
    std::vector <float> S2_tscr;
    uint64_t tscr_count;

    //! Total SK statistic for each poln/channel, post filtering
    std::vector<float> filtered_sum;

    //! Hits on filtered average for each channel
    std::vector<uint64_t> filtered_hits;

    //! Total SK statistic for each poln/channel, before filtering
    std::vector<float> unfiltered_sum;

    //! Hits on unfiltered SK statistic, same for each channel
    uint64_t unfiltered_hits;

    //! Number of samples integrated into tscr
    unsigned M_tscr;

    //! exicision thresholds for tscr
    std::vector<uint64_t> thresholds_tscr_m;
    std::vector<float> thresholds_tscr_lower;
    std::vector<float> thresholds_tscr_upper;

    //! samples zapped by type [0:all, 1:sk, 2:fscr, 3:tscr]
    std::vector<uint64_t> zap_counts;

    //! total number of samples processed
    uint64_t npart_total;

    //! flags for detection types [0:fscr, 1:tscr, 2:tscr]
    std::vector<bool> detection_flags;

    bool prepared;

    //! flag that indicates whether or not to report intermediate data products
    //! via the *_report EventEmitter objects.
    bool report;

    // //! Input TimeSeries that has not been dedispersed in some previous operation.
    // Reference::To<dsp::TimeSeries> zero_DM_input;

    //! HasInput continaer for zero_DM_input TimeSeries
    HasInput<TimeSeries> zero_DM_input_container;

    Reference::To<BufferingPolicy> zero_DM_buffering_policy;

    bool zero_DM;

  };

  class SpectralKurtosis::Engine : public Reference::Able
  {
  public:

      virtual void setup () = 0;

      virtual void compute (const TimeSeries* input, TimeSeries* output,
                            TimeSeries *output_tscr, unsigned tscrunch) = 0;

      virtual void reset_mask (BitSeries* output) = 0;

      virtual void detect_ft (const TimeSeries* input, BitSeries* output,
                              float upper_thresh, float lower_thresh) = 0;

      virtual void detect_fscr (const TimeSeries* input, BitSeries* output,
                                const float mu2, const float std_devs,
                                unsigned schan, unsigned echan) = 0;

      virtual void detect_tscr (const TimeSeries* input,
                                const TimeSeries * input_tscr,
                                BitSeries* output,
                                float upper, float lower) = 0;

      virtual int count_mask (const BitSeries* output) = 0;

      virtual float * get_estimates (const TimeSeries* input) = 0;

      virtual unsigned char * get_zapmask (const BitSeries* input) = 0;

      virtual void mask (BitSeries* mask, const TimeSeries * in, TimeSeries* out, unsigned M) = 0;

      virtual void insertsk (const TimeSeries* input, TimeSeries* out, unsigned M) = 0;

  };
}

#endif
