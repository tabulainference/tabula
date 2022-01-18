/*
 * YaoGarbler.cpp
 *
 */

#include "YaoGarbler.h"
#include "YaoGate.h"

#include "GC/ThreadMaster.hpp"
#include "GC/Processor.hpp"
#include "GC/Program.hpp"
#include "GC/Machine.hpp"
#include "GC/Secret.hpp"
#include "GC/Thread.hpp"
#include "Tools/MMO.hpp"
#include "YaoWire.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

static uint64_t time_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

thread_local YaoGarbler* YaoGarbler::singleton = 0;

YaoGarbler::YaoGarbler(int thread_num, YaoGarbleMaster& master) :
		GC::Thread<GC::Secret<YaoGarbleWire>>(thread_num, master),
		YaoCommon<YaoGarbleWire>(master),
		master(master),
		and_proc_timer(CLOCK_PROCESS_CPUTIME_ID),
		and_main_thread_timer(CLOCK_THREAD_CPUTIME_ID),
		player(master.N, 1, thread_num << 24),
		ot_ext(OTExtensionWithMatrix::setup(player,
				master.get_delta().get<__m128i>(), SENDER, true))
{
	prng.ReSeed();
	set_n_program_threads(master.machine.nthreads);
	this->init(*this);
}

YaoGarbler::~YaoGarbler()
{
#ifdef VERBOSE
	cerr << "Number of AND gates: " << counter << endl;
#endif
#ifdef YAO_TIMINGS
	cout << "AND time: " << and_timer.elapsed() << endl;
	cout << "AND process timer: " << and_proc_timer.elapsed() << endl;
	cout << "AND main thread timer: " << and_main_thread_timer.elapsed() << endl;
	cout << "AND prepare timer: " << and_prepare_timer.elapsed() << endl;
	cout << "AND wait timer: " << and_wait_timer.elapsed() << endl;
	for (auto& x : timers)
		cout << x.first << " time:" << x.second.elapsed() << endl;
#endif
}

void YaoGarbler::run(GC::Program& program)
{
        cout << "Start running garbler..." << endl;
        int i = 0;
	singleton = this;

	GC::BreakType b = GC::TIME_BREAK;

	uint64_t ms1 = time_ms(); 
	while(GC::DONE_BREAK != b)
	{
	  cout << "A " << time_ms()-ms1  << " " << time_ms() << endl;	  
		try
		{
			b = program.execute(processor, master.memory, -1);
		}		
		catch (needs_cleaning& e)
		{
			if (not continuous())
				throw runtime_error("run-time branching impossible with garbling at once");
			processor.PC--;
		}
	  cout << "B " << time_ms()-ms1  << " " << time_ms() << endl;	  		
	        send_write(*P, master.opts.gcs_saveprefix + "_" + to_string(i));
	  cout << "C " << time_ms()-ms1  << " " << time_ms() << endl;	  		
		gates.clear();
		output_masks.clear();
	  cout << "D " << time_ms()-ms1  << " " << time_ms() << endl;	  		
	  if (continuous())
	    process_receiver_inputs();
		i++;
		cout << "E " << time_ms()-ms1 << " " << time_ms() << endl;
	}

	uint64_t ms2 = time_ms();
	cout << "Garbler elapsed: " << ms2-ms1 << endl;
	//P->send_long(1, YaoCommon::DONE);
	//cout << "H " << time_ms() << endl;	  	
}

void YaoGarbler::post_run()
{
	if (not continuous())
	{
		P->send_long(1, YaoCommon::DONE);
		process_receiver_inputs();
	}
}

void YaoGarbler::send_write(Player& P, string fname)
{
  cout << "sending " << gates.size() <<  " gates and " <<
    output_masks.size() << " output masks at " << processor.PC << endl;
    
  P.send_long(1, YaoCommon::MORE);
  size_t size = gates.size();

  P.send_to_and_writefile(1, gates, fname);
  gates.allocate(2 * size);
  P.send_to_and_writefile(1, output_masks, fname+"mask");
}

void YaoGarbler::send(Player& P)
{
  cout << "sending " << gates.size() <<  " gates and " <<
    output_masks.size() << " output masks at " << processor.PC << endl;
    
  P.send_long(1, YaoCommon::MORE);
  size_t size = gates.size();

  P.send_to(1, gates);
  gates.allocate(2 * size);
  P.send_to(1, output_masks);  
}

void YaoGarbler::process_receiver_inputs()
{
  cout << "process receiver " << time_ms() << endl;
  while (not receiver_input_keys.empty())
	{
	  cout << "process receiver A " << time_ms() << endl;	  
	  vector<Key>& inputs = receiver_input_keys.front();
	  BitVector _;
	  ot_ext.extend_correlated(inputs.size(), _);
	  octetStream os;
	  cout << "process receiver B " << time_ms() << endl;	  	  
	  for (size_t i = 0; i < inputs.size(); i++)
	    os.serialize(inputs[i] ^ ot_ext.senderOutputMatrices[0][i]);
	  cout << "process receiver C " << time_ms() << endl;	  	  
	  player.send(os);
	  cout << "process receiver D " << time_ms() << endl;	  	  
	  
	  receiver_input_keys.pop_front();
	}
}

size_t YaoGarbler::data_sent()
{
	return super::data_sent() + player.comm_stats.total_data();
}
