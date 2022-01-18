/*
 * YaoEvaluator.cpp
 *
 */

#include "YaoEvaluator.h"

#include "GC/Machine.hpp"
#include "GC/Program.hpp"
#include "GC/Processor.hpp"
#include "GC/Secret.hpp"
#include "GC/Thread.hpp"
#include "GC/ThreadMaster.hpp"
#include "Tools/MMO.hpp"
#include "YaoWire.hpp"
#include <chrono>

using namespace std::chrono;

static uint64_t time_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

thread_local YaoEvaluator* YaoEvaluator::singleton = 0;

YaoEvaluator::YaoEvaluator(int thread_num, YaoEvalMaster& master) :
		Thread<GC::Secret<YaoEvalWire>>(thread_num, master),
		YaoCommon<YaoEvalWire>(master),
		master(master),
		player(N, 0, thread_num << 24),
		ot_ext(OTExtensionWithMatrix::setup(player, {}, RECEIVER, true))
{
	set_n_program_threads(master.machine.nthreads);
	this->init(*this);
}

void YaoEvaluator::pre_run()
{
	processor.out.activate(true);
	if (not continuous())
		receive_to_store(*P);
}

void YaoEvaluator::run(GC::Program& program)
{
	singleton = this;

	if (continuous())
		run(program, *P);
	else
	{
		run_from_store(program);
	}
}

void YaoEvaluator::run(GC::Program& program, Player& P)
{
        cout << "Start running evaluator..." << endl;
	uint64_t ms1 = time_ms();
        int i = 0;
	auto next = GC::TIME_BREAK;
	do
	{
	  cout << "A " << time_ms()-ms1 << endl;
		receive_load(P, master.opts.gcs_saveprefix + "_" + to_string(i));
	  cout << "B " << time_ms()-ms1 << endl;		
		try
		{
			next = program.execute(processor, master.memory, -1);
		}
		catch (needs_cleaning& e)
		{
		}
		i++;
	  cout << "C " << time_ms()-ms1 << endl;
	}
	while(GC::DONE_BREAK != next);

	uint64_t ms2 = time_ms();
	cout << "Evaluator elapsed " << ms2-ms1 << endl;
}

void YaoEvaluator::run_from_store(GC::Program& program)
{
	machine.reset_timer();
	do
	{
		gates_store.pop(gates);
		output_masks_store.pop(output_masks);
	}
	while(GC::DONE_BREAK != program.execute(processor, master.memory, -1));
}

bool YaoEvaluator::receive_load(Player& P, string fname)
{
  if (P.receive_long(0) == YaoCommon::DONE)
    return false;
	P.receive_player_and_readfile(0, gates, fname);
	P.receive_player_and_readfile(0, output_masks, fname+"mask");
	cout << "received " << gates.size() << " gates and " << output_masks.size()
	        << " output masks at " << processor.PC << endl;
	return true;
}


bool YaoEvaluator::receive(Player& P)
{
	if (P.receive_long(0) == YaoCommon::DONE)
		return false;

	P.receive_player(0, gates);
	P.receive_player(0, output_masks);
	cout << "received " << gates.size() << " gates and " << output_masks.size()
	        << " output masks at " << processor.PC << endl;
	return true;
}

void YaoEvaluator::receive_to_store(Player& P)
{
        cout << "Start running evaluator..." << endl;
	int i = 0;
	uint64_t t1 = time_ms();

	while (receive_load(P, master.opts.gcs_saveprefix + "_" + to_string(i)))
	{
	        t1 = time_ms();	  
		gates_store.push(gates);
		output_masks_store.push(output_masks);
		i += 1;
	}
	uint64_t t2 = time_ms();
	cout << "Evaluator elapsed: " << t2-t1 << endl;
}
