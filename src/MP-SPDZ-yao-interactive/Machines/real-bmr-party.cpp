/*
 * real-bmr-party.cpp
 *
 */

#include "BMR/RealProgramParty.hpp"
#include "Machines/SPDZ.hpp"

int main(int argc, const char** argv)
{
	RealProgramParty<Share<gf2n_long>>(argc, argv);
}
