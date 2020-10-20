/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_etemp.h"
#include "domain.h"
#include "error.h"
#include "compute.h"
#include "fix_store.h"
#include "force.h"
#include "group.h"
#include "modify.h"
#include "pair.h"
#include "update.h"
#include <cstring>

#define eV_to_Kelvin 11604.51812

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixElectronTemperature::FixElectronTemperature(LAMMPS *lmp, int narg,
                                               char **arg)
    : Fix(lmp, narg, arg) {
  if (narg != 4)
    error->all(FLERR, "Illegal fix adapt command");
  nevery = force->inumeric(FLERR, arg[3]);
  if (nevery < 0)
    error->all(FLERR, "Illegal fix adapt command");
  kelvin = 0;
  scalar_flag = 1;
}

/* ---------------------------------------------------------------------- */

FixElectronTemperature::~FixElectronTemperature()
= default;

/* ---------------------------------------------------------------------- */

int FixElectronTemperature::setmask() {
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixElectronTemperature::init() {
  if (strcmp(force->pair_style, "tensoralloy") != 0) {
    error->all(FLERR, "pair_style must be tensoralloy for fix etemp");
  }
  pair = dynamic_cast<PairTensorAlloy *>(force->pair);
  kelvin = pair->get_etemp() * eV_to_Kelvin;
}

/* ---------------------------------------------------------------------- */

void FixElectronTemperature::end_of_step() {
  if (nevery == 0)
    return;
  if (update->ntimestep % nevery)
    return;
  change_settings();
}

/* ---------------------------------------------------------------------- */

void FixElectronTemperature::change_settings() {
  modify->clearstep_compute();
  int ipe = modify->find_compute("thermo_temp");
  if (ipe == -1)
    error->all(FLERR,"compute thermo_pe does not exist.");
  Compute *c_temp = modify->compute[ipe];
  kelvin = c_temp->compute_scalar();
  pair->set_etemp(kelvin / 11604.51812);
  modify->addstep_compute(update->ntimestep + nevery);
}

/* ---------------------------------------------------------------------- */

double FixElectronTemperature::compute_scalar() {
  return kelvin;
}
