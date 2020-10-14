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

#include "compute_eentropy.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "update.h"
#include <mpi.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeElectronEntropy::ComputeElectronEntropy(LAMMPS *lmp, int narg,
                                               char **arg)
    : Compute(lmp, narg, arg) {
  if (narg != 3)
    error->all(FLERR, "Illegal compute eentropy command");

  scalar_flag = 1;
  extscalar = 1;

  // error check

  if (!atom->tensoralloy_flag)
    error->all(FLERR, "Compute eentropy requires atom style tensoralloy");
}

/* ---------------------------------------------------------------------- */

void ComputeElectronEntropy::init() {}

/* ---------------------------------------------------------------------- */

double ComputeElectronEntropy::compute_scalar() {
  invoked_scalar = update->ntimestep;
  int nlocal = atom->nlocal;
  double eentropy = 0.0;
  for (int i = 0; i < nlocal; i++) {
    eentropy += atom->eentropy[i];
  }
  MPI_Allreduce(&eentropy, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  return scalar;
}
