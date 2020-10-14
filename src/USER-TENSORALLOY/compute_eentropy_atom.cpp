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

/* ----------------------------------------------------------------------
   Contributing author: Andres Jaramillo-Botero
------------------------------------------------------------------------- */

#include "compute_eentropy_atom.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include <cstdlib>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeAtomicElectronEntropy::ComputeAtomicElectronEntropy(LAMMPS *lmp,
                                                           int narg, char **arg)
    : Compute(lmp, narg, arg) {
  if (narg != 3)
    error->all(FLERR, "Illegal compute eentropy/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  eentropy = NULL;

  // error check

  if (!atom->tensoralloy_flag)
    error->all(FLERR, "Compute eentropy requires atom style tensoralloy");
}

/* ---------------------------------------------------------------------- */

ComputeAtomicElectronEntropy::~ComputeAtomicElectronEntropy() {
  memory->destroy(eentropy);
}

/* ---------------------------------------------------------------------- */

void ComputeAtomicElectronEntropy::init() {
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style, "eentropy/atom") == 0)
      count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR, "More than one compute eentropy/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeAtomicElectronEntropy::compute_peratom() {
  invoked_peratom = update->ntimestep;
  if (atom->nmax > nmax) {
    memory->destroy(eentropy);
    nmax = atom->nmax;
    memory->create(eentropy, nmax, "compute/eentropy/atom:eentropy");
    vector_atom = eentropy;
  }
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (mass)
    for (int i = 0; i < nlocal; i++) {
      eentropy[i] = atom->eentropy[i];
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeAtomicElectronEntropy::memory_usage() {
  double bytes = nmax * sizeof(double);
  return bytes;
}
