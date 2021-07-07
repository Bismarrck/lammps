//
// Created by Xin Chen on 2019-06-11.
//
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

#include "pair_tensoralloy.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "fmt/format.h"
#include "force.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "utils.h"
#include <cmath>
#include <domain.h>
#include <vector>
#include <functional>

using namespace LAMMPS_NS;
using fmt::format;

using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;

#define MAXLINE 1024
#define eV_to_Kelvin 11604.51812
#define DOUBLE(x) static_cast<double>(x)
#define LOGFILE(x)                                                             \
  if (comm->me == 0) {                                                         \
    utils::logmesg(this->lmp, x);                                              \
  }

/* ---------------------------------------------------------------------- */

PairTensorAlloy::PairTensorAlloy(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  single_enable = 0;
  one_coeff = 1;
  manybody_flag = 1;
  comm_forward = 0;
  comm_reverse = 0;

  // Internal variables and pointers
  etemp = 0.0;
  use_hyper_thread = false;
  dynamic_bytes = 0;
  calc = nullptr;
  neigh_extra = 0.25;
  cutmax = 0.0;
  cutforcesq = 0.0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorAlloy::~PairTensorAlloy() {
  CallStatistics stats = calc->get_call_statistics();
  if (stats.num_calls > 0) {
    LOGFILE(format("Total session->run calls = {:.0f}/core\n", stats.num_calls))
    LOGFILE(format("Avg session->run cost: {:.2f} ms/core\n",
                   stats.elapsed / stats.num_calls))
    LOGFILE(format("Avg nnl_max: {:.0f}\n", stats.nnl_max_sum / stats.num_calls))
    LOGFILE(format("Avg nij_max: {:.0f}\n", stats.nij_max_sum / stats.num_calls))
  }
  if (allocated && calc) {
    delete calc;
    calc = nullptr;
  }
}

/* ----------------------------------------------------------------------
    The `compute` method
------------------------------------------------------------------------- */

void PairTensorAlloy::compute(int eflag, int vflag)
{
  // Initialze the flags
  ev_init(eflag, vflag);

  double neigh_coef =
      pow(cutmax / (neighbor->skin + cutmax), 3.0) + neigh_extra;
  calc->set_neigh_coef(neigh_coef);

  double etotal = 0.0;
  double vtotal[6] = {0, 0, 0, 0, 0, 0};

  Status status = calc->compute(
      atom->nlocal, atom->ntypes, atom->type, list->ilist, list->numneigh,
      list->firstneigh, atom->x, atom->f, atom->eentropy, etemp, etotal, eatom,
      vtotal, vatom);

  if (eflag) {
    eng_vdwl = etotal;
  }
  if (vflag_global) {
    for (int i = 0; i < 6; i ++) {
      virial[i] = vtotal[i];
    }
  }
  if (vflag_fdotr) {
    virial_fdotr_compute();
  }

  dynamic_bytes = 0;
  dynamic_bytes += calc->get_vap()->memory_usage();
}

/* ----------------------------------------------------------------------
   Allocate arrays
------------------------------------------------------------------------- */

void PairTensorAlloy::allocate() {
  allocated = 1;

  int n = atom->ntypes + 1;
  int i, j;

  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(setflag, n, n, "pair:setflag");
  for (i = 1; i < n; i++) {
    for (j = i; j < n; j++) {
      setflag[i][j] = 0;
    }
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTensorAlloy::settings(int narg, char ** /*arg*/) {
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   The implementation of `pair_coef
------------------------------------------------------------------------- */

void PairTensorAlloy::coeff(int narg, char **arg) {

  // Skip first one or two `*`
  int idx = 0;
  while (idx < narg) {
    if (strcmp(arg[idx], "*") == 0) {
      idx ++;
    }
    else break;;
  }
  const int istart = idx;

  // Read atom types from the lammps input file.
  std::vector<string> symbols;
  symbols.emplace_back("X");
  idx = istart + 1;
  while (idx < narg) {
    auto iarg = string(arg[idx]);
    if (iarg == "hyper") {
      auto val = string(arg[idx + 1]);
      if (val == "off") {
        use_hyper_thread = false;
      } else if (val == "on") {
        use_hyper_thread = true;
        LOGFILE(fmt::format("Hyper thread mode is enabled\n"))
      } else {
        error->all(FLERR, "'on/off' are valid values for key 'hyper'");
      }
      idx++;
    } else if (iarg == "etemp") {
      double kelvin = std::atof(string(arg[idx + 1]).c_str());
      etemp = kelvin / eV_to_Kelvin;
      if (comm->me == 0) {
        LOGFILE(fmt::format("Electron temperature is {:.4f} eV\n", etemp))
      }
      idx++;
    } else if (iarg == "neigh_extra") {
      neigh_extra = std::atof(string(arg[idx + 1]).c_str());
      if (comm->me == 0) {
        LOGFILE(fmt::format("Neigh_coef skin is {:.2f}\n", neigh_extra))
      }
      idx++;
    } else {
      symbols.emplace_back(iarg);
    }
    idx++;
  }

  allocate();

  // Load the graph model
  bool cpu0 = comm->me == 0;
  calc = new TensorAlloy(this->lmp, string(arg[istart]), symbols, atom->nlocal,
                         atom->ntypes, atom->type, cpu0);
  if (cpu0) {
    calc->collect_call_statistics();
  }

  // Set atomic masses
  double atom_mass[symbols.size()];
  for (int i = 1; i <= atom->ntypes; i++) {
    atom_mass[i] = 1.0;
  }
  atom->set_mass(atom_mass);

  // Set `setflag` which is required by Lammps.
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      setflag[i][j] = 1;
    }
  }

  // Set `cutmax`.
  cutmax = calc->get_model()->get_cutoff();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTensorAlloy::init_style() {
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style Tersoff requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style Tersoff requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTensorAlloy::init_one(int /*i*/, int /*j*/) {
  // single global cutoff = max of cut from all files read in
  // for funcfl could be multiple files
  // for setfl or fs, just one file
  cutforcesq = cutmax * cutmax;
  return cutmax;
}

/* ----------------------------------------------------------------------
   memory usage of tensors
------------------------------------------------------------------------- */

double PairTensorAlloy::tensors_memory_usage() {
  double bytes = 0.0;
  bytes += dynamic_bytes;
  return bytes;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairTensorAlloy::memory_usage() {
  double bytes = Pair::memory_usage();
  bytes += tensors_memory_usage();
  return bytes;
}
