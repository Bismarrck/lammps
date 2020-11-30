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
#include "fmt/format.h"
#include "force.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "utils.h"
#include <cmath>
#include <domain.h>
#include <map>
#include <vector>

using namespace LAMMPS_NS;

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
  cutmax = 0.0;
  cutforcesq = 0.0;
  single_enable = 0;
  one_coeff = 1;
  manybody_flag = 1;
  comm_forward = 0;
  comm_reverse = 0;

  // Internal variables and pointers
  ijtypes = nullptr;
  ijnums = nullptr;
  vap = nullptr;
  graph_model = nullptr;
  positions = nullptr;
  cell = nullptr;
  volume = nullptr;
  n_atoms_vap_tensor = nullptr;
  atom_masks = nullptr;
  pulay_stress = nullptr;
  etemperature = nullptr;
  nnl_max = nullptr;
  row_splits = nullptr;
  etemp = 0.0;
  neigh_extra = 0.05;
  num_calls = 0;
  nij_max_sum = 0;
  nnl_max_sum = 0;
  elapsed = 0.0;
  use_hyper_thread = false;
  dynamic_bytes = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorAlloy::~PairTensorAlloy() {

  LOGFILE(fmt::format("Total number of session->run calls = {:d}/core\n",
                      num_calls))
  LOGFILE(fmt::format("Average session->run cost: {:.2f} ms/core\n",
                      elapsed / num_calls))
  LOGFILE(fmt::format("Average nnl_max: {:.0f}\n", nnl_max_sum / num_calls))
  LOGFILE(fmt::format("Average nij_max: {:.0f}\n", nij_max_sum / num_calls))

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(ijtypes);
    memory->destroy(ijnums);

    delete positions;
    positions = nullptr;

    delete cell;
    cell = nullptr;

    delete volume;
    volume = nullptr;

    delete n_atoms_vap_tensor;
    n_atoms_vap_tensor = nullptr;

    delete nnl_max;
    nnl_max = nullptr;

    delete pulay_stress;
    pulay_stress = nullptr;

    delete etemperature;
    etemperature = nullptr;

    delete atom_masks;
    atom_masks = nullptr;

    delete row_splits;
    row_splits = nullptr;
  }

  if (vap) {
    delete vap;
    vap = nullptr;
  }

  if (graph_model) {
    delete graph_model;
    graph_model = nullptr;
  }
}

/* ----------------------------------------------------------------------
   Update the ASE-style lattice matrix.

   * h: the 3x3 lattice tensor
   * volume: the volume of the latticce
------------------------------------------------------------------------- */
template <typename T> double PairTensorAlloy::update_cell() {
  auto cell_ = cell->template tensor<T, 2>();
  cell_(0, 0) = static_cast<T>(domain->h[0]);
  cell_(0, 1) = static_cast<T>(0.0);
  cell_(0, 2) = static_cast<T>(0.0);
  cell_(1, 0) = static_cast<T>(domain->h[5]);
  cell_(1, 1) = static_cast<T>(domain->h[1]);
  cell_(1, 2) = static_cast<T>(0.0);
  cell_(2, 0) = static_cast<T>(domain->h[4]);
  cell_(2, 1) = static_cast<T>(domain->h[3]);
  cell_(2, 2) = static_cast<T>(domain->h[2]);

  // Compute the volume.
  double vol;
  if (domain->dimension == 3) {
    vol = domain->xprd * domain->yprd * domain->zprd;
  } else {
    vol = domain->xprd * domain->yprd;
  }
  return vol;
}

/* ----------------------------------------------------------------------
   The `compute` method.
------------------------------------------------------------------------- */

template <typename T>
void PairTensorAlloy::run(int eflag, int vflag, DataType dtype) {
  int i, j;
  int ii, jj, inum, jnum, itype, jtype;
  int ijtype;
  int nij, nij_max;
  int nnl = 0;
  int inc;
  double rsq;
  double rijx, rijy, rijz;
  double fx, fy, fz;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int *ivec, *jvec;
  int newton_pair;
  const int32 *local_to_vap_map, *vap_to_local_map;

  // Initialze the flags
  ev_init(eflag, vflag);

  // Update all tensors if needed.
  vap->build(atom->nlocal, atom->type);
  update_tensors<T>(dtype);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  newton_pair = force->newton_pair;
  local_to_vap_map = vap->get_local_to_vap_map();
  vap_to_local_map = vap->get_vap_to_local_map();

  // Cell and volume
  volume->flat<T>()(0) = DOUBLE(update_cell<T>());

  // Positions: will be removed later
  positions->tensor<T, 2>().setConstant(0.0f);

  // Determine nij_max
  // Lammps adds an extra `skin` to `cutoff`, so `nij_max` should be adjusted
  double neigh_coef =
      pow(cutmax / (neighbor->skin + cutmax), 3.0) + neigh_extra;
  nij_max = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    nij_max += numneigh[i];
  }
  nij_max = static_cast<int>(nij_max * neigh_coef);
  ivec = new int[nij_max];
  jvec = new int[nij_max];
  for (nij = 0; nij < nij_max; nij++) {
    ivec[nij] = 0;
    jvec[nij] = 0;
  }
  nij = 0;

  // Reset the counters
  for (i = 0; i < vap->get_n_atoms_vap(); i++) {
    for (j = 0; j < atom->ntypes + 1; j++) {
      ijnums[i][j] = 0;
    }
  }

  auto rdists = new Tensor(dtype, TensorShape({4, nij_max}));
  auto rdists_ = rdists->tensor<T, 2>();
  rdists_.setConstant(0.0);

  auto rmap = new Tensor(DT_INT32, TensorShape({nij_max, 5}));
  auto rmap_ = rmap->tensor<int32, 2>();
  rmap_.setConstant(0);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      rijx = atom->x[j][0] - atom->x[i][0];
      rijy = atom->x[j][1] - atom->x[i][1];
      rijz = atom->x[j][2] - atom->x[i][2];
      rsq = rijx * rijx + rijy * rijy + rijz * rijz;

      if (rsq < cutforcesq) {
        if (nij == nij_max) {
          error->all(FLERR, "tensoralloy: neigh_coef is too small");
        }

        jtype = atom->type[j];

        rdists_(0, nij) = DOUBLE(sqrt(rsq));
        rdists_(1, nij) = DOUBLE(rijx);
        rdists_(2, nij) = DOUBLE(rijy);
        rdists_(3, nij) = DOUBLE(rijz);
        ivec[nij] = i;
        jvec[nij] = j;

        ijtype = ijtypes[itype][jtype];
        inc = ijnums[local_to_vap_map[i]][ijtype];
        nnl = MAX(inc + 1, nnl);

        rmap_(nij, 0) = ijtype;
        rmap_(nij, 1) = local_to_vap_map[i];
        rmap_(nij, 2) = inc;
        rmap_(nij, 3) = 0;
        rmap_(nij, 4) = 1;
        nij += 1;
        ijnums[local_to_vap_map[i]][ijtype] += 1;
      }
    }
  }

  // Set the nnl_max
  nnl_max->flat<int32>()(0) = nnl + 1;

  std::vector<std::pair<string, Tensor>> feed_dict({
      {"Placeholders/positions", *positions},
      {"Placeholders/n_atoms_vap", *n_atoms_vap_tensor},
      {"Placeholders/nnl_max", *nnl_max},
      {"Placeholders/atom_masks", *atom_masks},
      {"Placeholders/cell", *cell},
      {"Placeholders/volume", *volume},
      {"Placeholders/pulay_stress", *pulay_stress},
      {"Placeholders/etemperature", *etemperature},
      {"Placeholders/row_splits", *row_splits},
      {"Placeholders/g2.v2g_map", *rmap},
      {"Placeholders/g2.rij", *rdists},
  });

  if (graph_model->is_angular()) {
    error->all(FLERR, "Angular part is not implemented yet!");
  }

  auto begin = std::chrono::high_resolution_clock::now();
  std::vector<Tensor> outputs = graph_model->run(feed_dict, error, false);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count());
  if (comm->me == 0) {
    num_calls++;
    elapsed += ms;
    nnl_max_sum += DOUBLE(nnl + 1);
    nij_max_sum += DOUBLE(nij);
  }

  if (graph_model->is_finite_temperature() &&
      strcmp(atom->atom_style, "tensoralloy") == 0) {
    int idx = graph_model->get_index_eentropy(true);
    auto eentropy_atom = outputs[idx].flat<T>();
    for (i = 1; i < vap->get_n_atoms_vap(); i++) {
      atom->eentropy[i] = eentropy_atom(i - 1);
    }
  }

  auto dEdrij =
      outputs[graph_model->get_index_partial_forces(false)].matrix<T>();
  for (nij = 0; nij < nij_max; nij++) {
    double rij = rdists_(0, nij);
    if (std::abs(rij) < 1e-6) {
      rijx = 0;
      rijy = 0;
      rijz = 0;
      fx = 0;
      fy = 0;
      fz = 0;
    } else {
      rijx = rdists_(1, nij);
      rijy = rdists_(2, nij);
      rijz = rdists_(3, nij);
      fx = dEdrij(0, nij) * rijx / rij + dEdrij(1, nij);
      fy = dEdrij(0, nij) * rijy / rij + dEdrij(2, nij);
      fz = dEdrij(0, nij) * rijz / rij + dEdrij(3, nij);
    }
    i = ivec[nij];
    j = jvec[nij];
    atom->f[i][0] += fx;
    atom->f[i][1] += fy;
    atom->f[i][2] += fz;
    atom->f[j][0] -= fx;
    atom->f[j][1] -= fy;
    atom->f[j][2] -= fz;
    if (evflag) {
      ev_tally_xyz(i, j, atom->nlocal, newton_pair, 0.0, 0.0, fx, fy, fz, rijx,
                   rijy, rijz);
    }
  }

  auto pe_atom =
      outputs[graph_model->get_index_variation_energy(true)].flat<T>();
  for (i = 1; i < vap->get_n_atoms_vap(); i++) {
    if (eflag) {
      ev_tally_full(vap_to_local_map[i], 2.0 * pe_atom(i - 1), 0.0, 0.0, 0.0,
                    0.0, 0.0);
    }
  }

  if (vflag_fdotr) {
    virial_fdotr_compute();
  }

  dynamic_bytes = 0;
  dynamic_bytes += rmap->TotalBytes();
  dynamic_bytes += rdists->TotalBytes();
  dynamic_bytes += vap->memory_usage();

  delete rmap;
  delete rdists;
  delete[] ivec;
  delete[] jvec;
}

/* ----------------------------------------------------------------------
   compute with different precision
------------------------------------------------------------------------- */

void PairTensorAlloy::compute(int eflag, int vflag) {
  if (graph_model->is_fp64()) {
    run<double>(eflag, vflag, DataType::DT_DOUBLE);
  } else {
    run<float>(eflag, vflag, DataType::DT_FLOAT);
  }
}

/* ----------------------------------------------------------------------
   Update tensors. Some may be reallocated.
------------------------------------------------------------------------- */

template <typename T> void PairTensorAlloy::update_tensors(DataType dtype) {
  int i, j;
  int n = atom->ntypes + 1;

  if (!vap->updated()) {
    return;
  }

  int n_atoms_vap = vap->get_n_atoms_vap();

  // Radial interaction counters
  memory->grow(ijnums, n_atoms_vap, n, "pair:tensoralloy:ijnums");
  for (i = 0; i < n_atoms_vap; i++) {
    for (j = 0; j < n; j++) {
      ijnums[i][j] = 0;
    }
  }

  // Positions
  if (positions) {
    delete positions;
    positions = nullptr;
  }
  positions = new Tensor(dtype, TensorShape({n_atoms_vap, 3}));
  positions->tensor<T, 2>().setConstant(0.f);

  // row splits tensor
  if (row_splits) {
    delete row_splits;
    row_splits = nullptr;
  }
  row_splits = new Tensor(DT_INT32, TensorShape({n}));
  row_splits->tensor<int32, 1>().setConstant(0);
  for (i = 0; i < n; i++) {
    row_splits->tensor<int32, 1>()(i) = vap->get_row_splits()[i];
  }

  // Atom masks tensor
  if (atom_masks) {
    delete atom_masks;
    atom_masks = nullptr;
  }
  atom_masks = new Tensor(dtype, TensorShape({n_atoms_vap}));
  auto masks = vap->get_atom_masks();
  for (i = 0; i < n_atoms_vap; i++) {
    atom_masks->tensor<T, 1>()(i) = static_cast<T>(masks[i]);
  }

  // N_atom_vap tensor
  if (n_atoms_vap_tensor == nullptr) {
    n_atoms_vap_tensor = new Tensor(DT_INT32, TensorShape());
  }
  n_atoms_vap_tensor->flat<int32>()(0) = n_atoms_vap;

  // nnl_max tensor
  if (nnl_max == nullptr) {
    nnl_max = new Tensor(DT_INT32, TensorShape());
  }

  // Pulay stress tensor
  if (pulay_stress == nullptr) {
    pulay_stress = new Tensor(dtype, TensorShape());
    pulay_stress->flat<T>()(0) = 0.0f;
  }

  // electron temperature tensor
  if (etemperature == nullptr) {
    etemperature = new Tensor(dtype, TensorShape());
    etemperature->flat<T>()(0) = etemp;
  }

  // Volume
  if (volume == nullptr) {
    volume = new Tensor(dtype, TensorShape());
  }

  // Lattice
  if (cell == nullptr) {
    cell = new Tensor(dtype, TensorShape({3, 3}));
    cell->tensor<T, 2>().setConstant(0.f);
  }
}

/* ----------------------------------------------------------------------
   Allocate arrays
------------------------------------------------------------------------- */

template <typename T> void PairTensorAlloy::allocate(DataType dtype) {
  if (vap == nullptr) {
    error->all(FLERR, "VAP is not succesfully initialized.");
  }

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

  // Radial interactions
  memory->create(ijtypes, n, n, "pair:tensoralloy:ijtypes");
  for (i = 1; i < n; i++) {
    ijtypes[i][i] = 0;
    int val = 1;
    for (j = 1; j < n; j++) {
      if (j != i) {
        ijtypes[i][j] = val;
        val++;
      }
    }
  }

  update_tensors<T>(dtype);
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
  // Read atom types from the lammps input file.
  std::vector<string> symbols;
  symbols.emplace_back("X");
  int idx = 1;
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

  // Load the graph model
  graph_model = new GraphModel(this->lmp, string(arg[0]), symbols, error,
                               use_hyper_thread, comm->me == 0);

  // Initialize the Virtual-Atom Map
  vap = new VirtualAtomMap(memory, graph_model->get_n_elements());
  vap->build(atom->nlocal, atom->type);
  LOGFILE("VAP initialized\n")

  // Allocate arrays and tensors.
  if (graph_model->is_fp64()) {
    allocate<double>(DataType::DT_DOUBLE);
  } else {
    allocate<float>(DataType::DT_FLOAT);
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
  cutmax = graph_model->get_cutoff();
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
  bytes += cell->TotalBytes();
  bytes += positions->TotalBytes();
  bytes += volume->TotalBytes();
  bytes += atom_masks->TotalBytes();
  bytes += nnl_max->TotalBytes();
  bytes += pulay_stress->TotalBytes();
  bytes += etemperature->TotalBytes();
  bytes += atom_masks->TotalBytes();
  bytes += dynamic_bytes;
  return bytes;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairTensorAlloy::memory_usage() {
  double bytes = Pair::memory_usage();
  bytes += tensors_memory_usage();
  bytes += vap->memory_usage();
  return bytes;
}
