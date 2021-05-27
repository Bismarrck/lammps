//
// Created by Xin Chen on 2020/11/30.
//

#include "tensoralloy.h"
#include "utils.h"
#include "memory.h"
#include "error.h"

#define eV_to_Kelvin 11604.51812
#define DOUBLE(x) static_cast<double>(x)

#if !defined(NEIGHMASK)
#define NEIGHMASK 0x3FFFFFFF
#endif

using namespace LAMMPS_NS;

using tensorflow::DataType;
using tensorflow::DT_DOUBLE;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::TensorShape;

/* ----------------------------------------------------------------------
   Constructor
------------------------------------------------------------------------- */

TensorAlloy::TensorAlloy(LAMMPS *lmp,
                         const string &graph_model_path,
                         const std::vector<string> &symbols, int nlocal,
                         int ntypes, int *itypes, bool verbose) : Pointers(lmp)
{
  // Initialize internal variables
  neigh_coef = 1.0;
  cutforcesq = 0.0;
  collect_statistics = false;

  // Set all internal pointers to null
  ijnums = nullptr;
  ijtypes = nullptr;
  n_atoms_vap_tensor = nullptr;
  nnl_max = nullptr;
  etemperature = nullptr;
  atom_masks = nullptr;
  row_splits = nullptr;
  graph_model = nullptr;
  vap = nullptr;

  // Load the graph model
  graph_model =
      new GraphModel(this->lmp, graph_model_path, symbols, false, verbose);

  // Initialize the Virtual-Atom Map
  vap = new VirtualAtomMap(this->lmp, graph_model->get_n_elements());
  vap->build(nlocal, itypes);
  if (verbose) {
    utils::logmesg(this->lmp, "VAP initialized\n");
  }

  // Allocate arrays and tensors.
  if (graph_model->is_fp64()) {
    allocate<double>(DT_DOUBLE, ntypes);
  } else {
    allocate<float>(DT_FLOAT, ntypes);
  }

  // Set `cutmax`.
  double rc = graph_model->get_cutoff();
  cutforcesq = rc * rc;
}

/* ----------------------------------------------------------------------
   Desctructor
------------------------------------------------------------------------- */

TensorAlloy::~TensorAlloy() {
  memory->destroy(ijtypes);
  memory->destroy(ijnums);

  delete n_atoms_vap_tensor;
  n_atoms_vap_tensor = nullptr;

  delete nnl_max;
  nnl_max = nullptr;

  delete etemperature;
  etemperature = nullptr;

  delete atom_masks;
  atom_masks = nullptr;

  delete row_splits;
  row_splits = nullptr;

  if (vap) {
    delete vap;
    vap = nullptr;
  }

  if (graph_model) {
    delete graph_model;
    graph_model = nullptr;
  }

  delete memory;
  memory = nullptr;
}

/* ----------------------------------------------------------------------
   Update tensors. Some may be reallocated.
------------------------------------------------------------------------- */

template <typename T>
void TensorAlloy::update_tensors(DataType dtype, int ntypes, double etemp) {
  int i, j;
  int n = ntypes + 1;

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

  // electron temperature tensor
  if (etemperature == nullptr) {
    etemperature = new Tensor(dtype, TensorShape());
  }
  etemperature->flat<T>()(0) = etemp;
}

/* ----------------------------------------------------------------------
   Allocate arrays
------------------------------------------------------------------------- */

template <typename T> void TensorAlloy::allocate(DataType dtype, int ntypes) {
  if (vap == nullptr) {
    error->all(FLERR, "VAP is not succesfully initialized");
  }

  int n = ntypes + 1;
  int i, j;

  // Radial interactions
  memory->create(ijtypes, n, n, "libtensoralloy:ijtypes");
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

  update_tensors<T>(dtype, ntypes, 0.0);
}

/* ----------------------------------------------------------------------
   Run once
------------------------------------------------------------------------- */

template <typename T>
Status TensorAlloy::run(DataType dtype, int nlocal, int ntypes, int *itypes,
                        const int *ilist, const int *numneigh, int **firstneigh,
                        double **x, double **f, double *eentropy, double etemp,
                        double &etotal, double *eatom, double *virial,
                        double **vatom) {
  int i, j;
  int ii, jj, jnum, itype, jtype;
  int ijtype;
  int nij, nij_max;
  int nnl = 0;
  int inc;
  double rsq;
  double rijx, rijy, rijz;
  double fx, fy, fz;
  int *ivec, *jvec, *jlist;
  double v[6] = {0, 0, 0, 0, 0, 0};
  const int32 *local_to_vap_map, *vap_to_local_map;

  // Update all tensors if needed.
  vap->build(nlocal, itypes);
  update_tensors<T>(dtype, ntypes, etemp);

  local_to_vap_map = vap->get_local_to_vap_map();
  vap_to_local_map = vap->get_vap_to_local_map();

  nij_max = 0;
  for (ii = 0; ii < nlocal; ii++) {
    i = ilist[ii];
    nij_max += numneigh[i];
  }
  nij_max = static_cast<int>(nij_max * MIN(neigh_coef, 1.0));
  ivec = new int[nij_max];
  jvec = new int[nij_max];
  for (nij = 0; nij < nij_max; nij++) {
    ivec[nij] = 0;
    jvec[nij] = 0;
  }
  nij = 0;

  // Reset the counters
  for (i = 0; i < vap->get_n_atoms_vap(); i++) {
    for (j = 0; j < ntypes + 1; j++) {
      ijnums[i][j] = 0;
    }
  }

  auto rdists = new Tensor(dtype, TensorShape({4, nij_max}));
  auto rdists_ = rdists->tensor<T, 2>();
  rdists_.setConstant(0.0);

  auto rmap = new Tensor(DT_INT32, TensorShape({nij_max, 5}));
  auto rmap_ = rmap->tensor<int32, 2>();
  rmap_.setConstant(0);

  for (ii = 0; ii < nlocal; ii++) {
    i = ilist[ii];
    i &= NEIGHMASK;
    itype = itypes[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      rijx = x[j][0] - x[i][0];
      rijy = x[j][1] - x[i][1];
      rijz = x[j][2] - x[i][2];
      rsq = rijx * rijx + rijy * rijy + rijz * rijz;

      if (rsq < cutforcesq) {
        if (nij == nij_max) {
          error->all(FLERR, "tensoralloy: neigh_coef is too small");
        }

        jtype = itypes[j];

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

  // Set the electron temperature
  etemperature->flat<T>()(0) = etemp;

  std::vector<std::pair<string, Tensor>> feed_dict({
      {"Placeholders/n_atoms_vap", *n_atoms_vap_tensor},
      {"Placeholders/nnl_max", *nnl_max},
      {"Placeholders/atom_masks", *atom_masks},
      {"Placeholders/etemperature", *etemperature},
      {"Placeholders/row_splits", *row_splits},
      {"Placeholders/g2.v2g_map", *rmap},
      {"Placeholders/g2.rij", *rdists},
  });

  if (graph_model->is_angular()) {
    error->all(FLERR, "Angular part is not implemented yet!");
  }

  auto begin = std::chrono::high_resolution_clock::now();
  tensorflow::Status status;
  std::vector<Tensor> outputs = graph_model->run(feed_dict, status, false);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count());
  if (collect_statistics) {
    call_stats.num_calls++;
    call_stats.elapsed += ms;
    call_stats.nnl_max_sum += DOUBLE(nnl + 1);
    call_stats.nij_max_sum += DOUBLE(nij);
  }

  if (graph_model->is_finite_temperature() && eentropy) {
    int idx = graph_model->get_index_eentropy(true);
    auto satom = outputs[idx].flat<T>();
    for (i = 1; i < vap->get_n_atoms_vap(); i++) {
      eentropy[vap_to_local_map[i]] = satom(i - 1);
    }
  }

  if (f) {
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
      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      f[j][0] -= fx;
      f[j][1] -= fy;
      f[j][2] -= fz;
      if (virial) {
        v[0] = rijx * fx;
        v[1] = rijy * fy;
        v[2] = rijz * fz;
        v[3] = rijx * fy;
        v[4] = rijx * fz;
        v[5] = rijy * fz;

        virial[0] += v[0];
        virial[1] += v[1];
        virial[2] += v[2];
        virial[3] += v[3];
        virial[4] += v[4];
        virial[5] += v[5];

        if (vatom) {
          if (i < nlocal) {
            vatom[i][0] += 0.5 * v[0];
            vatom[i][1] += 0.5 * v[1];
            vatom[i][2] += 0.5 * v[2];
            vatom[i][3] += 0.5 * v[3];
            vatom[i][4] += 0.5 * v[4];
            vatom[i][5] += 0.5 * v[5];
          }
          if (j < nlocal) {
            vatom[j][0] += 0.5 * v[0];
            vatom[j][1] += 0.5 * v[1];
            vatom[j][2] += 0.5 * v[2];
            vatom[j][3] += 0.5 * v[3];
            vatom[j][4] += 0.5 * v[4];
            vatom[j][5] += 0.5 * v[5];
          }
        }
      }
    }
  }

  auto pe = outputs[graph_model->get_index_variation_energy(true)].flat<T>();
  for (i = 1; i < vap->get_n_atoms_vap(); i++) {
    double evdwl = pe(i - 1);
    if (eatom) {
      eatom[vap_to_local_map[i]] = evdwl;
    }
    etotal += evdwl;
  }

  delete rmap;
  delete rdists;
  delete[] ivec;
  delete[] jvec;

  return status;
}

/* ----------------------------------------------------------------------
   The compute methods
------------------------------------------------------------------------- */

Status TensorAlloy::compute(int nlocal, int ntypes, int *itypes,
                            const int *ilist, const int *numneigh,
                            int **firstneigh, double **x, double **f,
                            double *eentropy, double etemp, double &etotal,
                            double *eatom, double *virial, double **vatom) {
  if (graph_model->is_fp64()) {
    return run<double>(DT_DOUBLE, nlocal, ntypes, itypes, ilist, numneigh,
                       firstneigh, x, f, eentropy, etemp, etotal, eatom, virial,
                       vatom);
  } else {
    return run<float>(DT_FLOAT, nlocal, ntypes, itypes, ilist, numneigh,
                      firstneigh, x, f, eentropy, etemp, etotal, eatom, virial,
                      vatom);
  }
}

Status TensorAlloy::compute(int nlocal, int ntypes, int *itypes,
                            const int *ilist, const int *numneigh,
                            int **firstneigh, double **x, double etemp,
                            double &etotal, double *eatom) {
  if (graph_model->is_fp64())
    return run<double>(DT_DOUBLE, nlocal, ntypes, itypes, ilist, numneigh,
                       firstneigh, x, nullptr, nullptr, etemp, etotal, eatom,
                       nullptr, nullptr);
  else
    return run<float>(DT_FLOAT, nlocal, ntypes, itypes, ilist, numneigh,
                      firstneigh, x, nullptr, nullptr, etemp, etotal, eatom,
                      nullptr, nullptr);
}
