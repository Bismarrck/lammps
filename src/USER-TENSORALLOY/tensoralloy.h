//
// Created by Xin Chen on 2020/11/30.
//

#ifndef LIBTENSORALLOY_TENSORALLOY_H
#define LIBTENSORALLOY_TENSORALLOY_H

#include "graph_model.h"
#include "virtual_atom_approach.h"
#include "pointers.h"
#include <string>

using std::string;
using tensorflow::DataType;
using tensorflow::Status;

namespace LAMMPS_NS {

struct CallStatistics {
  double nij_max_sum;
  double nnl_max_sum;
  double elapsed;
  double num_calls;
};

class TensorAlloy : Pointers {
public:
  TensorAlloy(class LAMMPS *lmp,
              const string &graph_model_path,
              const std::vector<string> &symbols,
              int nlocal,
              int ntypes,
              int *itypes,
              bool verbose);
  ~TensorAlloy();

  /// @brief Compute total energy (etotal), atomic energy (eatom)
  Status compute(int nlocal, int ntypes, int *itypes, const int *ilist,
                 const int *numneigh, int **firstneigh, double **x,
                 double etemp, double &etotal, double *eatom);

  /// @brief Compute total energy (etotal), atomic energy (eatom), forces (f)
  /// virial (unit is eV) and per-atom virial.
  Status compute(int nlocal, int ntypes, int *itypes, const int *ilist,
                 const int *numneigh, int **firstneigh, double **x, double **f,
                 double *eentropy, double etemp, double &etotal, double *eatom,
                 double *virial, double **vatom);

  /// @brief Manually set `neigh_coef`. For calculations outside lammps,
  /// `neigh_coef` should be 1.0
  void set_neigh_coef(double val) { neigh_coef = MAX(MIN(val, 1.0), 0.2); }

  /// @brief Collect sess->run related statistics
  void collect_call_statistics() { collect_statistics = true; }
  CallStatistics get_call_statistics() { return call_stats; }

  GraphModel *get_model() { return graph_model; }
  VirtualAtomMap *get_vap() { return vap; }

protected:
  GraphModel *graph_model;
  VirtualAtomMap *vap;

  double neigh_coef;
  double cutforcesq;

private:

  template <typename T>
  Status run(DataType dtype, int nlocal, int ntypes, int *itypes,
             const int *ilist, const int *numneigh, int **firstneigh,
             double **x, double **f, double *eentropy, double etemp,
             double &etotal, double *eatom, double *virial, double **vatom);

  template <typename T> void allocate(DataType dtype, int ntypes);
  template <typename T>
  void update_tensors(DataType dtype, int ntypes, double etemp);

  Tensor *n_atoms_vap_tensor;
  Tensor *nnl_max;
  Tensor *etemperature;
  Tensor *atom_masks;
  Tensor *row_splits;

  bool collect_statistics;
  CallStatistics call_stats{0, 0, 0, 0};

  int32 **ij_types;
  int32 **ij_nums;

  int *ij_pairs;
  Tensor *rdists, *rmap;

  int local_update(const int *ilist, const int *numneigh, int nlocal);
  template <typename T> int resize(int nij_max, DataType dtype);
};
} // namespace LIBTENSORALLOY_NS

#endif // LIBTENSORALLOY_TENSORALLOY_H
