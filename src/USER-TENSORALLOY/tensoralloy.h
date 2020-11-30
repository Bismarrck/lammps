//
// Created by Xin Chen on 2020/11/30.
//

#ifndef LIBTENSORALLOY_TENSORALLOY_H
#define LIBTENSORALLOY_TENSORALLOY_H

#include "tensoralloy_utils.h"
#include "graph_model.h"
#include "virtual_atom_approach.h"
#include <string>

using std::string;
using tensorflow::DataType;
using tensorflow::Status;

namespace LIBTENSORALLOY_NS {

class TensorAlloy {
public:
  TensorAlloy(const string &graph_model_path,
              const std::vector<string>& symbols,
              int nlocal,
              int ntypes,
              int *itypes,
              logger &logfun,
              logger &errfun,
              bool verbose);

  template <typename T>
  Status compute(int nlocal, int ntypes, int *itypes,
                 const int *ilist, const int *jlist, const int *numneigh,
                 int **firstneigh, double **x, double **f, double *eentropy,
                 double etemp, double &etotal, double *pe);

protected:
  GraphModel *graph_model;
  VirtualAtomMap *vap;
  Memory *memory;

  double neigh_coef;
  double cutforcesq, cutmax;

private:
  logger *err;

  template <typename T>
  Status run(DataType dtype, int nlocal, int ntypes, int *itypes, const int *ilist,
             const int *jlist, const int *numneigh, int **firstneigh, double **x,
             double **f, double *eentropy, double etemp, double &etotal, double *pe);

  template <typename T> void allocate(DataType dtype, int ntypes);
  template <typename T> void update_tensors(DataType dtype, int ntypes,
                                            double etemp);

  Tensor *cell;
  Tensor *positions;
  Tensor *volume;
  Tensor *n_atoms_vap_tensor;
  Tensor *nnl_max;
  Tensor *pulay_stress;
  Tensor *etemperature;
  Tensor *atom_masks;
  Tensor *row_splits;

  int num_calls;
  double elapsed;
  bool collect_statistics;
  double nij_max_sum;
  double nnl_max_sum;

  int32 **ijtypes;
  int32 **ijnums;
};
} // namespace LIBTENSORALLOY_NS

#endif // LIBTENSORALLOY_TENSORALLOY_H
