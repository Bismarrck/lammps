//
// Created by Xin Chen on 2020/11/30.
//

#ifndef LIBTENSORALLOY_TENSORALLOY_H
#define LIBTENSORALLOY_TENSORALLOY_H

#include "graph_model.h"
#include "tensoralloy_utils.h"
#include "virtual_atom_approach.h"
#include <string>

using std::string;
using tensorflow::DataType;
using tensorflow::Status;

namespace LIBTENSORALLOY_NS {

struct CallStatistics {
  double nij_max;
  double nnl_max;
  double elapsed;
  double num_calls;
};

class TensorAlloy {
public:
  TensorAlloy(const string &graph_model_path,
              const std::vector<string> &symbols, int nlocal, int ntypes,
              int *itypes, bool verbose,
              const logger& logfun,
              const logger& errfun);
  ~TensorAlloy();

  Status compute(int nlocal, int ntypes, int *itypes, const int *ilist,
                 const int *numneigh, int **firstneigh, double **x, double **f,
                 double *eentropy, double etemp, double &etotal, double *pe);

  void set_neigh_coef(double val) { neigh_coef = val; }

  CallStatistics get_call_statistics() { return call_stats; }

  GraphModel *get_model() { return graph_model; }
  VirtualAtomMap *get_vap() { return vap; }

protected:
  GraphModel *graph_model;
  VirtualAtomMap *vap;
  Memory *memory;

  double neigh_coef;
  double cutforcesq, cutmax;

private:
  logger err;
  logger log;

  template <typename T>
  Status run(DataType dtype, int nlocal, int ntypes, int *itypes,
             const int *ilist, const int *numneigh, int **firstneigh,
             double **x, double **f, double *eentropy, double etemp,
             double &etotal, double *pe);

  void allocate(int ntypes);
  template <typename T> void init(DataType dtype, int ntypes);
  template <typename T>
  void update_tensors(DataType dtype, int ntypes, double etemp);

  Tensor *cell;
  Tensor *positions;
  Tensor *volume;
  Tensor *n_atoms_vap_tensor;
  Tensor *nnl_max;
  Tensor *pulay_stress;
  Tensor *etemperature;
  Tensor *atom_masks;
  Tensor *row_splits;

  bool collect_statistics;
  CallStatistics call_stats;

  int32 **ijtypes;
  int32 **ijnums;
};
} // namespace LIBTENSORALLOY_NS

#endif // LIBTENSORALLOY_TENSORALLOY_H
