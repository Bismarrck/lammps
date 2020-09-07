//
// Created by Xin Chen on 2019-06-11.
//

#ifdef PAIR_CLASS

PairStyle(tensoralloy, PairTensorAlloy)

#else

#ifndef LMP_PAIR_TENSORALLOY_H
#define LMP_PAIR_TENSORALLOY_H

#include <chrono>
#include "pair.h"
#include "atom.h"
#include "graph_model.h"
#include "virtual_atom_approach.h"
#include "tensorflow/core/public/session.h"

namespace LAMMPS_NS {

using std::vector;
using tensorflow::DataType;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

class PairTensorAlloy : public Pair {
public:
  explicit PairTensorAlloy(class LAMMPS *);
  ~PairTensorAlloy() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  double memory_usage() override;

protected:
  GraphModel *graph_model;
  double cutforcesq, cutmax;

  int32 **ijtypes;
  int32 **ijnums;
  VirtualAtomMap *vap;

  template <typename T> double update_cell();
  template <typename T> void run(int eflag, int vflag, DataType dtype);
  template <typename T> void allocate(DataType dtype);
  template <typename T> void update_tensors(DataType dtype);

  bool use_hyper_thread;

  // Electron temperature (eV)
  double etemp;

  Tensor *cell;
  Tensor *positions;
  Tensor *volume;
  Tensor *n_atoms_vap_tensor;
  Tensor *nnl_max;
  Tensor *pulay_stress;
  Tensor *etemperature;
  Tensor *atom_masks;
  Tensor *row_splits;

private:
  int num_calls;
  double nnl_max_sum;
  double nij_max_sum;
  double neigh_extra;
  double elapsed;
  double dynamic_bytes;
  double tensors_memory_usage();
};
} // namespace LAMMPS_NS

#endif
#endif
