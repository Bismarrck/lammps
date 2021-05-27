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
#include "utils.h"
#include "tensoralloy.h"

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
  void allocate();

  void set_etemp(double new_etemp) { etemp = new_etemp; }
  double get_etemp() const { return etemp; }

protected:
  TensorAlloy *calc;

  double cutforcesq, cutmax;
  bool use_hyper_thread;
  double etemp;

private:

  double neigh_extra{};
  double dynamic_bytes;
  double tensors_memory_usage();
};
} // namespace LAMMPS_NS

#endif
#endif
