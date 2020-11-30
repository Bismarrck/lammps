//
// Created by Xin Chen on 2019-07-26.
//

#ifndef LMP_TENSORALLOY_GRAPH_MODEL_H
#define LMP_TENSORALLOY_GRAPH_MODEL_H

#include "pair.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/public/session.h"

namespace LAMMPS_NS {

using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

class GraphModel {

public:
  GraphModel(LAMMPS *lammps, const string &graph_model_path,
             const std::vector<string> &symbols, Error *error, bool serial_mode,
             bool verbose);

  ~GraphModel();

  int get_n_elements() const { return n_elements; }
  bool is_angular() const { return _angular; }
  bool is_fp64() const { return _fp64; }
  bool is_finite_temperature() const { return _is_finite_temperature; }
  double get_cutoff(bool angular = false) const {
    return angular ? acut : rcut;
  }

  std::vector<Tensor> run(const std::vector<std::pair<string, Tensor>> &,
                          Error *, bool);

  int get_index_variation_energy(bool atomic) const {
    return static_cast<int>(atomic) + (_is_finite_temperature ? 5 : 0);
  }

  int get_index_free_energy(bool atomic) const {
    if (!_is_finite_temperature) return -1;
    else return 5 + static_cast<int>(atomic);
  }

  int get_index_eentropy(bool atomic) const {
    if (!_is_finite_temperature) return -1;
    else return 3 + static_cast<int>(atomic);
  }

  int get_index_partial_forces(bool angular=false) const {
    if (angular) return _is_finite_temperature ? 7 : 3;
    else return 2;
  }

protected:
  double rcut;
  double acut;
  int n_elements;
  LAMMPS *lmp;
  std::vector<string> symbols;
  string filename;
  std::map<string, string> ops;
  std::unique_ptr<tensorflow::Session> session;

  Status load_graph(const string &filename, bool serial_mode);
  Status read_transformer_params(const Tensor &, const string &,
                                 const std::vector<string> &);
  Status read_ops(const Tensor &);

  bool decoded;

  bool _angular;
  bool _fp64;
  bool _is_finite_temperature;
};
} // namespace LAMMPS_NS

#endif // LMP_TENSORALLOY_GRAPH_MODEL_H
