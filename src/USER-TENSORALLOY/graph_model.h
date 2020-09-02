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
  bool angular() const { return use_angular; }
  bool use_fp64() const { return fp64; }
  double get_cutoff(bool angular = false) const {
    return angular ? acut : rcut;
  }

  std::vector<Tensor> run(const std::vector<std::pair<string, Tensor>> &,
                          Error *);

protected:
  bool use_angular;
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

  bool fp64;
  bool decoded;
};
} // namespace LAMMPS_NS

#endif // LMP_TENSORALLOY_GRAPH_MODEL_H
