//
// Created by Xin Chen on 2019-07-26.
//

#include <vector>

#include "error.h"
#include "fmt/format.h"
#include "lammps.h"
#include "utils.h"

#include "graph_model.h"
#include "jsoncpp/json/json.h"

#include "tensorflow/core/framework/tensor.h"

using namespace LAMMPS_NS;

using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

/* ----------------------------------------------------------------------
   Initialization.
------------------------------------------------------------------------- */

GraphModel::GraphModel(LAMMPS *lammps, const string &graph_model_path,
                       const std::vector<string> &symbols, Error *error,
                       bool serial_mode, bool verbose) {
  decoded = false;
  filename = graph_model_path;
  rcut = 0.0;
  acut = 0.0;
  use_angular = false;
  n_elements = 0;
  lmp = lammps;

  Status status = load_graph(graph_model_path, serial_mode);
  if (verbose) {
    utils::logmesg(lmp,
                   fmt::format("Read {}: {}\n", filename, status.ToString()));
  }

  std::vector<Tensor> outputs;
  status = session->Run({}, {"Transformer/params:0"}, {}, &outputs);
  if (!status.ok()) {
    auto message = "Decode graph model error: " + status.ToString();
    error->all(FLERR, message);
  }
  status = read_transformer_params(outputs[0], filename, symbols);
  if (!status.ok()) {
    error->all(FLERR, status.error_message());
  }

  outputs.clear();
  status = session->Run({}, {"Metadata/precision:0"}, {}, &outputs);
  fp64 = status.ok() && outputs[0].flat<string>().data()[0] == "high";
  if (verbose) {
    if (fp64) {
      utils::logmesg(lmp, "Graph model uses float64\n");
    } else {
      utils::logmesg(lmp, "Graph model uses float32\n");
    }
  }

  outputs.clear();
  status = session->Run({}, {"Metadata/ops:0"}, {}, &outputs);
  if (!status.ok()) {
    auto message = "Decode graph model error: " + status.ToString();
    error->all(FLERR, message);
  }
  status = read_ops(outputs[0]);
  if (!status.ok()) {
    error->all(FLERR, status.error_message());
  }

  decoded = true;
}

/* ----------------------------------------------------------------------
   Deallocation
------------------------------------------------------------------------- */

GraphModel::~GraphModel() { session.reset(); }

/* ----------------------------------------------------------------------
   Run
------------------------------------------------------------------------- */

std::vector<Tensor>
GraphModel::run(const std::vector<std::pair<string, Tensor>> &feed_dict,
                Error *error) {
  std::vector<Tensor> outputs;
  std::vector<string> run_ops(
      {ops["free_energy"], ops["dEdrij"], ops["atomic"]});
  if (use_angular) {
    run_ops.emplace_back(ops["dEdrijk"]);
  }
  Status status = session->Run(feed_dict, run_ops, {}, &outputs);
  if (!status.ok()) {
    auto message = "TensorAlloy internal error: " + status.ToString();
    error->all(FLERR, message);
  }
  return outputs;
}

/* ----------------------------------------------------------------------
   Load the graph model
------------------------------------------------------------------------- */

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status GraphModel::load_graph(const string &graph_file_name,
                              bool use_hyper_thread) {
  tensorflow::GraphDef graph_def;
  Status status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }

  // Initialize the session
  tensorflow::SessionOptions options;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(false);
  if (use_hyper_thread) {
    options.config.set_inter_op_parallelism_threads(2);
  } else {
    options.config.set_inter_op_parallelism_threads(1);
  }
  options.config.set_intra_op_parallelism_threads(1);

  session.reset(tensorflow::NewSession(options));
  Status session_create_status = session->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

/* ----------------------------------------------------------------------
   Read parameters for initializing a `UniversalTransformer`.
------------------------------------------------------------------------- */

Status
GraphModel::read_transformer_params(const Tensor &metadata,
                                    const string &graph_model_path,
                                    const std::vector<string> &lammps_symbols) {
  Json::Value jsonData;
  Json::Reader jsonReader;
  auto stream = metadata.flat<string>().data()[0];
  auto parse_status = jsonReader.parse(stream, jsonData, false);

  if (parse_status) {
    filename = graph_model_path;
    rcut = jsonData["rcut"].asDouble();
    acut = jsonData["acut"].asDouble();
    use_angular = jsonData["angular"].asBool();
    Json::Value graph_symbols = jsonData["elements"];
    n_elements = graph_symbols.size();

    auto size = lammps_symbols.size();
    for (int i = 1; i < size; i++) {
      if (lammps_symbols[i] != graph_symbols[i - 1].asString()) {
        auto message = "Elements mismatch at " + std::to_string(i);
        return Status(tensorflow::error::Code::INTERNAL, message);
      } else {
        symbols.push_back(lammps_symbols[i]);
      }
    }
  } else {
    auto message = "Could not decode transformer parameters";
    return Status(tensorflow::error::Code::INTERNAL, message);
  }
  return Status::OK();
}

/* ----------------------------------------------------------------------
   Read Ops
------------------------------------------------------------------------- */

Status GraphModel::read_ops(const Tensor &metadata) {
  Json::Value jsonData;
  Json::Reader jsonReader;
  auto stream = metadata.flat<string>().data()[0];
  auto parse_status = jsonReader.parse(stream, jsonData, false);

  if (parse_status) {
    Json::Value::iterator itr;
    for (itr = jsonData.begin(); itr != jsonData.end(); itr++) {
      ops.insert({itr.key().asString(), itr->asString()});
    }
  } else {
    auto message = "Could not decode ops";
    return Status(tensorflow::error::Code::INTERNAL, message);
  }
  if (ops.find("energy") == ops.end()) {
    auto message = "The total energy Op is missing";
    return Status(tensorflow::error::Code::INTERNAL, message);
  } else if (ops.find("dEdrij") == ops.end()) {
    auto message = "The radial partial force Op dE/drij is missing";
    return Status(tensorflow::error::Code::INTERNAL, message);
  } else if (use_angular && ops.find("dEdrijk") == ops.end()) {
    auto message = "The angular partial force Op dE/drijk is missing";
    return Status(tensorflow::error::Code::INTERNAL, message);
  } else {
    return Status::OK();
  }
}
