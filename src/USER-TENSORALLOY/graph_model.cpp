//
// Created by Xin Chen on 2019-07-26.
//

#include <vector>
#include <iostream>
#include <iomanip>

#include "error.h"

#include "jsoncpp/json/json.h"
#include "graph_model.h"

#include "tensorflow/core/framework/tensor.h"

using namespace LAMMPS_NS;

using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;

/* ----------------------------------------------------------------------
   Initialization.
------------------------------------------------------------------------- */

GraphModel::GraphModel(
        const string &graph_model_path,
        const std::vector<string>& symbols,
        Error *error,
        bool serial_mode)
{
    max_occurs_initialized = false;
    decoded = false;
    filename = graph_model_path;
    rcut = 0.0;
    acut = 0.0;
    use_angular = false;
    n_elements = 0;

    Status load_graph_status = load_graph(graph_model_path, serial_mode);
    std::cout << "Read " << graph_model_path << ": " << load_graph_status << std::endl;

    std::vector<Tensor> outputs;
    Status status = session->Run({}, {"Transformer/params:0"}, {}, &outputs);
    if (!status.ok()) {
        auto message = "Decode graph model error: " + status.ToString();
        error->all(FLERR, message.c_str());
    }
    status = read_transformer_params(outputs[0], graph_model_path, symbols);
    if (!status.ok()) {
        error->all(FLERR, status.error_message().c_str());
    }

    outputs.clear();
    status = session->Run({}, {"Metadata/precision:0"}, {}, &outputs);
    fp64 = status.ok() && outputs[0].flat<string>().data()[0] == "high";
    if (fp64) {
        std::cout << "Graph model uses float64" << std::endl;
    } else {
        std::cout << "Graph model uses float32" << std::endl;
    }

    outputs.clear();
    status = session->Run({}, {"Metadata/ops:0"}, {}, &outputs);
    if (!status.ok()) {
        auto message = "Decode graph model error: " + status.ToString();
        error->all(FLERR, message.c_str());
    }
    status = read_ops(outputs[0]);
    if (!status.ok()) {
        error->all(FLERR, status.error_message().c_str());
    }

    decoded = true;
}

/* ----------------------------------------------------------------------
   Deallocation
------------------------------------------------------------------------- */

GraphModel::~GraphModel()
{
    session.reset();
}

/* ----------------------------------------------------------------------
   Run
------------------------------------------------------------------------- */

Status GraphModel::run(
        const std::vector<std::pair<string, Tensor>> &feed_dict,
        std::vector<Tensor> &outputs)
{
    std::vector<string> run_ops({ops["free_energy"], ops["forces"], ops["stress"]});
    return session->Run(feed_dict, run_ops, {}, &outputs);
}

/* ----------------------------------------------------------------------
   Load the graph model
------------------------------------------------------------------------- */

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status GraphModel::load_graph(const string &graph_file_name, bool serial_mode) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }

    // Initialize the session
    tensorflow::SessionOptions options;
    options.config.set_allow_soft_placement(true);
    options.config.set_log_device_placement(false);

    if (serial_mode) {
        options.config.set_inter_op_parallelism_threads(1);
        options.config.set_intra_op_parallelism_threads(1);
    }

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

Status GraphModel::read_transformer_params(
        const Tensor& metadata,
        const string& graph_model_path,
        const std::vector<string>& lammps_symbols)
{
    Json::Value jsonData;
    Json::Reader jsonReader;
    auto stream = metadata.flat<string>().data()[0];
    auto parse_status = jsonReader.parse(stream, jsonData, false);

    if (parse_status) {
        filename = graph_model_path;
        rcut = jsonData["rcut"].asDouble();
        acut = jsonData["acut"].asDouble();
        use_angular = jsonData["angular"].asBool();
        Json::Value graph_model_symbols = jsonData["elements"];
        n_elements = graph_model_symbols.size();

        auto size = lammps_symbols.size();
        for (int i = 1; i < size; i++) {
            if (lammps_symbols[i] != graph_model_symbols[i - 1].asString()) {
                auto message = "Elements mismatch at index " + std::to_string(i);
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

Status GraphModel::read_ops(const Tensor& metadata)
{
    Json::Value jsonData;
    Json::Reader jsonReader;
    auto stream = metadata.flat<string>().data()[0];
    auto parse_status = jsonReader.parse(stream, jsonData, false);

    if (parse_status) {
        for( Json::Value::iterator itr = jsonData.begin() ; itr != jsonData.end() ; itr++ ) {
            ops.insert({itr.key().asString(), itr->asString()});
        }
    } else {
        auto message = "Could not decode ops";
        return Status(tensorflow::error::Code::INTERNAL, message);
    }
    if (ops.find("energy") == ops.end()) {
        auto message = "The total energy Op is missing";
        return Status(tensorflow::error::Code::INTERNAL, message);
    } else if (ops.find("forces") == ops.end()) {
        auto message = "The atomic force Op is missing";
        return Status(tensorflow::error::Code::INTERNAL, message);
    } else if (ops.find("stress") == ops.end()) {
        auto message = "The virial stress Op is missing";
        return Status(tensorflow::error::Code::INTERNAL, message);
    } else {
        return Status::OK();
    }
}

/* ----------------------------------------------------------------------
   Compute `max_occurs`.
------------------------------------------------------------------------- */

void GraphModel::compute_max_occurs(const int natoms, const int* atom_types)
{
    int size = n_elements + 1;
    for (int i = 0; i < size; i++) {
        max_occurs.push_back(0);
    }
    for (int i = 0; i < natoms; i++) {
        max_occurs[atom_types[i]] += 1;
    }
    for (int i = 1; i < size; i++) {
        if (max_occurs[i] == 0) {
            max_occurs[i] = 1;
        }
    }
    max_occurs_initialized = true;
}

/* ----------------------------------------------------------------------
   Print the details of the loaded graph model.
------------------------------------------------------------------------- */

void GraphModel::print()
{
    std::cout << "Graph model <" << filename << "> Metadata" << std::endl;
    std::cout << "  * rcut: " << std::setprecision(3) << rcut << std::endl;
    std::cout << "  * acut: " << std::setprecision(3) << acut << std::endl;
    std::cout << "  * angular: " << use_angular << std::endl;
    std::cout << "  * max_occurs: " << std::endl;
    if (max_occurs_initialized) {
        for (int i = 0; i < n_elements; i++)
            printf("    %2s: %4d\n", symbols[i].c_str(), max_occurs[i]);
    }
}
