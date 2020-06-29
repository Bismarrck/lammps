//
// Created by Xin Chen on 2019-07-26.
//

#include <vector>
#include <iostream>
#include <iomanip>

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

GraphModel::GraphModel()
{
    max_occurs_initialized = false;
    decoded = false;
    filename = "";
    rcut = 0.0;
    acut = 0.0;
    use_angular = false;
    n_elements = 0;
}

/* ----------------------------------------------------------------------
   Read the metadata of the graph model.
------------------------------------------------------------------------- */

Status GraphModel::read(
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
        cls = jsonData["class"].asString();
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
        auto message = "Could not decode the graph model.";
        return Status(tensorflow::error::Code::INTERNAL, message);
    }
    decoded = true;
    return Status::OK();
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
