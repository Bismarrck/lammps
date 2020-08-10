//
// Created by Xin Chen on 2019-07-26.
//

#ifndef LMP_TENSORALLOY_GRAPH_MODEL_H
#define LMP_TENSORALLOY_GRAPH_MODEL_H

#include "pair.h"

#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace LAMMPS_NS {

    using tensorflow::string;
    using tensorflow::Tensor;
    using tensorflow::Status;
    using tensorflow::int32;

    class GraphModel {

    public:
        GraphModel(const string& graph_model_path, const std::vector<string>& symbols, Error *error,
                bool serial_mode);
        ~GraphModel();

        Status read(const Tensor&, const string&, const std::vector<string>&);
        bool is_initialized() const { return decoded && max_occurs_initialized; }
        int get_n_elements() const { return n_elements; }
        bool angular() const { return use_angular; }
        bool use_fp64() const { return fp64; }
        double get_cutoff(bool angular=false) const { return angular ? acut : rcut; }
        unsigned int get_max_occur(int index) const { return max_occurs[index]; }

        std::unique_ptr<tensorflow::Session> session;

        void compute_max_occurs(int natoms, const int* atom_types);
        void print();

    protected:
        std::vector<unsigned int> max_occurs;
        bool use_angular;
        double rcut;
        double acut;
        int n_elements;
        std::vector<string> symbols;
        string filename;

        Status load_graph(const string& filename, bool serial_mode);
        bool fp64;

        bool decoded;
        bool max_occurs_initialized;
    };
}

#endif //LMP_TENSORALLOY_GRAPH_MODEL_H
