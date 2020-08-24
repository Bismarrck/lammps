//
// Created by Xin Chen on 2019-07-26.
//

#ifndef LMP_TENSORALLOY_GRAPH_MODEL_H
#define LMP_TENSORALLOY_GRAPH_MODEL_H

#include "pair.h"

#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace LAMMPS_NS {

    using tensorflow::string;
    using tensorflow::Tensor;
    using tensorflow::Status;
    using tensorflow::int32;

    class GraphModel {

    public:
        GraphModel();
        ~GraphModel() = default;;

        Status read(const Tensor&, const string&, const std::vector<string>&);
        bool is_initialized() const { return decoded && max_occurs_initialized; }
        int get_n_elements() const { return n_elements; }
        bool angular() const { return use_angular; }
        double get_cutoff() const { return rc; }
        unsigned int get_max_occur(int index) const { return max_occurs[index]; }
        int get_ndim(bool is_angular=false) const {
            if (is_angular) {
                return n_gamma * n_beta * n_zeta;
            } else {
                return n_omega * n_eta;
            }
        }

        bool use_universal_transformer() { return cls == "UniversalTransformer"; }
        string get_transformer_name() const { return cls; }

        void compute_max_occurs(int natoms, const int* atom_types);
        void print();

    protected:
        std::vector<unsigned int> max_occurs;
        bool use_angular;
        double rc;
        int n_elements;
        std::vector<string> symbols;
        int n_eta;
        int n_omega;
        int n_beta;
        int n_gamma;
        int n_zeta;
        string filename;
        string cls;

    private:
        bool decoded;
        bool max_occurs_initialized;

    };
}

#endif //LMP_TENSORALLOY_GRAPH_MODEL_H