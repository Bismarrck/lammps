//
// Created by Xin Chen on 2019-06-11.
//

#ifdef PAIR_CLASS

PairStyle(tensoralloy, PairTensorAlloy)

#else

#ifndef LMP_PAIR_TENSORALLOY_H
#define LMP_PAIR_TENSORALLOY_H

#include <atom.h>
#include "pair.h"
#include "virtual_atom_approach.h"
#include "graph_model.h"

#include "tensorflow/core/public/session.h"


namespace LAMMPS_NS {

    using tensorflow::Tensor;
    using tensorflow::string;
    using tensorflow::int32;
    using tensorflow::Status;
    using std::vector;

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
        // Virtual-atom approach variables
        GraphModel graph_model;
        int *g4_numneigh;
        double cutforcesq, cutmax;

        int32 *g2_offset_map;
        int32 *g4_offset_map;
        VirtualAtomMap *vap;

        void init_offset_maps();
        void read_graph_model(const string& filename, const vector<string>& symbols);

        template <typename T> double update_cell ();
        void get_shift_vector(int i, double &nx, double &ny, double &nz);
        const double get_interatomic_distance (unsigned int i, unsigned int j, bool square=true);
        int inline get_local_idx(const unsigned int i) {
            return atom->tag[i] - 1;
        }

        // Original Pair variables
        int nmax;
        void allocate();

    private:
        // TensorFlow variables and functions
        Status load_graph(const string& filename);
        std::unique_ptr<tensorflow::Session> session;
        bool serial_mode;

        Tensor *h_tensor;
        double h_inv[3][3];

        Tensor *R_tensor;
        Tensor *volume_tensor;
        Tensor *n_atoms_vap_tensor;
        Tensor *pulay_stress_tensor;
        Tensor *composition_tensor;
        Tensor *atom_mask_tensor;
        Tensor *row_splits_tensor;

        double dynamic_bytes;
        double tensors_memory_usage();
    };
}

#endif
#endif
