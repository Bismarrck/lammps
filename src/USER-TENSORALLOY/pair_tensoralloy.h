//
// Created by Xin Chen on 2019-06-11.
//

#ifdef PAIR_CLASS

PairStyle(tensoralloy, PairTensorAlloy)

#else

#ifndef LMP_PAIR_TENSORALLOY_H
#define LMP_PAIR_TENSORALLOY_H

#include "pair.h"
#include "virtual_atom_approach.h"

#include "tensorflow/core/public/session.h"


namespace LAMMPS_NS {

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

        // Virtual-Atom Approach
        int *max_occurs;
        bool use_angular;
        int n_eta;
        int n_omega;
        int n_beta;
        int n_gamma;
        int n_zeta;
        tensorflow::int32 *g2_offset_map;
        tensorflow::int32 *g4_offset_map;

        VirtualAtomMap *vap;

        template <typename T>
        void update_cell (tensorflow::Tensor *h, double &volume, double (&h_inv)[3][3]);

        int nmax;                   // allocated size of per-atom arrays
        double cutforcesq, cutmax;

        // potentials as file data

        int *map;                   // which element each atom type maps to

        void allocate();

        tensorflow::Status load_graph(const tensorflow::string& filename);


    private:

        std::unique_ptr<tensorflow::Session> session;

    };
}

#endif
#endif
