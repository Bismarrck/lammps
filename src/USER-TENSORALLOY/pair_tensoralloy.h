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
        PairTensorAlloy(class LAMMPS *);
        virtual ~PairTensorAlloy();
        virtual void compute(int, int);
        void settings(int, char **);
        void coeff(int, char **);
        void init_style();
        double init_one(int, int);

        int pack_forward_comm(int, int *, double *, int, int *);
        void unpack_forward_comm(int, int, double *);
        int pack_reverse_comm(int, int, double *);
        void unpack_reverse_comm(int, int *, double *);
        double memory_usage();

    protected:

        // Virtual-Atom Approach
        int *element_map;
        int *max_occurs;
        bool use_angular;
        int n_eta;
        int n_omega;
        int n_beta;
        int n_gamma;
        int n_zeta;
        double rc;

        VirtualAtomMap *vap;

        int nmax;                   // allocated size of per-atom arrays
        double cutforcesq,cutmax;

        // per-atom arrays

        double *rho,*fp;
        double **mu, **lambda;

        // potentials as array data

        int nrho,nr;
        int nfrho,nrhor,nz2r;
        int nu2r, nw2r;
        double **frho,**rhor,**z2r;
        double **u2r, **w2r;
        int *type2frho,**type2rhor,**type2z2r;
        int **type2u2r,**type2w2r;

        // potentials in spline form used for force computation

        double dr,rdr,drho,rdrho;
        double ***rhor_spline,***frho_spline,***z2r_spline;
        double ***u2r_spline, ***w2r_spline;

        // potentials as file data

        int *map;                   // which element each atom type maps to

        struct Setfl {
            char **elements;
            int nelements,nrho,nr;
            double drho,dr,cut;
            double *mass;
            double **frho,**rhor,***z2r;
            double ***u2r, ***w2r;
        };
        Setfl *setfl;

        void allocate();
        void array2spline();
        void interpolate(int, double, double *, double **);
        void grab(FILE *, char *, int, double *);

        void read_file(char *);
        void file2array();

    private:
        tensorflow::Session *session;

    };
}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: No matching element in ADP potential file

The ADP potential file does not contain elements that match the
requested elements.

E: Cannot open ADP potential file %s

The specified ADP potential file cannot be opened.  Check that the
path and name are correct.

E: Incorrect element names in ADP potential file

The element names in the ADP file do not match those requested.

*/
