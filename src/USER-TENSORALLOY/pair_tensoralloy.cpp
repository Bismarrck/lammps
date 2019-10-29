//
// Created by Xin Chen on 2019-06-11.
//
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cmath>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <domain.h>
#include <chrono>

#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "error.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "pair_tensoralloy.h"

using namespace LAMMPS_NS;

using tensorflow::int32;
using tensorflow::DT_INT32;
using tensorflow::DT_FLOAT;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;

typedef std::chrono::high_resolution_clock Clock;

#define MAXLINE 1024
#define IJ(i,j,n) i * n + j
#define IJK(i,j,k,n) i * n * n + j * n + k


/* ---------------------------------------------------------------------- */

PairTensorAlloy::PairTensorAlloy(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
    force->newton_pair = 0;
    force->newton = 0;

    cutmax = 0.0;
    cutforcesq = 0.0;

    g2_offset_map = nullptr;
    g4_offset_map = nullptr;
    vap = nullptr;
    session = nullptr;

    // Tensor pointers
    R_tensor = nullptr;
    h_tensor = nullptr;
    volume_tensor = nullptr;
    n_atoms_vap_tensor = nullptr;
    composition_tensor = nullptr;
    atom_mask_tensor = nullptr;
    pulay_stress_tensor = nullptr;
    row_splits_tensor = nullptr;

    // Use parallel mode by default.
    serial_mode = false;

    // set comm size needed by this Pair
    comm_forward = 0;
    comm_reverse = 0;

    single_enable = 0;
    one_coeff = 1;
    manybody_flag = 1;

    // Virial is handled by TensorAlloy.
    vflag_fdotr = 0;

    // Temporarily disable atomic energy.
    eflag_atom = 0;

    // Per-atom virial is not supported.
    vflag_atom = 0;

    // Disable `float64` by default.
    use_fp64 = true;

    // Set the variables to their default values
    dynamic_bytes = 0;
    nmax = -1;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            h_inv[i][j] = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorAlloy::~PairTensorAlloy()
{
    memory->destroy(g2_offset_map);
    delete [] g2_offset_map;

    if (graph_model.angular()) {
        memory->destroy(g4_offset_map);
        delete [] g4_offset_map;
    }

    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);

        delete R_tensor;
        R_tensor = nullptr;

        delete h_tensor;
        h_tensor = nullptr;

        delete volume_tensor;
        volume_tensor = nullptr;

        delete n_atoms_vap_tensor;
        n_atoms_vap_tensor = nullptr;

        delete pulay_stress_tensor;
        pulay_stress_tensor = nullptr;

        delete composition_tensor;
        composition_tensor = nullptr;

        delete atom_mask_tensor;
        atom_mask_tensor = nullptr;

        delete row_splits_tensor;
        row_splits_tensor = nullptr;
    }

    delete vap;
    vap = nullptr;
}

/* ----------------------------------------------------------------------
   Update the ASE-style lattice matrix.

   * h: the 3x3 lattice tensor
   * volume: the volume of the latticce
   * h_inv: the inverse of the lattice tensor
------------------------------------------------------------------------- */
template <typename T>
double PairTensorAlloy::update_cell()
{
    auto h_mapped = h_tensor->template tensor<T, 2>();
    h_mapped(0, 0) = static_cast<T> (domain->h[0]);
    h_mapped(0, 1) = static_cast<T> (0.0);
    h_mapped(0, 2) = static_cast<T> (0.0);
    h_mapped(1, 0) = static_cast<T> (domain->h[5]);
    h_mapped(1, 1) = static_cast<T> (domain->h[1]);
    h_mapped(1, 2) = static_cast<T> (0.0);
    h_mapped(2, 0) = static_cast<T> (domain->h[4]);
    h_mapped(2, 1) = static_cast<T> (domain->h[3]);
    h_mapped(2, 2) = static_cast<T> (domain->h[2]);

    // Compute the volume.
    double volume;
    if (domain->dimension == 3) {
        volume = domain->xprd * domain->yprd * domain->zprd;
    }
    else {
        volume = domain->xprd * domain->yprd;
    }

    h_inv[0][0] = domain->h_inv[0];
    h_inv[1][1] = domain->h_inv[1];
    h_inv[2][2] = domain->h_inv[2];
    h_inv[2][1] = domain->h_inv[3];
    h_inv[2][0] = domain->h_inv[4];
    h_inv[1][0] = domain->h_inv[5];
    h_inv[0][1] = 0.0;
    h_inv[0][2] = 0.0;
    h_inv[1][2] = 0.0;

    return volume;
}

/* ----------------------------------------------------------------------
   Calculate the shift vector (nx, ny, nz) of atom `i`.
------------------------------------------------------------------------- */

void PairTensorAlloy::get_shift_vector(const int i, double &nx, double &ny, double &nz)
{
    double nhx = atom->x[i][0] - atom->x[atom->tag[i] - 1][0];
    double nhy = atom->x[i][1] - atom->x[atom->tag[i] - 1][1];
    double nhz = atom->x[i][2] - atom->x[atom->tag[i] - 1][2];

    nx = round(nhx * h_inv[0][0] + nhy * h_inv[1][0] + nhz * h_inv[2][0]);
    ny = round(nhx * h_inv[0][1] + nhy * h_inv[1][1] + nhz * h_inv[2][1]);
    nz = round(nhx * h_inv[0][2] + nhy * h_inv[1][2] + nhz * h_inv[2][2]);
}

/* ----------------------------------------------------------------------
   Calculate the interatomic distance of (i, j).
------------------------------------------------------------------------- */

double PairTensorAlloy::get_interatomic_distance(unsigned int i, unsigned int j, bool square)
{
    double rsq;
    double rijx, rijy, rijz;

    rijx = atom->x[j][0] - atom->x[i][0];
    rijy = atom->x[j][1] - atom->x[i][1];
    rijz = atom->x[j][2] - atom->x[i][2];
    rsq = rijx*rijx + rijy*rijy + rijz*rijz;

    return square ? rsq : sqrt(rsq);
}

/* ----------------------------------------------------------------------
   The `compute` method.
------------------------------------------------------------------------- */

template <typename T>
void PairTensorAlloy::run_once(int eflag, int vflag, DataType dtype)
{
    unsigned int i, j, k;
    int ii, jj, kk, inum, jnum, itype, jtype, ktype, ilocal, igsl;
    int nij_max = 0;
    int nij = 0;
    int nijk = 0;
    int nijk_max = 0;
    int offset;
    double rsq;
    double rjk2, rik2;
    double volume;
    int i0, j0, k0;
    double jnx, jny, jnz;
    double knx, kny, knz;
    int *ilist, *jlist, *numneigh, **firstneigh;
    const int32 *index_map;
    bool **shortneigh;
    int model_N = graph_model.get_n_elements() + 1;
    bool use_timer = false;

    const char * use_timer_flag = getenv("USE_TENSORALLOY_TIMER");
    if (use_timer_flag && strcmp(use_timer_flag, "true") == 0) {
        use_timer  = true;
    }

    ev_init(eflag, vflag);

    // grow local arrays if necessary
    // need to be atom->nmax in length
    if (atom->nmax > nmax) {
        nmax = atom->nmax;
    }

    double **R = atom->x;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    index_map = vap->get_index_map();
    shortneigh = new bool* [inum];

    auto t_start = Clock::now();

    // Cell
    volume = update_cell<T>();
    auto h_tensor_matrix = h_tensor->matrix<T>();

    // Volume
    volume_tensor->flat<T>()(0) = static_cast<T> (volume);

    // Positions
    auto R_tensor_mapped = R_tensor->tensor<T, 2>();
    R_tensor_mapped.setConstant(0.0f);
    nij_max = (inum - 1) * inum / 2;

    for (ii = 0; ii < inum; ii++) {
        ilocal = ilist[ii];
        igsl = index_map[ilocal];
        jnum = numneigh[ilocal];
        nij_max += jnum;
        R_tensor_mapped(igsl, 0) = R[ilocal][0];
        R_tensor_mapped(igsl, 1) = R[ilocal][1];
        R_tensor_mapped(igsl, 2) = R[ilocal][2];
    }

    auto g2_shift_tensor = new Tensor(dtype, TensorShape({nij_max, 3}));
    auto g2_shift_tensor_mapped = g2_shift_tensor->tensor<T, 2>();
    g2_shift_tensor_mapped.setConstant(0);

    auto g2_ilist_tensor = new Tensor(DT_INT32, TensorShape({nij_max}));
    auto g2_ilist_tensor_mapped = g2_ilist_tensor->tensor<int32, 1>();
    g2_ilist_tensor_mapped.setConstant(0);

    auto g2_jlist_tensor = new Tensor(DT_INT32, TensorShape({nij_max}));
    auto g2_jlist_tensor_mapped = g2_jlist_tensor->tensor<int32, 1>();
    g2_jlist_tensor_mapped.setConstant(0);

    auto g2_v2gmap_tensor = new Tensor(DT_INT32, TensorShape({nij_max, 2}));
    auto g2_v2gmap_tensor_mapped = g2_v2gmap_tensor->tensor<int32, 2>();
    g2_v2gmap_tensor_mapped.setConstant(0);

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        i0 = get_local_idx(i);
        itype = atom->type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];
        shortneigh[i] = new bool [jnum + ii];

        for (jj = 0; jj < jnum + ii; jj++) {
            if (jj < jnum) {
                j = jlist[jj];
                j &= (unsigned)NEIGHMASK;
            } else {
                j = jj - jnum;
            }
            shortneigh[i][jj] = false;
            rsq = get_interatomic_distance(i, j);

            if (rsq < cutforcesq) {
                shortneigh[i][jj] = true;
                jtype = atom->type[j];
                j0 = get_local_idx(j);
                get_shift_vector(j, jnx, jny, jnz);

                g2_shift_tensor_mapped(nij, 0) = static_cast<T> (jnx);
                g2_shift_tensor_mapped(nij, 1) = static_cast<T> (jny);
                g2_shift_tensor_mapped(nij, 2) = static_cast<T> (jnz);
                g2_ilist_tensor_mapped(nij) = index_map[i0];
                g2_jlist_tensor_mapped(nij) = index_map[j0];

                offset = IJ(itype, jtype, model_N);
                g2_v2gmap_tensor_mapped(nij, 0) = index_map[i0];
                g2_v2gmap_tensor_mapped(nij, 1) = g2_offset_map[offset];
                nij += 1;

                if (graph_model.angular()) {
                    for (kk = jj + 1; kk < jnum + ii; kk++) {
                        if (kk < jnum) {
                            k = jlist[kk];
                            k &= (unsigned)NEIGHMASK;
                        } else {
                            k = kk - jnum;
                        }
                        rik2 = get_interatomic_distance(i, k);
                        rjk2 = get_interatomic_distance(j, k);
                        if (rik2 < cutforcesq && rjk2 < cutforcesq) {
                            nijk_max += 1;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::pair<string, Tensor>> feed_dict({
        {"Placeholders/positions", *R_tensor},
        {"Placeholders/cells", *h_tensor},
        {"Placeholders/n_atoms_plus_virt", *n_atoms_vap_tensor},
        {"Placeholders/composition", *composition_tensor},
        {"Placeholders/volume", *volume_tensor},
        {"Placeholders/mask", *atom_mask_tensor},
        {"Placeholders/row_splits", *row_splits_tensor},
        {"Placeholders/g2.ilist", *g2_ilist_tensor},
        {"Placeholders/g2.jlist", *g2_jlist_tensor},
        {"Placeholders/g2.shift", *g2_shift_tensor},
        {"Placeholders/g2.v2g_map", *g2_v2gmap_tensor},
        {"Placeholders/pulay_stress", *pulay_stress_tensor}
    });

    auto t_g2 = Clock::now();

    Tensor *g4_v2gmap_tensor = nullptr;
    Tensor *g4_ilist_tensor = nullptr;
    Tensor *g4_jlist_tensor = nullptr;
    Tensor *g4_klist_tensor = nullptr;
    Tensor *g4_ij_shift_tensor = nullptr;
    Tensor *g4_ik_shift_tensor = nullptr;
    Tensor *g4_jk_shift_tensor = nullptr;

    if (graph_model.angular()) {
        g4_ilist_tensor = new Tensor(DT_INT32, TensorShape({nijk_max}));
        g4_jlist_tensor = new Tensor(DT_INT32, TensorShape({nijk_max}));
        g4_klist_tensor = new Tensor(DT_INT32, TensorShape({nijk_max}));
        g4_ij_shift_tensor = new Tensor(dtype, TensorShape({nijk_max, 3}));
        g4_ik_shift_tensor = new Tensor(dtype, TensorShape({nijk_max, 3}));
        g4_jk_shift_tensor = new Tensor(dtype, TensorShape({nijk_max, 3}));
        g4_v2gmap_tensor = new Tensor(DT_INT32, TensorShape({nijk_max, 2}));

        auto g4_ilist_tensor_mapped = g4_ilist_tensor->tensor<int32, 1>();
        auto g4_jlist_tensor_mapped = g4_jlist_tensor->tensor<int32, 1>();
        auto g4_klist_tensor_mapped = g4_klist_tensor->tensor<int32, 1>();
        auto g4_ij_shift_tensor_mapped = g4_ij_shift_tensor->tensor<T, 2>();
        auto g4_ik_shift_tensor_mapped = g4_ik_shift_tensor->tensor<T, 2>();
        auto g4_jk_shift_tensor_mapped = g4_jk_shift_tensor->tensor<T, 2>();
        auto g4_v2gmap_tensor_mapped = g4_v2gmap_tensor->tensor<int32, 2>();

        g4_ilist_tensor_mapped.setConstant(0);
        g4_jlist_tensor_mapped.setConstant(0);
        g4_klist_tensor_mapped.setConstant(0);
        g4_ij_shift_tensor_mapped.setConstant(0.0f);
        g4_ik_shift_tensor_mapped.setConstant(0.0f);
        g4_jk_shift_tensor_mapped.setConstant(0.0f);
        g4_v2gmap_tensor_mapped.setConstant(0);

        for (ii = 0; ii < inum; ii++) {
            i = ilist[ii];
            i0 = get_local_idx(i);
            itype = atom->type[i];
            jlist = firstneigh[i];
            jnum = numneigh[i];

            for (jj = 0; jj < jnum + ii; jj++) {
                if (jj < jnum) {
                    j = jlist[jj];
                    j &= (unsigned)NEIGHMASK;
                } else {
                    j = jj - jnum;
                }

                if (shortneigh[i][jj]) {
                    jtype = atom->type[j];
                    j0 = get_local_idx(j);
                    get_shift_vector(j, jnx, jny, jnz);

                    for (kk = jj + 1; kk < jnum + ii; kk++) {
                        if (kk < jnum) {
                            k = jlist[kk];
                            k &= (unsigned)NEIGHMASK;
                        } else {
                            k = kk - jnum;
                        }
                        if (shortneigh[i][kk] && get_interatomic_distance(j, k) < cutforcesq) {
                            ktype = atom->type[k];
                            k0 = get_local_idx(k);
                            get_shift_vector(k, knx, kny, knz);

                            g4_ilist_tensor_mapped(nijk) = index_map[i0];
                            g4_jlist_tensor_mapped(nijk) = index_map[j0];
                            g4_klist_tensor_mapped(nijk) = index_map[k0];

                            g4_ij_shift_tensor_mapped(nijk, 0) = static_cast<T> (jnx);
                            g4_ij_shift_tensor_mapped(nijk, 1) = static_cast<T> (jny);
                            g4_ij_shift_tensor_mapped(nijk, 2) = static_cast<T> (jnz);

                            g4_ik_shift_tensor_mapped(nijk, 0) = static_cast<T> (knx);
                            g4_ik_shift_tensor_mapped(nijk, 1) = static_cast<T> (kny);
                            g4_ik_shift_tensor_mapped(nijk, 2) = static_cast<T> (knz);

                            g4_jk_shift_tensor_mapped(nijk, 0) = static_cast<T> (knx - jnx);
                            g4_jk_shift_tensor_mapped(nijk, 1) = static_cast<T> (kny - jny);
                            g4_jk_shift_tensor_mapped(nijk, 2) = static_cast<T> (knz - jnz);

                            offset = IJK(itype, jtype, ktype, model_N);
                            g4_v2gmap_tensor_mapped(nijk, 0) = index_map[i0];
                            g4_v2gmap_tensor_mapped(nijk, 1) = g4_offset_map[offset];

                            nijk += 1;
                        }
                    }
                }
            }
        }
        std::vector<std::pair<string, Tensor>> addon({
            {"Placeholders/g4.v2g_map", *g4_v2gmap_tensor},
            {"Placeholders/g4.ij.ilist", *g4_ilist_tensor},
            {"Placeholders/g4.ij.jlist", *g4_jlist_tensor},
            {"Placeholders/g4.ik.ilist", *g4_ilist_tensor},
            {"Placeholders/g4.ik.klist", *g4_klist_tensor},
            {"Placeholders/g4.jk.jlist", *g4_jlist_tensor},
            {"Placeholders/g4.jk.klist", *g4_klist_tensor},
            {"Placeholders/g4.shift.ij", *g4_ij_shift_tensor},
            {"Placeholders/g4.shift.ik", *g4_ik_shift_tensor},
            {"Placeholders/g4.shift.jk", *g4_jk_shift_tensor},
        });
        feed_dict.insert(std::end(feed_dict), std::begin(addon), std::end(addon));
    }

    auto t_g4 = Clock::now();

    std::vector<Tensor> outputs;
    std::vector<string> run_ops({
        "Output/Energy/energy:0",
        "Output/Forces/forces:0",
        "Output/Stress/Full/stress:0"});
    Status status = session->Run(feed_dict, run_ops, {}, &outputs);
    if (!status.ok()) {
        auto message = "TensorAlloy internal error: " + status.ToString();
        error->all(FLERR, message.c_str());
    }
    auto t_run = Clock::now();

    if (eflag_global) {
        eng_vdwl = outputs[0].scalar<T>().data()[0];
    }

    auto nn_gsl_forces = outputs[1].matrix<T>();
    const int32 *reverse_map = vap->get_reverse_map();
    for (igsl = 1; igsl < vap->get_n_atoms_vap(); igsl++) {
        ilocal = reverse_map[igsl];
        if (ilocal >= 0) {
            atom->f[ilocal][0] = static_cast<double> (nn_gsl_forces(igsl - 1, 0));
            atom->f[ilocal][1] = static_cast<double> (nn_gsl_forces(igsl - 1, 1));
            atom->f[ilocal][2] = static_cast<double> (nn_gsl_forces(igsl - 1, 2));
        }
    }

    // Lammps uses a special Voigt order: xx yy zz xy xz yz
    virial[0] = static_cast<double> (-outputs[2].matrix<T>()(0, 0));
    virial[1] = static_cast<double> (-outputs[2].matrix<T>()(1, 1));
    virial[2] = static_cast<double> (-outputs[2].matrix<T>()(2, 2));
    virial[3] = static_cast<double> (-outputs[2].matrix<T>()(1, 0));
    virial[4] = static_cast<double> (-outputs[2].matrix<T>()(2, 0));
    virial[5] = static_cast<double> (-outputs[2].matrix<T>()(2, 1));

    vflag_fdotr = 0;

    dynamic_bytes = 0;
    dynamic_bytes += g2_shift_tensor->TotalBytes();
    dynamic_bytes += g2_ilist_tensor->TotalBytes();
    dynamic_bytes += g2_jlist_tensor->TotalBytes();
    dynamic_bytes += g2_v2gmap_tensor->TotalBytes();

    auto t_efv = Clock::now();

    delete g2_shift_tensor;
    delete g2_ilist_tensor;
    delete g2_jlist_tensor;
    delete g2_v2gmap_tensor;

    for (i = 0; i < inum; i++)
        delete [] shortneigh[i];
    delete [] shortneigh;

    if (graph_model.angular()) {

        dynamic_bytes += g4_v2gmap_tensor->TotalBytes();
        dynamic_bytes += g4_ilist_tensor->TotalBytes();
        dynamic_bytes += g4_jlist_tensor->TotalBytes();
        dynamic_bytes += g4_klist_tensor->TotalBytes();
        dynamic_bytes += g4_ij_shift_tensor->TotalBytes();
        dynamic_bytes += g4_ik_shift_tensor->TotalBytes();
        dynamic_bytes += g4_jk_shift_tensor->TotalBytes();

        delete g4_v2gmap_tensor;
        delete g4_ilist_tensor;
        delete g4_jlist_tensor;
        delete g4_klist_tensor;
        delete g4_ij_shift_tensor;
        delete g4_ik_shift_tensor;
        delete g4_jk_shift_tensor;
    }

    if (use_timer) {
        auto t_stop = Clock::now();
        auto ms_1 = std::chrono::duration_cast<std::chrono::milliseconds>(t_g2 - t_start).count();
        auto ms_2 = std::chrono::duration_cast<std::chrono::milliseconds>(t_g4 - t_g2).count();
        auto ms_3 = std::chrono::duration_cast<std::chrono::milliseconds>(t_run - t_g4).count();
        auto ms_4 = std::chrono::duration_cast<std::chrono::milliseconds>(t_efv - t_run).count();
        auto ms_5 = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_efv).count();
        auto ms_6 = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_start).count();
        printf("nij_max %5d nijk_max %5d ms %5lld g2 %5lld g4 %5lld run %5lld efv %5lld mem %5lld\n",
               nij_max, nijk_max, ms_6, ms_1, ms_2, ms_3, ms_4, ms_5);
    }
}


void PairTensorAlloy::compute(int eflag, int vflag)
{
    if (use_fp64) {
        run_once<double>(eflag, vflag, DataType::DT_DOUBLE);
    } else {
        run_once<float>(eflag, vflag, DataType::DT_FLOAT);
    }
}

/* ----------------------------------------------------------------------
   allocate pair_style arrays
------------------------------------------------------------------------- */

template <typename T>
void PairTensorAlloy::allocate_with_dtype(DataType dtype)
{
    if (vap == nullptr) {
        error->all(FLERR, "VAP is not succesfully initialized.");
    }

    allocated = 1;

    // ntypes: the number of atom types
    // N: the number of atom types plus one because `atom->type[i] >= 1`.
    // model: the atom types read from the graph model.
    // lmp: the atom types read from LAMMPS input (data) file.
    int model_ntypes = graph_model.get_n_elements();
    int model_N = model_ntypes + 1;
    int lmp_ntypes = atom->ntypes;
    int lmp_N = lmp_ntypes + 1;
    int i, j;

    memory->create(setflag, lmp_N, lmp_N, "pair:setflag");
    for (i = 1; i < lmp_N; i++)
        for (j = i; j <= lmp_N; j++)
            setflag[i][j] = 0;
    memory->create(cutsq, lmp_N, lmp_N, "pair:cutsq");

    // Lattice
    h_tensor = new Tensor(dtype, TensorShape({3, 3}));
    h_tensor->tensor<T, 2>().setConstant(0.f);

    // Positions
    R_tensor = new Tensor(dtype, TensorShape({vap->get_n_atoms_vap(), 3}));
    R_tensor->tensor<T, 2>().setConstant(0.f);

    // Volume
    volume_tensor = new Tensor(dtype, TensorShape({}));

    // row splits tensor
    row_splits_tensor = new Tensor(DT_INT32, TensorShape({model_N}));
    row_splits_tensor->tensor<int32, 1>().setConstant(0);
    for (i = 0; i < model_N; i++) {
        row_splits_tensor->tensor<int32, 1>()(i) = vap->get_row_splits()[i];
    }

    // Atom masks tensor
    atom_mask_tensor = new Tensor(dtype, TensorShape({vap->get_n_atoms_vap()}));
    auto atom_mask_ptr = vap->get_atom_mask();
    for (i = 0; i < vap->get_n_atoms_vap(); i++) {
        atom_mask_tensor->tensor<T, 1>()(i) = static_cast<T>(atom_mask_ptr[i]);
    }

    // N_atom_vap tensor
    n_atoms_vap_tensor = new Tensor(DT_INT32, TensorShape());
    n_atoms_vap_tensor->flat<int32>()(0) = vap->get_n_atoms_vap();

    // Pulay stress tensor
    pulay_stress_tensor = new Tensor(dtype, TensorShape());
    pulay_stress_tensor->flat<T>()(0) = 0.0f;

    // Composition tensor
    composition_tensor = new Tensor(dtype, TensorShape({model_ntypes}));
    composition_tensor->flat<T>().setConstant(0.0);
    for (i = 0; i < atom->nlocal; i++) {
        composition_tensor->flat<T>()(atom->type[i] - 1) += 1.0f;
    }
}

void PairTensorAlloy::allocate()
{
    if (use_fp64) {
        allocate_with_dtype<double>(DataType::DT_DOUBLE);
    } else {
        allocate_with_dtype<float>(DataType::DT_FLOAT);
    }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTensorAlloy::settings(int narg, char **/*arg*/)
{
    if (narg > 0) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   Load the graph model
------------------------------------------------------------------------- */

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status PairTensorAlloy::load_graph(const string &graph_file_name) {
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
   Initialize the G2 and G4 offset maps
------------------------------------------------------------------------- */

void PairTensorAlloy::init_offset_maps()
{
    int n = graph_model.get_n_elements();
    int N = n + 1;
    int i, j, k;
    int pos, offset;
    int g2_size = graph_model.get_ndim(false);
    int g4_size = graph_model.get_ndim(true);
    int g2_ndim_per_atom = n * g2_size;
    int g4_ndim_per_atom = 0;
    if (graph_model.angular()) {
        g4_ndim_per_atom = (n + 1) * n / 2 * g4_size;
    }

    offset = 0;

    memory->create(g2_offset_map, N * N, "pair:g2_offset_map");
    for (i = 1; i < N; i++) {
        pos = IJ(i, i, N);
        g2_offset_map[pos] = offset;
        offset += g2_size;
        for (j = 1; j < N; j++) {
            if (j == i) {
                continue;
            }
            pos = IJ(i, j, N);
            g2_offset_map[pos] = offset;
            offset += g2_size;
        }
        offset += g4_ndim_per_atom;
    }

    size_t alloc_size = N * N * N;
    memory->create(g4_offset_map, alloc_size, "pair:g4_offset_map");
    offset = g2_ndim_per_atom;
    for (i = 1; i < N; i++) {
        for (j = 1; j < N; j++) {
            for (k = j; k < N; k++) {
                pos = IJK(i, j, k, N);
                g4_offset_map[pos] = offset;
                offset += g4_size;
            }
        }
        offset += g2_ndim_per_atom;
    }
}

/* ----------------------------------------------------------------------
   Read the graph model.
------------------------------------------------------------------------- */

void PairTensorAlloy::read_graph_model(
        const string& graph_model_path,
        const std::vector<string>& symbols)
{
    Status load_graph_status = load_graph(graph_model_path);
    std::cout << "Read " << graph_model_path << ": " << load_graph_status << std::endl;

    std::vector<Tensor> outputs;
    Status status = session->Run({}, {"Transformer/params:0"}, {}, &outputs);
    if (!status.ok()) {
        auto message = "Decode graph model error: " + status.ToString();
        error->all(FLERR, message.c_str());
    }

    status = graph_model.read(outputs[0], graph_model_path, symbols);
    if (!status.ok()) {
        error->all(FLERR, status.error_message().c_str());
    }

    outputs.clear();
    status = session->Run({}, {"Metadata/precision:0"}, {}, &outputs);
    if (status.ok() && outputs[0].flat<string>().data()[0] == "high") {
        use_fp64 = true;
    } else {
        use_fp64 = false;
    }
}

/* ----------------------------------------------------------------------
   The implementation of `pair_coef
------------------------------------------------------------------------- */

void PairTensorAlloy::coeff(int narg, char **arg)
{
    // Read atom types from the lammps input file.
    std::vector<string> symbols;
    symbols.emplace_back("X");
    for (int i = 1; i < narg; i++) {
        auto option = string(arg[i]);
        if (option.c_str()[0] == '-') {
            if (option == "--serial") {
                serial_mode = true;
                std::cout << "Warning: serial mode is used." << std::endl;
            } else {
                auto message = "Unrecognized option: " + option;
                error->all(FLERR, message.c_str());
            }
        } else {
            symbols.emplace_back(string(arg[i]));
        }
    }

    // Load the graph model
    read_graph_model(string(arg[0]), symbols);
    graph_model.compute_max_occurs(atom->natoms, atom->type);

    // Initialize the Virtual-Atom Map
    vap = new VirtualAtomMap(memory);
    vap->build(graph_model, atom->natoms, atom->type);
    std::cout << "VAP initialized." << std::endl;

    // Initialize the offset maps.
    init_offset_maps();

    // Allocate arrays and tensors.
    allocate();

    // Set atomic masses
    double atom_mass[symbols.size()];
    for (int i = 1; i <= atom->ntypes; i++) {
        atom_mass[i] = 1.0;
    }
    atom->set_mass(atom_mass);

    // Set `setflag` which is required by Lammps.
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = i; j <= atom->ntypes; j++) {
            setflag[i][j] = 1;
        }
    }

    // Set `cutmax`.
    cutmax = graph_model.get_cutoff();
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTensorAlloy::init_style()
{
    // convert read-in file(s) to arrays and spline them
    neighbor->request(this, instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTensorAlloy::init_one(int /*i*/, int /*j*/)
{
    // single global cutoff = max of cut from all files read in
    // for funcfl could be multiple files
    // for setfl or fs, just one file
    cutforcesq = cutmax * cutmax;
    return cutmax;
}

/* ----------------------------------------------------------------------
   memory usage of tensors
------------------------------------------------------------------------- */

double PairTensorAlloy::tensors_memory_usage()
{
    double bytes = 0.0;
    bytes += h_tensor->TotalBytes();
    bytes += R_tensor->TotalBytes();
    bytes += volume_tensor->TotalBytes();
    bytes += atom_mask_tensor->TotalBytes();
    bytes += pulay_stress_tensor->TotalBytes();
    bytes += atom_mask_tensor->TotalBytes();
    bytes += composition_tensor->TotalBytes();
    bytes += dynamic_bytes;
    return bytes;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairTensorAlloy::memory_usage()
{
    double bytes = Pair::memory_usage();
    bytes += tensors_memory_usage();
    bytes += vap->memory_usage();
    return bytes;
}
