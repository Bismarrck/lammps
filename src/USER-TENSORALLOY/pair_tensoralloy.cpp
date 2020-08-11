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

#include "pair_tensoralloy.h"
#include "tensoralloy_utils.h"

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
#define IJ2num(i,j,n) i * n + j
#define IJK2num(i,j,k,n) i * n * n + j * n + k
#define eV_to_Kelvin 11604.51812


/* ---------------------------------------------------------------------- */

PairTensorAlloy::PairTensorAlloy(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
    force->newton_pair = 0;
    force->newton = 0;

    cutmax = 0.0;
    cutforcesq = 0.0;

    etemp = 0.0;

    radial_interactions = nullptr;
    radial_counters = nullptr;
    vap = nullptr;
    graph_model = nullptr;

    // Tensor pointers
    R_tensor = nullptr;
    h_tensor = nullptr;
    volume_tensor = nullptr;
    n_atoms_vap_tensor = nullptr;
    atom_mask_tensor = nullptr;
    pulay_stress_tensor = nullptr;
    etemperature_tensor = nullptr;
    nnl_max_tensor = nullptr;
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

    // Set the variables to their default values
    dynamic_bytes = 0;
    nmax = -1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h_inv[i][j] = 0.0;
        }
    }
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorAlloy::~PairTensorAlloy()
{
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(radial_interactions);
        memory->destroy(radial_counters);

        delete R_tensor;
        R_tensor = nullptr;

        delete h_tensor;
        h_tensor = nullptr;

        delete volume_tensor;
        volume_tensor = nullptr;

        delete n_atoms_vap_tensor;
        n_atoms_vap_tensor = nullptr;

        delete nnl_max_tensor;
        nnl_max_tensor = nullptr;

        delete pulay_stress_tensor;
        pulay_stress_tensor = nullptr;

        delete etemperature_tensor;
        etemperature_tensor = nullptr;

        delete atom_mask_tensor;
        atom_mask_tensor = nullptr;

        delete row_splits_tensor;
        row_splits_tensor = nullptr;
    }

    if (vap) {
        delete vap;
        vap = nullptr;
    }

    if (graph_model) {
        delete graph_model;
        graph_model = nullptr;
    }
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
void PairTensorAlloy::run_once_universal(int eflag, int vflag, DataType dtype)
{
    unsigned int i, j, k;
    int ii, jj, kk, inum, jnum, itype, jtype, ktype;
    int i_local, i_vap;
    int ijtype;
    int nij_max = 0;
    int nij = 0;
    int nijk = 0;
    int nijk_max = 0;
    int nnl_max = 0;
    int offset;
    int inc;
    double rsq;
    double rjk2, rik2;
    double volume;
    int i0, j0, k0;
    double jnx, jny, jnz;
    double knx, kny, knz;
    int *ilist, *jlist, *numneigh, **firstneigh;
    const int32 *local_to_vap_map;
    bool **shortneigh;
    int model_N = graph_model->get_n_elements() + 1;
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
    local_to_vap_map = vap->get_local_to_vap_map();
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
        i_local = ilist[ii];
        i_vap = local_to_vap_map[i_local];
        jnum = numneigh[i_local];
        nij_max += jnum;
        R_tensor_mapped(i_vap, 0) = R[i_local][0];
        R_tensor_mapped(i_vap, 1) = R[i_local][1];
        R_tensor_mapped(i_vap, 2) = R[i_local][2];
    }

    // Reset the counters
    for (i = 0; i < vap->get_n_atoms_vap(); i++) {
        for (j = 0; j < atom->ntypes + 1; j++) {
            radial_counters[i][j] = 0;
        }
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

    auto g2_v2gmap_tensor = new Tensor(DT_INT32, TensorShape({nij_max, 5}));
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
                ijtype = radial_interactions[itype][jtype];

                g2_shift_tensor_mapped(nij, 0) = static_cast<T> (jnx);
                g2_shift_tensor_mapped(nij, 1) = static_cast<T> (jny);
                g2_shift_tensor_mapped(nij, 2) = static_cast<T> (jnz);
                g2_ilist_tensor_mapped(nij) = local_to_vap_map[i0];
                g2_jlist_tensor_mapped(nij) = local_to_vap_map[j0];

                inc = radial_counters[local_to_vap_map[i0]][ijtype];
                nnl_max = MAX(inc + 1, nnl_max);

                offset = IJ2num(itype, jtype, model_N);
                g2_v2gmap_tensor_mapped(nij, 0) = ijtype;
                g2_v2gmap_tensor_mapped(nij, 1) = local_to_vap_map[i0];
                g2_v2gmap_tensor_mapped(nij, 2) = inc;
                g2_v2gmap_tensor_mapped(nij, 3) = 0;
                g2_v2gmap_tensor_mapped(nij, 4) = 1;
                nij += 1;
                radial_counters[local_to_vap_map[i0]][ijtype] += 1;
            }
        }
    }

    // Set the nnl_max
    nnl_max_tensor->flat<int32>()(0) = static_cast<int32> (nnl_max + 1);

    std::vector<std::pair<string, Tensor>> feed_dict({
        {"Placeholders/positions", *R_tensor},
        {"Placeholders/n_atoms_vap", *n_atoms_vap_tensor},
        {"Placeholders/nnl_max", *nnl_max_tensor},
        {"Placeholders/atom_masks", *atom_mask_tensor},
        {"Placeholders/cell", *h_tensor},
        {"Placeholders/volume", *volume_tensor},
        {"Placeholders/pulay_stress", *pulay_stress_tensor},
        {"Placeholders/etemperature", *etemperature_tensor},
        {"Placeholders/row_splits", *row_splits_tensor},
        {"Placeholders/g2.v2g_map", *g2_v2gmap_tensor},
        {"Placeholders/g2.ilist", *g2_ilist_tensor},
        {"Placeholders/g2.jlist", *g2_jlist_tensor},
        {"Placeholders/g2.n1", *g2_shift_tensor},
    });

    auto t_g2 = Clock::now();

    Tensor *g4_v2gmap_tensor = nullptr;
    Tensor *g4_ilist_tensor = nullptr;
    Tensor *g4_jlist_tensor = nullptr;
    Tensor *g4_klist_tensor = nullptr;
    Tensor *g4_ij_shift_tensor = nullptr;
    Tensor *g4_ik_shift_tensor = nullptr;
    Tensor *g4_jk_shift_tensor = nullptr;

    if (graph_model->angular()) {
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

                            g4_ilist_tensor_mapped(nijk) = local_to_vap_map[i0];
                            g4_jlist_tensor_mapped(nijk) = local_to_vap_map[j0];
                            g4_klist_tensor_mapped(nijk) = local_to_vap_map[k0];

                            g4_ij_shift_tensor_mapped(nijk, 0) = static_cast<T> (jnx);
                            g4_ij_shift_tensor_mapped(nijk, 1) = static_cast<T> (jny);
                            g4_ij_shift_tensor_mapped(nijk, 2) = static_cast<T> (jnz);

                            g4_ik_shift_tensor_mapped(nijk, 0) = static_cast<T> (knx);
                            g4_ik_shift_tensor_mapped(nijk, 1) = static_cast<T> (kny);
                            g4_ik_shift_tensor_mapped(nijk, 2) = static_cast<T> (knz);

                            g4_jk_shift_tensor_mapped(nijk, 0) = static_cast<T> (knx - jnx);
                            g4_jk_shift_tensor_mapped(nijk, 1) = static_cast<T> (kny - jny);
                            g4_jk_shift_tensor_mapped(nijk, 2) = static_cast<T> (knz - jnz);

                            offset = IJK2num(itype, jtype, ktype, model_N);
                            g4_v2gmap_tensor_mapped(nijk, 0) = local_to_vap_map[i0];

                            nijk += 1;
                        }
                    }
                }
            }
        }
        feed_dict.emplace_back("Placeholders/g4.v2g_map", *g4_v2gmap_tensor);
        feed_dict.emplace_back("Placeholders/g4.ilist", *g4_ilist_tensor);
        feed_dict.emplace_back("Placeholders/g4.jlist", *g4_jlist_tensor);
        feed_dict.emplace_back("Placeholders/g4.klist", *g4_klist_tensor);
        feed_dict.emplace_back("Placeholders/g4.n1", *g4_ij_shift_tensor);
        feed_dict.emplace_back("Placeholders/g4.n2", *g4_ik_shift_tensor);
        feed_dict.emplace_back("Placeholders/g4.n3", *g4_jk_shift_tensor);
    }

    auto t_g4 = Clock::now();

    std::vector<Tensor> outputs;
    Status status = graph_model->run(feed_dict, outputs);

    if (!status.ok()) {
        auto message = "TensorAlloy internal error: " + status.ToString();
        error->all(FLERR, message.c_str());
    }
    auto t_run = Clock::now();

    if (eflag_global) {
        eng_vdwl = outputs[0].scalar<T>().data()[0];
    }

    auto F_vap = outputs[1].matrix<T>();
    const int32 *vap_to_local = vap->get_vap_to_local_map();
    for (i_vap = 1; i_vap < vap->get_n_atoms_vap(); i_vap++) {
        i_local = vap_to_local[i_vap];
        if (i_local >= 0) {
            atom->f[i_local][0] = static_cast<double> (F_vap(i_vap - 1, 0));
            atom->f[i_local][1] = static_cast<double> (F_vap(i_vap - 1, 1));
            atom->f[i_local][2] = static_cast<double> (F_vap(i_vap - 1, 2));
        }
    }

    // Lammps uses a special Voigt order: xx yy zz xy xz yz
    virial[0] = static_cast<double> (-outputs[2].flat<T>()(0));
    virial[1] = static_cast<double> (-outputs[2].flat<T>()(1));
    virial[2] = static_cast<double> (-outputs[2].flat<T>()(2));
    virial[3] = static_cast<double> (-outputs[2].flat<T>()(5));
    virial[4] = static_cast<double> (-outputs[2].flat<T>()(4));
    virial[5] = static_cast<double> (-outputs[2].flat<T>()(3));

    vflag_fdotr = 0;

    dynamic_bytes = 0;
    dynamic_bytes += g2_shift_tensor->TotalBytes();
    dynamic_bytes += g2_ilist_tensor->TotalBytes();
    dynamic_bytes += g2_jlist_tensor->TotalBytes();
    dynamic_bytes += g2_v2gmap_tensor->TotalBytes();

    delete g2_shift_tensor;
    delete g2_ilist_tensor;
    delete g2_jlist_tensor;
    delete g2_v2gmap_tensor;

    for (i = 0; i < inum; i++)
        delete [] shortneigh[i];
    delete [] shortneigh;

    if (graph_model->angular()) {

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
        auto ms_6 = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_start).count();
        printf("nij_max=%5d nijk_max=%5d nnl_max=%5d comput=%5lld g2=%5lld g4=%5lld run=%5lld\n",
               nij_max, nijk_max, nnl_max, ms_6, ms_1, ms_2, ms_3);
    }
}


/* ----------------------------------------------------------------------
   compute with different precision
------------------------------------------------------------------------- */

void PairTensorAlloy::compute(int eflag, int vflag)
{
//    if (graph_model->use_fp64()) {
//        run_once_universal<double>(eflag, vflag, DataType::DT_DOUBLE);
//    } else {
//        run_once_universal<float>(eflag, vflag, DataType::DT_FLOAT);
//    }
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
    int model_ntypes = graph_model->get_n_elements();
    int model_N = model_ntypes + 1;
    int lmp_ntypes = atom->ntypes;
    int lmp_N = lmp_ntypes + 1;
    int i, j;

    memory->create(cutsq, lmp_N, lmp_N, "pair:cutsq");
    memory->create(setflag, lmp_N, lmp_N, "pair:setflag");
    for (i = 1; i < lmp_N; i++) {
        for (j = i; j < lmp_N; j++) {
            setflag[i][j] = 0;
        }
    }

    // Radial interactions
    memory->create(radial_interactions, lmp_N, lmp_N, "pair:ta:radial");
    for (i = 1; i < lmp_N; i++) {
        radial_interactions[i][i] = 0;
        int val = 1;
        for (j = 1; j < lmp_N; j++) {
            if (j != i) {
                radial_interactions[i][j] = val;
                val ++;
            }
        }
    }

    // Radial interaction counters
    memory->create(radial_counters, vap->get_n_atoms_vap(), lmp_N, "pair:ta:rcount");
    for (i = 0; i < vap->get_n_atoms_vap(); i++) {
        for (j = 0; j < lmp_N; j++) {
            radial_counters[i][j] = 0;
        }
    }

    // Lattice
    h_tensor = new Tensor(dtype, TensorShape({3, 3}));
    h_tensor->tensor<T, 2>().setConstant(0.f);

    // Positions
    R_tensor = new Tensor(dtype, TensorShape({vap->get_n_atoms_vap(), 3}));
    R_tensor->tensor<T, 2>().setConstant(0.f);

    // Volume
    volume_tensor = new Tensor(dtype, TensorShape());

    // row splits tensor
    row_splits_tensor = new Tensor(DT_INT32, TensorShape({model_N}));
    row_splits_tensor->tensor<int32, 1>().setConstant(0);
    for (i = 0; i < model_N; i++) {
        row_splits_tensor->tensor<int32, 1>()(i) = vap->get_row_splits()[i];
    }

    // Atom masks tensor
    atom_mask_tensor = new Tensor(dtype, TensorShape({vap->get_n_atoms_vap()}));
    auto atom_mask_ptr = vap->get_atom_masks();
    for (i = 0; i < vap->get_n_atoms_vap(); i++) {
        atom_mask_tensor->tensor<T, 1>()(i) = static_cast<T>(atom_mask_ptr[i]);
    }

    // N_atom_vap tensor
    n_atoms_vap_tensor = new Tensor(DT_INT32, TensorShape());
    n_atoms_vap_tensor->flat<int32>()(0) = vap->get_n_atoms_vap();

    // nnl_max tensor
    nnl_max_tensor = new Tensor(DT_INT32, TensorShape());

    // Pulay stress tensor
    pulay_stress_tensor = new Tensor(dtype, TensorShape());
    pulay_stress_tensor->flat<T>()(0) = 0.0f;

    // electron temperature tensor
    etemperature_tensor = new Tensor(dtype, TensorShape());
    etemperature_tensor->flat<T>()(0) = 0.0f;
}

void PairTensorAlloy::allocate()
{
    if (graph_model->use_fp64()) {
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
   The implementation of `pair_coef
------------------------------------------------------------------------- */

void PairTensorAlloy::coeff(int narg, char **arg)
{
    // Read atom types from the lammps input file.
    std::vector<string> symbols;
    symbols.emplace_back("X");
    int idx = 1;
    while (idx < narg) {
        auto iarg = string(arg[idx]);
        if (iarg == "serial") {
            auto val = string(arg[idx + 1]);
            if (val == "off") {
                serial_mode = false;
            } else if (val == "on") {
                serial_mode = true;
                if (comm->me == 0) {
                    std::cout << "Serial mode is enabled" << std::endl;
                }
            } else {
                error->all(FLERR, "'on/off' are available values for key 'serial'");
            }
            idx ++;
        } else if (iarg == "etemp") {
            double kelvin = std::atof(string(arg[idx + 1]).c_str());
            etemp = kelvin / eV_to_Kelvin;
            if (comm->me == 0) {
                std::cout << "Electron temperature is " << etemp << " (eV)" << std::endl;
            }
            idx ++;
        } else {
            symbols.emplace_back(iarg);
        }
        idx ++;
    }

    // Load the graph model
    graph_model = new GraphModel(
            string(arg[0]),
            symbols,
            error,
            serial_mode,
            comm->me == 0);
    graph_model->compute_max_occurs(atom->natoms, atom->type);

    // Initialize the Virtual-Atom Map
    vap = new VirtualAtomMap(memory);
    vap->build(graph_model, atom->natoms, atom->type);

    if (comm->me) {
        std::cout << "VAP initialized." << std::endl;
    }

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
    cutmax = graph_model->get_cutoff();
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
    bytes += nnl_max_tensor->TotalBytes();
    bytes += pulay_stress_tensor->TotalBytes();
    bytes += etemperature_tensor->TotalBytes();
    bytes += atom_mask_tensor->TotalBytes();
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
