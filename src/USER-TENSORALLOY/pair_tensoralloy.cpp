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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <domain.h>
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "jsoncpp/json/json.h"
#include "cnpy/cnpy.h"
#include "pair_tensoralloy.h"

using namespace LAMMPS_NS;

using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::DT_INT32;
using tensorflow::DT_FLOAT;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;

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
    use_angular = false;
    n_eta = 0;
    n_omega = 0;
    n_beta = 0;
    n_gamma = 0;
    n_zeta = 0;

    g2_offset_map = nullptr;
    g4_offset_map = nullptr;
    g4_numneigh = nullptr;
    max_occurs = nullptr;
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

    // Disable log by default.
    verbose = false;

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
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorAlloy::~PairTensorAlloy()
{
    if (vap) {
        delete vap;
        vap = nullptr;
    }

    memory->destroy(max_occurs);
    delete [] max_occurs;

    memory->destroy(g2_offset_map);
    delete [] g2_offset_map;

    if (use_angular) {
        memory->destroy(g4_offset_map);
        delete [] g4_offset_map;

        memory->destroy(g4_numneigh);
        delete [] g4_numneigh;
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
    double boxxlo, boxxhi, boxylo, boxyhi, boxzlo, boxzhi;
    double boxxy, boxxz, boxyz;

    if (domain->triclinic == 0) {
        boxxlo = domain->boxlo[0];
        boxxhi = domain->boxhi[0];
        boxylo = domain->boxlo[1];
        boxyhi = domain->boxhi[1];
        boxzlo = domain->boxlo[2];
        boxzhi = domain->boxhi[2];
        boxxy = 0.0;
        boxxz = 0.0;
        boxyz = 0.0;
    } else {
        boxxlo = domain->boxlo_bound[0];
        boxxhi = domain->boxhi_bound[0];
        boxylo = domain->boxlo_bound[1];
        boxyhi = domain->boxhi_bound[1];
        boxzlo = domain->boxlo_bound[2];
        boxzhi = domain->boxhi_bound[2];
        boxxy = domain->xy;
        boxxz = domain->xz;
        boxyz = domain->yz;
    }

    // Copied from `ase.io.lammpsdata.read_lammps_data`
    double xhilo = (boxxhi - boxxlo) - abs(boxxy) - abs(boxxz);
    double yhilo = (boxyhi - boxylo) - abs(boxyz);
    double zhilo = boxzhi - boxzlo;
    auto h_mapped = h_tensor->template tensor<T, 2>();
    h_mapped(0, 0) = static_cast<T> (xhilo);
    h_mapped(0, 1) = static_cast<T> (0.0);
    h_mapped(0, 2) = static_cast<T> (0.0);
    h_mapped(1, 0) = static_cast<T> (boxxy);
    h_mapped(1, 1) = static_cast<T> (yhilo);
    h_mapped(1, 2) = static_cast<T> (0.0);
    h_mapped(2, 0) = static_cast<T> (boxxz);
    h_mapped(2, 1) = static_cast<T> (boxyz);
    h_mapped(2, 2) = static_cast<T> (zhilo);

    // Compute the volume.
    double volume;
    if (domain->dimension == 3) {
        volume = domain->xprd * domain->yprd * domain->zprd;
    }
    else {
        volume = domain->xprd * domain->yprd;
    }

    // Compute the inverse matrix of `h`
    double ivol = 1.0 / volume;
    h_inv[0][0] = (h_mapped(1, 1) * h_mapped(2, 2) - h_mapped(2, 1) * h_mapped(1, 2)) * ivol;
    h_inv[0][1] = (h_mapped(0, 2) * h_mapped(2, 1) - h_mapped(0, 1) * h_mapped(2, 2)) * ivol;
    h_inv[0][2] = (h_mapped(0, 1) * h_mapped(1, 2) - h_mapped(0, 2) * h_mapped(1, 1)) * ivol;
    h_inv[1][0] = (h_mapped(1, 2) * h_mapped(2, 0) - h_mapped(1, 0) * h_mapped(2, 2)) * ivol;
    h_inv[1][1] = (h_mapped(0, 0) * h_mapped(2, 2) - h_mapped(0, 2) * h_mapped(2, 0)) * ivol;
    h_inv[1][2] = (h_mapped(1, 0) * h_mapped(0, 2) - h_mapped(0, 0) * h_mapped(1, 2)) * ivol;
    h_inv[2][0] = (h_mapped(1, 0) * h_mapped(2, 1) - h_mapped(2, 0) * h_mapped(1, 1)) * ivol;
    h_inv[2][1] = (h_mapped(2, 0) * h_mapped(0, 1) - h_mapped(0, 0) * h_mapped(2, 1)) * ivol;
    h_inv[2][2] = (h_mapped(0, 0) * h_mapped(1, 1) - h_mapped(1, 0) * h_mapped(0, 1)) * ivol;

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

double PairTensorAlloy::get_interatomic_distance(const int i, const int j, bool square)
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

void PairTensorAlloy::compute(int eflag, int vflag)
{
    unsigned int i, j, k;
    int ii, jj, kk, inum, jnum, itype, jtype, ktype, ilocal, igsl;
    int nij_max = 0;
    int nij = 0;
    int nijk = 0;
    int nijk_max = 0;
    int offset;
    double rsq;
    double volume;
    int i0, j0, k0;
    double jnx, jny, jnz;
    double knx, kny, knz;
    int *ilist, *jlist, *numneigh, **firstneigh;
    const int32 *index_map;
    bool **shortneigh;

    // The atom type starts from 1 in LAMMPS. So `n_lmp_types` should be `ntypes + 1`.
    int n_lmp_types = atom->ntypes + 1;

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

    // Cell
    volume = update_cell<float>();
    auto h_tensor_matrix = h_tensor->matrix<float>();

    // Volume
    volume_tensor->flat<float>()(0) = static_cast<float> (volume);

    // Positions
    auto R_tensor_mapped = R_tensor->tensor<float, 2>();
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

    auto g2_shift_tensor = new Tensor(DT_FLOAT, TensorShape({nij_max, 3}));
    auto g2_shift_tensor_mapped = g2_shift_tensor->tensor<float, 2>();
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
        itype = atom->type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];
        shortneigh[i] = new bool [jnum + ii];

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= (unsigned)NEIGHMASK;
            shortneigh[ii][jj] = false;
            rsq = get_interatomic_distance(i, j, true);

//        for (jj = 0; jj < jnum + ii; jj++) {
//            if (jj < jnum) {
//                j = jlist[jj];
//                j &= (unsigned)NEIGHMASK;
//            } else {
//                j = jj - jnum;
//            }
//            shortneigh[ii][jj] = false;
//            rsq = get_interatomic_distance(i, j);

            if (rsq < cutforcesq) {
                shortneigh[ii][jj] = true;
                jtype = atom->type[j];
                j0 = get_local_idx(j);
                get_shift_vector(j, jnx, jny, jnz);

                g2_shift_tensor_mapped(nij, 0) = static_cast<float> (jnx);
                g2_shift_tensor_mapped(nij, 1) = static_cast<float> (jny);
                g2_shift_tensor_mapped(nij, 2) = static_cast<float> (jnz);
                g2_ilist_tensor_mapped(nij) = index_map[i];
                g2_jlist_tensor_mapped(nij) = index_map[j0];

                offset = IJ(itype, jtype, n_lmp_types);
                g2_v2gmap_tensor_mapped(nij, 0) = index_map[i];
                g2_v2gmap_tensor_mapped(nij, 1) = g2_offset_map[offset];
                nij += 1;

                if (use_angular) {
                    g4_numneigh[i] += 1;
                }

                if (i < inum && j < inum) {
                    // `j > i` is always true in Lammps neighbour list.
                    g2_shift_tensor_mapped(nij, 0) = 0.0f;
                    g2_shift_tensor_mapped(nij, 1) = 0.0f;
                    g2_shift_tensor_mapped(nij, 2) = 0.0f;
                    g2_ilist_tensor_mapped(nij) = index_map[j];
                    g2_jlist_tensor_mapped(nij) = index_map[i];

                    offset = IJ(jtype, itype, n_lmp_types);
                    g2_v2gmap_tensor_mapped(nij, 0) = index_map[j];
                    g2_v2gmap_tensor_mapped(nij, 1) = g2_offset_map[offset];
                    nij += 1;

                    if (use_angular) {
                        g4_numneigh[j] += 1;
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

    if (verbose) {
        std::cout << "Cell: \n" << std::setprecision(6) << h_tensor->matrix<float>() << std::endl;
        std::cout << "Volume: " << std::setprecision(6) << volume << std::endl;
        std::cout << "positions: \n" << std::setprecision(6) << R_tensor->matrix<float>() << std::endl;
        std::cout << "Nij_max: " << nij_max << std::endl;
        std::cout << "composition: \n" << composition_tensor->vec<float>() << std::endl;
        std::cout << "Nij: " << nij << std::endl;
        std::cout << "Atom mask: \n" << atom_mask_tensor->vec<float>() << std::endl;
        std::cout << "Row splits: \n" << row_splits_tensor->vec<int32>() << std::endl;
    }

    Tensor *g4_v2gmap_tensor = nullptr;
    Tensor *g4_ilist_tensor = nullptr;
    Tensor *g4_jlist_tensor = nullptr;
    Tensor *g4_klist_tensor = nullptr;
    Tensor *g4_ij_shift_tensor = nullptr;
    Tensor *g4_ik_shift_tensor = nullptr;
    Tensor *g4_jk_shift_tensor = nullptr;

    if (use_angular) {

        for (i = 0; i < atom->nlocal; i++) {
            jnum = g4_numneigh[i];
            nijk_max += (jnum - 1) * jnum / 2;
        }

        if (verbose) {
            std::cout << "Nijk_max: " << nijk_max << std::endl;
        }

        g4_ilist_tensor = new Tensor(DT_INT32, TensorShape({nijk_max}));
        g4_jlist_tensor = new Tensor(DT_INT32, TensorShape({nijk_max}));
        g4_klist_tensor = new Tensor(DT_INT32, TensorShape({nijk_max}));
        g4_ij_shift_tensor = new Tensor(DT_FLOAT, TensorShape({nijk_max, 3}));
        g4_ik_shift_tensor = new Tensor(DT_FLOAT, TensorShape({nijk_max, 3}));
        g4_jk_shift_tensor = new Tensor(DT_FLOAT, TensorShape({nijk_max, 3}));
        g4_v2gmap_tensor = new Tensor(DT_INT32, TensorShape({nijk_max, 2}));

        auto g4_ilist_tensor_mapped = g4_ilist_tensor->tensor<int32, 1>();
        auto g4_jlist_tensor_mapped = g4_jlist_tensor->tensor<int32, 1>();
        auto g4_klist_tensor_mapped = g4_klist_tensor->tensor<int32, 1>();
        auto g4_ij_shift_tensor_mapped = g4_ij_shift_tensor->tensor<float, 2>();
        auto g4_ik_shift_tensor_mapped = g4_ik_shift_tensor->tensor<float, 2>();
        auto g4_jk_shift_tensor_mapped = g4_jk_shift_tensor->tensor<float, 2>();
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

                if (shortneigh[ii][jj]) {

                    jtype = atom->type[j];
                    j0 = get_local_idx(j);
                    get_shift_vector(j, jnx, jny, jnz);

                    for (kk = jj + 1; kk < jnum; kk++) {
                        k = jlist[kk];
                        k &= (unsigned)NEIGHMASK;

//                        rikx = R[k][0] - xi0;
//                        riky = R[k][1] - yi0;
//                        rikz = R[k][2] - zi0;
//                        rik2 = rikx * rijx + riky * riky + rikz * rikz;
//
//                        rjkx = R[j][0] - R[k][0];
//                        rjky = R[j][2] - R[k][1];
//                        rjkz = R[j][1] - R[k][2];
//                        rjk2 = rjkx * rjkx + rjky * rjky + rjkz * rjkz;

                        if (shortneigh[ii][kk]) {

                            ktype = atom->type[k];
                            k0 = get_local_idx(k);
                            get_shift_vector(k, knx, kny, knz);

                            g4_ilist_tensor_mapped(nijk) = index_map[i0];
                            g4_jlist_tensor_mapped(nijk) = index_map[j0];
                            g4_klist_tensor_mapped(nijk) = index_map[k0];

                            g4_ij_shift_tensor_mapped(nijk, 0) = static_cast<float> (jnx);
                            g4_ij_shift_tensor_mapped(nijk, 1) = static_cast<float> (jny);
                            g4_ij_shift_tensor_mapped(nijk, 2) = static_cast<float> (jnz);

                            g4_ik_shift_tensor_mapped(nijk, 0) = static_cast<float> (knx);
                            g4_ik_shift_tensor_mapped(nijk, 1) = static_cast<float> (kny);
                            g4_ik_shift_tensor_mapped(nijk, 2) = static_cast<float> (knz);

                            g4_jk_shift_tensor_mapped(nijk, 0) = static_cast<float> (knx - jnx);
                            g4_jk_shift_tensor_mapped(nijk, 1) = static_cast<float> (kny - jny);
                            g4_jk_shift_tensor_mapped(nijk, 2) = static_cast<float> (knz - jnz);

                            offset = IJK(itype, jtype, ktype, n_lmp_types);
                            g4_v2gmap_tensor_mapped(nijk, 0) = index_map[i0];
                            g4_v2gmap_tensor_mapped(nijk, 1) = g4_offset_map[offset];

                            nijk += 1;

                            printf("%2d, %2d, %2d, ( % 2.0f, % 2.0f, % 2.0f ), ( % 2.0f, % 2.0f, % 2.0f ), ( % 2.0f, % 2.0f, % 2.0f )\n",
                                    index_map[i0], index_map[j0], index_map[k0],
                                    jnx, jny, jnz, knx, kny, knz, knx - jnx, kny - jny, knz - jnz);

                            if (j < inum || k < inum) {

                                g4_ilist_tensor_mapped(nijk) = index_map[j0];
                                g4_jlist_tensor_mapped(nijk) = index_map[i0];
                                g4_klist_tensor_mapped(nijk) = index_map[k0];

                                g4_ij_shift_tensor_mapped(nijk, 0) = static_cast<float> (jnx);
                                g4_ij_shift_tensor_mapped(nijk, 1) = static_cast<float> (jny);
                                g4_ij_shift_tensor_mapped(nijk, 2) = static_cast<float> (jnz);

                                g4_ik_shift_tensor_mapped(nijk, 0) = static_cast<float> (knx);
                                g4_ik_shift_tensor_mapped(nijk, 1) = static_cast<float> (kny);
                                g4_ik_shift_tensor_mapped(nijk, 2) = static_cast<float> (knz);

                                g4_jk_shift_tensor_mapped(nijk, 0) = static_cast<float> (knx - jnx);
                                g4_jk_shift_tensor_mapped(nijk, 1) = static_cast<float> (kny - jny);
                                g4_jk_shift_tensor_mapped(nijk, 2) = static_cast<float> (knz - jnz);

                                offset = IJK(itype, jtype, ktype, n_lmp_types);
                                g4_v2gmap_tensor_mapped(nijk, 0) = index_map[j0];
                                g4_v2gmap_tensor_mapped(nijk, 1) = g4_offset_map[offset];

                                nijk += 1;

                            }
                        }
                    }
                }
            }
        }

        std::cout << "nijk_max: " << nijk_max << std::endl;
        std::cout << "nijk: " << nijk << std::endl;

        cnpy::npy_save<int32>(
                "g4.ilist.npy",
                g4_ilist_tensor->flat<int32>().data(),
                {static_cast<size_t>(nijk_max)},
                "w");
        std::cout << "g4.ilist.npy" << std::endl;
        cnpy::npy_save<int32>(
                "g4.jlist.npy",
                g4_jlist_tensor->flat<int32>().data(),
                {static_cast<size_t>(nijk_max)},
                "w");
        std::cout << "g4.jlist.npy" << std::endl;
        cnpy::npy_save<int32>(
                "g4.klist.npy",
                g4_klist_tensor->flat<int32>().data(),
                {static_cast<size_t>(nijk_max)},
                "w");
        std::cout << "g4.klist.npy" << std::endl;
        cnpy::npy_save<float>(
                "g4.ijSlist.npy",
                g4_ij_shift_tensor->flat<float>().data(),
                {static_cast<size_t>(nijk_max), 3},
                "w");
        std::cout << "g4.ijSlist.npy" << std::endl;
        cnpy::npy_save<float>(
                "g4.ikSlist.npy",
                g4_ik_shift_tensor->flat<float>().data(),
                {static_cast<size_t>(nijk_max), 3},
                "w");
        std::cout << "g4.ikSlist.npy" << std::endl;
        cnpy::npy_save<float>(
                "g4.jkSlist.npy",
                g4_jk_shift_tensor->flat<float>().data(),
                {static_cast<size_t>(nijk_max), 3},
                "w");
        std::cout << "g4.jkSlist.npy" << std::endl;
        cnpy::npy_save<int32>(
                "g4.v2g_map.npy",
                g4_v2gmap_tensor->flat<int32>().data(),
                {static_cast<size_t>(nijk_max), 2},
                "w");
        std::cout << "g4.v2g_map.npy" << std::endl;

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

    std::vector<Tensor> outputs;
    std::vector<string> run_ops({
        "Output/Energy/energy:0",
        "Output/Forces/forces:0",
        "Output/Stress/Voigt/stress:0"});
    Status status = session->Run(feed_dict, run_ops, {}, &outputs);

    if (verbose) {
        std::cout << status.ToString() << std::endl;
    }
    else if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        error->all(FLERR, "TensorAlloy internal error");
    }

    if (eflag_global) {
        eng_vdwl = outputs[0].scalar<float>().data()[0];
    }

    auto nn_gsl_forces = outputs[1].matrix<float>();
    const int32 *reverse_map = vap->get_reverse_map();

    for (igsl = 1; igsl < vap->get_n_atoms_vap(); igsl++) {
        ilocal = reverse_map[igsl];
        if (ilocal >= 0) {
            atom->f[ilocal][0] = static_cast<double> (nn_gsl_forces(igsl - 1, 0));
            atom->f[ilocal][1] = static_cast<double> (nn_gsl_forces(igsl - 1, 1));
            atom->f[ilocal][2] = static_cast<double> (nn_gsl_forces(igsl - 1, 2));
        }
    }

    auto nn_virial = outputs[2].vec<float>();
    for (i = 0; i < 6; i++) {
        virial[i] = static_cast<double> (nn_virial(i)) * volume * (-1.0);
    }
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

    if (use_angular) {

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
}

/* ----------------------------------------------------------------------
   allocate pair_style arrays
------------------------------------------------------------------------- */

void PairTensorAlloy::allocate()
{
    if (vap == nullptr) {
        error->all(FLERR, "VAP is not succesfully initialized.");
    }

    allocated = 1;

    int n = atom->ntypes;
    int i, j;

    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (i = 1; i <= n; i++)
        for (j = i; j <= n; j++)
            setflag[i][j] = 0;
    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

    // Lattice
    h_tensor = new Tensor(DT_FLOAT, TensorShape({3, 3}));
    h_tensor->tensor<float, 2>().setConstant(0.f);

    // Positions
    R_tensor = new Tensor(DT_FLOAT, TensorShape({vap->get_n_atoms_vap(), 3}));
    R_tensor->tensor<float, 2>().setConstant(0.f);

    // Volume
    volume_tensor = new Tensor(DT_FLOAT, TensorShape({}));

    // row splits tensor
    row_splits_tensor = new Tensor(DT_INT32, TensorShape({n + 1}));
    row_splits_tensor->tensor<int32, 1>().setConstant(0);
    for (i = 0; i < n + 1; i++) {
        row_splits_tensor->tensor<int32, 1>()(i) = vap->get_row_splits()[i];
    }

    // Atom masks tensor
    atom_mask_tensor = new Tensor(DT_FLOAT, TensorShape({vap->get_n_atoms_vap()}));
    auto atom_mask_ptr = vap->get_atom_mask();
    for (i = 0; i < vap->get_n_atoms_vap(); i++) {
        atom_mask_tensor->tensor<float, 1>()(i) = atom_mask_ptr[i];
    }

    // N_atom_vap tensor
    n_atoms_vap_tensor = new Tensor(DT_INT32, TensorShape());
    n_atoms_vap_tensor->flat<int32>()(0) = vap->get_n_atoms_vap();

    // Pulay stress tensor
    pulay_stress_tensor = new Tensor(DT_FLOAT, TensorShape());
    pulay_stress_tensor->flat<float>()(0) = 0.0f;

    // Composition tensor
    composition_tensor = new Tensor(DT_FLOAT, TensorShape({n}));
    composition_tensor->flat<float>().setConstant(0.0);
    for (i = 0; i < atom->nlocal; i++) {
        composition_tensor->flat<float>()(atom->type[i] - 1) += 1.0f;
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
    options.config.set_use_per_session_threads(true);
    options.config.set_log_device_placement(false);
    options.config.set_inter_op_parallelism_threads(0);
    options.config.set_intra_op_parallelism_threads(1);

    session.reset(tensorflow::NewSession(options));
    Status session_create_status = session->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

/* ----------------------------------------------------------------------
   Read the coefficients
------------------------------------------------------------------------- */

void PairTensorAlloy::coeff(int narg, char **arg)
{
    int i, j, k, n_elements;
    int *element_map;
    std::vector<string> symbols;
    symbols.emplace_back("X");

    // Read Lammps element order
    std::cout << "Lammps atom types: ";
    for (i = 1; i < narg; i++) {
        string element = string(arg[i]);
        std::cout << element << " ";
        symbols.emplace_back(element);
    }
    std::cout << std::endl;
    memory->create(element_map, symbols.size(), "pair:element_map");
    for (i = 0; i < symbols.size() + 1; i++) {
        element_map[i] = 0;
    }

    // Load the graph model
    string graph_model_path(arg[0]);
    Status load_graph_status = load_graph(graph_model_path);
    std::cout << "Read " << graph_model_path << ": " << load_graph_status << std::endl;

    // Decode the transformer
    std::vector<Tensor> outputs;
    Status status = session->Run({}, {"Transformer/params:0"}, {}, &outputs);
    std::cout << "Recover model metadata: " << status << std::endl;

    Json::Value jsonData;
    Json::Reader jsonReader;
    auto decoded = outputs[0].flat<string>();
    auto parse_status = jsonReader.parse(decoded.data()[0], jsonData, false);

    if (parse_status)
    {
        std::cout << "Successfully parsed tensor <Transformer/params:0>" << std::endl;

        cutmax = jsonData["rc"].asDouble();
        // cutmax = 3.0;
        use_angular = jsonData["angular"].asBool();
        n_eta = jsonData["eta"].size();
        n_omega = jsonData["omega"].size();
        n_beta = jsonData["beta"].size();
        n_gamma = jsonData["gamma"].size();
        n_zeta = jsonData["zeta"].size();

        Json::Value model_elements = jsonData["elements"];
        n_elements = model_elements.size();

        std::cout << "Graph model elements: ";
        for (i = 0; i < n_elements; i++) {
            std::cout << model_elements[i].asString() << " ";
        }
        std::cout << std::endl;

        for (i = 1; i < symbols.size(); i++) {
            if (symbols[i] != model_elements[i - 1].asString()) {
                error->all(FLERR, "Elements misorder.");
            }
        }

        std::cout << "rc: " << std::setprecision(3) << cutmax << std::endl;
        std::cout << "angular: " << use_angular << std::endl;
        std::cout << "n_eta: " << n_eta << std::endl;
        std::cout << "n_omega: " << n_omega << std::endl;
        std::cout << "n_beta: " << n_beta << std::endl;
        std::cout << "n_gamma: " << n_gamma << std::endl;
        std::cout << "n_zeta: " << n_zeta << std::endl;
        std::cout << "map: ";
        for (i = 1; i < symbols.size(); i++) {
            std::cout << i << "->" << element_map[i] << " ";
        }
        std::cout << std::endl;
    }
    else
    {
        n_elements = 0;
        error->all(FLERR, "Could not decode tensor <Transformer/params:0>");
    }

    int32 n_lmp_types = n_elements + 1;

    if (narg != 1 + n_elements)
        error->all(FLERR,"Incorrect args for pair coefficients");

    memory->create(max_occurs, n_lmp_types, "pair:max_occurs");
    for (i = 0; i < n_lmp_types; i++)
        max_occurs[i] = 0;
    for (i = 0; i < atom->natoms; i++)
        max_occurs[atom->type[i]] ++;
    for (i = 1; i < n_lmp_types; i++)
        if (max_occurs[i] == 0)
            max_occurs[i] = 1;
    for (i = 0; i < n_lmp_types; i++)
        std::cout << "MaxOccur of " << symbols[i] << ": " << max_occurs[i] << std::endl;

    // Initialize the Virtual-Atom Map
    vap = new VirtualAtomMap(memory, symbols.size(), max_occurs, atom->natoms, atom->type);
    vap->print();

    int32 g2_size = n_eta * n_omega;
    int32 g4_size = n_beta * n_gamma * n_zeta;
    int32 g2_ndim_per_atom = n_elements * g2_size;
    int32 g4_ndim_per_atom = 0;

    if (use_angular) {
        g4_ndim_per_atom = (n_elements + 1) * n_elements / 2 * g4_size;
    }

    int32 offset = 0;
    int32 pos = 0;

    memory->create(g2_offset_map, n_lmp_types * n_lmp_types, "pair:g2_offset_map");
    for (i = 1; i < n_lmp_types; i++) {
        pos = IJ(i, i, n_lmp_types);
        g2_offset_map[i * n_lmp_types + i] = offset;
        std::cout << symbols[i] << symbols[i] << ": " << g2_offset_map[pos] << std::endl;
        offset += g2_size;
        for (j = 1; j < n_lmp_types; j++) {
            if (j == i) {
                continue;
            }
            pos = IJ(i, j, n_lmp_types);
            g2_offset_map[pos] = offset;
            std::cout << symbols[i] << symbols[j] << ": " << g2_offset_map[pos] << std::endl;
            offset += g2_size;
        }
        offset += g4_ndim_per_atom;
    }

    if (use_angular) {
        size_t alloc_size = n_lmp_types * n_lmp_types * n_lmp_types;
        memory->create(g4_offset_map, alloc_size, "pair:g4_offset_map");
        offset = g2_ndim_per_atom;
        for (i = 1; i < n_lmp_types; i++) {
            for (j = 1; j < n_lmp_types; j++) {
                for (k = j; k < n_lmp_types; k++) {
                    pos = IJK(i, j, k, n_lmp_types);
                    g4_offset_map[pos] = offset;
                    std::cout << symbols[i] << symbols[j] << symbols[k] << ": " << g4_offset_map[pos] << std::endl;
                    offset += g4_size;
                }
            }
            offset += g2_ndim_per_atom;
        }

        memory->create(g4_numneigh, atom->nlocal, "pair:g4_numneigh");
        for (i = 0; i < atom->nlocal; i++)
            g4_numneigh[i] = 0;
    }

    // Allocate arrays and tensors.
    allocate();

    // Set atomic masses
    double atom_mass[n_lmp_types];
    for (i = 0; i < n_lmp_types; i++) {
        atom_mass[i] = 1.0;
    }
    atom->set_mass(atom_mass);

    // setflag
    for (i = 1; i < n_lmp_types; i++) {
        for (j = i; j < n_lmp_types; j++) {
            setflag[i][j] = 1;
        }
    }
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
