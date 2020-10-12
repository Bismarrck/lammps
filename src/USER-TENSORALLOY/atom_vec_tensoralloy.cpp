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

#include "atom_vec_tensoralloy.h"
#include "atom.h"
#include "error.h"
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecTensorAlloy::AtomVecTensorAlloy(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = 0;
  mass_type = 1;
  atom->tensoralloy_flag = 1;
  atom->eentropy_flag = 1;
  eentropy = nullptr;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "eentropy";
  fields_copy = (char *) "eentropy";
  fields_comm = (char *) "eentropy";
  fields_comm_vel = (char *) "eentropy";
  fields_border = (char *) "eentropy";
  fields_border_vel = (char *) "eentropy";
  fields_exchange = (char *) "eentropy";
  fields_restart = (char * ) "eentropy";
  fields_create = (char *) "eentropy";
  fields_data_atom = (char *) "id type x";
  fields_data_vel = (char *) "id v";

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecTensorAlloy::grow_pointers()
{
  eentropy = atom->eentropy;
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecTensorAlloy::property_atom(char *name)
{
  if (strcmp(name,"eentropy") == 0) return 0;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecTensorAlloy::pack_property_atom(int index, double *buf,
                                            int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = eentropy[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}
