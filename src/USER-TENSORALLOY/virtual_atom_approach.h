//
// Created by Xin Chen on 2019-07-02.
//

#ifndef LIBTENSORALLOY_VAP_H
#define LIBTENSORALLOY_VAP_H

#include "pointers.h"
#include <iostream>
#include <tensorflow/core/platform/default/integral_types.h>
#include <vector>

namespace LAMMPS_NS {

using tensorflow::int32;

class VirtualAtomMap : Pointers {
public:
  VirtualAtomMap(class LAMMPS *lmp, int);
  ~VirtualAtomMap();

  void build(int, const int *);

  const int32 *get_row_splits() const { return splits; }
  const int32 *get_local_to_vap_map() const { return local_to_vap_map; }
  const int32 *get_vap_to_local_map() const { return vap_to_local_map; }
  const float *get_atom_masks() const { return atom_masks; }
  int get_n_atoms_vap() const { return n_atoms_vap; }
  bool updated() const { return is_domain_changed; }

  double memory_usage() const { return total_bytes; };

protected:
  // Internal variables
  int32 n_atoms_vap;
  int32 *element_map;
  int32 *offsets;
  int32 *local_to_vap_map;
  int32 *vap_to_local_map;
  int32 *splits;
  float *atom_masks;
  int n_elements;
  int *max_occurs;

private:
  int curr_nlocal;
  int *curr_itypes;
  bool is_domain_changed;
  bool build_max_occurs(int nlocal, const int *atom_types);

  double total_bytes;
};
} // namespace LAMMPS_NS

#endif // LMP_TENSORALLOY_VAP_H
