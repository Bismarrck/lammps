/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Axel Kohlmeyer (Temple U),
                         Ryan S. Elliott (UMN),
                         Ellad B. Tadmor (UMN),
                         Yaser Afshar (UMN)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2 of the License, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
   more details.

   You should have received a copy of the GNU General Public License along with
   this program; if not, see <https://www.gnu.org/licenses>.

   Linking LAMMPS statically or dynamically with other modules is making a
   combined work based on LAMMPS. Thus, the terms and conditions of the GNU
   General Public License cover the whole combination.

   In addition, as a special exception, the copyright holders of LAMMPS give
   you permission to combine LAMMPS with free software programs or libraries
   that are released under the GNU LGPL and with code included in the standard
   release of the "kim-api" under the CDDL (or modified versions of such code,
   with unchanged license). You may copy and distribute such a system following
   the terms of the GNU GPL for LAMMPS and the licenses of the other code
   concerned, provided that you include the source code of that other code
   when and as the GNU GPL requires distribution of source code.

   Note that people who make modified versions of LAMMPS are not obligated to
   grant this special exception for their modified versions; it is their choice
   whether to do so. The GNU General Public License gives permission to release
   a modified version without this exception; this exception also makes it
   possible to release a modified version which carries forward this exception.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Designed for use with the kim-api-2.1.0 (and newer) package
------------------------------------------------------------------------- */

#include "kim_init.h"

#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_store_kim.h"
#include "input.h"
#include "kim_units.h"
#include "modify.h"
#include "universe.h"
#include "variable.h"

#include <cstring>

extern "C" {
#include "KIM_SimulatorHeaders.h"
}

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

void KimInit::command(int narg, char **arg)
{
  if ((narg < 2) || (narg > 3)) error->all(FLERR, "Illegal 'kim init' command");

  if (domain->box_exist)
    error->all(FLERR, "Must use 'kim init' command before "
                      "simulation box is defined");
  char *model_name = utils::strdup(arg[0]);
  char *user_units = utils::strdup(arg[1]);
  if (narg == 3) {
    if (strcmp(arg[2], "unit_conversion_mode")==0) unit_conversion_mode = true;
    else {
      error->all(FLERR, fmt::format("Illegal 'kim init' command.\nThe argument "
                                    "followed by unit_style {} is an optional "
                                    "argument and when is used must "
                                    "be unit_conversion_mode", user_units));
    }
  } else unit_conversion_mode = false;

  char *model_units;
  KIM_Model *pkim = nullptr;

  if (universe->me == 0) std::remove("kim.log");
  if (universe->nprocs > 1) MPI_Barrier(universe->uworld);

  determine_model_type_and_units(model_name, user_units, &model_units, pkim);

  write_log_cite(model_name);

  do_init(model_name, user_units, model_units, pkim);
}

/* ---------------------------------------------------------------------- */

namespace {
void get_kim_unit_names(
    char const * const system,
    KIM_LengthUnit & lengthUnit,
    KIM_EnergyUnit & energyUnit,
    KIM_ChargeUnit & chargeUnit,
    KIM_TemperatureUnit & temperatureUnit,
    KIM_TimeUnit & timeUnit,
    Error * error)
{
  if (strcmp(system, "real") == 0) {
    lengthUnit = KIM_LENGTH_UNIT_A;
    energyUnit = KIM_ENERGY_UNIT_kcal_mol;
    chargeUnit = KIM_CHARGE_UNIT_e;
    temperatureUnit = KIM_TEMPERATURE_UNIT_K;
    timeUnit = KIM_TIME_UNIT_fs;
  } else if (strcmp(system, "metal") == 0) {
    lengthUnit = KIM_LENGTH_UNIT_A;
    energyUnit = KIM_ENERGY_UNIT_eV;
    chargeUnit = KIM_CHARGE_UNIT_e;
    temperatureUnit = KIM_TEMPERATURE_UNIT_K;
    timeUnit = KIM_TIME_UNIT_ps;
  } else if (strcmp(system, "si") == 0) {
    lengthUnit = KIM_LENGTH_UNIT_m;
    energyUnit = KIM_ENERGY_UNIT_J;
    chargeUnit = KIM_CHARGE_UNIT_C;
    temperatureUnit = KIM_TEMPERATURE_UNIT_K;
    timeUnit = KIM_TIME_UNIT_s;
  } else if (strcmp(system, "cgs") == 0) {
    lengthUnit = KIM_LENGTH_UNIT_cm;
    energyUnit = KIM_ENERGY_UNIT_erg;
    chargeUnit = KIM_CHARGE_UNIT_statC;
    temperatureUnit = KIM_TEMPERATURE_UNIT_K;
    timeUnit = KIM_TIME_UNIT_s;
  } else if (strcmp(system, "electron") == 0) {
    lengthUnit = KIM_LENGTH_UNIT_Bohr;
    energyUnit = KIM_ENERGY_UNIT_Hartree;
    chargeUnit = KIM_CHARGE_UNIT_e;
    temperatureUnit = KIM_TEMPERATURE_UNIT_K;
    timeUnit = KIM_TIME_UNIT_fs;
  } else if (strcmp(system, "lj") == 0 ||
             strcmp(system, "micro") ==0 ||
             strcmp(system, "nano")==0) {
    error->all(FLERR, fmt::format("LAMMPS unit_style {} not supported "
                                  "by KIM models", system));
  } else {
    error->all(FLERR, "Unknown unit_style");
  }
}
}  // namespace

void KimInit::determine_model_type_and_units(char * model_name,
                                             char * user_units,
                                             char ** model_units,
                                             KIM_Model *&pkim)
{
  KIM_LengthUnit lengthUnit;
  KIM_EnergyUnit energyUnit;
  KIM_ChargeUnit chargeUnit;
  KIM_TemperatureUnit temperatureUnit;
  KIM_TimeUnit timeUnit;
  int units_accepted;
  KIM_Collections * collections;
  KIM_CollectionItemType itemType;

  int kim_error = KIM_Collections_Create(&collections);
  if (kim_error)
    error->all(FLERR, "Unable to access KIM Collections to find Model");

  auto logID = fmt::format("{}_Collections", comm->me);
  KIM_Collections_SetLogID(collections, logID.c_str());

  kim_error = KIM_Collections_GetItemType(collections, model_name, &itemType);
  if (kim_error) error->all(FLERR, "KIM Model name not found");
  KIM_Collections_Destroy(&collections);

  if (KIM_CollectionItemType_Equal(itemType,
                                   KIM_COLLECTION_ITEM_TYPE_portableModel)) {
    get_kim_unit_names(user_units, lengthUnit, energyUnit,
                       chargeUnit, temperatureUnit, timeUnit, error);
    int kim_error = KIM_Model_Create(KIM_NUMBERING_zeroBased,
                                     lengthUnit,
                                     energyUnit,
                                     chargeUnit,
                                     temperatureUnit,
                                     timeUnit,
                                     model_name,
                                     &units_accepted,
                                     &pkim);

    if (kim_error) error->all(FLERR, "Unable to load KIM Simulator Model");

    model_type = MO;

    if (units_accepted) {
      logID = fmt::format("{}_Model", comm->me);
      KIM_Model_SetLogID(pkim, logID.c_str());
      *model_units = utils::strdup(user_units);
      return;
    } else if (unit_conversion_mode) {
      KIM_Model_Destroy(&pkim);
      int const num_systems = 5;
      char const * const systems[num_systems]
          = {"metal", "real", "si", "cgs", "electron"};
      for (int i=0; i < num_systems; ++i) {
        get_kim_unit_names(systems[i], lengthUnit, energyUnit,
                           chargeUnit, temperatureUnit, timeUnit, error);
        kim_error = KIM_Model_Create(KIM_NUMBERING_zeroBased,
                                     lengthUnit,
                                     energyUnit,
                                     chargeUnit,
                                     temperatureUnit,
                                     timeUnit,
                                     model_name,
                                     &units_accepted,
                                     &pkim);
        if (units_accepted) {
          logID = fmt::format("{}_Model", comm->me);
          KIM_Model_SetLogID(pkim, logID.c_str());
          *model_units = utils::strdup(systems[i]);
          return;
        }
        KIM_Model_Destroy(&pkim);
      }
      error->all(FLERR, "KIM Model does not support any lammps unit system");
    } else {
      KIM_Model_Destroy(&pkim);
      error->all(FLERR, "KIM Model does not support the requested unit system");
    }
  } else if (KIM_CollectionItemType_Equal(
             itemType, KIM_COLLECTION_ITEM_TYPE_simulatorModel)) {
    KIM_SimulatorModel * simulatorModel;
    kim_error = KIM_SimulatorModel_Create(model_name, &simulatorModel);
    if (kim_error)
      error->all(FLERR, "Unable to load KIM Simulator Model");
    model_type = SM;

    logID = fmt::format("{}_SimulatorModel", comm->me);
    KIM_SimulatorModel_SetLogID(simulatorModel, logID.c_str());

    int sim_fields;
    int sim_lines;
    char const * sim_field;
    char const * sim_value;
    KIM_SimulatorModel_GetNumberOfSimulatorFields(simulatorModel, &sim_fields);
    KIM_SimulatorModel_CloseTemplateMap(simulatorModel);
    for (int i=0; i < sim_fields; ++i) {
      KIM_SimulatorModel_GetSimulatorFieldMetadata(
          simulatorModel, i, &sim_lines, &sim_field);

      if (0 == strcmp(sim_field, "units")) {
        KIM_SimulatorModel_GetSimulatorFieldLine(
          simulatorModel, i, 0, &sim_value);
        *model_units = utils::strdup(sim_value);
        break;
      }
    }
    KIM_SimulatorModel_Destroy(&simulatorModel);

    if ((! unit_conversion_mode) && (strcmp(*model_units, user_units)!=0)) {
      error->all(FLERR, fmt::format("Incompatible units for KIM Simulator Model"
                                    ", required units = {}", *model_units));
    }
  }
}

/* ---------------------------------------------------------------------- */

void KimInit::do_init(char *model_name, char *user_units, char *model_units,
                      KIM_Model *&pkim)
{
  // create storage proxy fix. delete existing fix, if needed.

  int ifix = modify->find_fix("KIM_MODEL_STORE");
  if (ifix >= 0) modify->delete_fix(ifix);
  modify->add_fix("KIM_MODEL_STORE all STORE/KIM");
  ifix = modify->find_fix("KIM_MODEL_STORE");

  FixStoreKIM *fix_store = (FixStoreKIM *) modify->fix[ifix];
  fix_store->setptr("model_name", (void *) model_name);
  fix_store->setptr("user_units", (void *) user_units);
  fix_store->setptr("model_units", (void *) model_units);

  // Begin output to log file
  input->write_echo("#=== BEGIN kim init ==================================="
                    "=======\n");

  KIM_SimulatorModel * simulatorModel;
  if (model_type == SM) {
    int kim_error =
      KIM_SimulatorModel_Create(model_name, &simulatorModel);
    if (kim_error)
      error->all(FLERR, "Unable to load KIM Simulator Model");

    auto logID = fmt::format("{}_SimulatorModel", comm->me);
    KIM_SimulatorModel_SetLogID(simulatorModel, logID.c_str());

    char const *sim_name, *sim_version;
    KIM_SimulatorModel_GetSimulatorNameAndVersion(
        simulatorModel, &sim_name, &sim_version);

    if (0 != strcmp(sim_name, "LAMMPS"))
      error->all(FLERR, "Incompatible KIM Simulator Model");

    if (comm->me == 0) {
      std::string mesg("# Using KIM Simulator Model : ");
      mesg += model_name;
      mesg += "\n";
      mesg += "# For Simulator             : ";
      mesg += std::string(sim_name) + " " + sim_version + "\n";
      mesg += "# Running on                : LAMMPS ";
      mesg += lmp->version;
      mesg += "\n";
      mesg += "#\n";

      utils::logmesg(lmp, mesg);
    }

    fix_store->setptr("simulator_model", (void *) simulatorModel);

    // need to call this to have access to (some) simulator model init data.

    KIM_SimulatorModel_CloseTemplateMap(simulatorModel);
  }

  // Define unit conversion factor variables and print to log
  if (unit_conversion_mode) do_variables(user_units, model_units);

  // set units

  std::string cmd("units ");
  cmd += model_units;
  input->one(cmd);

  // Set the skin and timestep default values as
  // 2.0 Angstroms and 1.0 femtosecond

  std::string skin_cmd =
    (strcmp(model_units, "real") == 0) ? "neighbor 2.0 bin   # Angstroms":
    (strcmp(model_units, "metal") == 0) ? "neighbor 2.0 bin   # Angstroms":
    (strcmp(model_units, "si") == 0) ? "neighbor 2e-10 bin   # meters":
    (strcmp(model_units, "cgs") == 0) ? "neighbor 2e-8 bin   # centimeters":
    "neighbor 3.77945224 bin   # Bohr";
  std::string step_cmd =
    (strcmp(model_units, "real") == 0) ? "timestep 1.0       # femtoseconds":
    (strcmp(model_units, "metal") == 0) ? "timestep 1.0e-3    # picoseconds":
    (strcmp(model_units, "si") == 0) ? "timestep 1e-15       # seconds":
    (strcmp(model_units, "cgs") == 0) ? "timestep 1e-15      # seconds":
    "timestep 1.0              # femtoseconds";
  input->one(skin_cmd);
  input->one(step_cmd);

  if (model_type == SM) {
    int sim_fields, sim_lines;
    char const *sim_field, *sim_value;
    KIM_SimulatorModel_GetNumberOfSimulatorFields(simulatorModel, &sim_fields);

    // init model

    for (int i=0; i < sim_fields; ++i) {
      KIM_SimulatorModel_GetSimulatorFieldMetadata(
          simulatorModel, i, &sim_lines, &sim_field);
      if (0 == strcmp(sim_field, "model-init")) {
        for (int j=0; j < sim_lines; ++j) {
          KIM_SimulatorModel_GetSimulatorFieldLine(
              simulatorModel, i, j, &sim_value);
          input->one(sim_value);
        }
        break;
      }
    }

    // reset template map.
    KIM_SimulatorModel_OpenAndInitializeTemplateMap(simulatorModel);
  } else if (model_type == MO) {
    int numberOfParameters;
    KIM_Model_GetNumberOfParameters(pkim, &numberOfParameters);

    std::string mesg = "\nThis model has ";
    if (numberOfParameters) {
      KIM_DataType kim_DataType;
      int extent;
      char const *str_name = nullptr;
      char const *str_desc = nullptr;

      mesg += std::to_string(numberOfParameters) + " mutable parameters. \n";

      int max_len(0);
      for (int i = 0; i < numberOfParameters; ++i) {
        KIM_Model_GetParameterMetadata(pkim, i, &kim_DataType,
        &extent, &str_name, &str_desc);
        max_len = MAX(max_len, (int)strlen(str_name));
      }
      max_len = MAX(18, max_len + 1);
      mesg += fmt::format(" No.      | {:<{}} | data type  | extent\n",
                          "Parameter name", max_len);
      mesg += fmt::format("{:-<{}}\n", "-", max_len + 35);
      for (int i = 0; i < numberOfParameters; ++i) {
        KIM_Model_GetParameterMetadata(pkim, i, &kim_DataType,
        &extent, &str_name, &str_desc);
        auto data_type = std::string("\"");
        data_type += KIM_DataType_ToString(kim_DataType) + std::string("\"");
        mesg += fmt::format(" {:<8} | {:<{}} | {:<10} | {}\n", i + 1, str_name,
                            max_len, data_type, extent);
      }
    } else mesg += "No mutable parameters.\n";

    KIM_Model_Destroy(&pkim);
    input->write_echo(mesg);
  }

  // End output to log file
  input->write_echo("#=== END kim init ====================================="
                    "=======\n\n");
}

/* ---------------------------------------------------------------------- */

void KimInit::do_variables(const std::string &from, const std::string &to)
{
  // refuse conversion from or to reduced units

  if ((from == "lj") || (to == "lj"))
    error->all(FLERR, "Cannot set up conversion variables for 'lj' units");

  // get index to internal style variables. create, if needed.
  // set conversion factors for newly created variables.
  double conversion_factor;
  int ier;
  std::string var_str;
  int v_unit;
  const char *units[] = {"mass",
                         "distance",
                         "time",
                         "energy",
                         "velocity",
                         "force",
                         "torque",
                         "temperature",
                         "pressure",
                         "viscosity",
                         "charge",
                         "dipole",
                         "efield",
                         "density",
                         nullptr};

  input->write_echo(fmt::format("# Conversion factors from {} to {}:\n",
                                from, to));

  auto variable = input->variable;
  for (int i = 0; units[i] != nullptr; ++i) {
    var_str = std::string("_u_") + units[i];
    v_unit = variable->find(var_str.c_str());
    if (v_unit < 0) {
      variable->set(var_str + " internal 1.0");
      v_unit = variable->find(var_str.c_str());
    }
    ier = lammps_unit_conversion(units[i], from, to,
                                 conversion_factor);
    if (ier != 0)
      error->all(FLERR, fmt::format("Unable to obtain conversion factor: "
                                    "unit = {}; from = {}; to = {}",
                                    units[i], from, to));

    variable->internal_set(v_unit, conversion_factor);
    input->write_echo(fmt::format("variable {:<15s} internal {:<15.12e}\n",
                                  var_str, conversion_factor));
  }
  input->write_echo("#\n");
}

/* ---------------------------------------------------------------------- */

void KimInit::write_log_cite(char *model_name)
{
  KIM_Collections * collections;
  int err = KIM_Collections_Create(&collections);
  if (err) return;

  auto logID = fmt::format("{}_Collections", comm->me);
  KIM_Collections_SetLogID(collections, logID.c_str());

  int extent;
  if (model_type == MO) {
    err = KIM_Collections_CacheListOfItemMetadataFiles(
      collections, KIM_COLLECTION_ITEM_TYPE_portableModel,
      model_name, &extent);
  } else if (model_type == SM) {
    err = KIM_Collections_CacheListOfItemMetadataFiles(
      collections, KIM_COLLECTION_ITEM_TYPE_simulatorModel,
      model_name, &extent);
  } else {
    error->all(FLERR, "Unknown model type");
  }

  if (err) {
    KIM_Collections_Destroy(&collections);
    return;
  }

  for (int i = 0; i < extent; ++i) {
    char const * fileName;
    int availableAsString;
    char const * fileString;
    err = KIM_Collections_GetItemMetadataFile(
        collections, i, &fileName, nullptr, nullptr,
        &availableAsString, &fileString);
    if (err) continue;

    if (0 == strncmp("kimcite", fileName, 7)) {
      if ((lmp->citeme) && (availableAsString)) lmp->citeme->add(fileString);
    }
  }

  KIM_Collections_Destroy(&collections);
}
