# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sustainable Power Systems Laboratory (https://sps-lab.org/)
# Part of DYNAGNN: Lazy public module exports

from importlib import import_module

__all__ = [
    "default_dynawo_execution_path",
    "run_dynawo_job",
    "build_dyd_single",
    "build_par_single",
    "default_event_time",
    "lib_from_fault_type",
    "write_event_files",
    "build_dyd_id_to_staticid_map",
    "build_static_id_to_dynamic_id_map",
    "build_graph_from_iidm",
    "build_pyg_graph_from_iidm",
    "build_graph_from_directory",
    "build_pyg_graph_from_directory",
    "write_electrical_distance_csv_from_iidm",
    "write_electrical_distance_csv_from_network",
    "run_kpi",
    "run_actions_detection",
    "run_disconnections_detection",
    "generate_curves",
    "generate_curves_for_operating_point",
    "process_kpi_operating_point",
    "process_actions_operating_point",
    "process_disconnections_operating_point",
    "build_generator_snom_tables",
    "build_generator_snom_for_operating_point",
    "build_generator_snom_matrix",
    "SNOM_DIR",
    "build_dataset_split",
    "load_split_settings",
    "SplitSettings",
    "SplitSummary",
    "DATA_DIR",
    "KPI_DIR",
    "ACTIONS_DIR",
    "DISCONNECTIONS_DIR",
    "DATASET_DIR",
    "OP_GRAPHS_DIR",
    "OP_ELECTRIC_DISTANCE_DIR",
]

_SYMBOL_TO_MODULE = {
    "default_dynawo_execution_path": "dynawo_runner",
    "run_dynawo_job": "dynawo_runner",
    "build_dyd_single": "event_files",
    "build_par_single": "event_files",
    "default_event_time": "event_files",
    "lib_from_fault_type": "event_files",
    "write_event_files": "event_files",
    "build_dyd_id_to_staticid_map": "dyd_mapping",
    "build_static_id_to_dynamic_id_map": "dyd_mapping",
    "build_graph_from_iidm": "graph_construction",
    "build_pyg_graph_from_iidm": "graph_construction",
    "build_graph_from_directory": "graph_construction",
    "build_pyg_graph_from_directory": "graph_construction",
    "write_electrical_distance_csv_from_iidm": "electric_distance",
    "write_electrical_distance_csv_from_network": "electric_distance",
    "run_kpi": "kpi",
    "run_actions_detection": "actions_detection",
    "run_disconnections_detection": "disconnections_detection",
    "generate_curves": "curve_generation",
    "generate_curves_for_operating_point": "curve_generation",
    "process_kpi_operating_point": "kpi",
    "process_actions_operating_point": "actions_detection",
    "process_disconnections_operating_point": "disconnections_detection",
    "build_generator_snom_tables": "generator_snom",
    "build_generator_snom_for_operating_point": "generator_snom",
    "build_generator_snom_matrix": "generator_snom",
    "SNOM_DIR": "paths",
    "build_dataset_split": "dataset_split",
    "load_split_settings": "dataset_split",
    "SplitSettings": "dataset_split",
    "SplitSummary": "dataset_split",
    "DATA_DIR": "paths",
    "KPI_DIR": "paths",
    "ACTIONS_DIR": "paths",
    "DISCONNECTIONS_DIR": "paths",
    "DATASET_DIR": "paths",
    "OP_GRAPHS_DIR": "paths",
    "OP_ELECTRIC_DISTANCE_DIR": "paths",
}


def __getattr__(name):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
