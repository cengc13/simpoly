# SPDX-License-Identifier: MIT

import os.path

import click

from simpoly.poly_arena import experiment
from simpoly.poly_arena.simulation import protocol
from simpoly.poly_arena.simulation import tools as sim_tools


def prepare_simulation(
    directory: str,
    model_type: str,
    poly_id: str,
    n_atoms: int,
    temp_k: float,
    pressure_atm: float,
    mlff_path: str | None = None,
    time_prefactor: float = 1.0,
    seed: int = 42,
    protocol_name: str = "21step",
    cool_temp_end_k: float = 100.0,
    cool_temp_step_k: float = 20.0,
    cool_time_ps: float = 150.0,
    restart_path: str | None = None,
    mlff_atom_types: list[str] | None = None,
) -> None:
    working_dir = os.path.abspath(directory)
    os.makedirs(working_dir, exist_ok=True)
    print(f"Preparing {protocol_name} simulation in {working_dir} for polymer {poly_id}")

    needs_build = protocol_name in ("21step", "21step_then_cooling")
    if protocol_name == "cooling" and restart_path is None:
        raise ValueError(
            "--restart-path is required for --protocol cooling; "
            "use --protocol 21step_then_cooling to build from scratch."
        )

    if needs_build:
        df = experiment.load_data()
        poly_data = df.loc[poly_id]

        protocol.create_lammps_input(
            directory=working_dir,
            smiles=str(poly_data["smiles"]),
            end_groups=(str(poly_data["end_group_0"]), str(poly_data["end_group_1"])),
            density=0.5,  # initial density
            temperature=temp_k,
            n_tot=n_atoms,
            n_ru_per_chain=10,
            seed=seed,
        )
        lammps_data_path = os.path.join(working_dir, "system.data")

    if model_type == "pcff":
        config = protocol.get_pcff_config()
        header_fn = protocol.pcff_header_fn
        if needs_build:
            config["data_file"] = lammps_data_path
        else:
            config["data_file"] = os.path.abspath(restart_path)  # type: ignore[arg-type]

    elif model_type == "mlff":
        if mlff_path is None:
            raise ValueError("mlff_path is required when model_type is 'mlff'")

        if needs_build:
            metal_data_path = os.path.join(working_dir, "data.lmps")
            atom_types = sim_tools.rewrite_full_to_metal_data(lammps_data_path, metal_data_path)
            data_file = metal_data_path
        else:
            if not mlff_atom_types:
                raise ValueError(
                    "--mlff-atom-types is required for --protocol cooling with --model-type mlff "
                    "(e.g. 'C C C H' matching the checkpoint's pair_coeff order)."
                )
            atom_types = list(mlff_atom_types)
            data_file = os.path.abspath(restart_path)  # type: ignore[arg-type]

        config = protocol.get_mlff_config(
            mlff_path=mlff_path,
            data_path=data_file,
            atom_types=atom_types,
        )
        config["data_file"] = data_file
        header_fn = protocol.mlff_header_fn
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    config["start_command"] = "read_data" if needs_build else "read_restart"
    config["n_thermo_freq"] = 250
    config["poly_id"] = poly_id
    config["temp_target"] = temp_k
    config["n_atoms"] = n_atoms
    config["seed"] = seed

    sim_protocol: protocol.LAMMPSProtocol
    if protocol_name == "21step":
        sim_protocol = protocol.build_21steps_protocol(
            temp_final_k=temp_k,
            p_final_atm=pressure_atm,
            seed=seed,
            pressure_couple="aniso",
            units=config["units"],
            time_step=config["time_step"],
            time_prefactor=time_prefactor,
        )
        lammps_filename = "21steps.in"
    elif protocol_name == "cooling":
        sim_protocol = protocol.build_tg_cooling_protocol(
            temp_start_k=temp_k,
            temp_end_k=cool_temp_end_k,
            temp_step_k=cool_temp_step_k,
            npt_time_ps=cool_time_ps,
            p_final_atm=pressure_atm,
            pressure_couple="aniso",
            units=config["units"],
            time_step=config["time_step"],
            time_prefactor=time_prefactor,
        )
        lammps_filename = "tg_cooling.in"
    elif protocol_name == "21step_then_cooling":
        sim_protocol = protocol.build_21step_then_cooling_protocol(
            temp_start_k=temp_k,
            temp_end_k=cool_temp_end_k,
            temp_step_k=cool_temp_step_k,
            npt_time_ps=cool_time_ps,
            p_final_atm=pressure_atm,
            seed=seed,
            pressure_couple="aniso",
            units=config["units"],
            time_step=config["time_step"],
            time_prefactor=time_prefactor,
        )
        lammps_filename = "21step_then_cooling.in"
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    blocks = sim_protocol.render()
    header = header_fn(config)
    lammps_input = "\n\n".join([header, blocks])

    # Write input file to disk
    lammps_path = os.path.join(working_dir, lammps_filename)
    with open(lammps_path, "w") as f:
        f.write(lammps_input)

    print("Done")


@click.command()
@click.option(
    "--directory",
    default=None,
    type=click.Path(),
    help="Working directory for simulation files (default: {poly_id}_n{n_atoms}_T{temp_k}_s{seed})",
)
@click.option(
    "--model-type",
    default="pcff",
    type=click.Choice(["pcff", "mlff"], case_sensitive=False),
    help="Force field model type (default: pcff)",
)
@click.option(
    "--poly-id",
    required=True,
    type=str,
    help="Polymer identifier",
)
@click.option(
    "--n-atoms",
    default=100,
    type=int,
    help="Number of atoms in the system (default: 100)",
)
@click.option(
    "--temp-k",
    default=300.0,
    type=float,
    help="Temperature in Kelvin (default: 300.0)",
)
@click.option(
    "--pressure-atm",
    default=1.0,
    type=float,
    help="Pressure in atmospheres (default: 1.0)",
)
@click.option(
    "--mlff-path",
    default=None,
    type=click.Path(exists=True),
    help="Path to MLFF model (required when model-type is mlff)",
)
@click.option(
    "--time-prefactor",
    default=1.0,
    type=float,
    help="Time prefactor for simulation speed (default: 1.0, use 0.1 for 10x faster)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducibility (default: 42)",
)
@click.option(
    "--protocol",
    "protocol_name",
    default="21step",
    type=click.Choice(["21step", "cooling", "21step_then_cooling"], case_sensitive=False),
    help=(
        "Which LAMMPS protocol to emit: '21step' (Polymatic-style equilibration), "
        "'cooling' (Tg cooling scan from --temp-k to --cool-temp-end-k; assumes a "
        "pre-equilibrated restart), or '21step_then_cooling' (equilibrate then cool "
        "in one input). Default: 21step."
    ),
)
@click.option(
    "--cool-temp-end-k",
    default=100.0,
    type=float,
    help="Final temperature (K) for the cooling scan (default: 100.0).",
)
@click.option(
    "--cool-temp-step-k",
    default=20.0,
    type=float,
    help="Temperature decrement (K) between cooling stages (default: 20.0).",
)
@click.option(
    "--cool-time-ps",
    default=150.0,
    type=float,
    help="NPT duration (ps) at each cooling stage (default: 150.0; paper protocol).",
)
@click.option(
    "--restart-path",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Path to a LAMMPS restart file. Required (and only used) for "
        "--protocol cooling; the cooling block then continues from this "
        "equilibrated state instead of rebuilding the system."
    ),
)
@click.option(
    "--mlff-atom-types",
    default=None,
    type=str,
    help=(
        "Space-separated atom types for MLFF pair_coeff (e.g. 'C C C H'). "
        "Required for --protocol cooling with --model-type mlff because EMC "
        "is skipped and type information cannot be auto-detected."
    ),
)
def main(
    directory: str | None,
    model_type: str,
    poly_id: str,
    n_atoms: int,
    temp_k: float,
    pressure_atm: float,
    mlff_path: str | None,
    time_prefactor: float,
    seed: int,
    protocol_name: str,
    cool_temp_end_k: float,
    cool_temp_step_k: float,
    cool_time_ps: float,
    restart_path: str | None,
    mlff_atom_types: str | None,
) -> None:
    """Prepare a LAMMPS simulation (21-step, cooling, or combined) for polymer systems."""
    # Generate default directory name if not provided
    if directory is None:
        directory = f"{poly_id}_n{n_atoms}_T{temp_k}_s{seed}"

    atom_types_list = mlff_atom_types.split() if mlff_atom_types else None

    prepare_simulation(
        directory=directory,
        model_type=model_type,
        poly_id=poly_id,
        n_atoms=n_atoms,
        temp_k=temp_k,
        pressure_atm=pressure_atm,
        mlff_path=mlff_path,
        time_prefactor=time_prefactor,
        seed=seed,
        protocol_name=protocol_name.lower(),
        cool_temp_end_k=cool_temp_end_k,
        cool_temp_step_k=cool_temp_step_k,
        cool_time_ps=cool_time_ps,
        restart_path=restart_path,
        mlff_atom_types=atom_types_list,
    )


if __name__ == "__main__":
    main()
