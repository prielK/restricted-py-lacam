import csv
import os
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout
import re
from loguru import logger

from pycam import LaCAM, get_grid, get_scenario, save_configs_for_visualizer, validate_mapf_solution


def extract_log_times(log_output):
    preprocessing_time = None
    initial_solution_time = None
    optimal_solution_time = None
    initial_solution_cost = None
    optimal_solution_cost = None

    # Regex patterns to capture log entries
    preprocessing_pattern = re.compile(r"(\d+)ms\s+start solving MAPF")
    initial_solution_pattern = re.compile(r"(\d+)ms\s+initial solution found, cost=(\d+)")
    optimal_solution_pattern = re.compile(
        r"(\d+)ms\s+(suboptimal solution|reach optimal solution), cost=(\d+)"
    )

    print("Extracting log times from output...")
    print(log_output)  # Debugging: Print the log output

    for line in log_output.splitlines():
        print(f"Parsing log line: {line}")  # Debugging: Show each line being parsed
        if match := preprocessing_pattern.search(line):
            preprocessing_time = int(match.group(1))
        elif match := initial_solution_pattern.search(line):
            initial_solution_time = int(match.group(1))
            initial_solution_cost = int(match.group(2))
        elif match := optimal_solution_pattern.search(line):
            optimal_solution_time = int(match.group(1))
            optimal_solution_cost = int(match.group(3))

    print(
        f"Captured Times: Preprocessing: {preprocessing_time}, Initial: {initial_solution_time}, Optimal: {optimal_solution_time}"
    )
    return (
        preprocessing_time,
        initial_solution_time,
        optimal_solution_time,
        initial_solution_cost,
        optimal_solution_cost,
    )


def run_experiment(
    map_file,
    scen_file,
    num_agents,
    restrict_directions,
    output_file,
    seed=0,
    time_limit_ms=100000,
    verbose=1,
    flg_star=True,
):
    grid = get_grid(map_file)
    starts, goals = get_scenario(scen_file, num_agents)
    planner = LaCAM()

    print(
        f"Running with {len(starts)} agents on {os.path.basename(map_file)} with restrict_directions={restrict_directions}"
    )

    # Capture output during the solver run
    log_buffer = StringIO()
    logger.remove()  # Remove default logger
    logger.add(log_buffer, format="{message}")  # Add logger to StringIO buffer

    with StringIO() as buf, redirect_stdout(buf):
        solution = planner.solve(
            grid=grid,
            starts=starts,
            goals=goals,
            seed=seed,
            time_limit_ms=time_limit_ms,
            flg_star=flg_star,
            verbose=verbose,
            restrict_directions=restrict_directions,
        )
        restricted_directions = planner.restricted_directions
        validate_mapf_solution(grid, starts, goals, solution, restricted_directions)
        save_configs_for_visualizer(solution, output_file)

    # Get the log output from loguru
    log_output = log_buffer.getvalue()

    # Extract log times from the captured output
    (
        preprocessing_time,
        initial_solution_time,
        optimal_solution_time,
        initial_solution_cost,
        optimal_solution_cost,
    ) = extract_log_times(log_output)

    return {
        "map_file": map_file,
        "scen_file": scen_file,
        "num_agents": num_agents,
        "restrict_directions": restrict_directions,
        "preprocessing_time": preprocessing_time,
        "initial_solution_time": initial_solution_time,
        "optimal_solution_time": optimal_solution_time,
        "initial_solution_cost": initial_solution_cost,
        "optimal_solution_cost": optimal_solution_cost,
    }


def run_all_experiments():
    map_files = [
        Path(__file__).parent / "assets" / "random-32-32-10.map",
        Path(__file__).parent / "assets" / "random-32-32-10-nowalls.map",
    ]
    scen_files = [
        Path(__file__).parent / "assets" / "random-32-32-10-random-1.scen",
        Path(__file__).parent / "assets" / "random-32-32-10-nowalls.scen",
    ]
    agent_counts = [5, 10, 15]
    results = []

    for map_file, scen_file in zip(map_files, scen_files):
        for num_agents in agent_counts:
            for restrict_directions in [True, False]:
                output_file = f"results_{os.path.basename(map_file)}_{num_agents}_agents_{restrict_directions}.txt"
                result = run_experiment(
                    map_file=map_file,
                    scen_file=scen_file,
                    num_agents=num_agents,
                    restrict_directions=restrict_directions,
                    output_file=output_file,
                )
                results.append(result)

    with open("experiment_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "map_file",
            "scen_file",
            "num_agents",
            "restrict_directions",
            "preprocessing_time",
            "initial_solution_time",
            "optimal_solution_time",
            "initial_solution_cost",
            "optimal_solution_cost",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    run_all_experiments()
