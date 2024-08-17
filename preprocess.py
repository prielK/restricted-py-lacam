import numpy as np
import re
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq


def heuristic(a, b):
    """
    Computes the Manhattan distance heuristic between two points.

    Parameters:
    a (tuple): The coordinates of the first point (y, x).
    b (tuple): The coordinates of the second point (y, x).

    Returns:
    int: The Manhattan distance between the two points.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, grid):
    """
    Implements the A* pathfinding algorithm to find the shortest path between two points on a grid.

    Parameters:
    start (tuple): The starting coordinates (y, x).
    goal (tuple): The goal coordinates (y, x).
    grid (np.ndarray): The grid to search on (True for open space, False for wall).

    Returns:
    int: The length of the shortest path found, or a large number if no path is found.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return float("inf")  # Return a very high cost if no path found

    # Reconstruct path to get its length
    path_length = 0
    current = goal
    while current != start:
        current = came_from[current]
        path_length += 1

    return path_length


def count_neighbors_in_layer(tile, layer, grid):
    """
    Counts the number of neighboring tiles in the same layer.

    Parameters:
    tile (tuple): The coordinates of the tile (y, x).
    layer (set): The set of coordinates representing the current layer.
    grid (np.ndarray): The grid being processed.

    Returns:
    int: The number of neighbors in the same layer.
    """
    return len([n for n in get_neighbors(tile, grid) if n in layer])


def is_edge_tile(tile, outer_layer, grid):
    """
    Determines if a tile is an edge tile in the outer layer.

    Parameters:
    tile (tuple): The coordinates of the tile (y, x).
    outer_layer (set): The set of coordinates representing the outer layer.
    grid (np.ndarray): The grid being processed.

    Returns:
    bool: True if the tile is an edge tile, False otherwise.
    """
    return count_neighbors_in_layer(tile, outer_layer, grid) < 2


def is_legal_tile(tile, layer, grid):
    """
    Checks if a tile is legal based on its connectivity within a layer.

    Parameters:
    tile (tuple): The coordinates of the tile (y, x).
    layer (set): The set of coordinates representing the current layer.
    grid (np.ndarray): The grid being processed.

    Returns:
    bool: True if the tile is legal, False otherwise.
    """
    return count_neighbors_in_layer(tile, layer, grid) >= 2


def get_neighbors(coord, grid):
    """
    Returns the valid neighbors of a tile on the grid.

    Parameters:
    coord (tuple): The coordinates of the tile (y, x).
    grid (np.ndarray): The grid being processed.

    Returns:
    list: A list of neighboring coordinates (y, x) that are open (True).
    """
    y, x = coord
    neighbors = []
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in deltas:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx]:
            neighbors.append((ny, nx))
    return neighbors


def remove_single_neighbor_tiles(grid):
    """
    Removes tiles from the grid that have only one neighbor, making them inaccessible.

    Parameters:
    grid (np.ndarray): The grid being processed.

    Returns:
    tuple: A modified grid and a set of unrestricted tiles that were removed.
    """
    grid_copy = grid.copy()
    unrestricted_tiles = set()
    while True:
        to_remove = set()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid_copy[y, x] and len(get_neighbors((y, x), grid_copy)) < 2:
                    to_remove.add((y, x))
        if not to_remove:
            break
        unrestricted_tiles.update(to_remove)
        for y, x in to_remove:
            grid_copy[y, x] = False
    return grid_copy, unrestricted_tiles


def goes_around_wall_corner(tile, grid):
    """
    Determines if a tile is adjacent to a wall corner.

    Parameters:
    tile (tuple): The coordinates of the tile (y, x).
    grid (np.ndarray): The grid being processed.

    Returns:
    bool: True if the tile is adjacent to a wall corner, False otherwise.
    """
    val = False
    if tile[0] - 1 >= 0 and tile[1] - 1 >= 0:
        if not grid[tile[0] - 1, tile[1] - 1]:
            val = True
    if tile[0] - 1 >= 0 and tile[1] + 1 < grid.shape[1]:
        if not grid[tile[0] - 1, tile[1] + 1]:
            val = True
    if tile[0] + 1 < grid.shape[0] and tile[1] - 1 >= 0:
        if not grid[tile[0] + 1, tile[1] - 1]:
            val = True
    if tile[0] + 1 < grid.shape[0] and tile[1] + 1 < grid.shape[1]:
        if not grid[tile[0] + 1, tile[1] + 1]:
            val = True
    return val


def adjacent_to_wall(tile, grid):
    """
    Counts the number of adjacent walls to a given tile.

    Parameters:
    tile (tuple): The coordinates of the tile (y, x).
    grid (np.ndarray): The grid being processed.

    Returns:
    int: The number of adjacent walls.
    """
    adjacent_walls = 0
    if tile[0] - 1 >= 0:
        if not grid[tile[0] - 1, tile[1]]:
            adjacent_walls += 1
    if tile[0] + 1 < grid.shape[0]:
        if not grid[tile[0] + 1, tile[1]]:
            adjacent_walls += 1
    if tile[1] - 1 >= 0:
        if not grid[tile[0], tile[1] - 1]:
            adjacent_walls += 1
    if tile[1] + 1 < grid.shape[1]:
        if not grid[tile[0], tile[1] + 1]:
            adjacent_walls += 1
    return adjacent_walls


def find_outermost_path(start, goal, grid, current_layer=None, prev_layer=None):
    """
    Finds the path from start to goal that stays as close to the edge of the grid as possible.

    Parameters:
    start (tuple): The starting coordinates (y, x).
    goal (tuple): The goal coordinates (y, x).
    grid (np.ndarray): The grid being processed.
    current_layer (set, optional): The current layer being processed.
    prev_layer (set, optional): The previous layer to avoid overlap.

    Returns:
    list: The path from start to goal, represented as a list of coordinates.
    """

    def distance_to_prev_layer(coord, prev_layer):
        y, x = coord
        if prev_layer is None:
            return min(y, grid.shape[0] - 1 - y, x, grid.shape[1] - 1 - x)
        min_distance = float("inf")
        for ly, lx in prev_layer:
            distance = abs(y - ly) + abs(x - lx)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def distance_to_edge(coord):
        y, x = coord
        return min(y, grid.shape[0] - 1 - y, x, grid.shape[1] - 1 - x)

    def is_valid_path_tile(tile):
        return grid[tile[0], tile[1]]

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid):
            if not is_valid_path_tile(neighbor):
                continue

            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = (
                    new_cost
                    + astar(start, goal, grid)
                    + heuristic(goal, neighbor)
                    + distance_to_edge(neighbor) * 10
                )  # Bias towards edges

                if distance_to_prev_layer(neighbor, prev_layer) <= 1:
                    priority -= 1  # Very strong preference for border tiles
                else:
                    priority += distance_to_prev_layer(neighbor, prev_layer)
                    adjacent_walls = adjacent_to_wall(neighbor, grid)
                    if adjacent_walls > 0:
                        priority -= 10 * adjacent_walls
                    if goes_around_wall_corner(neighbor, grid):
                        priority -= 1
                    if current_layer and prev_layer:
                        if neighbor in current_layer or neighbor in prev_layer:
                            priority += 10**10

                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    path = []
    if goal in came_from:
        current = goal
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()

    return path


def extract_layer(grid, prev_layer=None):
    """
    Extracts the outermost layer of tiles from the grid.

    Parameters:
    grid (np.ndarray): The grid being processed.
    prev_layer (set, optional): The previous layer to avoid overlap.

    Returns:
    set: A set of coordinates representing the outermost layer.
    """
    if prev_layer is None:
        # For the outermost layer
        true_indices = np.argwhere(grid)
        if true_indices.size == 0:
            return set()
        min_y, min_x = true_indices.min(axis=0)
        max_y, max_x = true_indices.max(axis=0)
        relevant_grid = grid[min_y : max_y + 1, min_x : max_x + 1]
        layer = set(
            (i + min_y, j + min_x)
            for i in range(relevant_grid.shape[0])
            for j in range(relevant_grid.shape[1])
            if relevant_grid[i, j]
            and (
                i == 0
                or i == relevant_grid.shape[0] - 1
                or j == 0
                or j == relevant_grid.shape[1] - 1
                or any(not relevant_grid[ny, nx] for ny, nx in get_neighbors((i, j), relevant_grid))
            )
        )
    else:
        # For subsequent layers
        layer = set()
        for y, x in prev_layer:
            for ny, nx in get_neighbors((y, x), grid):
                if grid[ny, nx] and (ny, nx) not in prev_layer:
                    layer.add((ny, nx))

    while len([tile for tile in layer if is_edge_tile(tile, layer, grid)]) >= 1:
        # Ensure edge tiles are connected
        layer = connect_edge_tiles(layer, prev_layer, grid)

    return layer


def get_farthest_tile(start_tile, layer, grid):
    """
    Finds the farthest tile from the start_tile within a layer using BFS.

    Parameters:
    start_tile (tuple): The starting coordinates (y, x).
    layer (set): The set of coordinates representing the current layer.
    grid (np.ndarray): The grid being processed.

    Returns:
    tuple: The coordinates of the farthest tile (y, x).
    """
    from collections import deque

    farthest_tile = start_tile
    max_distance = -1

    # BFS to explore the layer
    queue = deque([(start_tile, 0)])
    visited = set([start_tile])

    while queue:
        current_tile, distance = queue.popleft()

        # Update farthest_tile if the current distance is greater
        if distance > max_distance:
            max_distance = distance
            farthest_tile = current_tile

        # Check all neighbors in the current layer
        for neighbor in get_neighbors(current_tile, grid):
            if neighbor in layer and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return farthest_tile


def connect_edge_tiles(layer, prev_layer, grid):
    """
    Connects edge tiles within a layer to ensure the layer is fully connected.

    Parameters:
    layer (set): The set of coordinates representing the current layer.
    prev_layer (set): The set of coordinates representing the previous layer.
    grid (np.ndarray): The grid being processed.

    Returns:
    set: The updated layer with connected edge tiles.
    """
    edge_tiles = [tile for tile in layer if is_edge_tile(tile, layer, grid)]
    connected = set()
    for y, x in edge_tiles:
        if (y, x) in connected:
            continue

        min_distance = float("inf")
        closest_tile = None

        for y2, x2 in edge_tiles:
            if (y2, x2) == (y, x) or (y2, x2) in connected:
                continue
            distance = abs(y - y2) + abs(x - x2)
            if distance < min_distance:
                min_distance = distance
                closest_tile = (y2, x2)

        if not closest_tile:
            closest_tile = get_farthest_tile((y, x), layer, grid)

        if closest_tile:
            path = find_outermost_path(
                (y, x),
                closest_tile,
                grid,
                current_layer=layer,
                prev_layer=prev_layer,
            )
            path2 = find_outermost_path(
                closest_tile,
                (y, x),
                grid,
                current_layer=layer,
                prev_layer=prev_layer,
            )
            best_path = []
            paths_dict_list = [
                {"path": path, "connected_count": 0},
                {"path": path2, "connected_count": 0},
            ]
            for p in paths_dict_list:
                temp_layer = layer.copy()
                temp_layer.update(p["path"])
                if is_legal_tile((y, x), layer, grid):
                    p["connected_count"] += 1
                if is_legal_tile(closest_tile, layer, grid):
                    p["connected_count"] += 1
            max_connected_count = max(p["connected_count"] for p in paths_dict_list)
            candidates = [p for p in paths_dict_list if p["connected_count"] == max_connected_count]
            best_path = max(candidates, key=lambda p: len(p["path"]))["path"]

            layer.update(best_path)
            if is_legal_tile((y, x), layer, grid):
                connected.add((y, x))
            if is_legal_tile(closest_tile, layer, grid):
                connected.add(closest_tile)

    # Post-process to ensure layer is continuous
    final_layer = set()
    for y, x in layer:
        if is_legal_tile((y, x), layer, grid):
            final_layer.add((y, x))
    return final_layer


def identify_layers(grid):
    """
    Identifies and extracts all the layers within the grid, starting from the outermost layer.

    Parameters:
    grid (np.ndarray): The grid being processed.

    Returns:
    tuple: A list of layers (each layer is a set of coordinates) and a set of unrestricted tiles.
    """
    layers = []
    modified_grid = grid.copy()
    all_unrestricted_tiles = set()
    prev_outer_layer = None

    while modified_grid.any():
        # Remove single neighbor tiles and gather unrestricted tiles
        modified_grid, unrestricted_tiles = remove_single_neighbor_tiles(modified_grid)
        all_unrestricted_tiles.update(unrestricted_tiles)

        if prev_outer_layer is None:
            outer_layer = extract_layer(modified_grid)
        else:
            outer_layer = extract_layer(modified_grid, prev_outer_layer)

        if len(outer_layer) > 0:
            outer_layer = sorted(outer_layer, key=lambda t: (t[0], t[1]))
            layers.append(outer_layer)
            prev_outer_layer = outer_layer

            # Mark the tiles in the current outer layer as processed
            for y, x in outer_layer:
                modified_grid[y, x] = False

            # Remove tiles trapped between the current and previous layers
            min_y, min_x = min(y for y, _ in outer_layer), min(x for _, x in outer_layer)
            max_y, max_x = max(y for y, _ in outer_layer), max(x for _, x in outer_layer)
            if len(layers) > 1:
                prev_min_y, prev_min_x = min(y for y, _ in layers[-2]), min(
                    x for _, x in layers[-2]
                )
                prev_max_y, prev_max_x = max(y for y, _ in layers[-2]), max(
                    x for _, x in layers[-2]
                )

                for y in range(grid.shape[0]):
                    for x in range(grid.shape[1]):
                        if (
                            prev_min_y <= y <= prev_max_y
                            and prev_min_x <= x <= prev_max_x
                            and not (min_y <= y <= max_y and min_x <= x <= max_x)
                        ):
                            modified_grid[y, x] = False
            else:
                for y in range(grid.shape[0]):
                    for x in range(grid.shape[1]):
                        if not (min_y <= y <= max_y and min_x <= x <= max_x):
                            modified_grid[y, x] = False
        else:
            break

    return layers, all_unrestricted_tiles


def assign_layer_directions(layer, direction, grid):
    """
    Assigns movement directions to each tile in the layer to create a flow around the layer.

    Parameters:
    layer (list): The list of tiles in the current layer, ordered from outermost to innermost.
    direction (str): The desired flow direction ("clockwise" or "counterclockwise").
    grid (np.ndarray): The grid being processed.

    Returns:
    tuple: A dictionary mapping tiles to movement directions and the sorted layer.
    """
    layer_directions = {tile: [] for tile in layer}
    sorted_layer = []  # To keep the ordered list of tiles
    visited = set()
    backtrack_sections = []  # Store sections that need backtracking directions

    def get_initial_directions(tile):
        # Dynamically determine initial directions based on position in the layer
        y, x = tile
        if direction == "clockwise":
            if y == min(t[0] for t in layer):  # Top row
                return [("right", 1, 0), ("down", 0, 1)]
            elif x == max(t[1] for t in layer):  # Right column
                return [("down", 0, 1), ("left", -1, 0)]
            elif y == max(t[0] for t in layer):  # Bottom row
                return [("left", -1, 0), ("up", 0, -1)]
            elif x == min(t[1] for t in layer):  # Left column
                return [("up", 0, -1), ("right", 1, 0)]
        else:  # Counterclockwise
            if y == min(t[0] for t in layer):  # Top row
                return [("left", -1, 0), ("down", 0, 1)]
            elif x == max(t[1] for t in layer):  # Right column
                return [("up", 0, -1), ("left", -1, 0)]
            elif y == max(t[0] for t in layer):  # Bottom row
                return [("right", 1, 0), ("up", 0, -1)]
            elif x == min(t[1] for t in layer):  # Left column
                return [("down", 0, 1), ("right", 1, 0)]
        # Default fallback for tiles in the middle of the layer
        return [("right", 1, 0), ("down", 0, 1), ("left", -1, 0), ("up", 0, -1)]

    start_tile = min(layer, key=lambda t: (t[0], t[1]))
    current_tile = start_tile
    sorted_layer.append(current_tile)
    visited.add(current_tile)
    neighbors = [n for n in get_neighbors(current_tile, grid) if n in layer]

    # Sort neighbors so that those with empty lists in layer_directions come first
    neighbors.sort(key=lambda n: len(layer_directions.get(n, [])))

    stack = []
    last_dir = None

    def process_neighbor(neighbor, dir, dx, dy):
        nonlocal current_tile, last_dir
        last_dir = (dir, dx, dy)
        layer_directions[current_tile].append(dir)
        current_tile = neighbor
        sorted_layer.append(current_tile)
        visited.add(current_tile)

    found_initial = False
    initial_directions = get_initial_directions(current_tile)
    for dir, dx, dy in initial_directions:
        if found_initial:
            break
        for neighbor in neighbors:
            if neighbor == (current_tile[0] + dy, current_tile[1] + dx):
                process_neighbor(neighbor, dir, dx, dy)
                found_initial = True
                break
    if not found_initial:
        for neighbor in neighbors:
            if found_initial:
                break
            for dir, dx, dy in get_initial_directions(current_tile):
                if neighbor == (current_tile[0] + dy, current_tile[1] + dx):
                    process_neighbor(neighbor, dir, dx, dy)
                    found_initial = True
                    break

    # Assign directions to the rest of the tiles in the layer
    while True:
        neighbors = [
            n for n in get_neighbors(current_tile, grid) if n in layer and n not in visited
        ]

        # Sort neighbors so that those with empty lists in layer_directions come first
        neighbors.sort(key=lambda n: len(layer_directions.get(n, [])))

        if len(neighbors) == 2:  # Handle cases where a tile has exactly two neighbors
            # Prioritize the direction that maintains the flow (clockwise/counterclockwise)
            if layer_directions[neighbors[0]] or layer_directions[neighbors[1]]:
                # If one neighbor already has a direction, assign to the other neighbor
                for dir, dx, dy in get_initial_directions(current_tile):
                    if neighbors[0] == (current_tile[0] + dy, current_tile[1] + dx):
                        process_neighbor(neighbors[0], dir, dx, dy)
                    elif neighbors[1] == (current_tile[0] + dy, current_tile[1] + dx):
                        process_neighbor(neighbors[1], dir, dx, dy)
                continue

        if len(neighbors) > 1:
            # If there are multiple neighbors, push current state onto stack
            stack.append((current_tile, last_dir, neighbors[1:]))

        found_next_tile = False
        if neighbors:
            next_tile = neighbors[0]
            for dir, dx, dy in get_initial_directions(current_tile):
                if next_tile == (current_tile[0] + dy, current_tile[1] + dx):
                    process_neighbor(next_tile, dir, dx, dy)
                    found_next_tile = True
                    break

        if not found_next_tile:
            # Backtrack and store backtrack sections
            if stack:
                current_tile, last_dir, remaining_neighbors = stack.pop()
                for next_tile in remaining_neighbors:
                    if next_tile not in visited:
                        backtrack_sections.append((current_tile, next_tile, last_dir))
                        break
            else:
                break

        # Stop if all tiles are visited
        if len(visited) == len(layer):
            break

    # Assign directions to the backtrack sections
    for current_tile, next_tile, last_dir in backtrack_sections:
        for dir, dx, dy in get_initial_directions(current_tile):
            if next_tile == (current_tile[0] + dy, current_tile[1] + dx):
                layer_directions[current_tile].append(dir)
                sorted_layer.append(next_tile)
                break

    # Ensure any tiles left without a direction are assigned one
    for tile in layer:
        if not layer_directions[tile]:  # If the tile has no direction assigned
            neighbors = [n for n in get_neighbors(tile, grid) if n in layer]
            for neighbor in neighbors:
                for dir, dx, dy in get_initial_directions(tile):
                    if neighbor == (tile[0] + dy, tile[1] + dx):
                        layer_directions[tile].append(dir)
                        break

    # Fix tiles that only have directions pointing back to their incoming direction
    opposite_directions = {
        "up": ("down", (0, -1)),
        "down": ("up", (0, 1)),
        "left": ("right", (-1, 0)),
        "right": ("left", (1, 0)),
    }
    for tile in layer:
        directions = layer_directions[tile]
        if len(directions) == 1:
            direction = directions[0]
            opposite_direction = opposite_directions[direction]
            # Check if the only direction points back to where the tile was reached from
            layer_neighbors = [n for n in get_neighbors(tile, grid) if n in layer]
            for neighbor in layer_neighbors:
                if (
                    neighbor
                    == (
                        tile[0] + opposite_direction[1][1],
                        tile[1] + opposite_direction[1][0],
                    )
                    and opposite_direction[0] in layer_directions[neighbor]
                ):
                    # Find a different neighbor that doesn't point back to this tile
                    multi_access = False
                    for alt_neighbor in layer_neighbors:
                        if alt_neighbor != neighbor:
                            for alt_dir in layer_directions[alt_neighbor]:
                                temp_dir = opposite_directions[alt_dir]
                                if tile == (
                                    alt_neighbor[0] + temp_dir[1][1],
                                    alt_neighbor[1] + temp_dir[1][0],
                                ):
                                    multi_access = True
                                    break
                    if not multi_access:
                        for alt_neighbor in layer_neighbors:
                            if alt_neighbor != neighbor:
                                alt_dir = opposite_directions[opposite_direction[0]]
                                if not tile == (
                                    alt_neighbor[0] + alt_dir[1][1],
                                    alt_neighbor[1] + alt_dir[1][0],
                                ):
                                    # Replace the problematic direction
                                    layer_directions[tile] = []
                                    for dir, dx, dy in [
                                        ("right", 1, 0),
                                        ("down", 0, 1),
                                        ("left", -1, 0),
                                        ("up", 0, -1),
                                    ]:
                                        if alt_neighbor == (tile[0] + dy, tile[1] + dx):
                                            layer_directions[tile].append(dir)
                                            break
                                    break
    # Fix tiles that have no direction to any of their layer neighbors
    for tile in layer:
        layer_neighbors = [n for n in get_neighbors(tile, grid) if n in layer]
        has_direction_to_neighbors = any(
            (tile[0] + dy, tile[1] + dx) in layer_neighbors
            for dir in layer_directions[tile]
            for dx, dy in [
                (
                    (1, 0)
                    if dir == "right"
                    else (0, 1) if dir == "down" else (-1, 0) if dir == "left" else (0, -1)
                )
            ]
        )

        if not has_direction_to_neighbors and layer_neighbors:
            # Add a direction pointing to one of the layer neighbors
            for neighbor in layer_neighbors:
                for dir, dx, dy in [
                    ("right", 1, 0),
                    ("down", 0, 1),
                    ("left", -1, 0),
                    ("up", 0, -1),
                ]:
                    if neighbor == (tile[0] + dy, tile[1] + dx):
                        layer_directions[tile].append(dir)
                        break

    return layer_directions, sorted_layer


def ensure_access_to_unrestricted_tiles(grid, all_layer_directions, all_unrestricted_tiles):
    """
    Ensures that unrestricted tiles have valid access paths by adding the appropriate directions.

    Parameters:
    grid (np.ndarray): The grid being processed.
    all_layer_directions (dict): The dictionary of all layer directions.
    all_unrestricted_tiles (set): The set of unrestricted tiles.

    Returns:
    dict: The updated dictionary of all layer directions.
    """
    for y, x in all_unrestricted_tiles:
        neighbors = get_neighbors((y, x), grid)
        for ny, nx in neighbors:
            if (ny, nx) in all_layer_directions:
                if ny > y:
                    direction = "up"
                elif ny < y:
                    direction = "down"
                elif nx > x:
                    direction = "left"
                elif nx < x:
                    direction = "right"
                all_layer_directions[(ny, nx)].append(direction)

    return all_layer_directions


def handle_inter_layer_movement(layers, all_layer_directions, original_grid):
    """
    Handles movement between different layers by assigning appropriate directions.

    Parameters:
    layers (list): The list of layers in the grid.
    all_layer_directions (dict): The dictionary of all layer directions.
    original_grid (np.ndarray): The original grid being processed.

    Returns:
    dict: The updated dictionary of all layer directions.
    """

    def within_bounds(y, x, grid):
        return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]

    for layer in layers:
        for index, (y, x) in enumerate(layer):
            if (y, x) not in all_layer_directions:
                continue  # Skip if the tile is not in directions

            if index % 2 == 0:
                for neighbor in get_neighbors((y, x), original_grid):
                    if neighbor not in layer and within_bounds(
                        neighbor[0], neighbor[1], original_grid
                    ):
                        if neighbor[0] - y == 0:
                            if neighbor[1] - x > 0:
                                direction = "right"
                            else:
                                direction = "left"
                        else:
                            if neighbor[0] - y > 0:
                                direction = "down"
                            else:
                                direction = "up"
                        all_layer_directions[(y, x)].append(direction)

    return all_layer_directions


def identify_unreachable_tiles(grid, all_layer_directions):
    """
    Identifies unreachable tiles in the grid by performing a DFS traversal.

    Parameters:
    grid (np.ndarray): The grid being processed.
    all_layer_directions (dict): The dictionary of all layer directions.

    Returns:
    set: The set of unreachable tiles.
    """
    visited = set()
    stack = [list(all_layer_directions.keys())[0]]

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            y, x = current
            neighbors = get_neighbors((y, x), grid)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)

    unreachable_tiles = set(zip(*np.where(grid))) - visited
    return unreachable_tiles


def preprocess_map(grid: np.ndarray) -> dict:
    """
    Preprocesses the grid by identifying layers, assigning movement directions, and handling inter-layer movement.

    Parameters:
    grid (np.ndarray): The grid being processed.

    Returns:
    dict: A dictionary containing the movement directions for each tile in the grid.
    """
    grid_copy = grid.copy()
    layers, all_unrestricted_tiles = identify_layers(grid_copy)
    processed_layers = []

    # Extract SCCs from each layer and store large connected components
    for layer in layers:
        sccs = tarjan_scc_without_back_edges(layer, grid)
        for scc in sccs:
            if len(scc) >= 4:
                processed_layers.append(scc)
            else:
                all_unrestricted_tiles.update(scc)

    direction_order = ["clockwise", "counterclockwise"]
    all_layer_directions = {}

    # Assign directions to each layer
    for i, layer in enumerate(layers):
        direction = direction_order[i % 2]
        layer_directions, sorted_layer = assign_layer_directions(layer, direction, grid)
        layers[i] = sorted_layer  # Replace the original layer with the sorted layer
        all_layer_directions.update(layer_directions)

    # Handle movement between different layers
    all_layer_directions = handle_inter_layer_movement(layers, all_layer_directions, grid)

    # Ensure access to unrestricted tiles
    all_layer_directions = ensure_access_to_unrestricted_tiles(
        grid, all_layer_directions, all_unrestricted_tiles
    )

    # Add all 4 directions for non-restricted tiles
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] and (
                (y, x) not in all_layer_directions or all_layer_directions[y, x] == []
            ):
                all_layer_directions[(y, x)] = []
                for neighbor in get_neighbors((y, x), grid):
                    if grid[neighbor[0], neighbor[1]]:
                        if neighbor[0] - y == 0:
                            if neighbor[1] - x > 0:
                                direction = "right"
                            else:
                                direction = "left"
                        else:
                            if neighbor[0] - y > 0:
                                direction = "down"
                            else:
                                direction = "up"
                        all_layer_directions[(y, x)].append(direction)

    return {k: v for k, v in sorted(all_layer_directions.items()) if v}


def read_map_file(map_file: str) -> np.ndarray:
    """
    Reads a map file and converts it into a grid representation.

    Parameters:
    map_file (str): The path to the map file.

    Returns:
    np.ndarray: A boolean grid representation of the map where True indicates an open space and False indicates a wall.
    """
    width, height = 0, 0
    with open(map_file, "r") as f:
        for row in f:
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))
            if width > 0 and height > 0:
                break

        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1

    assert y == height, f"Map format seems strange, check {map_file}"
    return grid


def visualize_layers(grid, layers):
    """
    Visualizes the identified layers on the grid using distinct colors for each layer.

    Parameters:
    grid (np.ndarray): The grid being processed.
    layers (list): A list of layers, where each layer is a set of coordinates.
    """
    layer_colors = list(mcolors.TABLEAU_COLORS.values())
    img = np.ones((grid.shape[0], grid.shape[1], 3))  # Initialize as white
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not grid[y, x]:
                img[y, x] = [0, 0, 0]  # Black for walls

    for idx, layer in enumerate(layers):
        color = mcolors.to_rgb(layer_colors[idx % len(layer_colors)])
        for y, x in layer:
            img[y, x] = color

    plt.imshow(img)
    plt.grid(True, which="both", color="black", linewidth=1)
    plt.xticks(ticks=np.arange(-0.5, grid.shape[1], 1), labels=[])
    plt.yticks(ticks=np.arange(-0.5, grid.shape[0], 1), labels=[])
    plt.show()


def visualize_directions(grid, directions):
    """
    Visualizes the movement directions on the grid using arrows.

    Parameters:
    grid (np.ndarray): The grid being processed.
    directions (dict): A dictionary mapping each tile to its movement directions.
    """
    fig, ax = plt.subplots()
    img = np.ones((grid.shape[0], grid.shape[1], 3))  # Initialize as white
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not grid[y, x]:
                img[y, x] = [0, 0, 0]  # Black for walls

    ax.imshow(img, extent=(0, grid.shape[1], grid.shape[0], 0))

    direction_vectors = {"up": (0, -0.4), "down": (0, 0.4), "left": (-0.4, 0), "right": (0.4, 0)}

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x]:
                cell_directions = directions.get((y, x), [])
                for dir in cell_directions:
                    dx, dy = direction_vectors[dir]
                    ax.arrow(
                        x + 0.5,
                        y + 0.5,
                        dx,
                        dy,
                        head_width=0.1,
                        head_length=0.1,
                        fc="red",
                        ec="red",
                    )

    plt.grid(True, which="both", color="black", linewidth=1)
    plt.xticks(ticks=np.arange(0, grid.shape[1]), labels=[])
    plt.yticks(ticks=np.arange(0, grid.shape[0]), labels=[])
    plt.show()


def visualize_directions_with_colors(grid, directions, layers):
    """
    Visualizes the movement directions with layer colors for easier distinction between different layers.

    Parameters:
    grid (np.ndarray): The grid being processed.
    directions (dict): A dictionary mapping each tile to its movement directions.
    layers (list): A list of layers, where each layer is a set of coordinates.
    """
    fig, ax = plt.subplots()
    layer_colors = list(mcolors.TABLEAU_COLORS.values())
    img = np.ones((grid.shape[0], grid.shape[1], 3))  # Initialize as white

    # First, color the layers
    for idx, layer in enumerate(layers):
        color = mcolors.to_rgb(layer_colors[idx % len(layer_colors)])
        for y, x in layer:
            img[y, x] = color
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not grid[y, x]:
                img[y, x] = [0, 0, 0]  # Black for walls

    ax.imshow(img, extent=(0, grid.shape[1], grid.shape[0], 0))

    # Then, add direction arrows on top of the colored layers
    direction_vectors = {"up": (0, -0.4), "down": (0, 0.4), "left": (-0.4, 0), "right": (0.4, 0)}

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x]:
                cell_directions = directions.get((y, x), [])
                for dir in cell_directions:
                    dx, dy = direction_vectors[dir]
                    ax.arrow(
                        x + 0.5,
                        y + 0.5,
                        dx,
                        dy,
                        head_width=0.1,
                        head_length=0.1,
                        fc="red",  # Color of the arrow head
                        ec="red",  # Edge color of the arrow
                    )

    plt.grid(True, which="both", color="black", linewidth=1)
    plt.xticks(ticks=np.arange(0, grid.shape[1]), labels=[])
    plt.yticks(ticks=np.arange(0, grid.shape[0]), labels=[])
    plt.show()


def create_directed_graph_without_back_edges(layer, grid):
    """
    Creates a directed graph for the layer while avoiding direct back edges that could form 2-node SCCs.

    Parameters:
    layer (set): The set of tiles in the current layer.
    grid (np.ndarray): The grid being processed.

    Returns:
    dict: A directed graph mapping each tile to its neighbors.
    """
    graph = {}
    for tile in layer:
        neighbors = get_neighbors(tile, grid)
        graph[tile] = []

        for neighbor in neighbors:
            if neighbor in layer:
                graph[tile].append(neighbor)

    return graph


def tarjan_scc_without_back_edges(layer, grid):
    """
    Identifies all strongly connected components (SCCs) in a layer using Tarjan's algorithm,
    avoiding back edges that would create trivial 2-node SCCs.

    Parameters:
    layer (set): The set of tiles in the current layer.
    grid (np.ndarray): The grid being processed.

    Returns:
    list: A list of SCCs where each SCC is a set of tiles.
    """
    index = 0
    stack = []
    indices = {}
    lowlink = {}
    on_stack = set()
    sccs = []
    graph = create_directed_graph_without_back_edges(layer, grid)

    def strongconnect(node, previous_node=None):
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph[node]:
            if neighbor == previous_node:
                continue  # Skip back edge to the previous node

            if neighbor not in indices:
                strongconnect(neighbor, node)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.add(w)
                if w == node:
                    break
            sccs.append(scc)

    for tile in layer:
        if tile not in indices:
            strongconnect(tile)

    return sccs


if __name__ == "__main__":
    map_file_path = r"py-lacam-pibt\assets\random-32-32-10.map"
    # Read the map file to create the grid
    grid = read_map_file(map_file_path)

    # Preprocess the map
    all_layer_directions = preprocess_map(grid)

    layers, _ = identify_layers(grid)
    processed_layers = []
    # Extract SCCs from each layer and visualize
    for layer in layers:
        sccs = tarjan_scc_without_back_edges(layer, grid)
        for scc in sccs:
            if len(scc) >= 4:
                processed_layers.append(scc)

    # print("\nLayers Visualization:")
    # visualize_layers(grid, processed_layers)

    # print("\nDirections Visualization:")
    # visualize_directions(grid, all_layer_directions)

    print("\nDirections and Colors Visualization:")
    visualize_directions_with_colors(grid, all_layer_directions, processed_layers)
