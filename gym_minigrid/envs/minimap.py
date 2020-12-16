from gym_minigrid.minigrid import Goal
from gym_minigrid.register import register
from gym_minigrid.index_mapping import malmo_object_to_index, minigrid_index_mapping
from pathlib import Path

import numpy as np
from .numpymap import NumpyMap


RESOURCES_DIR = (Path(__file__).parent / './resources').resolve()


def fix_levers_on_same_level(same_level, above_level):
    """
    Input: 3D numpy array with malmo_object_to_index mapping

    Returns:
        3D numpy array where 3 channels represent 
        object index, color index, state index 
        for minigrid
    """
    lever_idx = malmo_object_to_index['lever']
    condition = above_level == lever_idx 
    minimap_array = np.where(condition, above_level, same_level) 
    return minimap_array


def fix_jump_locations(same_level, above_level, minigrid_index_mapping, jump_index=11):
    """
    Input: 3D numpy array with malmo_object_to_index mapping

    Returns:
        1. 3D numpy array where 3 channels represent 
        object index, color index, state index 
        for minigrid
        2. updated minigrid_index_mapping
    
    Notation for jump location 
            index = 11
            object = box
            color = grey # like walls
            toggletimes = 1

        NOTE: toggle to a box is a substitute for jump action
    """

    
    wall_idx = malmo_object_to_index['stained_hardened_clay']
    empty_idx = malmo_object_to_index['air']

    condition1 = same_level == wall_idx 
    condition2 = above_level == empty_idx

    minigrid_index_mapping['object_mapping'][jump_index] = 'box'
    minigrid_index_mapping['color_mapping'][jump_index] = 'white' #'grey0' # minigrid_index_mapping['color_mapping'][wall_idx]
    minigrid_index_mapping['toggletimes_mapping'][jump_index] = 1

    # Elementwise product of two bool arrays for AND
    minimap_array = np.where(condition1 * condition2, jump_index, same_level) # jump_index is broadcasted!

    return minimap_array, minigrid_index_mapping


def fill_outside_with_wall(same_level, below_level):
    empty_idx = malmo_object_to_index['air']
    wall_idx = malmo_object_to_index['stained_hardened_clay']

    condition1 = below_level == empty_idx
    condition2 = same_level == empty_idx

    minimap_array = np.where(condition1 * condition2, wall_idx, same_level)

    return minimap_array


def get_minimap_from_voxel(raw_map, minigrid_index_mapping, jump_index=11):
    """
    Aggregates minimap obtained from different same_level and above_map transforms
    Input: 3D numpy array with malmo_object_to_index mapping

    Returns:
        1. 3D numpy array where 3 channels represent 
        object index, color index, state index 
        for minigrid
        2. updated minigrid_index_mapping
    
    Functioning:
        * fixes jump locations
        * fixes levers in same level

        NOTE: toggle to a box is a substitute for jump action
    """
    below_level = raw_map[0]
    same_level = raw_map[1]
    above_level = raw_map[2]
    
    minimap_array, modified_index_mapping = fix_jump_locations(
        same_level, above_level, minigrid_index_mapping, jump_index)
    minimap_array = fix_levers_on_same_level(minimap_array, above_level)
    minimap_array = fill_outside_with_wall(minimap_array, below_level)

    return minimap_array, modified_index_mapping



class Minimap(NumpyMap):

    green_victim_colors = {'green', 'green2', 'inv_green2'}
    yellow_victim_colors = {'yellow', 'indianyellow', 'inv_indianyellow'}
    active_victim_colors = {
        'green', 'green2', 'inv_green2',
        'yellow', 'indianyellow', 'inv_indianyellow',
    }

    def __init__(self, 
        raw_map_path=Path(RESOURCES_DIR, 'raw_map.npy'), 
        agent_pos=(4, 25), agent_dir=0,
        yellow_victim_lifetime=5*60, total_game_duration=10*60,
        seconds_per_step=0.5, random_victim_colors=False):

        self.total_game_duration = total_game_duration
        self.seconds_per_step = seconds_per_step
        self.yellow_death_step = np.ceil(yellow_victim_lifetime / seconds_per_step)
        self.random_victim_colors = random_victim_colors

        raw_map = np.load(raw_map_path)
        minimap_array, modified_index_mapping = get_minimap_from_voxel(
            raw_map, minigrid_index_mapping, jump_index=11)

        super().__init__(
            modified_index_mapping, minimap_array, 
            agent_pos=agent_pos, agent_dir=agent_dir)


    @property
    def time(self):
        return self.step_count * self.seconds_per_step


    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        if self.random_victim_colors:
            for i in range(width):
                for j in range(height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == 'goal':
                        new_color = np.random.choice(['green', 'yellow'], p=[0.7, 0.3])

                        if new_color == 'green':
                            self.put_obj(Goal(new_color, toggletimes=4), i, j)
                        elif new_color == 'yellow':
                            self.put_obj(Goal(new_color, toggletimes=7), i, j)


    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Swap yellow victims with red victims
        if self.step_count == self.yellow_death_step:
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == 'goal' and cell.color in self.yellow_victim_colors:
                        self.put_obj(Goal('red', toggletimes=0), i, j)

        # Check if there are any green/yellow victims left
        if self.step_count > self.yellow_death_step:
            no_more_victims = True
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == 'goal' and cell.color in self.active_victim_colors:
                         no_more_victims = False
                         break
            done = no_more_victims

        # Check if we've reached the total game duration
        if self.time >= self.total_game_duration:
            done = True

        return obs, reward, done, info



class MinimapForSparky(Minimap):

    def __init__(self, 
        raw_map_path=Path(RESOURCES_DIR, 'sparky_map.npy'), 
        agent_pos=(4, 25), agent_dir=0,
        yellow_victim_lifetime=5*60, total_game_duration=10*60,
        seconds_per_step=0.5, random_victim_colors=False):

        super().__init__(
            raw_map_path=raw_map_path, 
            agent_pos=agent_pos, agent_dir=agent_dir,
            yellow_victim_lifetime=yellow_victim_lifetime, total_game_duration=total_game_duration,
            seconds_per_step=0.5, random_victim_colors=random_victim_colors)



class MinimapForFalcon(Minimap):

    def __init__(self, 
        raw_map_path=Path(RESOURCES_DIR, 'falcon_map.npy'), 
        agent_pos=(2, 10), agent_dir=0,
        yellow_victim_lifetime=5*60, total_game_duration=10*60,
        seconds_per_step=0.5, random_victim_colors=False):

        raw_map = np.load(raw_map_path)
        sub_raw_map = raw_map[:, :, 10:]
        minimap_array, modified_index_mapping = get_minimap_from_voxel(
            sub_raw_map, minigrid_index_mapping, jump_index=11)

        self.total_game_duration = total_game_duration
        self.seconds_per_step = seconds_per_step
        self.yellow_death_step = np.ceil(yellow_victim_lifetime / seconds_per_step)
        self.random_victim_colors = random_victim_colors

        super(Minimap, self).__init__(
            modified_index_mapping, minimap_array, 
            agent_pos=agent_pos, agent_dir=agent_dir,
        )



register(
    id='MiniGrid-MinimapForMinecraft-v0',
    entry_point='gym_minigrid.envs:Minimap'
)

register(
    id='MiniGrid-MinimapForSparky-v0',
    entry_point='gym_minigrid.envs:MinimapForSparky'
)

register(
    id='MiniGrid-MinimapForFalcon-v0',
    entry_point='gym_minigrid.envs:MinimapForFalcon'
)
