import numpy as np



### MiniGrid

# Used to map objects to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}



# Map of color names to RGB values
COLORS = {
    'red'       : np.array([255, 0, 0]),
    'green'     : np.array([0, 255, 0]),
    'blue'      : np.array([0, 0, 255]),
    'purple'    : np.array([112, 39, 195]),
    'yellow'    : np.array([255, 255, 0]),
    'grey'      : np.array([100, 100, 100]),
    'cyan'      : np.array([0, 255, 255]),
    'brown'     : np.array([203, 169, 134]), #[203, 169, 134] [168, 147, 125]
    'white'     : np.array([255, 255, 255]),
    'black'     : np.array([0, 0, 0]),
    'green2'    : np.array([63, 165, 76]),
    'blue2'     : np.array([51, 153,245]),
    'yellow2'   : np.array([255, 196, 75]),
    'grey0'     : np.array([200, 200, 200]),
    'grey2'     : np.array([180, 180, 150]),
    'grey3'     : np.array([20, 20, 20]),
    'darkbrown' : np.array([101, 67, 33]),
    'indianyellow': np.array([222, 159, 75]),
    # 'inv_brown' : np.array([255-203, 255-169, 255-134])
}

INV_COLORS = dict(zip(list(map(lambda x: 'inv_' + x, COLORS.keys())), 255 - np.array(list(COLORS.values()))))

COLORS.update(dict(zip(INV_COLORS.keys(), INV_COLORS.values())))

COLOR_NAMES = sorted(list(COLORS.keys()) + list(INV_COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'       : 0,
    'green'     : 1,
    'blue'      : 2,
    'purple'    : 3,
    'yellow'    : 4,
    'grey'      : 5,
    'cyan'      : 6,
    'brown'     : 7,
    'white'     : 8,
    'black'     : 9,
    'green2'    : 10,
    'blue2'     : 11,
    'yellow2'   : 12,
    'grey0'     : 13,
    'grey2'     : 14,
    'grey3'     : 15,
    'darkbrown' : 16, # np.array([101, 67, 33]),
    'indianyellow': 17, # np.array([222, 159, 75]),
    # 'inv_brown' : 16,
}

num_unique_colors = len(COLOR_TO_IDX)

COLOR_TO_IDX.update(dict(zip(INV_COLORS.keys(), [x for x in range(num_unique_colors, 2*num_unique_colors)])))

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))



# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}


# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

DIR_TO_8_VEC = [
    # Down (positive Y)
    np.array((0, 1)),
    np.array((-1, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    np.array((-1, -1)),
    # Up (negative Y)
    np.array((0, -1)),
    np.array((1, -1)),
    # Pointing right (positive X)
    np.array((1, 0)),
    np.array((1, 1)),
]

# {
#     # Pointing right (positive X)
#     270 : np.array((1, 0)),
#     # Down (positive Y)
#     0 : 
#     # Pointing left (negative X)
#     90 : np.array((-1, 0)),
#     # Up (negative Y)
#     180 : np.array((0, -1)),
#     # corners
#     225 : np.array((1, 1)),
#     135 : np.array((-1, 1)),
#     315 : np.array((1, -1)),
#     45 : np.array((-1, -1)),
# }

### MiniGrid Miscellaneous (Vidhi)

# Minecraft Notations
malmo_object_to_index = {
    # Basics
    'air'                       :     1, 
    'wooden_door'               :     2, 
    'dark_oak_door'             :     2,
    # 'wool' : 3, wall  : 4, 
    'lever'                     :     5, 
    # 'ball': 6,'box' : 7, 
    # 'goal': 8,
    'fire'                      :     9,
    'player'                    :     10, 

    # Walls
    'stained_hardened_clay'     :     4, 
    'clay'                      :     4,      # 42, 
    'iron_block'                :     4,      # 43, 
    'quartz_block'              :     4,      # 44,
    'redstone_wire'             :     4,      # 45,
    'gravel'                    :     4,      # 46, 
    'sticky_piston'             :     4,
    'piston_head'               :     4,
    'hardened_clay'             :     4,
    'stone'                     :     4,
    'bone_block'                :     4,
    'stained_glass_pane'        :     4,
    'monster_egg'               :     4,
    'cobblestone_wall'          :     4,
    'cobblestone'               :     4,
    'bedrock'                   :     30,

    # Goals
    'redstone_block'            :     80,
    'gold_block'                :     81,
    'prismarine'                :     82,
    'bone_block'                :     83,
    'wool'                      :     83,  
}

raw_map_colors = { 
        0   : 'white',    # [0, 0, 0]     ,
        1   : 'grey3',    # [255,255, 255],
        2   : 'grey',     # [63, 165, 76] ,
        3   : 'blue2',    # [51, 153,245],
        30  : 'purple',
        4   : 'grey0',     # [98, 67, 67],
        5   : 'cyan', 

        80  : 'red',
        81  : 'yellow',
        82  : 'green',
        83  : 'white',

        9   : 'yellow',
        10  : 'red',
        255 : 'brown' ,
}


visibility_colors = {
        0:  [0]*3, # [50, 50, 50],   # [200,200,200],
        1:  [255]*3, # [200, 200, 200],
        10: [0, 0, 255]
}


# Gym-minigrid Notations

INT_TO_OBJECT_numpymap = {
            0   :    'unseen',
            1   :    'empty',
            2   :    'door',

            30  :    'wall',

            4   :    'wall',
            5   :    'key',

            80  :    'goal',
            81  :    'goal',
            82  :    'goal',
            83  :    'goal', 

            9   :    'lava',
            10  :    'agent',
            255 :    'box',
        }

OBJECT_TO_INT_numpymap = dict(zip(IDX_TO_OBJECT.values(), IDX_TO_OBJECT.keys()))

OBJECT_TO_INT_LIST_numpymap = {
        'unseen'    : [0],
        'empty'     : [1],
        'door'      : [2],
        'wall'      : [30, 4,],
        'key'       : [5],
        'goal'      : [81, 82, 80, 83],
        'lava'      : [9],
        'agent'     : [10],
        'box'       : [255, 11]
    }

# # TO RECOMPUTE OBJECT_TO_INT_LIST_numpymap
# 
# for key, value in IDX_TO_OBJECT.items():
#     if OBJECT_TO_IDX_LIST.get(value, None) is None:
#         OBJECT_TO_IDX_LIST[value]=[key]
#     else:
#         OBJECT_TO_IDX_LIST[value].append(key)


minigrid_index_mapping = {

        'object_mapping' : INT_TO_OBJECT_numpymap,

        'color_mapping'  : {

            2   :    'inv_blue',

            30  :    'grey2',

            4   :    'inv_darkbrown', # 'grey',
            5   :    'yellow2',

            80  :    'inv_red',
            81  :    'inv_indianyellow', # 'red', 
            82  :    'inv_green2',
            83  :    'inv_blue', # 'white',

            255 :    'brown',

        },

        'toggletimes_mapping' : {

            # 80  :    0,
            81  :    5,
            82  :    2,
            # 83  :    0, 

            255 :    1,

        },

}


def main():
    from pprint import pprint
    print('\n\nraw_map_colors')
    pprint(raw_map_colors)

    print('\n\nmalmo_object_to_index')
    pprint(malmo_object_to_index)
    
    print('\n\nminigrid_index_mapping')
    pprint(minigrid_index_mapping)


if __name__ == '__main__':
    main()
