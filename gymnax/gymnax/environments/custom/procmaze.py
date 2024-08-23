from functools import partial

import jax as jx
from jax import jit, vmap
from jax import numpy as jnp
from jax import jit, vmap

from gymnax.gymnax.environments import environment, spaces
from flax import struct


@struct.dataclass
class EnvState:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    time: int


class ProcMaze:
    def __init__(self, grid_size=10, goal_reward=False, timeout=None):
        super.__init__()
        self.move_map = jnp.asarray([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])
        self.timeout = timeout

        self._num_actions = 5
        self.grid_size = grid_size
        self.channels ={
            'player':0,
            'goal':1,
            'wall':2,
            'empty':3
        }

    def step_env(self, action, env_state):
        goal, wall_grid, pos, t = env_state

        # Move if the new position is on the grid and open
        new_pos = jnp.clip(pos+self.move_map[action], 0, self.grid_size-1)
        pos = jnp.where(jnp.logical_not(wall_grid[new_pos[0], new_pos[1]]), new_pos, pos)

        # Treminated if we reach the goal
        terminal = jnp.array_equal(pos, goal)
        if(self.timeout is not None):
            terminal = jnp.logical_or(terminal, t>=self.timeout)

        reward = -1

        t+=1

        env_state = goal, wall_grid, pos, t

        return env_state, self.get_observation(env_state), reward, terminal, {}

    def reset_env(self, key):
        def push(stack, top, x):
            stack= stack.at[top].set(x)
            top+=1
            return stack, top

        def pop(stack, top):
            top-=1
            return stack[top], top

        #takes and flattened index, returns neighbours as (x,y) pair
        def neighbours(cell):
            coord_tuple = jnp.unravel_index(cell, (self.grid_size, self.grid_size))
            coord_array = jnp.stack(list(coord_tuple))
            return coord_array+self.move_map

        #takes (x,y) pair
        def can_expand(cell, visited):
            # A neighbour can be expanded as long as it is on the grid, it has not been visited, and it's only visited neighbour is the current cell
            flat_cell = jnp.ravel_multi_index((cell[0],cell[1]),(self.grid_size,self.grid_size),mode='clip')
            not_visited = jnp.logical_not(visited[flat_cell])
            ns = neighbours(flat_cell)
            ns_on_grid = jnp.all(jnp.logical_and(ns>=0,ns<=self.grid_size-1), axis=1)
            flat_ns = jnp.ravel_multi_index((ns[:,0],ns[:,1]),(self.grid_size,self.grid_size),mode='clip')
            # Only count neighbours which are actually on the grid
            only_one_visited_neighbor = jnp.equal(jnp.sum(jnp.logical_and(visited[flat_ns],ns_on_grid)),1)
            on_grid = jnp.all(jnp.logical_and(cell>=0,cell<=self.grid_size-1))
            return jnp.logical_and(jnp.logical_and(not_visited,only_one_visited_neighbor),on_grid)
        can_expand = vmap(can_expand, in_axes=(0,None))

        wall_grid = jnp.ones((self.grid_size, self.grid_size), dtype=bool)

        #Visited node map
        visited = jnp.zeros(self.grid_size*self.grid_size, dtype=bool)

        #big enough to hold every location in the grid, indices should be flattened to store here
        stack = jnp.zeros(self.grid_size*self.grid_size, dtype=int)
        top = 0

        #Location of current cell being searched
        key, subkey = jx.random.split(key)
        curr = jx.random.choice(subkey, self.grid_size, (2,))
        flat_curr = jnp.ravel_multi_index((curr[0],curr[1]),(self.grid_size,self.grid_size),mode='clip')
        wall_grid = wall_grid.at[curr[0], curr[1]].set(False)

        visited = visited.at[flat_curr].set(True)
        stack, top = push(stack,top, flat_curr)

        def cond_fun(carry):
            visited, stack, top, wall_grid, key = carry
            #continue until stack is empty
            return top!=0

        def body_fun(carry):
            visited, stack, top, wall_grid, key = carry
            curr, top = pop(stack,top)
            ns = neighbours(curr)
            flat_ns = jnp.ravel_multi_index((ns[:,0],ns[:,1]),(self.grid_size,self.grid_size),mode='clip')

            expandable = can_expand(ns,visited)

            has_expandable_neighbour = jnp.any(expandable)

            # This will all be used only conditioned on has_expandable neighbor
            _stack, _top = push(stack, top, curr)
            key, subkey = jx.random.split(key)
            selected = jx.random.choice(subkey, flat_ns,p=expandable/jnp.sum(expandable))
            _stack, _top = push(_stack, _top, selected)
            _wall_grid = wall_grid.at[jnp.unravel_index(selected, (self.grid_size, self.grid_size))].set(False)
            _visited = visited.at[selected].set(True)

            stack = jnp.where(has_expandable_neighbour, _stack, stack)
            top = jnp.where(has_expandable_neighbour, _top, top)
            wall_grid = jnp.where(has_expandable_neighbour, _wall_grid, wall_grid)
            visited = jnp.where(has_expandable_neighbour, _visited, visited)
            return (visited, stack, top, wall_grid, key)



        key, subkey = jx.random.split(key)
        carry = (visited, stack, top, wall_grid, subkey)
        visited, stack, top, wall_grid, key = jx.lax.while_loop(cond_fun, body_fun, carry)

        flat_open = jnp.logical_not(jnp.ravel(wall_grid))

        key, subkey = jx.random.split(key)
        pos = jx.random.choice(subkey, self.grid_size*self.grid_size, p=flat_open/jnp.sum(flat_open))
        pos = jnp.stack(list(jnp.unravel_index(pos, (self.grid_size, self.grid_size))))
        key, subkey = jx.random.split(key)
        goal = jx.random.choice(subkey, self.grid_size*self.grid_size, p=flat_open/jnp.sum(flat_open))
        goal = jnp.stack(list(jnp.unravel_index(goal, (self.grid_size, self.grid_size))))

        wall_grid = wall_grid.at[goal[0], goal[1]].set(False)
        wall_grid = wall_grid.at[pos[0], pos[1]].set(False)
        t=0
        env_state = goal, wall_grid, pos, t
        return env_state

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        goal, wall_grid, pos, t = env_state
        obs = jnp.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs = obs.at[pos[0],pos[1],self.channels['player']].set(True)
        obs = obs.at[goal[0],goal[1],self.channels['goal']].set(True)
        obs = obs.at[:,:,self.channels['wall']].set(wall_grid)
        obs = obs.at[:,:,self.channels['empty']].set(jnp.logical_not(wall_grid))
        return obs

    def num_actions(self):
        return self._num_actions