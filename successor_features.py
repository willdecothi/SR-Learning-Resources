import numpy as np
import numpy.matlib 
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt


class BVC:
    def __init__(self, distance_pref, angular_pref, distance_std, angular_std):
        """ A boundary vector cell as defined by Hartley et al (2000)

        Args:
            distance_pref (float): the distal tuning preference
            angular_pref (float): the angular tuning preference 
            distance_std (float): std dev of gaussian around distance_pref
            angular_std (float): std dev of gaussian around angular_pref
        """        
        self.distance_pref = distance_pref
        self.angular_pref = angular_pref
        self.distance_std = distance_std
        self.angular_std = angular_std
        self.rate_map = None

class SuccessorFeature:
    def __init__(self):
        self.weights = None
        self.rate_map = None

class SuccessorEigenvector:
    def __init__(self):
        self.weights = None
        self.rate_map = None

class Environment:
    def __init__(self, enclosure_type):
        """Defines the size and boundaries in the environment

        Args:
            enclosure_type (string): One of the enclosure_types listed below
        """
        if enclosure_type == 'small_square':
            grid_map = np.zeros([100, 100, 3], np.uint8)
            walls = [((0,0),(0,100)),((0,100),(100,100)),((100,0),(100,100)),((0,0),(100,0))]
            for wall in walls:
                wall_start, wall_end = wall
                cv.line(grid_map, wall_start, wall_end,(0,255,0),1)
            grid_map = grid_map[:, :, 1].astype('bool')
            self.map =  grid_map
            self.walls = walls

        elif enclosure_type == 'square':
            grid_map = np.zeros([200, 200, 3], np.uint8)
            walls = [((0,0),(0,200)),((0,200),(200,200)),((200,0),(200,200)),((0,0),(200,0))]
            for wall in walls:
                wall_start, wall_end = wall
                cv.line(grid_map, wall_start, wall_end,(0,255,0),1)
            grid_map = grid_map[:, :, 1].astype('bool')
            self.map =  grid_map
            self.walls = walls

        elif enclosure_type == 'barrier':
            grid_map = np.zeros([200, 200, 3], np.uint8)
            walls = [((0,0),(0,200)),((0,200),(200,200)),((200,0),(200,200)),((0,0),(200,0)),
    		((100,0),(100,100))]
            for wall in walls:
                wall_start, wall_end = wall
                cv.line(grid_map, wall_start, wall_end,(0,255,0),1)
            grid_map = grid_map[:, :, 1].astype('bool')
            self.map =  grid_map
            self.walls = walls

        elif enclosure_type == 'four_rooms':
            grid_map = np.zeros([200, 200, 3], np.uint8)
            walls = [((0,0),(0,200)),((0,200),(200,200)),((200,0),(200,200)),((0,0),(200,0)),
    		((100,60),(100,140)),((60,100),(140,100)),
    		((100,0),(100,40)),((100,160),(100,200)),((0,100),(40,100)),((160,100),(200,100))]
            for wall in walls:
                wall_start, wall_end = wall
                cv.line(grid_map, wall_start, wall_end,(0,255,0),1)
            grid_map = grid_map[:, :, 1].astype('bool')
            self.map =  grid_map
            self.walls = walls
        else:
            print('Please select existing enclosure.')

    def generate_trajectory(self, n_samples):
        """Generates a random walk according to Raudies and Hasselmo (2012) Figure 2

        Args:
            n_samples (int): the length of the random walk

        Returns:
            arrays: position [x,y] and head direction [radians] data
        """        
        print('Simulating trajectory')
        mu = 0 # distribution mean for turns
        sigma = 0.2 # standard deviation for turns
        b = 16 # Rayleigh velocity parameter
        direction = 2*np.pi*np.random.rand(1) # initial direction
        dt = 1/50 # sampling rate
        # create random turn and velocity samples
        random_turns = np.random.normal(mu,sigma,n_samples)
        random_velocities = np.random.rayleigh(b,n_samples)
        # allocate memory for x, y, and z components of position and velocity
        positions = np.zeros((n_samples,2))
        y, x = np.argwhere(self.map>= 0)[np.random.randint(len(np.argwhere(self.map>= 0)))] + 0.5
        head_directions = np.zeros(n_samples)
        positions[0] = [x,y]
        head_directions[0] = direction

        for step in tqdm(np.arange(1, n_samples), position=0,leave=True):
            v = random_velocities[step]
            [distance_to_wall,angle_to_wall] = self._min_distance_angle(x,y,direction)
            # update speed and turn angle
            if distance_to_wall<2 and np.abs(angle_to_wall) < (np.pi/2):
            # if close to wall slow down and follow wall or turn away
                angle = np.sign(angle_to_wall) * (np.pi/2 - np.abs(angle_to_wall)) + random_turns[step]
                v = v - 0.5*np.max([0,(v-5)])
            else:
                angle = random_turns[step]

            # check for collisions with wall
            motion_vector = v*dt*np.array([np.cos(direction),np.sin(direction)]).squeeze()
            intersection_points = np.zeros((len(self.walls),2))
            for wall_i, wall in enumerate(self.walls):
                wall_start, wall_end = np.array(wall,dtype='float64')
                intersection_points[wall_i] = wall_intersection(positions[step-1],motion_vector,wall_start,wall_end)
                positions[step] = positions[step-1] + motion_vector
            if np.any(~np.isnan(intersection_points)): # next position is out of bounds stay in position
                positions[step] = positions[step-1]
            # iterate
            x, y = positions[step]
            direction = direction + angle
            head_directions[step] = direction
        return positions, head_directions

    def _min_distance_angle(self, x, y, direction):
        """Calculate the (approximate) shortest distance and direction from current position to the walls 

        Args:
            x (float): x coordinate
            y (float): y coordinate
            direction (float): head direction

        Returns:
            distance_to_wall (float), angle_to_wall (float): distance and direction (radians) to the nearest wall 
        """        
        angles = np.arange(0,2,0.1)*np.pi
        radius = 300.0
        distances_to_walls = np.full((len(self.walls),len(angles)),np.nan)
        position = np.array([x,y],dtype='float64')
        for angle_i, angle in enumerate(angles):
            bearing = np.array([radius*np.cos(angle), radius*np.sin(angle)])
            for wall_i, wall in enumerate(self.walls):
                wall_start, wall_end = np.array(wall,dtype='float64')
                intersection_point = wall_intersection(position,bearing,wall_start,wall_end)
                distances_to_walls[wall_i, angle_i] = np.sqrt(np.sum((intersection_point - position)**2))
        shortest_distances = np.nanmin(distances_to_walls, axis=0)

        distance_to_wall = np.min(shortest_distances)
        angle_to_wall = wrapToPi(direction - angles[shortest_distances == distance_to_wall])
        angle_to_wall = np.random.choice(angle_to_wall)
        return distance_to_wall, angle_to_wall

def generate_bvcs(env):
    """Creates a list of BVC objects specified by preferred_angles and preferred_distances

    Args:
        env (Environment): instantiation of the Environment class

    Returns:
        [bvcs]: list of BVCs with length n_cells
    """    
    n_cells = 160
    preferred_angles = np.matlib.repmat(np.array(np.pi * np.arange(0, 16) / 8), 1, 10).T
    preferred_distances = (np.matlib.repmat(np.array([3.6, 11.1, 19.1, 27.7, 37.0, 47.0, 57.7, 69.3, 81.7, 95.0]),
                  n_cells // 10, 1)).T.flatten()
    bvcs = []
    for i in np.arange(n_cells):
        c = BVC(preferred_distances[i],preferred_angles[i],preferred_distances[i]/12 + 8,np.array([np.pi / 16]))
        bvcs.append(c)

    # initialize firing map
    rate_maps = np.zeros([len(bvcs), np.shape(env.map)[0], np.shape(env.map)[1]])
    rate_maps[:] = np.nan

    [y, x] = np.where(env.map >= 0)

    print('Building BVC rate maps')
    for idx in tqdm(np.arange(len(y)), position=0, leave=True):
        rate_maps[:, y[idx], x[idx]] = bvc_activity(x[idx]+0.5, y[idx]+0.5, bvcs, env)

    rate_maps[np.isnan(rate_maps)] = 0
    for idx in tqdm(np.arange(n_cells), position=0, leave=True):
        bvcs[idx].rate_map = np.squeeze(rate_maps[idx,:,:])
        bvcs[idx].rate_map = bvcs[idx].rate_map / np.amax(bvcs[idx].rate_map)
    return bvcs

def bvc_activity(x, y, bvcs, env):
    """Returns population vector (firing rates) at a position [x,y] in env

    Args:
        x (float): x coordinate
        y (float): y coordinate
        bvcs (list): list of BVC objects
        env (Environment): generated via the environment class

    Returns:
        population_vector: a vector of length bvcs with the firing rate of each bvcs at position [x,y] in env
    """
    angles = np.arange(0,2,0.02)*np.pi
    radius = 300.0
    distances_to_walls = np.full((len(env.walls),len(angles)),np.nan)
    position = np.array([x,y],dtype='float64')
    for angle_i, angle in enumerate(angles):
        bearing = np.array([radius*np.cos(angle), radius*np.sin(angle)])
        for wall_i, wall in enumerate(env.walls):
            wall_start, wall_end = np.array(wall,dtype='float64')
            intersection_point = wall_intersection(position,bearing,wall_start,wall_end)
            distances_to_walls[wall_i, angle_i] = np.sqrt(np.sum((intersection_point - position)**2))

    shortest_distances = np.nanmin(distances_to_walls, axis=0)
    population_vector = bvc_firing_rate(angles, shortest_distances, bvcs)
    return population_vector

def bvc_firing_rate(angles, distances_to_wall, bvcs):
    """Calculates current firing of the BVCs given the distances and directions to the walls according to Hartley et al (2000) 

    Args:
        angles (array): range of angles across which to integrate firing rate contributions
        distances_to_wall (array): distances to the nearest wall corresponding to each angle
        bvcs (list): list of BVC objects  

    Returns:
        [rates]: vector of same length as bvcs with corresponding firing rates 
    """    
    rates = np.zeros(len(bvcs))
    for i in np.arange(len(bvcs)):
        rates[i] = np.sum(np.exp( - (distances_to_wall-bvcs[i].distance_pref) ** 2 / (2 * bvcs[i].distance_std ** 2)) *
                          np.exp(- wrapToPi(angles - bvcs[i].angular_pref) ** 2 / (2 * bvcs[i].angular_std ** 2)) /
                          (2 * np.pi * bvcs[i].distance_std * bvcs[i].angular_std))
    return rates

def wall_intersection(position,bearing,wall_start,wall_end):
    """Calculates the intersection point of two line segments position + bearing and wall_start to wall_end

    Args:
        position ([x,y]): current location
        bearing ([x,y]): projection along bearing (we will only find intersection along the length of the bearing)
        wall_start ([x,y]): start point of the wall
        wall_end ([x,y]): end point of the wall (must be increasing, we will only find intersections along the length of the wall)

    Returns:
        [x,y]: the location of the intersection (nan if none exists)
    """    
    wall_bearing = wall_end-wall_start
    # np.cross is ridiculously slow so lets do cross products manually
    bearing_cross_product = bearing[0]*wall_bearing[1] - bearing[1]*wall_bearing[0]
    position_to_wall_start = wall_start-position
    proportion_along_wall = (position_to_wall_start[0]*bearing[1] - position_to_wall_start[1]*bearing[0])/bearing_cross_product
    proportion_along_bearing = (position_to_wall_start[0]*wall_bearing[1] - position_to_wall_start[1]*wall_bearing[0])/bearing_cross_product
    # proportion_along_wall = np.cross(position_to_wall_start,bearing) /  np.cross(bearing,wall_bearing)
    # proportion_along_bearing = np.cross(position_to_wall_start,wall_bearing) /  np.cross(bearing,wall_bearing)

    if ((proportion_along_wall > 0) & (proportion_along_wall < 1) & (proportion_along_bearing > 0) & (proportion_along_bearing < 1)):
        intersection_point = wall_start + proportion_along_wall*(wall_end-wall_start)
    else:
        intersection_point = np.array([np.nan,np.nan])
    return intersection_point

def train_model(cells, M, trajectory, alpha = 1e-4, gamma=0.995, time_lag=1):
    """Implements the successor feature learning rule given the behaviour in trajectory according to de Cothi & Barry (2020)

    Args:
        cells (list): list of basis features (e.g. BVCs)
        M (matrix): of size n_cells by n_cells
        trajectory (array): of [x,y] locations in the trajectory
        alpha (float, optional): Learning rate. Defaults to 1e-4.
        gamma (float, optional): Discount factor. Defaults to 0.995.
        time_lag (int, optional): number of time steps between updates. Defaults to 1.

    Returns:
        M: The learnt successor matrix 
    """    
    total_steps = len(trajectory) - time_lag
    print('')
    print('Training BVC-SR model')
    for t in tqdm(np.arange(total_steps-1)):
        x = trajectory[t, 0]
        y = trajectory[t, 1]
        next_x = trajectory[t+time_lag, 0]
        next_y = trajectory[t+time_lag, 1]
        firing_rates = get_firing_rates(int(np.round(x-0.5)), int(np.round(y-0.5)), cells)
        next_firing_rates = get_firing_rates(int(np.round(next_x-0.5)), int(np.round(next_y-0.5)), cells)
        M = sr_update(firing_rates, next_firing_rates, M, alpha, gamma)

    return M

def sr_update(firing_rates, next_firing_rates, M, alpha, gamma):
    """The sr learning rule

    Args:
        firing_rates (vector): current firing rate of basis features 
        next_firing_rates (vector): firing rate of basis features at the next time step]
        M (matrix): the SR matrix
        alpha (float): the learning rate takes values in (0,1)
        gamma (float): the discount factor takes values in (0,1)

    Returns:
        updated_M: the updated successor matrix
    """    
    firing_rates = firing_rates[np.newaxis]
    next_firing_rates = next_firing_rates[np.newaxis]
    updated_M = M + alpha * (firing_rates.T + gamma * (M @ next_firing_rates.T) - (M @ firing_rates.T)) @ firing_rates
    # R = R + r_alpha * M * phi.T * (r + gamma * np.dot(M * n_phi.T, R) - np.dot(M * phi.T,R.T))
    return updated_M

def get_firing_rates(x, y, cells):
    """Fetches the population vector of cells from the closest bin in the rate map (actually calculating from scratch takes way too long for BVCs)

    Args:
        x (int): index of bin along x-axis
        y (int): index of bin along y-axis
        cells (list): list of basis features (e.g. BVCs)

    Returns:
        vector: vector of cell firing rates
    """    
    n_cells = len(cells)
    population_vector = np.zeros([n_cells])
    for i in np.arange(n_cells):
        population_vector[i] = cells[i].rate_map[y,x]
    population_vector[np.isnan(population_vector)] = 0
    return population_vector

def calculate_successor_features(basis_cells, M, threshold=True):
    """Calculate the successor features of the the basis features given the learnt successor matrix M

    Args:
        basis_cells (list): of basis features (e.g. BVCs)
        M (matrix): The learnt successor matrix
        threshold (bool, optional): Whether to threshold the successor features at 80%. Defaults to True.

    Returns:
        list: of the SuccessorFeatures
    """    
    print('')
    print('Calculating successor features')
    successor_features = []
    for i in tqdm(np.arange(len(basis_cells))):
        # initialise successor feature
        c = SuccessorFeature()
        c.rate_map = np.zeros(np.shape(basis_cells[i].rate_map))
        # get successor feature weights
        c.weights = M[i, :]
        # sum inputs
        for j in np.arange(len(basis_cells)):
            c.rate_map += basis_cells[j].rate_map * c.weights[j]
        # normalize and threshold maps
        c.rate_map = c.rate_map / np.amax(c.rate_map)
        if threshold == True:
            c.rate_map = np.maximum(c.rate_map - 0.8, 0)
        successor_features.append(c)
    return successor_features

def calculate_successor_eigenvectors(basis_cells, M, threshold=True):
    """Calulate the eigenvectors of the successor matrix and project them back onto the basis set

    Args:
        basis_cells (list): of basis features (e.g. BVCs)
        M (matrix): the learnt successor matrix
        threshold (bool, optional): Determines whether to threshold the projected eigenvectors to be non-negative. Defaults to True.

    Returns:
        list: of SuccessorEigenvectors
    """    
    print('')
    print('Calculating successor eigenvectors')
    successor_eigenvectors = []
    for i in tqdm(np.arange(len(basis_cells))):
        # initialise successor eigenvector
        c = SuccessorEigenvector()
        c.rate_map = np.zeros(np.shape(basis_cells[i].rate_map))
        # get eigenvectors
        _, v = np.linalg.eig(M)
        c.weights = np.real(v[:,i])
        # sum inputs
        for j in np.arange(len(basis_cells)):
            c.rate_map += basis_cells[j].rate_map * c.weights[j]
        # normalize and threshold maps
        c.rate_map = c.rate_map / np.amax(c.rate_map)
        if threshold == True:
            c.rate_map = np.maximum(c.rate_map, 0)
        successor_eigenvectors.append(c)
    return successor_eigenvectors

def plot_cells(cells, n=10, m=16, cell_type = ''):
    """plots the rate_map attributes of cells

    Args:
        cells (list): of features (e.g. BVCs, SuccessorFeatures, SuccessorEigenvectors)
        n (int, optional): number of subplot rows. Defaults to 10.
        m (int, optional): number of subplot columns. Defaults to 16.
        cell_type (str, optional): goes into title of each subplot. Defaults to ''.

    Returns:
        fig: figure of all the rate_maps
    """    
    fig = plt.figure(figsize=(20,20))
    for i in np.arange(n*m):
        ax = plt.subplot(n, m, i+1)
        ax.imshow(cells[i].rate_map, cmap='jet')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(cell_type+str(i+1))
    plt.subplots_adjust(hspace=0.5,wspace=0.3)
    plt.show()
    return fig

def wrapTo2Pi_element(angle):
    '''
    Converts angle in radians to range from 0 to 2pi
    '''
    angle =  angle % (np.pi*2)
    angle = (angle + (np.pi*2)) % (np.pi*2 )
    if (angle > (np.pi*2)):
        angle -= np.pi*2
    return angle

def wrapTo2Pi(angle):
    vfunc = np.vectorize(wrapTo2Pi_element)
    return vfunc(angle)

def wrapToPi_element(angle):
    '''
    Converts angle in radians range from -pi to pi
    '''
    angle =  angle % (np.pi*2)
    angle = (angle + (np.pi*2)) % (np.pi*2 )
    if (angle > np.pi):
        angle -= np.pi*2
    return angle

def wrapToPi(angle):
    vfunc = np.vectorize(wrapToPi_element)
    return vfunc(angle)
