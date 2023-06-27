from dataclasses import dataclass

@dataclass
class MomentsBundle():
    """
    Store the internalised exact learning data, ready for fitting
    """
    num_dims: int
    valid_s_domain: ??
    min_x: [] # a vector of length num_dims
    max_x: [] # a vector of length num_dims
    is_distribution: bool
    interpolating_function: ??? # A scipy function
    max_moment_power: int # How many moment derivatives to take? 
    q: 
    dq:
    ddq:
    dddq:

@dataclass
class ExactLearningResult():
    """
    Store an exact learning result
    """
    equation: str
    num_dims: int

    # Might have a method to yeild TeXForm etc.



def ingest(x,y) -> MomentsBundle:
    """
    Take x and y as plt.plot would
    Interpolate, generate this function as part of moments bundle
    Determine settings (distribution etc.?)
    Convert to a dataset, via complex integration for distributions?
    """

    # determine dimensionality of data

    # dig out an example of interpolation (i.e. Schrodinger)

    # dig out an example of integration (i.e. moment fitting)
