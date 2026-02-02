import numpy as np
import torch

def wide_cardioid_beam_pattern(facting_direction, phi, base_level=2.0):
    """calculate the microphone pattern at each direction, given the current microphone direction

    Parameters
    ----------
    facting_direction : float
        microphone direction
    phi : torch.tensor, direction to query the microphone beam pattern
        a grid of data to query the microphone direction

    Returns
    -------
    gain: torch.tensor, microphone gain patterns at each query direction
    """
    # Wide cardioid pattern
    main_lobe_gain = (1 + torch.cos(phi-facting_direction)) / 2

    # Combine main lobe with base level
    if not base_level:
        base_level = 1.0
        
    gain = main_lobe_gain + base_level
    gain /= torch.max(gain)

    return gain