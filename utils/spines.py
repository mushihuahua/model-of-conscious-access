
from Params import Params
params = Params()

def normalised_spine_count(k: int, chi_raw: int):
    """Calculate and return the normalised spine count for area `k`
    
       Inputs:
        `k`: Cortical area.
        `χ_raw`: The spine count for area `k`.
    """

    chi_normalised = (chi_raw - params.chi_raw_min) / (params.chi_raw_max - params.chi_raw_min)
    return chi_normalised

def spine_count_gradient(k: int, chi_raw: int, population: str):
    """Calculate and return the spine count modulation variable for area `k`"""

    z_min = params.z_min if population == "E" else params.z_min_I

    z_k = z_min + normalised_spine_count(k, chi_raw) * (1 - z_min)
    return z_k

def normalised_spine_count_long_range(spine_counts):
    """Calculate and return the normalised spine count for each area
    
       Inputs:
        `spine_counts`: The spine counts for all areas.
    """

    chi_normalised = (spine_counts - params.chi_raw_min) / (params.chi_raw_max - params.chi_raw_min)
    return chi_normalised

def spine_count_gradient_long_range(spine_counts, population: str):
    """Calculate and return the spine count modulation variable for all areas"""

    z_min = params.z_min if population == "E" else params.z_min_I

    z_k = z_min + normalised_spine_count_long_range(spine_counts) * (1 - z_min)
    return z_k

