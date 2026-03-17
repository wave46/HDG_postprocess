import numpy as np


def double_softplus(x, xmin, xmax, w, width):
    """
    this routine constrains value x between xmin and xmax
    using paradigm of softplus function
    for xmin it is a typical softplus 
    f(x) = xmin+width*ln(1+exp((x-xmin)/w)
    w here and after = w*xmin(or max), where w is defined inside the function 
    parameter width states for the region where smoothening is applied xmax+-width*w
    for xmax it is somewhat inversed softplus:
    f(x) = width*ln(1+exp(xmax/width))-width*ln(1+exp(-(x-xmax)/width))
    for x>= xmax+width*w*xmax : f(x)=xmax
    for xmax-width*w*xmax<=x<xmax+width*w*xmax : f(x) = w*xmax*ln(1+exp(1/w))-width*w*xmax*ln(1+exp(-(x-xmax)/(w*xmax))
    for xmin+width*w*xmin<=x<xmax-width*w*xmax : f(x) = x
    for xmin-width*w*xmin<=x<xmin+width*w*xmin : f(x) = xmin + w*xmin*ln(1+exp((x-xmin)/(w*xmin))
    x<xmin-width*w*xmin : f(x) = xmin
    """
    result = np.zeros_like(x)
    xmax_band = w * width * xmax
    xmin_band = w * width * xmin
    xmax_scale = w * xmax
    xmin_scale = w * xmin
    upper_start = xmax - xmax_band
    upper_stop = xmax + xmax_band
    lower_start = xmin - xmin_band
    lower_stop = xmin + xmin_band

    above_upper = x >= upper_stop
    result[above_upper] = xmax

    near_upper = (x >= upper_start) & (x < upper_stop)
    result[near_upper] = xmax - xmax_scale * np.log(1 + np.exp(-(x[near_upper] - xmax) / xmax_scale))

    middle = (x >= lower_stop) & (x < upper_start)
    result[middle] = x[middle]

    near_lower = (x >= lower_start) & (x < lower_stop)
    result[near_lower] = xmin + xmin_scale * np.log(1 + np.exp((x[near_lower] - xmin) / xmin_scale))

    below_lower = x < lower_start
    result[below_lower] = xmin

    return result


def softplus(x, xmin, w, width):
    """
    this routine limits value x with xmin
    using paradigm of softplus function 
    f(x) = xmin+width*ln(1+exp((x-xmin)/width)
    w here and after = w*xmin(or max), where w is defined inside the function 
    parameter width states for the region where smoothening is applied xmax+-width*w
    for x>=xmin-width*w*xmin : f(x) = xmin + w*xmin*ln(1+exp((x-xmin)/(w*xmin))
    x<xmin-width*w*xmin : f(x) = xmin
    """

    result = np.zeros_like(x)
    xmin_band = w * width * xmin
    xmin_scale = w * xmin
    lower_start = xmin - xmin_band
    lower_stop = xmin + xmin_band

    above_lower = x >= lower_stop
    result[above_lower] = x[above_lower]

    near_lower = (x >= lower_start) & (x < lower_stop)
    result[near_lower] = xmin + xmin_scale * np.log(1 + np.exp((x[near_lower] - xmin) / xmin_scale))

    below_lower = x < lower_start
    result[below_lower] = xmin

    return result
