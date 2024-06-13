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

    index_1 = x >= xmax + w * width * xmax
    result[index_1] = xmax

    index_2 = (x >= xmax - w * width * xmax) * (x < xmax + w * width * xmax)
    result[index_2] = xmax - w * xmax * np.log(
        1 + np.exp(-(x[index_2] - xmax) / (w * xmax))
    )

    index_3 = (x >= xmin + w * width * xmin) * (x < xmax - w * width * xmax)
    result[index_3] = x[index_3]

    index_4 = (x >= xmin - w * width * xmin) * (x < xmin + w * width * xmin)
    result[index_4] = xmin + w * xmin * np.log(
        1 + np.exp((x[index_4] - xmin) / (w * xmin))
    )

    index_5 = x < xmin - w * width * xmin
    result[index_5] = xmin

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

    index_1 = x >= xmin + w * width * xmin
    result[index_1] = x[index_1]

    index_2 = (x >= xmin - w * width * xmin) * (x < xmin + w * width * xmin)
    result[index_2] = xmin + w * xmin * np.log(
        1 + np.exp((x[index_2] - xmin) / (w * xmin))
    )

    index_3 = x < xmin - w * width * xmin
    result[index_3] = xmin

    return result
