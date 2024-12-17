import numpy as np
import lmfit

def lmfit_gaussian(data: np.ndarray, dA: float, bin_size: int = 1, center=None) -> lmfit.Parameters:
    """
    Fit a 2D Gaussian to a rectangular data array. Assumes that the location of
    the peak is relatively close to the middle of the data array, and that `dA`,
    which provides a scaling factor between data array coordinates and the
    physical quantities related to thosse coordinates, is uniform and square. The
    data array is interpreted to have x-coordinates correspond to column index
    values.

    Raises RuntimeError if the fit fails.

    Parameters
    ----------
    data : np.ndarray[ndim=2]
        2D data array.
    dA : float-like
        Relates the 1x1 data array cell size to an appropriate physical
        quantity. When `data` is, e.g., pixel brightness values taken from a
        camera, dA corresponds to the area of a single pixel.

    Returns
    -------
    fitparams : lmfit.Parameters
        Final parameters of the fit:
            A: overall height of the Gaussian
            x0: x-coordinate of the center
            sx: width along the axis parallel to `theta`
            y0: y-coordimate of the center
            sy: width along the axis perpendicular to `theta`
            theta: rotation of the Gaussian
            B: vertical offset
    """
    i0, j0 = [k // 2 for k in data.shape]
    x0 = np.sqrt(dA) * j0
    x = np.sqrt(dA) * np.arange(data.shape[1])
    y0 = np.sqrt(dA) * i0
    y = np.sqrt(dA) * np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    downsample = bin_size # take every `downsample`-th element on both axes
    sampler = slice(None, None, downsample)
    D = data[sampler, sampler].flatten()
    X = X[sampler, sampler].flatten()
    Y = Y[sampler, sampler].flatten()

    # # 07/12 21:30
    # x0 = X[D.argmax()]
    # y0 = Y[D.argmax()]

    # 07/25 
    if center is not None:
        x0 = center['x0']
        y0 = center['y0']
    else:
        x0 = X[D.argmax()]
        y0 = Y[D.argmax()]

    print(f"x0: {x0}, y0: {y0}")

    def model(
        params: lmfit.Parameters,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        A = params["A"].value
        x0 = params["x0"].value
        sx = params["sx"].value
        y0 = params["y0"].value
        sy = params["sy"].value
        th = params["theta"].value
        B = params["B"].value
        Xrel = X - x0
        Yrel = Y - y0
        Xrot = np.cos(th) * Xrel + np.sin(th) * Yrel
        Yrot = -np.sin(th) * Xrel + np.cos(th) * Yrel
        return A * np.exp(-(Xrot / sx)**2 - (Yrot / sy)**2) + B
        # return A * np.exp(- (Xrel / sx)**2 - (Yrel / sy)**2) + B
        # return A * np.exp(-2 * (Xrel ** 2 + Yrel ** 2) / (w0**2)) + B

    def residual(
        params: lmfit.Parameters,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        m = model(params, X, Y)
        return (D - m)**2

    params = lmfit.Parameters()
    params.add("A", value=data.max())
    params.add("x0", value=x0)
    params.add("sx", value=(x.max() - x.min()) / 20, min=0.0)    
    params.add("y0", value=y0)
    params.add("sy", value=(y.max() - y.min()) / 20, min=0.0)
    params.add("theta", value=0.0, min=0.0, max=np.pi / 2)
    params.add("B", value=data.mean())
    params["B"].min = 0.0

    # 07/12 21:30
    # params.add("w0", value=(x.max() - x.min()) / 10, min=0.0)

    fit = lmfit.minimize(residual, params, args=(X, Y, D))
    if not fit.success:
        raise RuntimeError("Error in Gaussian fit")
    return fit.params, X, Y, D

