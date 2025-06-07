import numpy as np
import lmfit

def lmfit_gaussian(data: np.ndarray, dA: float, bin_size: int, fit_param: dict[str, float]) -> tuple[lmfit.Parameters, np.ndarray, np.ndarray, np.ndarray]:
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
    bin_size : int
        Downsample the data by taking every `bin_size`-th element on both axes.
    fit_param : dict[str, float]
        Initial values for the fit parameters. The keys are:
            'x0': x-coordinate of the center (default: None, will be set to the
                   x-coordinate of the maximum value in `data`)
            'y0': y-coordinate of the center (default: None, will be set to the
                   y-coordinate of the maximum value in `data`)
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
    X : np.ndarray
        X-coordinates of the downsampled data points, in physical units.
    Y : np.ndarray
        Y-coordinates of the downsampled data points, in physical units.
    D : np.ndarray
        Downsampled data values, flattened to a 1D array.
    """
    # i0, j0 = [k // 2 for k in data.shape]
    # x0 = np.sqrt(dA) * j0
    x = np.sqrt(dA) * np.arange(data.shape[1])
    # y0 = np.sqrt(dA) * i0
    y = np.sqrt(dA) * np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    downsample = bin_size # take every `downsample`-th element on both axes
    sampler = slice(None, None, downsample)
    D = data[sampler, sampler].flatten()
    X = X[sampler, sampler].flatten()
    Y = Y[sampler, sampler].flatten()

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
        return A * np.exp(-Xrot ** 2 / (2 * sx ** 2) - Yrot ** 2 / (2 * sy ** 2)) + B

    def residual(
        params: lmfit.Parameters,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        m = model(params, X, Y)
        return (D - m)**2

    x0 = fit_param['x0'] if fit_param['x0'] != None else X[D.argmax()]
    y0 = fit_param['y0'] if fit_param['y0'] != None else Y[D.argmax()]
    sx = fit_param['sx'] if fit_param['sx'] != None else (x.max() - x.min()) / 3
    sy = fit_param['sy'] if fit_param['sy'] != None else (y.max() - y.min()) / 3
    theta = fit_param['theta'] if fit_param['theta'] != None else 0.0

    params = lmfit.Parameters()
    params.add("A", value=data.max(), min=0)
    params.add("B", value=data.min(), min=0)
    params.add("x0", value=x0)
    params.add("sx", value=sx)
    params.add("y0", value=y0)
    params.add("sy", value=sy)
    params.add("theta", value=theta, min=-np.pi/2, max=np.pi / 2)

    fit = lmfit.minimize(residual, params, args=(X, Y, D), max_nfev=10000, method='leastsq')
    print("wx:", round(fit.params['sx'].value * np.sqrt(2), 2), " wy:", round(fit.params['sy'].value * np.sqrt(2), 2))
    if not fit.success:
        raise RuntimeError("Error in Gaussian fit")
    return fit.params, X, Y, D

