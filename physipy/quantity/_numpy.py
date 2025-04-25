from typing import Callable, Union

import numpy as np

from .quantity import Dimension, DimensionError, Quantity, quantify

HANDLED_FUNCTIONS: dict[Callable, Callable] = {}


def implements(np_function: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def implements_like(np_function: Callable) -> Callable:
    # explicit name to notify that "like=" must be used to trigger the interface
    def decorator(func: Callable) -> Callable:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements_like(np.arange)
def np_arange(*args, **kwargs):
    if len(args) == 0:
        availables = {
            k: kwargs.pop(k) for k in ["start", "step", "stop"] if k in kwargs
        }
        first_dimension = next(iter(availables.values())).dimension
        for k, v in availables.items():
            if v.dimension != first_dimension:
                raise DimensionError(first_dimension, v.dimension)
        start = availables.pop("start", Quantity(0, first_dimension))
        step = availables.pop("step", Quantity(1, first_dimension))
        stop = availables.pop("stop", Quantity(1, first_dimension))
    elif len(args) == 1:
        stop = quantify(args[0])
        start = quantify(kwargs.pop("start", Quantity(0, stop.dimension)))
        step = quantify(kwargs.pop("step", Quantity(1, stop.dimension)))
    elif len(args) == 2:
        start, stop = quantify(args[0]), quantify(args[1])
        step = quantify(kwargs.pop("step", Quantity(1, start.dimension)))
    elif len(args) == 3:
        start, stop, step = args
        start, stop, step = quantify(start), quantify(stop), quantify(step)
    else:
        raise TypeError(
            f"arange() accepts 1, 2, or 3 arguments. Got {len(args)}"
        )
    if not (start.dimension == step.dimension):
        raise DimensionError(start.dimension, step.dimension)
    if not (step.dimension == stop.dimension):
        raise DimensionError(step.dimension, stop.dimension)
    raw_array = np.arange(start.value, stop.value, step.value, **kwargs)
    return Quantity(raw_array, start.dimension)


@implements(np.unique)
def np_unique(ar, *args, **kwargs):
    result = np.unique(ar.value, *args, **kwargs)

    if isinstance(result, tuple):
        uniques = Quantity(result[0], ar.dimension)
        return (uniques,) + result[1:]
    else:
        uniques = Quantity(result, ar.dimension)
        return uniques


@implements(np.asanyarray)
def np_asanyarray(a):
    return Quantity(np.asanyarray(a.value), a.dimension)


@implements(np.amax)
def np_amax(q):
    return Quantity(np.amax(q.value), q.dimension, favunit=q.favunit)


@implements(np.amin)
def np_amin(q):
    return Quantity(np.amin(q.value), q.dimension, favunit=q.favunit)


@implements(np.append)
def np_append(arr, values, **kwargs):
    values = quantify(values)
    if not arr.dimension == values.dimension:
        raise DimensionError(arr.dimension, values.dimension)
    return Quantity(
        np.append(arr.value, values.value, **kwargs), arr.dimension
    )


@implements(np.argmax)
def np_argmax(a, **kwargs):
    return Quantity(np.argmax(a.value, **kwargs), a.dimension)


@implements(np.nanmin)
def np_nanmin(a, **kwargs):
    return Quantity(np.nanmin(a.value, **kwargs), a.dimension)


@implements(np.nanmax)
def np_nanmax(a, **kwargs):
    return Quantity(np.nanmax(a.value, **kwargs), a.dimension)


@implements(np.nanargmin)
def np_nanargmin(a, **kwargs):
    return np.nanargmin(a.value, **kwargs)


@implements(np.nanargmax)
def np_nanargmax(a, **kwargs):
    return np.nanargmax(a.value, **kwargs)


@implements(np.nansum)
def np_nansum(a, **kwargs):
    return Quantity(np.nansum(a.value, **kwargs), a.dimension)


@implements(np.nanmean)
def np_nanmean(a, **kwargs):
    return Quantity(np.nanmean(a.value, **kwargs), a.dimension)


@implements(np.nanmedian)
def np_nanmedian(a, **kwargs):
    return Quantity(np.nanmedian(a.value, **kwargs), a.dimension)


@implements(np.nanvar)
def np_nanvar(a, **kwargs):
    return Quantity(np.nanvar(a.value, **kwargs), a.dimension**2)


@implements(np.nanstd)
def np_nanstd(a, **kwargs):
    return Quantity(np.nanstd(a.value, **kwargs), a.dimension)


@implements(np.nanpercentile)
def np_nanpercentile(a, *args, **kwargs):
    return Quantity(np.nanpercentile(a.value, *args, **kwargs), a.dimension)


@implements(np.nanprod)
def np_nanprod(a, axis=None, **kwargs):
    if axis is None:
        n = a.size
    elif isinstance(axis, int):
        n = a.shape[axis]
    elif isinstance(axis, tuple):
        n = np.prod([a.shape[i] for i in axis])
    else:
        raise ValueError(
            "Axis type not handled, use None, int or tuple of int."
        )
    # should the dimension be len(a)-number of nan ?
    return Quantity(
        np.nanprod(a.value, axis=axis, **kwargs), a.dimension ** (n)
    )


@implements(np.nancumsum)
def np_nancumsum(a, **kwargs):
    return Quantity(np.nancumsum(a.value, **kwargs), a.dimension)


# np.nancumprod : cant have an array with different dimensions


@implements(np.array_equal)
def np_array_equal(a1, a2, *args, **kwargs):
    a1 = quantify(a1)
    a2 = quantify(a2)
    return (
        np.array_equal(a1.value, a2.value, *args, **kwargs)
        and a1.dimension == a2.dimension
    )


@implements(np.argsort)
def np_argsort(a, **kwargs):
    return np.argsort(a.value, **kwargs)


@implements(np.sort)
def np_sort(a, **kwargs):
    return Quantity(np.sort(a.value, **kwargs), a.dimension)


@implements(np.argmin)
def np_argmin(a, **kwargs):
    return Quantity(np.argmin(a.value, **kwargs), a.dimension)


@implements(np.around)
def np_around(a, **kwargs):
    return Quantity(np.around(a.value, **kwargs), a.dimension)


@implements(np.atleast_1d)
def np_atleast_1d(*arys):
    res = [Quantity(np.atleast_1d(arr.value), arr.dimension) for arr in arys]
    return res if len(res) > 1 else res[0]


@implements(np.atleast_2d)
def np_atleast_2d(*arys):
    res = [Quantity(np.atleast_2d(arr.value), arr.dimension) for arr in arys]
    return res if len(res) > 1 else res[0]


@implements(np.atleast_3d)
def np_atleast_3d(*arys):
    res = [Quantity(np.atleast_3d(arr.value), arr.dimension) for arr in arys]
    return res if len(res) > 1 else res[0]


@implements(np.average)
def np_average(q):
    return Quantity(np.average(q.value), q.dimension, favunit=q.favunit)


# np.block : todo


@implements(np.broadcast_to)
def np_broadcast_to(array, *args, **kwargs):
    return Quantity(
        np.broadcast_to(array.value, *args, **kwargs), array.dimension
    )


@implements(np.broadcast_arrays)
def np_broadcast_arrays(*args, **kwargs):
    qargs = [quantify(a) for a in args]
    # get arrays values
    arrs = [qarg.value for qarg in qargs]
    # get broadcasted arrays
    res = np.broadcast_arrays(*arrs, **kwargs)
    return [Quantity(r, q.dimension) for r, q in zip(res, qargs)]


@implements(np.linalg.norm)
def np_linalg_norm(x, *args, **kwargs):
    return Quantity(np.linalg.norm(x.value, *args, **kwargs), x.dimension)


@implements(np.linalg.lstsq)
def np_linalg_lstsq(a, b, **kwargs):
    a = quantify(a)
    b = quantify(b)
    sol = np.linalg.lstsq(a.value, b.value, **kwargs)
    return Quantity(sol, b.dimension / a.dimension)


@implements(np.linalg.inv)
def np_inv(a):
    return Quantity(np.linalg.inv(a.value), 1 / a.dimension)


@implements(np.linalg.eig)
def np_eig(a):
    eigenvalues, eigenvectors = np.linalg.eig(a.value)
    return Quantity(eigenvalues, a.dimension), eigenvectors


@implements(np.diag)
def np_diag(v, *args, **kwargs):
    return Quantity(np.diag(v.value, *args, **kwargs), v.dimension)


@implements(np.flip)
def np_flip(m, axis=None):
    return Quantity(
        np.flip(m.value, axis=axis),
        m.dimension,
        symbol=m.symbol,
        favunit=m.favunit,
    )


@implements(np.fliplr)
def np_fliplr(m):
    return Quantity(
        np.fliplr(m.value), m.dimension, symbol=m.symbol, favunit=m.favunit
    )


@implements(np.flipud)
def np_flipud(m):
    return Quantity(
        np.flipud(m.value), m.dimension, symbol=m.symbol, favunit=m.favunit
    )


# random function are not supported
# see https://github.com/numpy/numpy/issues/19382
# @implements(np.random.normal)
# def np_random_normal(loc=0.0, scale=1.0, **kwargs):
#    loc = quantify(loc)
#    scale = quantify(scale)
#    if not loc.dimension == scale.dimension:
#        raise DimensionError(loc.dimension, scale.dimension)
#    return Quantity(np.random.normal(loc=loc.value, scale=scale.value,
#                                     **kwargs))


@implements(np.polyfit)
def np_polyfit(x, y, deg, *args, **kwargs):
    x = quantify(x)
    y = quantify(y)
    p = np.polyfit(x.value, y.value, deg, *args, **kwargs)
    qp = tuple(
        Quantity(coef, y.dimension / x.dimension ** (deg - i))
        for i, coef in enumerate(p)
    )
    return qp


@implements(np.round)
def np_round(a, *args, **kwargs):
    return Quantity(np.round(a.value, *args, **kwargs), a.dimension)


@implements(np.polyval)
def np_polyval(p, x):
    p_values = tuple(quantify(coef).value for coef in p)
    x = quantify(x)
    res = np.polyval(p_values, x.value)
    return Quantity(res, p[-1].dimension)


@implements(np.clip)
def np_clip(a, a_min, a_max, *args, **kwargs):
    a_min = quantify(a_min)
    a_max = quantify(a_max)
    if a.dimension != a_min.dimension:
        raise DimensionError(a.dimension, a_min.dimension)
    if a.dimension != a_max.dimension:
        raise DimensionError(a.dimension, a_max.dimension)
    return Quantity(
        np.clip(a.value, a_min.value, a_max.value, *args, **kwargs),
        a.dimension,
    )


@implements(np.copyto)
def np_copyto(dst, src, **kwargs):
    dst = quantify(dst)
    src = quantify(src)
    if dst.dimension != src.dimension:
        raise DimensionError(dst.dimension, src.dimension)
    return np.copyto(dst.value, src.value, **kwargs)


@implements(np.piecewise)
def np_piecewise(x, condlist, funclist, *args, **kwargs):
    newfunc = [
        lambda x_, f=f: f(x_ * x._SI_unitary_quantity) if callable(f) else f
        for f in funclist
    ]
    res = [
        quantify(f(x)).dimension if callable(f) else quantify(f).dimension
        for f in funclist
    ]
    if not len(set(res)) == 1:
        raise DimensionError(
            "All functions should return a Quantity with same dimension"
        )
    raw = np.piecewise(x.value, condlist, newfunc, *args, **kwargs)
    return Quantity(raw, res[0])


@implements(np.column_stack)
def np_column_stack(tup):
    dim = tup[0].dimension
    for arr in tup:
        if arr.dimension != dim:
            raise DimensionError(arr.dimension, dim)
    return Quantity(np.column_stack(tuple(arr.value for arr in tup)), dim)


@implements(np.compress)
def np_compress(condition, a, **kwargs):
    return Quantity(np.compress(condition, a.value, **kwargs), a.dimension)


@implements(np.concatenate)
def np_concatenate(tup, *args, **kwargs):
    dim = tup[0].dimension
    for arr in tup:
        if arr.dimension != dim:
            raise DimensionError(arr.dimension, dim)
    return Quantity(
        np.concatenate(tuple(arr.value for arr in tup), *args, **kwargs), dim
    )


@implements(np.copy)
def np_copy(a, **kwargs):
    return Quantity(np.copy(a.value, **kwargs), a.dimension)


# np.copyto todo
# np.count_nonzero


@implements(np.count_nonzero)
def np_count_nonzero(a, *args, **kwargs):
    return np.count_nonzero(a.value, *args, **kwargs)


@implements(np.cross)
def np_cross(a, b, **kwargs):
    return Quantity(np.cross(a.value, b.value), a.dimension * b.dimension)


# np.cumprod : cant have an array with different dimensions


@implements(np.cumsum)
def np_cumsum(a, **kwargs):
    return Quantity(np.cumsum(a.value, **kwargs), a.dimension)


@implements(np.histogram)
def np_histogram(a, bins=10, range=None, density=None, weights=None, **kwargs):
    if range is not None:
        range = (quantify(range[0]), quantify(range[1]))
        if not range[0].dimension == range[1].dimension:
            raise DimensionError(range[0].dimension, range[1].dimension)
    hist, bin_edges = np.histogram(
        a.value, bins=bins, range=range, density=density, weights=weights
    )
    return hist, Quantity(bin_edges, a.dimension)


@implements(np.histogram2d)
def np_histogram2d(x, y, bins=10, range=None, weights=None, **kwargs):
    x = quantify(x)
    y = quantify(y)
    hist, xedges, yedges = np.histogram2d(
        x.value, y.value, bins=bins, range=range, weights=weights
    )
    return hist, Quantity(xedges, x.dimension), Quantity(yedges, y.dimension)


@implements(np.diagonal)
def np_diagonal(a, **kwargs):
    return Quantity(np.diagonal(a.value, **kwargs), a.dimension)


@implements(np.diff)
def np_diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    if prepend != np._NoValue:
        if prepend.dimension != a.dimension:
            raise DimensionError(a.dimension, prepend.dimension)
    if append != np._NoValue:
        if append.dimension != a.dimension:
            raise DimensionError(a.dimension, append.dimension)
    return Quantity(
        np.diff(a.value, n=n, axis=axis, prepend=prepend, append=append),
        a.dimension,
    )


@implements(np.apply_along_axis)
def np_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    res = np.apply_along_axis(func1d, axis, arr.value, *args, **kwargs)
    return res


@implements(np.ndim)
def np_ndim(a):
    return np.ndim(a.value)


@implements(np.dot)
def np_dot(a, b, **kwargs):
    a = quantify(a)
    b = quantify(b)
    return Quantity(
        np.dot(a.value, b.value, **kwargs), a.dimension * b.dimension
    )


@implements(np.cov)
def np_cov(m, y=None, *args, **kwargs):
    m = quantify(m)
    if y is not None:
        y = quantify(y)
        raw = np.cov(m.value, y.value, *args, **kwargs)
        return Quantity(raw, m.dimension * y.dimension)
    raw = np.cov(m.value, y, *args, **kwargs)
    return Quantity(raw, m.dimension**2)


@implements(np.max)
def np_max(qarr, *args, **kwargs):
    return Quantity(np.max(qarr.value, *args, **kwargs), qarr.dimension)


@implements(np.min)
def np_min(qarr, *args, **kwargs):
    return Quantity(np.min(qarr.value, *args, **kwargs), qarr.dimension)


@implements(np.percentile)
def np_percentile(a, *args, **kwargs):
    a = quantify(a)
    return Quantity(np.percentile(a.value, *args, **kwargs), a.dimension)


@implements(np.searchsorted)
def np_searchsorted(a, v, *args, **kwargs):
    a = quantify(a)
    v = quantify(v)
    if not a.dimension == v.dimension:
        raise DimensionError(a.dimension, v.dimension)
    return np.searchsorted(a.value, v.value, *args, **kwargs)


@implements(np.stack)
def np_stack(arrays, *args, **kwargs):
    d = quantify(arrays[0]).dimension
    ars = []
    for ar in arrays:
        qar = quantify(ar)
        if qar.dimension == d:
            ars.append(qar.value)
        else:
            raise DimensionError(d, qar.dimension)
    return Quantity(np.stack(ars, *args, **kwargs), d)


@implements(np.insert)
def np_insert(arr, obj, values, *args, **kwargs):
    arr = quantify(arr)
    values = quantify(values)
    if not arr.dimension == values.dimension:
        raise DimensionError(arr.dimension, values.dimension)
    return Quantity(
        np.insert(arr.value, obj, values.value, *args, **kwargs),
        arr.dimension,
        favunit=arr.favunit,
    )


@implements(np.dstack)
def np_dstack(tup):
    dim = tup[0].dimension
    for arr in tup:
        if arr.dimension != dim:
            raise DimensionError(arr.dimension, dim)
    return Quantity(np.dstack(tuple(arr.value for arr in tup)), dim)


@implements(np.tile)
def np_tile(A, reps):
    return Quantity(np.tile(A.value, reps), A.dimension)


@implements(np.prod)
def np_prod(a, **kwargs):
    return Quantity(np.prod(a.value), a.dimension ** (len(a)))


# @implements(np.ediff1d)
# def np_ediff1d(ary, to_end=None, to_begin=None):
#    if not ary.dimension == to_end.dimension:
#        raise DimensionError(ary.dimension, to_end.dimension)
#    if not to_begin is None:
#        if not ary.dimension == to_begin.dimension:
#             raise DimensionError(ary.dimension, to_begin.dimension)
#    return Quantity(np.ediff1d(ary.value, to_end, to_begin))


@implements(np.may_share_memory)
def np_may_share_memory(a, b, max_work=None):
    if not isinstance(b, Quantity):
        return np.may_share_memory(a.value, b, max_work=max_work)
    if not isinstance(a, Quantity):
        return np.may_share_memory(a, b.value, max_work=max_work)
    return np.may_share_memory(a.value, b.value, max_work=max_work)


@implements(np.sum)
def np_sum(q, **kwargs):
    return Quantity(np.sum(q.value, **kwargs), q.dimension, favunit=q.favunit)


@implements(np.mean)
def np_mean(q, **kwargs):
    return Quantity(np.mean(q.value, **kwargs), q.dimension, favunit=q.favunit)


@implements(np.std)
def np_std(q, *args, **kwargs):
    return Quantity(
        np.std(q.value, *args, **kwargs), q.dimension, favunit=q.favunit
    )


@implements(np.median)
def np_median(q):
    return Quantity(np.median(q.value), q.dimension, favunit=q.favunit)


@implements(np.var)
def np_var(q, *args, **kwargs):
    return Quantity(np.var(q.value, *args, **kwargs), q.dimension**2)


@implements(np.rollaxis)
def np_rollaxis(a, axis, start=0):
    return Quantity(
        np.rollaxis(a.value, axis, start=0),
        a.dimension,
        symbol=a.symbol,
        favunit=a.favunit,
    )


@implements(np.trapz)
def np_trapz(q, x=None, dx=1, **kwargs):
    # if not isinstance(q.value,np.ndarray):
    #        raise TypeError("Quantity value must be array-like to integrate.")
    q = quantify(q)
    if x is None:
        dx = quantify(dx)
        return Quantity(
            np.trapz(q.value, x=None, dx=dx.value, **kwargs),
            q.dimension * dx.dimension,
        )
    else:
        x = quantify(x)
        return Quantity(
            np.trapz(q.value, x=x.value, **kwargs),
            q.dimension * x.dimension,
        )


try:
    if not hasattr(np, "trapezoid"):
        raise AttributeError(
            "np.trapezoid is not available in this NumPy version."
        )

    @implements(np.trapezoid)
    def np_trapz(q, x=None, dx=1, **kwargs):
        q = quantify(q)
        if x is None:
            dx = quantify(dx)
            return Quantity(
                np.trapezoid(q.value, x=None, dx=dx.value, **kwargs),
                q.dimension * dx.dimension,
            )
        else:
            x = quantify(x)
            return Quantity(
                np.trapezoid(q.value, x=x.value, **kwargs),
                q.dimension * x.dimension,
            )

except AttributeError as e:
    print("When trying to declare np.trapz wrapper:")
    print(f"AttributeError: {e}")
except Exception as e:
    print("When trying to declare np.trapz wrapper:")
    print(f"An unexpected error occurred: {e}")


@implements(np.bincount)
def np_bincount(x, weights=None, **kwargs):
    x = quantify(x)
    if weights is None:
        return np.bincount(x.value, **kwargs)
    weights = quantify(weights)
    return Quantity(
        np.bincount(x.value, weights=weights.value, **kwargs),
        weights.dimension,
    )


@implements(np.transpose)
def np_transpose(a, axes=None):
    return Quantity(
        np.transpose(a.value, axes=axes),
        a.dimension,
        favunit=a.favunit,
        symbol=a.symbol,
    )


@implements(np.rot90)
def np_rot90(m, k=1, axes=(0, 1)):
    return Quantity(
        np.rot90(m.value, k=k, axes=axes),
        m.dimension,
        favunit=m.favunit,
        symbol=m.symbol,
    )


@implements(np.angle)
def np_angle(x, *args, **kwargs):
    return np.angle(x.value, *args, **kwargs)


@implements(np.lib.stride_tricks.sliding_window_view)
def np_lib_stride_tricks_sliding_window_view(x, *args, **kwargs):
    raw = np.lib.stride_tricks.sliding_window_view(x.value, *args, **kwargs)
    return Quantity(raw, x.dimension, favunit=x.favunit, symbol=x.symbol)


# @implements(np.all)
# def np_all(a, *args, **kwargs):
#    # should dimension also be checked ?
#    return np.all(a.value)


@implements(np.ones_like)
def np_ones_like(a, **kwargs):
    return np.ones_like(a.value, **kwargs)


@implements(np.zeros_like)
def np_zeros_like(a, **kwargs):
    return np.zeros_like(a.value, **kwargs)


@implements(np.zeros)
@implements_like(np.zeros)
def np_zeros(shape, dtype=float, order="C", *, like=None):
    like = quantify(like)
    return Quantity(np.zeros(shape, dtype=dtype, order=order), like.dimension)


@implements(np.full_like)
def np_full_like(a, fill_value, **kwargs):
    a = quantify(a)
    fill_value = quantify(fill_value)
    return Quantity(
        np.full_like(a.value, fill_value.value, **kwargs), fill_value.dimension
    )


@implements(np.empty_like)
def np_empty_like(prototype, **kwargs):
    return np.empty_like(prototype.value, **kwargs)


@implements(np.expand_dims)
def np_expand_ims(a, axis):
    return Quantity(np.expand_dims(a.value, axis), a.dimension)


@implements(np.shape)
def np_shape(a):
    return np.shape(a.value)


# _linspace = decorate_with_various_unit(("A", "A"), "A")(np.linspace)


@implements(np.linspace)
def np_linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    start = quantify(start)
    stop = quantify(stop)
    if not start.dimension == stop.dimension:
        raise DimensionError(start.dimension, stop.dimension)
    return Quantity(
        np.linspace(
            start.value,
            stop.value,
            num=num,
            endpoint=endpoint,
            retstep=retstep,
            dtype=dtype,
            axis=axis,
        ),
        start.dimension,
    )


@implements(np.corrcoef)
def np_corrcoef(x, y=None, *args, **kwargs):
    x = quantify(x)
    if y is not None:
        y = quantify(y).value
    return np.corrcoef(x.value, y, *args, **kwargs)


@implements(np.take)
def np_take(a, *args, **kwargs):
    return Quantity(np.take(a.value, *args, **kwargs), a.dimension)


@implements(np.squeeze)
def np_squeeze(a, *args):
    return Quantity(np.squeeze(a.value, *args), a.dimension)


@implements(np.repeat)
def np_repeat(a, *args, **kwargs):
    return Quantity(np.repeat(a.value, *args, **kwargs), a.dimension)


@implements(np.roll)
def np_roll(a, *args, **kwargs):
    return Quantity(np.roll(a.value, *args, **kwargs), a.dimension)


@implements(np.meshgrid)
def np_meshgrid(*xi, **kwargs):
    xiq = [quantify(x) for x in xi]
    res = np.meshgrid(*(xi.value for xi in xiq), **kwargs)
    return tuple(Quantity(r, q.dimension) for r, q in zip(res, xiq))


@implements(np.real)
def np_real(a):
    return Quantity(np.real(a.value), a.dimension)


@implements(np.isclose)
def np_isclose(a, b, rtol=1e-05, atol=None, equal_nan=False):
    a = quantify(a)
    b = quantify(b)
    if not (a.dimension == b.dimension):
        raise DimensionError(a.dimension, b.dimension)
    if atol is None:
        atol = Quantity(1e-08, a.dimension)
    if not (atol.dimension == a.dimension):
        raise DimensionError(atol.dimension, b.dimension)
    return np.isclose(
        a.value, b.value, rtol=rtol, atol=atol.value, equal_nan=equal_nan
    )


@implements(np.allclose)
def np_allclose(a, b, rtol=1e-05, atol=None, *args, **kwargs):
    # absolute(a - b) <= (atol + rtol * absolute(b))
    a = quantify(a)
    b = quantify(b)
    rtol = quantify(rtol)
    if not (a.dimension == b.dimension):
        raise DimensionError(a.dimension, b.dimension)
    # override default atol from 1e-8 to None, so comparing two quantities of
    # the same dimension uses the "right" atol accordingly
    if atol is None:
        atol = Quantity(1e-8, a.dimension)
    else:
        atol = quantify(atol)
        if not atol.dimension == a.dimension:
            raise DimensionError(atol.dimension, a.dimension)
    if not (Dimension(None) == rtol.dimension):
        raise DimensionError(Dimension(None), rtol.dimension)
    return np.allclose(
        a.value, b.value, rtol=rtol.value, atol=atol.value, *args, **kwargs
    )


@implements(np.ravel)
def np_ravel(a, *args, **kwargs):
    return Quantity(np.ravel(a.value, *args, **kwargs), a.dimension)


@implements(np.reshape)
def np_reshape(a, *args, **kwargs):
    return Quantity(np.reshape(a.value, *args, **kwargs), a.dimension)


@implements(np.interp)
def np_interp(x, xp, fp, left=None, right=None, *args, **kwargs):
    x = quantify(x)
    xp = quantify(xp)
    fp_is_quantity = isinstance(fp, Quantity)
    fp = quantify(fp)
    if not x.dimension == xp.dimension:
        raise DimensionError(x.dimension, xp.dimension)
    if left is not None:
        left = quantify(left)
        if not left.dimension == fp.dimension:
            raise DimensionError(left.dimension, xp.dimension)
        left_v = left.value
    else:
        left_v = left
    if right is not None:
        right = quantify(right)
        if not left.dimension == fp.dimension:
            raise DimensionError(right.dimension, xp.dimension)
        right_v = right.value
    else:
        right_v = right

    res = np.interp(
        x.value, xp.value, fp.value, left_v, right_v, *args, **kwargs
    )
    return Quantity(res, fp.dimension) if fp_is_quantity else res


# @implements(np.asarray)
# def np_array(a):
#    print("np_array implm phyispy")
#    return np.asarray(a.value)*m
#
#
# @implements(np.empty)
# def np_empty(shape, dtype=float, order='C'):
#    return np.empty(shape,dtype=float, order=order)
#


@implements(np.full)
def np_full(shape, fill_value, *args, **kwargs):
    return Quantity(
        np.full(shape, fill_value.value, *args, **kwargs),
        fill_value.dimension,
        # not passing symbol throug since an array cannot be a favunit
        favunit=fill_value.favunit,
    )


@implements(np.fft.fft)
def np_fft_fft(a, *args, **kwargs):
    """Numpy fft.fft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifft)
def np_fft_ifft(a, *args, **kwargs):
    """Numpy fft.ifft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.fft2)
def np_fft_fft2(a, *args, **kwargs):
    """Numpy fft.fft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifft2)
def np_fft_ifft2(a, *args, **kwargs):
    """Numpy fft.ifft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.fftn)
def np_fft_fftn(a, *args, **kwargs):
    """Numpy fft.fftn wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifftn)
def np_fft_ifftn(a, *args, **kwargs):
    """Numpy fft.ifftn wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.rfft)
def np_fft_rfft(a, *args, **kwargs):
    """Numpy fft.rfft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.rfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.irfft)
def np_fft_irfft(a, *args, **kwargs):
    """Numpy fft.irfft wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.irfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.rfft2)
def np_fft_rfft2(a, *args, **kwargs):
    """Numpy fft.rfft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.rfft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.irfft2)
def np_fft_irfft2(a, *args, **kwargs):
    """Numpy fft.irfft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.irfft2(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.rfftn)
def np_fft_rfftn(a, *args, **kwargs):
    """Numpy fft.ifft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.rfftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.irfftn)
def np_fft_irfftn(a, *args, **kwargs):
    """Numpy fft.irfftn wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.irfftn(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.hfft)
def np_fft_hfft(a, *args, **kwargs):
    """Numpy fft.httf wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.hfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ihfft)
def np_fft_ihfft(a, *args, **kwargs):
    """Numpy fft.ihfft2 wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ihfft(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


# @implements(np.fft.fftfreq)
# def np_fft_fftfreq(n, d=1.0):
#    """No need because fftfreq is only a division which is already handled by quantities"""
#


@implements(np.fft.fftshift)
def np_fft_fftshift(a, *args, **kwargs):
    """Numpy fft.fftshift wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.fftshift(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.fft.ifftshift)
def np_fft_ifftshift(a, *args, **kwargs):
    """Numpy fft.ifftshift wrapper for Quantity objects.
    Drop dimension, compute result and add it back."""
    res = np.fft.ifftshift(a.value, *args, **kwargs)
    return Quantity(res, a.dimension)


@implements(np.convolve)
def np_convolve(a, v, *args, **kwargs):
    a = quantify(a)
    v = quantify(v)
    res = np.convolve(a.value, v.value, **kwargs)
    return Quantity(res, a.dimension * v.dimension)


@implements(np.gradient)
def np_gradient(f, *varargs, **kwargs):
    if len(varargs) > 1:
        raise NotImplementedError(
            "High dimension not implemented (but very doable"
        )
    dx = quantify(varargs[0])
    f = quantify(f)
    return Quantity(
        np.gradient(f.value, dx.value, **kwargs), f.dimension / dx.dimension
    )


@implements(np.vstack)
def np_vstack(tup):
    dim = tup[0].dimension
    new_tup = []
    for t in tup:
        t = quantify(t)
        if not t.dimension == dim:
            raise DimensionError(dim, t.dimension)
        new_tup.append(t.value)
    return Quantity(np.vstack(new_tup), dim)


@implements(np.hstack)
def np_hstack(tup):
    dim = tup[0].dimension
    new_tup = []
    for t in tup:
        t = quantify(t)
        if not t.dimension == dim:
            raise DimensionError(dim, t.dimension)
        new_tup.append(t.value)
    return Quantity(np.hstack(new_tup), dim)


@implements(np.where)
def np_where(cond, x, y):
    x = quantify(x)
    y = quantify(y)
    if not x.dimension == y.dimension:
        raise DimensionError(x.dimension, y.dimension)
    return Quantity(np.where(cond, x.value, y.value), x.dimension)


@implements(np.outer)
def np_outer(a, b, *args, **kwargs):
    a = quantify(a)
    b = quantify(b)
    return Quantity(
        np.outer(a.value, b.value, *args, **kwargs), a.dimension * b.dimension
    )


# ufuncs

# 2 in : same dimension ---> out : same dim as in
same_dim_out_2 = (
    "add",
    "subtract",
    "hypot",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "remainder",
    "mod",
    "fmod",
)
# 2 in : same dim ---> out : not a quantity
same_dim_in_2_nodim_out = (
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    "equal",
    "floor_divide",
)
# 1 in :
same_dim_in_1_nodim_out = ("sign", "isfinite", "isinf", "isnan")
# 2 in : any ---> out : depends
skip_2 = (
    "multiply",
    "divide",
    "true_divide",
    "copysign",
    "nextafter",
    "matmul",
)
# 1 in : any ---> out : depends
special_dict = (
    "sqrt",
    "power",
    "reciprocal",
    "square",
    "cbrt",
    "modf",
    "arctan2",
)
# 1 in : no dim ---> out : no dim
no_dim_1 = ("exp", "log", "exp2", "log2", "log10", "expm1", "log1p")
# 2 in : no dim ---> out : no dim
no_dim_2 = ("logaddexp", "logaddexp2")
# 1 in : dimless or angle ---> out : dimless
angle_1 = ("cos", "sin", "tan", "cosh", "sinh", "tanh")
# 1 in : any --> out : same
same_out = (
    "ceil",
    "conjugate",
    "conj",
    "floor",
    "rint",
    "trunc",
    "fabs",
    "negative",
    "absolute",
)
# 1 in : dimless -> out : dimless
inv_angle_1 = (
    "arcsin",
    "arccos",
    "arctan",
    "arcsinh",
    "arccosh",
    "arctanh",
)
# 1 in : dimless -> dimless
deg_rad = ("deg2rad", "rad2deg")


not_implemented_yet = ("isreal", "iscomplex", "signbit", "ldexp", "frexp")
cant_be_implemented = (
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
)


ufunc_2_args = same_dim_out_2 + skip_2 + no_dim_2
unary_ufuncs = (
    no_dim_1
    + angle_1
    + same_out
    + inv_angle_1
    + special_dict
    + deg_rad
    + same_dim_in_1_nodim_out
)
implemented_ufuncs = (
    same_dim_out_2
    + same_dim_in_2_nodim_out
    + same_dim_in_1_nodim_out
    + skip_2
    + special_dict
    + no_dim_1
    + no_dim_2
    + angle_1
    + same_out
    + inv_angle_1
    + deg_rad
)
