from dataclasses import dataclass
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


@implements(np.ptp)
def np_ptp(a, *args, **kwargs):
    # peak-to-peak (max - min) keeps the input's dimension
    return Quantity(
        np.ptp(a.value, *args, **kwargs), a.dimension, favunit=a.favunit
    )


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


@implements(np.nanquantile)
def np_nanquantile(a, *args, **kwargs):
    return Quantity(np.nanquantile(a.value, *args, **kwargs), a.dimension)


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

@implements(np.iscomplex)
def np_iscomplex(a):
    return np.iscomplex(a.value)

@implements(np.iscomplexobj)
def np_iscomplexobj(a):
    return np.iscomplexobj(a.value)

@implements(np.isreal)
def np_isreal(a):
    return np.isreal(a.value)

@implements(np.isrealobj)
def np_isrealobj(a):
    return np.isrealobj(a.value)




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


# split family : divide an array into sub-arrays, each keeping the dimension
# (and symbol/favunit) of the input.
def _split_like(func, ary, *args, **kwargs):
    return [
        Quantity(part, ary.dimension, symbol=ary.symbol, favunit=ary.favunit)
        for part in func(ary.value, *args, **kwargs)
    ]


@implements(np.split)
def np_split(ary, *args, **kwargs):
    return _split_like(np.split, ary, *args, **kwargs)


@implements(np.array_split)
def np_array_split(ary, *args, **kwargs):
    return _split_like(np.array_split, ary, *args, **kwargs)


@implements(np.hsplit)
def np_hsplit(ary, *args, **kwargs):
    return _split_like(np.hsplit, ary, *args, **kwargs)


@implements(np.vsplit)
def np_vsplit(ary, *args, **kwargs):
    return _split_like(np.vsplit, ary, *args, **kwargs)


@implements(np.dsplit)
def np_dsplit(ary, *args, **kwargs):
    return _split_like(np.dsplit, ary, *args, **kwargs)


@implements(np.linalg.norm)
def np_linalg_norm(x, *args, **kwargs):
    return Quantity(np.linalg.norm(x.value, *args, **kwargs), x.dimension)


@implements(np.linalg.lstsq)
def np_linalg_lstsq(a, b, **kwargs):
    a = quantify(a)
    b = quantify(b)
    # np.linalg.lstsq returns a 4-tuple, not a single array : the
    # least-squares solution, the residuals, the rank, and the singular
    # values. Each carries its own dimension and must be wrapped separately.
    solution, residuals, rank, singular_values = np.linalg.lstsq(
        a.value, b.value, **kwargs
    )
    return (
        # solves a @ x = b  ->  x has dimension b / a
        Quantity(solution, b.dimension / a.dimension),
        # residuals are sums of squared errors ||b - a @ x||^2
        Quantity(residuals, b.dimension**2),
        # rank is a plain integer
        rank,
        # singular values of a share a's dimension
        Quantity(singular_values, a.dimension),
    )


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


@implements(np.diagflat)
def np_diagflat(v, *args, **kwargs):
    return Quantity(np.diagflat(v.value, *args, **kwargs), v.dimension)


@implements(np.tril)
def np_tril(m, *args, **kwargs):
    return Quantity(np.tril(m.value, *args, **kwargs), m.dimension)


@implements(np.triu)
def np_triu(m, *args, **kwargs):
    return Quantity(np.triu(m.value, *args, **kwargs), m.dimension)


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


@implements(np.cumulative_sum)
def np_cumulative_sum(a, *args, **kwargs):
    # numpy>=2.0 array-API spelling of cumsum : same dimension as input
    return Quantity(np.cumulative_sum(a.value, *args, **kwargs), a.dimension)


@implements(np.cumprod)
def np_cumprod(a, *args, **kwargs):
    # each partial product would carry a different power of the dimension, so
    # the result is only representable as a single Quantity when dimensionless
    if not a.is_dimensionless():
        raise DimensionError(a.dimension, Dimension(None))
    return Quantity(np.cumprod(a.value, *args, **kwargs), a.dimension)


@implements(np.cumulative_prod)
def np_cumulative_prod(a, *args, **kwargs):
    if not a.is_dimensionless():
        raise DimensionError(a.dimension, Dimension(None))
    return Quantity(np.cumulative_prod(a.value, *args, **kwargs), a.dimension)


@implements(np.histogram)
def np_histogram(a, bins=10, range=None, density=None, weights=None, **kwargs):
    a = quantify(a)
    if range is not None:
        low, high = quantify(range[0]), quantify(range[1])
        if not low.dimension == high.dimension:
            raise DimensionError(low.dimension, high.dimension)
        if not low.dimension == a.dimension:
            raise DimensionError(a.dimension, low.dimension)
        # forward the bare magnitudes : np.histogram cannot consume Quantity
        # edges (np.result_type has no implementation for them).
        range = (low.value, high.value)
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
    # prepend/append must be stripped to their magnitude before the inner
    # np.diff call : forwarding a Quantity into np.diff(a.value, ...) would
    # re-dispatch here with a bare ndarray and fail on `a.dimension`.
    prepend_value = prepend
    if prepend is not np._NoValue:
        prepend = quantify(prepend)
        if prepend.dimension != a.dimension:
            raise DimensionError(a.dimension, prepend.dimension)
        prepend_value = prepend.value
    append_value = append
    if append is not np._NoValue:
        append = quantify(append)
        if append.dimension != a.dimension:
            raise DimensionError(a.dimension, append.dimension)
        append_value = append.value
    return Quantity(
        np.diff(
            a.value, n=n, axis=axis, prepend=prepend_value, append=append_value
        ),
        a.dimension,
    )


# --- dimension-preserving transforms : run on the magnitudes, keep the unit ---


@implements(np.ediff1d)
def np_ediff1d(ary, to_end=None, to_begin=None):
    ary = quantify(ary)
    # to_end / to_begin are inserted verbatim, so they must share the dimension
    extras = {}
    for name, val in (("to_end", to_end), ("to_begin", to_begin)):
        if val is not None:
            val = quantify(val)
            if not val.dimension == ary.dimension:
                raise DimensionError(ary.dimension, val.dimension)
            extras[name] = val.value
    return Quantity(np.ediff1d(ary.value, **extras), ary.dimension)


@implements(np.nan_to_num)
def np_nan_to_num(x, *args, **kwargs):
    return Quantity(np.nan_to_num(x.value, *args, **kwargs), x.dimension)


@implements(np.trim_zeros)
def np_trim_zeros(filt, *args, **kwargs):
    return Quantity(np.trim_zeros(filt.value, *args, **kwargs), filt.dimension)


@implements(np.fix)
def np_fix(x, *args, **kwargs):
    return Quantity(np.fix(x.value, *args, **kwargs), x.dimension)


@implements(np.real_if_close)
def np_real_if_close(a, *args, **kwargs):
    return Quantity(np.real_if_close(a.value, *args, **kwargs), a.dimension)


@implements(np.sort_complex)
def np_sort_complex(a):
    return Quantity(np.sort_complex(a.value), a.dimension)


@implements(np.resize)
def np_resize(a, new_shape):
    return Quantity(np.resize(a.value, new_shape), a.dimension)


@implements(np.take_along_axis)
def np_take_along_axis(arr, indices, axis):
    return Quantity(
        np.take_along_axis(arr.value, indices, axis), arr.dimension
    )


@implements(np.unstack)
def np_unstack(x, *args, **kwargs):
    return tuple(
        Quantity(part, x.dimension, symbol=x.symbol, favunit=x.favunit)
        for part in np.unstack(x.value, *args, **kwargs)
    )


def _block_values(node):
    # recurse np.block's nested list/tuple grid, collecting one shared dimension
    if isinstance(node, (list, tuple)):
        children = [_block_values(child) for child in node]
        dimension = children[0][1]
        for _, dim in children[1:]:
            if not dim == dimension:
                raise DimensionError(dimension, dim)
        return [value for value, _ in children], dimension
    node = quantify(node)
    return node.value, node.dimension


@implements(np.block)
def np_block(arrays):
    values, dimension = _block_values(arrays)
    return Quantity(np.block(values), dimension)


@implements(np.extract)
def np_extract(condition, arr):
    arr = quantify(arr)
    return Quantity(np.extract(condition, arr.value), arr.dimension)


@implements(np.choose)
def np_choose(a, choices, *args, **kwargs):
    qchoices = [quantify(c) for c in choices]
    dimension = qchoices[0].dimension
    for c in qchoices[1:]:
        if not c.dimension == dimension:
            raise DimensionError(dimension, c.dimension)
    index = a.value if isinstance(a, Quantity) else a
    return Quantity(
        np.choose(index, [c.value for c in qchoices], *args, **kwargs),
        dimension,
    )


# --- index-returning : the result is a plain integer array, no dimension ---


@implements(np.nonzero)
def np_nonzero(a):
    return np.nonzero(a.value)


@implements(np.argwhere)
def np_argwhere(a):
    return np.argwhere(a.value)


@implements(np.flatnonzero)
def np_flatnonzero(a):
    return np.flatnonzero(a.value)


# --- in-place writes : mutate the magnitudes through `.value`, return None ---
# (the inserted values must match the target's dimension)


@implements(np.put_along_axis)
def np_put_along_axis(arr, indices, values, axis):
    arr = quantify(arr)
    values = quantify(values)
    if not arr.dimension == values.dimension:
        raise DimensionError(arr.dimension, values.dimension)
    np.put_along_axis(arr.value, indices, values.value, axis)


@implements(np.place)
def np_place(arr, mask, vals):
    arr = quantify(arr)
    vals = quantify(vals)
    if not arr.dimension == vals.dimension:
        raise DimensionError(arr.dimension, vals.dimension)
    np.place(arr.value, mask, vals.value)


@implements(np.put)
def np_put(a, ind, v, *args, **kwargs):
    a = quantify(a)
    v = quantify(v)
    if not a.dimension == v.dimension:
        raise DimensionError(a.dimension, v.dimension)
    np.put(a.value, ind, v.value, *args, **kwargs)


@implements(np.putmask)
def np_putmask(a, mask, values):
    a = quantify(a)
    values = quantify(values)
    if not a.dimension == values.dimension:
        raise DimensionError(a.dimension, values.dimension)
    np.putmask(a.value, mask, values.value)


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


@implements(np.quantile)
def np_quantile(a, *args, **kwargs):
    return Quantity(np.quantile(a.value, *args, **kwargs), a.dimension)


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


@implements(np.delete)
def np_delete(arr, obj, *args, **kwargs):
    return Quantity(
        np.delete(arr.value, obj, *args, **kwargs),
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

@implements(np.sinc)
def np_sinc(x):
    if x.dimension!=Dimension(None):
        raise DimensionError(x.dimension, Dimension(None))
    return np.sinc(x.value)



# np.trapz was renamed np.trapezoid in NumPy 2.0 (np.trapz still exists there
# but is deprecated and slated for removal); register whichever names this
# NumPy version exposes.
if hasattr(np, "trapz"):
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

if hasattr(np, "trapezoid"):
    @implements(np.trapezoid)
    def np_trapezoid(q, x=None, dx=1, **kwargs):
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


# Canonical, version-agnostic trapezoid integrator. Prefer the modern name;
# fall back to the deprecated np.trapz on NumPy < 2.0. Use this (instead of
# hardcoding either name) in physipy code and tests so they stay portable
# across NumPy versions. It dispatches through __array_function__, so Quantity
# units are preserved either way.
# getattr (not np.trapz directly) so NumPy 2.x stubs, which dropped trapz,
# don't flag a missing attribute.
trapezoid: Callable = getattr(np, "trapezoid", None) or getattr(np, "trapz")


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


@implements(np.moveaxis)
def np_moveaxis(a, source, destination):
    return Quantity(
        np.moveaxis(a.value, source, destination),
        a.dimension,
        favunit=a.favunit,
        symbol=a.symbol,
    )


@implements(np.swapaxes)
def np_swapaxes(a, axis1, axis2):
    return Quantity(
        np.swapaxes(a.value, axis1, axis2),
        a.dimension,
        favunit=a.favunit,
        symbol=a.symbol,
    )


@implements(np.rollaxis)
def np_rollaxis(a, axis, start=0):
    return Quantity(
        np.rollaxis(a.value, axis, start),
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

@implements(np.pad)
def np_pad(array, pad_width, mode="constant", **kwargs):
    if "constant_values" in kwargs:
        cval = quantify(kwargs['constant_values'])
        if cval.dimension != array.dimension:
            raise DimensionError(array.dimension, cval.dimension)
        kwargs["constant_values"] = cval.value
    if "end_values" in kwargs:
        eval = quantify(kwargs['end_values'])
        if eval.dimension != array.dimension:
            raise DimensionError(array.dimension, eval.dimension)
        kwargs["end_values"] = eval.value
    padded = np.pad(array.value, pad_width, mode=mode, **kwargs)
    return Quantity(padded, array.dimension, favunit=array.favunit)

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
    return tuple(Quantity(r, q.dimension).set_favunit(q.favunit) for r, q in zip(res, xiq))


@implements(np.real)
def np_real(a):
    return Quantity(np.real(a.value), a.dimension)


@implements(np.imag)
def np_imag(a):
    return Quantity(np.imag(a.value), a.dimension)


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
    return Quantity(res, fp.dimension).set_favunit(fp.favunit) if fp_is_quantity else res


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


# ---------------------------------------------------------------------------
# Introspection : which numpy functions does physipy support ?
# ---------------------------------------------------------------------------
# Two dispatch mechanisms feed this :
#   - __array_function__ handlers, registered in HANDLED_FUNCTIONS via
#     @implements (np.concatenate, np.unique, np.linalg.norm, np.fft.*, ...)
#   - __array_ufunc__ ufuncs, listed by name in implemented_ufuncs
#     (np.add, np.sin, np.sqrt, ...)

# numpy namespaces scanned to enumerate the public, non-ufunc function surface.
_NUMPY_FUNC_NAMESPACES = (("", np), ("linalg", np.linalg), ("fft", np.fft))

# Array functions that can't be expressed with physipy's data model. A Quantity
# is a single ndarray of magnitudes sharing ONE dimension, so any function whose
# natural output is a single array with *per-element heterogeneous* dimensions
# has no faithful representation. The polynomial-coefficient family is the main
# example : a 1-D coefficient array is inherently heterogeneous (c_n carries y,
# c_{n-1} carries y/x, ...), which is exactly why np.polyfit returns a *tuple* of
# separate Quantities rather than one array. These are reported as
# ``not_applicable`` (not ``missing``) so coverage reflects only dimension-
# relevant functions. See also the ufunc-level :data:`cant_be_implemented`.
cant_be_implemented_functions = (
    "vander",  # columns x**0, x**1, ... x**(n-1) : one dimension per column
    "poly",  # coefficients from roots : heterogeneous coefficient array
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyint",
    "polyder",
    "roots",  # input is a (heterogeneous) coefficient array
)


def _qualified(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _public_numpy_functions() -> dict[str, Callable]:
    """Map ``qualified_name -> callable`` for numpy's public, non-ufunc functions."""
    funcs: dict[str, Callable] = {}
    for prefix, ns in _NUMPY_FUNC_NAMESPACES:
        for name in getattr(ns, "__all__", dir(ns)):
            if name.startswith("_"):
                continue
            obj = getattr(ns, name, None)
            if not callable(obj) or isinstance(obj, (type, np.ufunc)):
                continue
            funcs[_qualified(prefix, name)] = obj
    return funcs


def _canonical_numpy_ufuncs() -> dict[str, np.ufunc]:
    """Map ``canonical_name -> ufunc`` for every numpy ufunc, deduped over aliases.

    numpy>=2.0 exposes alias ufuncs (``np.abs`` *is* ``np.absolute``, ``np.acos``
    *is* ``np.arccos``, ...). Aliases share the same object, so keying on
    ``ufunc.__name__`` collapses them onto their canonical name -- which is also
    the name physipy dispatches on in ``Quantity.__array_ufunc__``.
    """
    ufuncs: dict[str, np.ufunc] = {}
    for name in dir(np):
        obj = getattr(np, name)
        if isinstance(obj, np.ufunc):
            ufuncs[obj.__name__] = obj
    return ufuncs


def supported_numpy_functions(*, names: bool = False):
    """Return every numpy callable physipy can apply to ``Quantity`` objects.

    Unifies both dispatch mechanisms : the ``__array_function__`` handlers
    registered via :func:`implements` and the ``__array_ufunc__`` ufuncs listed
    in :data:`implemented_ufuncs`.

    Parameters
    ----------
    names : bool, optional
        When ``True``, return a sorted list of names instead of the callables.

    Returns
    -------
    set[Callable] or list[str]
        The supported callables, or their sorted names when ``names`` is set.
    """
    ufuncs = {getattr(np, n) for n in implemented_ufuncs if hasattr(np, n)}
    supported = set(HANDLED_FUNCTIONS) | ufuncs
    if names:
        return sorted(getattr(f, "__name__", repr(f)) for f in supported)
    return supported


@dataclass(frozen=True)
class _CoverageGroup:
    """Implemented vs. missing names for one numpy dispatch family."""

    implemented: tuple[str, ...]
    missing: tuple[str, ...]
    not_applicable: tuple[str, ...] = ()

    @property
    def n_relevant(self) -> int:
        """Implemented + missing (excludes the not-applicable functions)."""
        return len(self.implemented) + len(self.missing)

    @property
    def ratio(self) -> float:
        """Fraction of dimension-relevant functions that are implemented."""
        return len(self.implemented) / self.n_relevant if self.n_relevant else 1.0


@dataclass(frozen=True)
class NumpyCoverage:
    """Snapshot of physipy's numpy coverage, computed against the running numpy."""

    ufuncs: _CoverageGroup
    array_functions: _CoverageGroup
    numpy_version: str

    def summary(self) -> str:
        lines = [f"physipy numpy coverage (numpy {self.numpy_version})"]
        for label, grp in (
            ("ufuncs", self.ufuncs),
            ("array functions", self.array_functions),
        ):
            extra = f", {len(grp.not_applicable)} n/a" if grp.not_applicable else ""
            lines.append(
                f"  {label:16s}: {len(grp.implemented):3d}/{grp.n_relevant:3d} "
                f"implemented ({grp.ratio:6.1%}){extra}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def numpy_coverage() -> NumpyCoverage:
    """Compare the running numpy's public API against what physipy implements.

    ufuncs are deduplicated over numpy-2.0 aliases and split into ``implemented``
    (in :data:`implemented_ufuncs`), ``not_applicable`` (declared in
    :data:`cant_be_implemented`) and ``missing`` (everything else).

    array functions are the public, non-ufunc callables of ``numpy``,
    ``numpy.linalg`` and ``numpy.fft``, split into ``implemented`` (in
    :data:`HANDLED_FUNCTIONS`), ``not_applicable`` (declared in
    :data:`cant_be_implemented_functions`) and ``missing`` (everything else).
    """
    # ufuncs
    canonical = _canonical_numpy_ufuncs()
    impl_set = set(implemented_ufuncs)
    na_set = set(cant_be_implemented)
    uf_impl, uf_missing, uf_na = [], [], []
    for name in canonical:
        if name in impl_set:
            uf_impl.append(name)
        elif name in na_set:
            uf_na.append(name)
        else:
            uf_missing.append(name)

    # array functions
    public = _public_numpy_functions()
    handled = set(HANDLED_FUNCTIONS)
    fn_na_set = set(cant_be_implemented_functions)
    fn_impl, fn_missing, fn_na = [], [], []
    for name, obj in public.items():
        if obj in handled:
            fn_impl.append(name)
        elif name in fn_na_set:
            fn_na.append(name)
        else:
            fn_missing.append(name)

    return NumpyCoverage(
        ufuncs=_CoverageGroup(
            implemented=tuple(sorted(uf_impl)),
            missing=tuple(sorted(uf_missing)),
            not_applicable=tuple(sorted(uf_na)),
        ),
        array_functions=_CoverageGroup(
            implemented=tuple(sorted(fn_impl)),
            missing=tuple(sorted(fn_missing)),
            not_applicable=tuple(sorted(fn_na)),
        ),
        numpy_version=np.__version__,
    )
