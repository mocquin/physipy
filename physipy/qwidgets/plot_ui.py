from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as ipyw

import physipy
from physipy import *
from physipy.quantity.utils import *
from physipy.quantity.calculus import ndvectorize
from physipy.qwidgets.qipywidgets import QuantityTextSlider

setup_matplotlib()


class WrappedFunction1D():

    def __init__(self, func, xmin, xmax, *args, num=100,
                 xfavunit=None, yfavunit=None,
                 xlim=None, ylim=None, **kwargs):

        self.func = func
        self.name = func.name
        self.xfavunit = xfavunit
        self.yfavunit = yfavunit
        self.xmin = xmin
        self.xmax = xmax
        self.num = num

        axis_list = ["stretch", "auto", "fixed"]
        self.xaxis_ddw = ipyw.Dropdown(
            options=axis_list, description="X-Axis scaling:")
        self.yaxis_ddw = ipyw.Dropdown(
            options=axis_list, description="Y-Axis scaling:")

        # Todo : handle xlim/ylim with fixed, stretched, etc

        self.ech_x = np.linspace(xmin, xmax, num=num)
        # get favunit for inputs
        import inspect
        sig = inspect.signature(func)
        if not sig.return_annotation == inspect._empty:
            init_values = []
            for k, v in sig.parameters.items():
                init_values.append((v.name, v.annotation, v.name))
            # add favunit for x for plotting
            self.ech_x.favunit = init_values[0][1]

        self.params = args
        self.params_slider_dict = OrderedDict()

        self.state_dict = {}

        self.pargs = {}
        self.sliders_dict = {}

        def _update_data(change):
            # print("update data")
            data = self.data()
            self.line.set_label(self.label)
            self.line.set_data(data)
            _update_cursor_data(None)

            self.ax.relim()
            cur_xlims = self.ax.get_xlim()
            cur_ylims = self.ax.get_ylim()

            # X
            if self.xaxis_scale == "auto":
                self.ax.autoscale_view(scaley=False)
            elif self.xaxis_scale == "stretch":
                new_lims = [self.ax.dataLim.x0,
                            self.ax.dataLim.x0 + self.ax.dataLim.width]
                new_lims = [
                    new_lims[0] if new_lims[0] < cur_xlims[0] else cur_xlims[0],
                    new_lims[1] if new_lims[1] > cur_xlims[1] else cur_xlims[1],
                ]
                self.ax.set_xlim(new_lims)

            if self.yaxis_scale == "auto":
                self.ax.autoscale_view(scalex=False)
            elif self.yaxis_scale == "stretch":
                new_lims = [self.ax.dataLim.y0,
                            self.ax.dataLim.y0 + self.ax.dataLim.height]
                new_lims = [
                    new_lims[0] if new_lims[0] < cur_ylims[0] else cur_ylims[0],
                    new_lims[1] if new_lims[1] > cur_ylims[1] else cur_ylims[1],
                ]
                self.ax.set_ylim(new_lims)

            self.ax.legend()
            # self.ax.autoscale_view()

        sliders_list = []
        for k, v in kwargs.items():
            self.pargs[k] = v
            # slider = ipyw.FloatSlider(v[0], min=v[0], max=v[-1], description=k, step=0.001)
            slider = QuantityTextSlider(v[0],
                                        min=v[0],
                                        max=v[-1],
                                        description=k,
                                        step=(v[-1] - v[0]) / 1000)  # we override a default step
            slider.observe(_update_data, names="value")

            self.sliders_dict[k] = slider

            sliders_list.append(slider)

        def _update_cursor_data(change):
            y = self.func(self.get_xcursor(), **self.get_pvalues())
            self.cursor_hline.set_data(
                asqarray([self.xmin, self.xmax]), asqarray([y, y]))
            self.cursor_vline.set_data(
                asqarray([self.get_xcursor(), self.get_xcursor()]), asqarray([0 * y, y]))

        self.cursor_slider = QuantityTextSlider(
            xmin, min=xmin, max=xmax, description="Cursor", step=(
                xmax - xmin) / 1000)
        self.cursor_slider.observe(_update_cursor_data, names="value")

        # print(self.sliders_list)
        self.sliders_list = sliders_list
        self.sliders_box = ipyw.VBox(
            sliders_list + [self.cursor_slider] + [self.xaxis_ddw, self.yaxis_ddw])
        self.out_w = ipyw.Output()

        # print(self.sliders_box)
        self.plot()
        self.plot_cursor()

        # self.state = {k:v for (k,v) in zip(self.)}

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def xaxis_scale(self):
        return self.xaxis_ddw.value

    @property
    def yaxis_scale(self):
        return self.yaxis_ddw.value

    def get_xcursor(self):
        return self.cursor_slider.value

    def add_integral(self, ximin, ximax):
        from physipy import quad
        args = tuple((v for v in self.get_pvalues().values()))
        func = self.func
        integ, prec = quad(func, ximin, ximax, args=args)

        from matplotlib.patches import Polygon

        def compute_integral_polygon(xmin, xmax, func, ax=self.ax, **kwargs):
            xmin = quantify(xmin)
            xmax = quantify(xmax)
            if ax is None:
                ax = plt.gca()
            ix = np.linspace(xmin, xmax)
            # iy = func(ix, **kwargs)
            try:
                iy = func(ix, *args, **kwargs)
            except BaseException:
                iy = physipy.quantity.utils.asqarray(
                    [func(x, *args, **kwargs) for x in ix])

            verts = [(xmin.value, 0),
                     *zip((ix.value), iy.value),
                     (xmax.value, 0)]
            poly = Polygon(verts, facecolor='0.9', edgecolor='0.5', alpha=0.5)
            ax.text(0.5 * (xmin.value + xmax.value),
                    np.max(iy.value / 5),
                    r"$\int_{x=" + f"{xmin}" + "}^{x=" + f"{xmax}" + "}" +
                    f"{self.label}" + "\\mathrm{d}x$=" + f"{integ:.2f}",
                    horizontalalignment='center',
                    fontsize=9)
            res = ax.add_patch(poly)
            return res
        polygon = compute_integral_polygon(ximin, ximax, func)
        self.integral_polygon = polygon
        return self.integral_polygon

    @property
    def label(self):
        params = ",".join(
            [
                k + "=" + f"{v.value:~}" for k,
                v in zip(
                    self.sliders_dict.keys(),
                    self.sliders_dict.values())])
        return f"{self.name}({params})"

    def get_pvalue(self, pname):
        return self.sliders_dict[pname].value

    def get_pvalues(self):
        res = {k: self.sliders_dict[k].value for k in self.sliders_dict.keys()}
        return res

    def __repr__(self):
        # display(self.sliders_box)
        display(ipyw.HBox([self.out_w,
                           self.sliders_box]))
        return ""

    def data(self):
        return (self.ech_x, self.func(self.ech_x, **self.get_pvalues()))

    def plot(self, ax=None, **kwargs):
        with self.out_w:
            if ax is None:
                fig, ax = plt.subplots()
                ax.grid("both")
            line, = ax.plot(*self.data(), label=self.label, **kwargs)
            self.ax = ax
            self.fig = fig
            self.fig.tight_layout()
            self.line = line
            ax.legend()

    def plot_cursor(self, **kwargs):
        ax = self.ax
        xx = (self.get_xcursor(), self.get_pvalues())
        y = self.func(self.get_xcursor(), **self.get_pvalues())
        with self.out_w:
            self.cursor_hline, = ax.plot(
                asqarray([self.xmin, self.xmax]), asqarray([y, y]))
            self.cursor_vline, = ax.plot(
                asqarray([self.get_xcursor(), self.get_xcursor()]), asqarray([0 * y, y]))


if __name__ == "__main__":

    @name_eq("Myfunc")
    def func(x1, x2, x3):
        return x1 * x2 + 3 * x3

    wf = WrappedFunction1D(func, 0 * s, 5 * s,
                           x2=(0 * m, 5 * m),
                           x3=(0 * m * s, 5 * m * s))
