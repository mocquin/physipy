import ipywidgets as ipyw

from .qipywidgets import QuantityText, QuantityTextSlider

from physipy import set_favunit, all_units, Quantity
from physipy.quantity.utils import hard_favunit
from IPython.display import display


def ui_widget_decorate(inits_values, kind="Text"):
    """
    Decorator to create a UI for a function.

     Example
     -------

     # as a function
     def disk_PSA_cart(x, y, R, h):
         return x*y*R/h

     ui = ui_widget_decorate([("x", 1*m),
                              ("y", 1*m),
                              ("R", 1*m, "Radius"),
                              ("h", 1*m, "distance")])(disk_PSA_carth)

    # as a decorator
     @ui_widget_decorate([("x", 1*m),
                          ("y", 1*m),
                          ("R", 1*m, "Radius"),    # alias Radius
                          ("h", 1*m, "distance")]) # alias distance
     def disk_PSA_cart(x, y, R, h):
         return x*y*R/h

     """
    if kind == "Text":
        w = QuantityText
    elif kind == "TextSlider":
        w = QuantityTextSlider
    else:
        raise ValueError()

    def decorator_func(func):
        # create a widget list for all inputs,
        # with an initial value, and a label
        qwidget_list = []

        for initq in inits_values:
            # param name
            pname = initq[0]

            # initial value
            initial_value = initq[1]
            # check if the initial is a favunit, and set itself as favunit
            if hard_favunit(initial_value, all_units.values()):
                initial_value.favunit = initial_value

            # if provided, use alias instead of param name
            if len(initq) == 3:
                pname = initq[2]

            # the widget is created here
            widget = w(initial_value,
                       description=pname)

            qwidget_list.append(widget)

        # wrap function to display result
        def display_func(*args, **kwargs):
            res = func(*args, **kwargs)
            display(res)
            return res

        # wrap all inputs widgets in a VBox
        input_ui = ipyw.VBox(qwidget_list)

        # create output widget, using inputs widgets
        out = ipyw.interactive_output(display_func, {
                                      k: qwidget_list[i] for i, k in enumerate([l[0] for l in inits_values])})

        # if func has a "name" attribute, create a Label for display, else use
        # default function __name__
        if hasattr(func, "name"):
            wlabel = ipyw.Label(func.name + ":")
        else:
            wlabel = ipyw.Label(func.__name__ + ":")

        # if func has a "latex" attribute, append it to Label
        if hasattr(func, "latex"):
            wlabel = ipyw.HBox([wlabel, ipyw.Label(func.latex)])

        # wrap all ui with Labels, inputs, and result
        ui = ipyw.VBox([wlabel, input_ui, out])

        return ui
    return decorator_func


def ui_widget_decorate_from_annotations(func, kind="Text"):
    """
     Example
     -------
     def disk_PSA_cart(x:mm, y:mm, R:mm, h:mm)-> msr:
         return x*y*R/h

     ui = ui_widget_decorate_from_annotations(disk_PSA_carth)
     ui

     """

    # recreating an inits_values list based on annotations
    # then reusing ui_widget_decorate

    import inspect
    sig = inspect.signature(func)
    inits_values = []

    for k, v in sig.parameters.items():
        # only turn quantity annotations to widgets,
        # standards params are "passed"
        if isinstance(v.annotation, Quantity):
            inits_values.append((v.name,
                                 v.annotation,  # initial value
                                 v.name))

    # fyi : to get retun annotation
    # sig.return_annotation
    if not sig.return_annotation == inspect._empty:
        func = set_favunit(sig.return_annotation)(func)

    return ui_widget_decorate(inits_values, kind=kind)(func)


def FunctionUI(tab_name, function_dict, kind="Text"):
    acc = ipyw.Accordion()

    sections_uis = []
    i = 0
    for section_name, section_functions_list in function_dict.items():
        function_uis = []
        # loop over all functions in section and generate ui
        for func in section_functions_list:
            ui = ui_widget_decorate_from_annotations(func, kind=kind)
            function_uis.append(ui)
        box = ipyw.VBox(function_uis)
        sections_uis.append(box)
        acc.set_title(i, section_name)
        i += 1

    acc.children = sections_uis

    tab = ipyw.Tab()
    tab.children = [acc]
    tab.set_title(0, tab_name)

    return tab
