from ipywidgets import Layout
from traitlets import TraitError
from physipy import quantify, Dimension, Quantity, units
import ipywidgets as ipyw
import traitlets
from numpy import pi



class QuantityText(ipyw.Box, ipyw.ValueWidget, ipyw.DOMWidget):
    # dimension trait : a Dimension instance
    dimension = traitlets.Instance(Dimension, allow_none=False)
    # value trait : a Quantity instance
    value = traitlets.Instance(Quantity, allow_none=False)
    # value_number : float value of quantity
    value_number = traitlets.Float(allow_none=True)
    # string trait text
    display_val = traitlets.Unicode(allow_none=True)
    # description
    description = traitlets.Unicode(allow_none=True)
    
    def __init__(self, value=0.0, disabled=False, 
                 continuous_update=True, description="Quantity:",
                 fixed_dimension=False,
                 **kwargs):
        
        # context for parsing
        self.context = {**units, "pi":pi}
        self.description = description
        

        # quantity work
        # set dimension
        value = quantify(value)
        self.dimension = value.dimension
        # if true, any change in value must have same dimension as initial dimension
        self.fixed_dimension = fixed_dimension
        
        # set quantity
        self.value = value
        # set text widget
        self.text = ipyw.Text(value=str(self.value),
                              placeholder='Type python exp',
                              description=self.description,#'Set to:',
                              disabled=disabled,
                              continuous_update=continuous_update,
                              layout=Layout(width='auto',
                                            margin="0px 0px 0px 0px",
                                            padding="0px 0px 0px 0px",
                                            border="solid gray"),
                             style={'description_width': '130px'})
        
        # link text value and display_val unicode trait
        traitlets.link((self.text, "value"), (self, "display_val"))
        super().__init__(**kwargs)
        self.children = [self.text]

        # on_submit observe
        def text_update_values(wdgt):
            # get expression entered
            expression = wdgt.value
            # eval expression with unit context
            try:
                res = eval(expression, self.context)
                res = quantify(res)
                # update quantity value
                self.value = res
                # update display_value
                self.display_val = str(self.value)
            except:
                self.display_val = str(self.value)
        self.text.on_submit(text_update_values)

    # update value_number and text on quantity value change
    @traitlets.observe("value")
    def _update_display_val(self, proposal):
        self.value_number = self.value.value
        self.dimension = self.value.dimension
        self.display_val = f'{str(self.value)}'

    @traitlets.validate('value')
    def _valid_value(self, proposal):
        if self.fixed_dimension and proposal['value'].dimension != self.dimension:
            raise TraitError('value and parity should be consistent')
        return proposal['value']
    

class FDQuantityText(QuantityText):
    
    def __init__(self, value=0.0, disabled=False, 
                 continuous_update=True, description="Quantity:", *args,
                 **kwargs):
        
        super().__init__(value=value, disabled=disabled,
                         continuous_update=continuous_update, 
                         description=description,
                         fixed_dimension=True, 
                         *args, 
                         **kwargs)


    
    
        
def ui_widget_decorate(inits_values):
    """inits_values contains list of tuples : 
     - quantity init value
     - str description
     - name from signature"""

    def decorator_func(func):
        qwidget_list = []
        for initq in inits_values:
            qwidget_list.append(QuantityText(initq[1], description=initq[2]))
        def display_func(*args, **kwargs):
            res = func(*args, **kwargs)
            display(res)
            return res
        input_ui = ipyw.VBox(qwidget_list)
        out = ipyw.interactive_output(display_func,
                                     {k:qwidget_list[i] for i, k in enumerate([l[0] for l in inits_values])})
        if hasattr(func, "name"):
            wlabel = ipyw.Label(func.name)
        else:
            wlabel = ipyw.Label(func._name__)
        if hasattr(func, "latex"):
            wlabel = ipyw.HBox([wlabel, ipyw.Label(func.latex)])
        ui = ipyw.VBox([wlabel, input_ui, out])
        return ui        
    return decorator_func
    
            