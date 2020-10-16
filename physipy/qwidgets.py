import ipywidgets as ipyw
import physipy

import numpy as np
from physipy import quantify, Quantity, DimensionError, units, Dimension
from physipy.quantity.dimension import SI_UNIT_SYMBOL
pi = np.pi



class QuantityFloatSlider():
    
    def __init__(self, value=0.0, min=None, max=None, step=None, description="", disabled=False,
                continuous_update=True, orientation="horizontal", readout=True, readout_format=".1f", 
                constraint_dimension=True):

        # turn value into a Quantity object (if not already)
        self.value_q = quantify(value)
        
        ## Turn min, max and step into Quantity objects, and check dimensions
        # min value
        if min is not None:
            self.min_q = quantify(min)
            if not self.min_q.dimension == self.value_q.dimension:
                raise DimensionError(self.min_q.dimension, self.value_q.dimension)
        else:
            self.min_q = Quantity(0.0, self.value_q.dimension)
        # max value
        if max is not None:
            self.max_q = quantify(max)
            if not self.max_q.dimension == self.value_q.dimension:
                raise DimensionError(self.max_q.dimension, self.value_q.dimension)
        else:
            self.max_q = Quantity(100.0, self.value_q.dimension)        
        # step value
        if step is not None:
            self.step_q = quantify(step)
            if not self.step_q.dimension == self.value_q.dimension:
                raise DimensionError(self.step_q.dimension, self.value_q.dimension)
        else:
            self.step_q = Quantity(0.1, self.value_q.dimension)       
        
        # should probably remove some of these
        self.value          = self.value_q.value
        self.dimension      = self.value_q.dimension
        self.favunit        = self.value_q.favunit
        self.favunit.symbol = self.value_q.favunit.symbol

        self.min_value  = self.min_q.value
        self.max_value  = self.max_q.value
        self.step_value = self.step_q.value
        
        # FloatSlider with values (Float values, not Quantity objects)
        self.floatslider = ipyw.FloatSlider(value=self.value,
                                            min=self.min_value,
                                            max=self.max_value,
                                            step=self.step_value,
                                            description=description,
                                            disabled=disabled,
                                            continuous_update=continuous_update,
                                            orientation=orientation,
                                            readout=False,  # to disable displaying readout value
                                            readout_format=readout_format)
        
        # Label to display the Quantity (str(Quantity) gives a dimensionfull repr)
        self.label = ipyw.Label(value=str(self.value_q))
        
        # list of favorite units (aka "favunit") that have same dimension to select from
        self.favunit_str_list = [u_str for u_str, u_q in units.items() ]#if self.dimension == u_q.dimension]
        self.favunit_str_list.append("None")
        self.favunit_dd = ipyw.Dropdown(
                            options=self.favunit_str_list,
                            value=str(self.favunit.symbol),
                            description='Favunit:',
                         )
        
        
        # selection of favunit
        def update_favunit_on_favunit_dd_change(change):
            self.favunit = units[change.new]
            self.value_q.favunit = self.favunit
            self.label.value = str(self.value_q)
        self.favunit_dd.observe(update_favunit_on_favunit_dd_change, names="value")
        
        
        # update value and label on slider change
        def update_label_on_slider_change(change):
            self.value_q = Quantity(change.new, self.dimension, favunit=self.favunit)
            self.label.value = str(self.value_q)
        self.floatslider.observe(update_label_on_slider_change, names="value")
    
    
        # text area parsing
        self.text = ipyw.Text(value='',
                              placeholder='Type python exp',
                              description='Set to:',
                              disabled=False,
                             continuous_update=True)
        
        # update value on text change
        def update_text(change):
            self.text.value = self.label.value
            
        self.label.observe(update_text, names="value")
        
        # 
        # define a context to parse a python expression into a final quantity
        # user can use "2*pi*m" to define 6.28*m quantity
        context = {**units, "pi":pi}
        def update_values(wdgt):
            expression = wdgt.value
            res = eval(expression, context)
            if res.dimension != self.value_q.dimension:
                self.label.value = str(self.value_q)
                self.floatslider.value = self.label.value
                self.text.value = self.label.value
            else:
                self.value_q = res
                self.label.value = str(self.value_q)

            
            #self.value_q = res
        self.text.on_submit(update_values)
    
        # wrapping in a box #self.label
        self.box = ipyw.HBox([
            self.text,
            self.floatslider,
            self.favunit_dd,
            ])
        
       #ipyw.Label(value=str(self.value_q)),
       #ipyw.Label(value=str(self.value)),
       #ipyw.Label(value=str(self.dimension)),
       #ipyw.Label(value=str(self.favunit)),
       #ipyw.Label(value=str(self.favunit.symbol)),
       #
       #
       ## to fill with attributes
       #self.debug_box = ipyw.VBox([

       #])


    def __repr__(self):
        """Text display"""
        return repr(self.box)


    def _ipython_display_(self, **kwargs):
        """Interactive display"""
        return self.box._ipython_display_(**kwargs)
    

