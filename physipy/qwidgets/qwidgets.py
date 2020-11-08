import ipywidgets as ipyw
import physipy
from ipywidgets import Layout


import numpy as np
from physipy import quantify, Quantity, DimensionError, units, Dimension
from physipy.quantity.dimension import SI_UNIT_SYMBOL
pi = np.pi


"""
TODO : 
 - parsing : add possibility to parse '5m' or '5 m' in text area
"""


class QuantityFloatSlider():
    
    def __init__(self, value=0.0, min=None, max=None, step=None, description="", disabled=False,
                continuous_update=True, orientation="horizontal", readout=True, readout_format=".1f", 
                constraint_dimension=True):

        
        #### READ PARAMS
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
        
        
        ####### WIDGETS
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
                                            readout_format=readout_format,
                                            layout=Layout(width="50%",
                                                          margin="0px",
                                                          border="solid red"))
        self.labelsmin = ipyw.Label(value=str(Quantity(self.floatslider.min,
                                                       self.dimension,
                                                       favunit=self.favunit)),
                                    layout=Layout(margin="0px",
                                                  padding='0px 0px 0px 0px',
                                                  border="solid gray"))
        self.labelsmax = ipyw.Label(value=str(Quantity(self.floatslider.max,
                                                       self.dimension,
                                                       favunit=self.favunit)),
                                    layout=Layout(margin="0px",
                                                  border="solid red"))
        self.slider_box = ipyw.HBox([
            self.labelsmin,
            self.floatslider,
            self.labelsmax,
            ],
            layout=Layout(width="300px",
                         border="solid blue",
                         padding="0px 0px 0px 0px",
                         margin="0px"))
        
        # DROPDOWN for favunit list of favorite units (aka "favunit") that have same dimension to select from
        self.favunit_str_list = [u_str for u_str, u_q in units.items() ]#if self.dimension == u_q.dimension]
        self.favunit_str_list.append("-")
        self.favunit_dd = ipyw.Dropdown(
                            options=self.favunit_str_list,
                            value=str(self.favunit.symbol),
                            description='Favunit:',
                            layout=Layout(width="20%",
                                          padding="0px 0px 0px 0px",
                                          margin="0px 0px 0px 0px",
                                          border="solid red")
                                    )


        # TEXT area parsing
        self.text = ipyw.Text(value=str(self.value_q),
                              placeholder='Type python exp',
                              description='',#'Set to:',
                              disabled=False,
                              continuous_update=True,
                              layout=Layout(width='30%',
                                            margin="0px 0px 0px 0px",
                                            padding="0px 0px 0px 0px",
                                            border="solid red"))

        # wrapping in a box #self.label
        self.box = ipyw.HBox([
            self.text,
            self.slider_box,
            self.favunit_dd,
            ],
        layout=Layout(widht='100%',
                     margin="0px 0px 0px 0px",
                     padding="0px 0px 0px 0px",
                     border="solid pink",
                     ))
        

        
        #### LINKING WIDGETS
        # 3 actions : 
        # 1. expression in text area
        # 2. slider move
        # 3. change favunit
        
        ### 1. expression in
        # define a context to parse a python expression into a final quantity
        # user can use "2*pi*m" to define 6.28*m quantity
        context = {**units, "pi":pi}
        def text_update_values(wdgt):
            # get expression entered
            expression = wdgt.value
            # eval expression with unit context
            res = eval(expression, context)
            res = quantify(res)
            res.favunit = self.favunit
            # if expression result has same dimension
            if res.dimension == self.value_q.dimension:
                self.value_q = res
                # udpate slider position
                self.floatslider.value = self.value_q.value
                # update text 
                self.text.value = str(self.value_q)
                #self.label.value = str(self.value_q)
            else:
                self.text.value="Result must have same dim"
                pass
                #self.label.value = str(self.value_q)
                #self.floatslider.value = self.label.value
                #self.text.value = self.label.value
        # On submit of text area
        self.text.on_submit(text_update_values)
        
        ### 2. slider
        def update_label_on_slider_change(change):
            self.value_q = Quantity(change.new, self.dimension, favunit=self.favunit)
            self.text.value = str(self.value_q)
        self.floatslider.observe(update_label_on_slider_change, names="value")
            
        
        ### 3. Change favunit
        # selection of favunit
        def update_favunit_on_favunit_dd_change(change):
            # retrieve new favunit q
            self.favunit = units[change.new]
            # update quantity favunit
            self.value_q.favunit = self.favunit
            
            # slider bounds
            self.labelsmin.value = str(Quantity(self.floatslider.min,
                                                self.dimension,
                                                favunit=self.favunit))
            
            self.labelsmax.value = str(Quantity(self.floatslider.max,
                                                self.dimension,
                                                favunit=self.favunit))
            
                                       
            self.text.value = str(self.value_q)
        self.favunit_dd.observe(update_favunit_on_favunit_dd_change, names="value")
        

        # update value on text change
        def update_text(change):
            self.text.value = self.label.value
        
    @property
    def min(self):
        return Quantity(self.floatslider.min, self.dimension)
    
    @property
    def max(self):
        return Quantity(self.floatslider.max, self.dimension)

    def __repr__(self):
        """Text display"""
        return repr(self.box)


    def _ipython_display_(self, **kwargs):
        """Interactive display"""
        return self.box._ipython_display_(**kwargs)
    

