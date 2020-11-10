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



class QuantityText():
    
    def __init__(self, qvalue=0.0, disabled=False, continuous_update=True):
        
        self.qvalue = quantify(qvalue)
        self.text = ipyw.Text(value=str(self.qvalue),
                              placeholder='Type python exp',
                              description='',#'Set to:',
                              disabled=disabled,
                              continuous_update=continuous_update,
                              layout=Layout(width='25%',
                                            margin="0px 0px 0px 0px",
                                            padding="0px 0px 0px 0px",
                                            border="solid gray"))
        
        self.text.on_submit(self.text_update_values)
        self.context = {**units, "pi":pi}

    def text_update_values(self, wdgt):
        # get expression entered
        expression = wdgt.value
        # eval expression with unit context
        try:
            res = eval(expression, self.context)
            res = quantify(res)
            self.qvalue = res
            # update text 
            self.text.value = str(self.qvalue)
        except:
            self.text.value = "Python expr only from QuantityText"
        
    def __repr__(self):
        """Text display"""
        return repr(self.text)


    def _ipython_display_(self, **kwargs):
        """Interactive display"""
        return self.text._ipython_display_(**kwargs)


class FDQuantityText(QuantityText):
    """Fixed Dimension Quantity"""
    
    def __init__(self, qvalue=0.0, disabled=False, continuous_update=True):
        
        super().__init__(qvalue, disabled, continuous_update)
        self.text.on_submit(self.text_update_values)

    # override text_update to check if input has same dim
    def text_update_values(self, wdgt):
        # get expression entered
        expression = wdgt.value
        # eval expression with unit context
        try:
            res = eval(expression, self.context)
            res = quantify(res)
            # update text 
            if res.dimension == self.qvalue.dimension:
                self.qvalue = res
                # update text 
                self.text.value = str(self.qvalue)
                #self.label.value = str(self.value_q)
            else:
                #self.text.value="Result must have same dim"
                self.text.value = str(self.qvalue)
        except:
            self.text.value = "Python expr only from FDQuantityText"

            
            
class abs_QuantityText():
    
    def __init__(self, qvalue=0.0, disabled=False, continuous_update=True):
        self.qvalue = quantify(qvalue)
        self.text = ipyw.Text(value=str(self.qvalue),
                              placeholder='Type python exp',
                              description='',#'Set to:',
                              disabled=disabled,
                              continuous_update=continuous_update,
                              layout=Layout(width='20%',
                                            margin="0px 0px 0px 0px",
                                            padding="0px 0px 0px 0px",
                                            border="solid black"))
        
    
        
    def __repr__(self):
        """Text display"""
        return repr(self.text)


    def _ipython_display_(self, **kwargs):
        """Interactive display"""
        return self.text._ipython_display_(**kwargs)
    
            
            
class FDQuantitySlider():
    
    def __init__(self, qvalue=0.0, min=None, max=None, step=None, description="", disabled=False,
                continuous_update=True, orientation="horizontal", readout=True, readout_format=".1f", 
                constraint_dimension=True):
        
        #### READ PARAMS
        # turn value into a Quantity object (if not already)
        self.qvalue = quantify(qvalue)
        
        ## Turn min, max and step into Quantity objects, and check dimensions
        # min value
        if min is not None:
            qmin = quantify(min)
            if not qmin.dimension == self.qvalue.dimension:
                raise DimensionError(qmin.dimension, self.qvalue.dimension)
        else:
            qmin = Quantity(0.0, self.qvalue.dimension)
        # max value
        if max is not None:
            qmax = quantify(max)
            if not qmax.dimension == self.qvalue.dimension:
                raise DimensionError(qmax.dimension, self.qvalue.dimension)
        else:
            qmax = Quantity(100.0, self.qvalue.dimension)        
        # step value
        if step is not None:
            self.qstep = quantify(step)
            if not self.qstep.dimension == self.qvalue.dimension:
                raise DimensionError(self.qstep.dimension, self.qvalue.dimension)
        else:
            self.qstep = Quantity(0.1, self.qvalue.dimension)     
            
        qmin.favunit = self.qvalue.favunit
        qmax.favunit = self.qvalue.favunit
        
        self.context = {**units, "pi":pi}

        
    
        ####### WIDGETS
        # FloatSlider with values (Float values, not Quantity objects)
        self.floatslider = ipyw.FloatSlider(value=self.qvalue.value,
                                            min=qmin.value,
                                            max=qmax.value,
                                            step=self.qstep.value,
                                            description=description,
                                            disabled=disabled,
                                            continuous_update=continuous_update,
                                            orientation=orientation,
                                            readout=False,  # to disable displaying readout value
                                            readout_format=readout_format,
                                            layout=Layout(width="30%",
                                                          margin="0px",
                                                          border="solid #3295a8"))

        self.label = ipyw.Label(value=str(self.qvalue))

        self.slider_box = ipyw.HBox([
            self.floatslider,
            self.label,
            ],
            layout=Layout(width="100%",
                         border="solid #c77700",
                         padding="0px 0px 0px 0px",
                         margin="0px"))
    
        def update_label_on_slider_change(change):
            self.qvalue = Quantity(change.new, self.qvalue.dimension, favunit=self.qvalue.favunit)
            self.label.value = str(self.qvalue)
        self.floatslider.observe(update_label_on_slider_change, names="value")


 
    
    
    
    def __repr__(self):
        """Text display"""
        return repr(self.slider_box)


    def _ipython_display_(self, **kwargs):
        """Interactive display"""
        return self.slider_box._ipython_display_(**kwargs)    
    
    
    
    
    
    
class FDQuantitySliderWithBounds():
    
    def __init__(self, qvalue=0.0, min=None, max=None, step=None, description="", disabled=False,
                continuous_update=True, orientation="horizontal", readout=True, readout_format=".1f", 
                constraint_dimension=True):
        
        #### READ PARAMS
        # turn value into a Quantity object (if not already)
        self.qvalue = quantify(qvalue)
        
        ## Turn min, max and step into Quantity objects, and check dimensions
        # min value
        if min is not None:
            qmin = quantify(min)
            if not qmin.dimension == self.qvalue.dimension:
                raise DimensionError(qmin.dimension, self.qvalue.dimension)
        else:
            qmin = Quantity(0.0, self.qvalue.dimension)
        # max value
        if max is not None:
            qmax = quantify(max)
            if not qmax.dimension == self.qvalue.dimension:
                raise DimensionError(qmax.dimension, self.qvalue.dimension)
        else:
            qmax = Quantity(100.0, self.qvalue.dimension)        
        # step value
        if step is not None:
            self.qstep = quantify(step)
            if not self.qstep.dimension == self.qvalue.dimension:
                raise DimensionError(self.qstep.dimension, self.qvalue.dimension)
        else:
            self.qstep = Quantity(0.1, self.qvalue.dimension)     
            
        qmin.favunit = self.qvalue.favunit
        qmax.favunit = self.qvalue.favunit
        
        self.context = {**units, "pi":pi}

        
    
        ####### WIDGETS
        # FloatSlider with values (Float values, not Quantity objects)
        self.floatslider = ipyw.FloatSlider(value=self.qvalue.value,
                                            min=qmin.value,
                                            max=qmax.value,
                                            step=self.qstep.value,
                                            description=description,
                                            disabled=disabled,
                                            continuous_update=continuous_update,
                                            orientation=orientation,
                                            readout=False,  # to disable displaying readout value
                                            readout_format=readout_format,
                                            layout=Layout(width="30%",
                                                          margin="0px",
                                                          border="solid #3295a8"))

        self.minw = abs_QuantityText(qmin)
        self.maxw = abs_QuantityText(qmax)
        self.label = ipyw.Label(value=str(self.qvalue))

        self.slider_box = ipyw.HBox([
            self.minw.text,
            self.floatslider,
            self.label,
            self.maxw.text,
            ],
            layout=Layout(width="100%",
                         border="solid #c77700",
                         padding="0px 0px 0px 0px",
                         margin="0px"))
    
        def update_label_on_slider_change(change):
            self.qvalue = Quantity(change.new, self.qvalue.dimension, favunit=self.qvalue.favunit)
            self.label.value = str(self.qvalue)
        self.floatslider.observe(update_label_on_slider_change, names="value")

        
        
        self.maxw.text.on_submit(self.check_max)
        self.minw.text.on_submit(self.check_min)
        
    def check_max(self, wdgt):
        # get expression entered
        expression = wdgt.value
        # eval expression with unit context
        try:
            res = eval(expression, self.context)
            res = quantify(res)
            if res < self.minw.qvalue:
                # forbiden : reset to old value 
                self.maxw.text.value = str(self.maxw.qvalue)
            else:
                self.maxw.qvalue = res
                # update text 
                self.maxw.text.value = str(self.maxw.qvalue)
                self.floatslider.max = self.maxw.qvalue.value
        except:
            self.maxw.text.value = str(self.maxw.qvalue)
    def check_min(self, wdgt):
        # get expression entered
        expression = wdgt.value
        # eval expression with unit context
        try:
            res = eval(expression, self.context)
            res = quantify(res)
            if res > self.maxw.qvalue:
                # forbiden : reset to old value 
                self.minw.text.value = str(self.minw.qvalue)
            else:
                self.minw.qvalue = res
                # update text 
                self.minw.text.value = str(self.minw.qvalue)
                self.floatslider.min = self.minw.qvalue.value

        except:
            self.minw.text.value = str(self.minw.qvalue)

    
    def __repr__(self):
        """Text display"""
        return repr(self.slider_box)


    def _ipython_display_(self, **kwargs):
        """Interactive display"""
        return self.slider_box._ipython_display_(**kwargs)    
    
    
    
    
    
    
    
    
    
    
        
#lass QuantityFloatSlider():
#   
#   def __init__(self, value=0.0, min=None, max=None, step=None, description="", disabled=False,
#               continuous_update=True, orientation="horizontal", readout=True, readout_format=".1f", 
#               constraint_dimension=True):

#       #### READ PARAMS
#       # turn value into a Quantity object (if not already)
#       self.qvalue = quantify(value)
#       
#       ## Turn min, max and step into Quantity objects, and check dimensions
#       # min value
#       if min is not None:
#           self.qmin = quantify(min)
#           if not self.qmin.dimension == self.qvalue.dimension:
#               raise DimensionError(self.qmin.dimension, self.qvalue.dimension)
#       else:
#           self.qmin = Quantity(0.0, self.qvalue.dimension)
#       # max value
#       if max is not None:
#           self.qmax = quantify(max)
#           if not self.qmax.dimension == self.qvalue.dimension:
#               raise DimensionError(self.qmax.dimension, self.qvalue.dimension)
#       else:
#           self.qmax = Quantity(100.0, self.qvalue.dimension)        
#       # step value
#       if step is not None:
#           self.qstep = quantify(step)
#           if not self.qstep.dimension == self.qvalue.dimension:
#               raise DimensionError(self.qstep.dimension, self.qvalue.dimension)
#       else:
#           self.qstep = Quantity(0.1, self.qvalue.dimension)       
#       
#       
#       ####### WIDGETS
#       # FloatSlider with values (Float values, not Quantity objects)
#       self.floatslider = ipyw.FloatSlider(value=self.qvalue.value,
#                                           min=self.qmin.value,
#                                           max=self.qmax.value,
#                                           step=self.qstep.value,
#                                           description=description,
#                                           disabled=disabled,
#                                           continuous_update=continuous_update,
#                                           orientation=orientation,
#                                           readout=False,  # to disable displaying readout value
#                                           readout_format=readout_format,
#                                           layout=Layout(width="50%",
#                                                         margin="0px",
#                                                         border="solid red"))
#       self.labelsmin = ipyw.Label(value=str(Quantity(self.floatslider.min,
#                                                      self.dimension,
#                                                      favunit=self.favunit)),
#                                   layout=Layout(margin="0px",
#                                                 padding='0px 0px 0px 0px',
#                                                 border="solid gray"))
#       self.labelsmax = ipyw.Label(value=str(Quantity(self.floatslider.max,
#                                                      self.dimension,
#                                                      favunit=self.favunit)),
#                                   layout=Layout(margin="0px",
#                                                 border="solid red"))
#       self.slider_box = ipyw.HBox([
#           self.labelsmin,
#           self.floatslider,
#           self.labelsmax,
#           ],
#           layout=Layout(width="300px",
#                        border="solid blue",
#                        padding="0px 0px 0px 0px",
#                        margin="0px"))
#       
#       # DROPDOWN for favunit list of favorite units (aka "favunit") that have same dimension to select from
#       self.favunit_str_list = [u_str for u_str, u_q in units.items() ]#if self.dimension == u_q.dimension]
#       self.favunit_str_list.append("-")
#       self.favunit_dd = ipyw.Dropdown(
#                           options=self.favunit_str_list,
#                           value=str(self.favunit.symbol),
#                           description='Favunit:',
#                           layout=Layout(width="20%",
#                                         padding="0px 0px 0px 0px",
#                                         margin="0px 0px 0px 0px",
#                                         border="solid red")
#                                   )


#       # TEXT area parsing
#       self.text = ipyw.Text(value=str(self.value_q),
#                             placeholder='Type python exp',
#                             description='',#'Set to:',
#                             disabled=False,
#                             continuous_update=True,
#                             layout=Layout(width='30%',
#                                           margin="0px 0px 0px 0px",
#                                           padding="0px 0px 0px 0px",
#                                           border="solid red"))

#       # wrapping in a box #self.label
#       self.box = ipyw.HBox([
#           self.text,
#           self.slider_box,
#           self.favunit_dd,
#           ],
#       layout=Layout(widht='100%',
#                    margin="0px 0px 0px 0px",
#                    padding="0px 0px 0px 0px",
#                    border="solid pink",
#                    ))
#       

#       
#       #### LINKING WIDGETS
#       # 3 actions : 
#       # 1. expression in text area
#       # 2. slider move
#       # 3. change favunit
#       
#       ### 1. expression in
#       # define a context to parse a python expression into a final quantity
#       # user can use "2*pi*m" to define 6.28*m quantity
#       context = {**units, "pi":pi}
#       def text_update_values(wdgt):
#           # get expression entered
#           expression = wdgt.value
#           # eval expression with unit context
#           res = eval(expression, context)
#           res = quantify(res)
#           res.favunit = self.favunit
#           # if expression result has same dimension
#           if res.dimension == self.value_q.dimension:
#               self.value_q = res
#               # udpate slider position
#               self.floatslider.value = self.value_q.value
#               # update text 
#               self.text.value = str(self.value_q)
#               #self.label.value = str(self.value_q)
#           else:
#               self.text.value="Result must have same dim"
#               pass
#               #self.label.value = str(self.value_q)
#               #self.floatslider.value = self.label.value
#               #self.text.value = self.label.value
#       # On submit of text area
#       self.text.on_submit(text_update_values)
#       
#       ### 2. slider
#       def update_label_on_slider_change(change):
#           self.value_q = Quantity(change.new, self.dimension, favunit=self.favunit)
#           self.text.value = str(self.value_q)
#       self.floatslider.observe(update_label_on_slider_change, names="value")
#           
#       
#       ### 3. Change favunit
#       # selection of favunit
#       def update_favunit_on_favunit_dd_change(change):
#           # retrieve new favunit q
#           self.favunit = units[change.new]
#           # update quantity favunit
#           self.value_q.favunit = self.favunit
#           
#           # slider bounds
#           self.labelsmin.value = str(Quantity(self.floatslider.min,
#                                               self.dimension,
#                                               favunit=self.favunit))
#           
#           self.labelsmax.value = str(Quantity(self.floatslider.max,
#                                               self.dimension,
#                                               favunit=self.favunit))
#           
#                                      
#           self.text.value = str(self.value_q)
#       self.favunit_dd.observe(update_favunit_on_favunit_dd_change, names="value")
#       

#       # update value on text change
#       def update_text(change):
#           self.text.value = self.label.value
#       
#   @property
#   def min(self):
#       return Quantity(self.floatslider.min, self.dimension)
#   
#   @property
#   def max(self):
#       return Quantity(self.floatslider.max, self.dimension)

#   def __repr__(self):
#       """Text display"""
#       return repr(self.box)


#   def _ipython_display_(self, **kwargs):
#       """Interactive display"""
#       return self.box._ipython_display_(**kwargs)


