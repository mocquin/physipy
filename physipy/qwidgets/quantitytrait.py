from traitlets import TraitType, HasTraits
from traitlets import Undefined
from traitlets.traitlets import _validate_bounds
import traitlets

from physipy import Quantity, Dimension, DimensionError, quantify


class QuantityTrait(TraitType):
    """A trait for Quantity.
    
    Templated from https://github.com/ipython/traitlets/blob/2bb2597224ca5ae485761781b11c06141770f110/traitlets/traitlets.py#L2064
    
    
    class Float(TraitType):
    "A float trait."

    default_value = 0.0
    info_text = 'a float'

    def __init__(self, default_value=Undefined, allow_none=False, **kwargs):
        self.min = kwargs.pop('min', -float('inf'))
        self.max = kwargs.pop('max', float('inf'))
        super(Float, self).__init__(default_value=default_value,
                                    allow_none=allow_none, **kwargs)

    def validate(self, obj, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            self.error(obj, value)
        return _validate_bounds(self, obj, value)

    def from_string(self, s):
        if self.allow_none and s == 'None':
            return None
        return float(s)
    
    
    
    """

    #default_value = 0.0
    info_text = 'a Quantity trait'

    def __init__(self, default_value=Undefined, allow_none=False, **kwargs):
        # make default value a Quantity object
        #default_value = quantify(default_value)
        # retrieve dimension
        default_dim = default_value.dimension
        
        ## allow 'min' and 'max' on creation
        # if no min value, set "-inf"*dimension
        proposed_min = kwargs.pop('min',
                              Quantity(-float('inf'),
                                              default_dim))
        if proposed_min.dimension == default_dim:
            self.min = proposed_min
        else:
            raise DimensionError(proposed_min.dimension, default_dim)
        # if no max value, set "inf"*dimension
        proposed_max = kwargs.pop('max', 
                              Quantity(float('inf'),
                                             default_dim))
        if proposed_max.dimension == default_dim:
            self.max = proposed_max
        else:
            raise DimensionError(proposed_max.dimension, default_dim)
        
        
        super(QuantityTrait, self).__init__(default_value=default_value,
                                    allow_none=allow_none, **kwargs)
    
    def validate(self, obj, value):
        # if value is a float, create a dimensionless quanatity
        if isinstance(value, float):
            value = Quantity(value, Dimension(None))
        if not isinstance(value, Quantity):
            self.error(obj, value)
        return _validate_bounds(self, obj, value)


    
class CQuantityTrait(QuantityTrait):
    """
    A casting version of the QuantityTrait trait.
    Based on : 
    https://github.com/ipython/traitlets/blob/2bb2597224ca5ae485761781b11c06141770f110/traitlets/traitlets.py#L2089
    
    class CFloat(Float):
        "A casting version of the float trait."

        def validate(self, obj, value):
            try:
                value = float(value)
            except Exception:
                self.error(obj, value)
            return _validate_bounds(self, obj, value)

    """    
    def validate(self, obj, value):
        try:
            value = quantify(value)
        except Exception:
            self.error(obj, value)
        return _validate_bounds(self, obj, value)

    
class TraitedQuantity(HasTraits):
    quantity = CQuantityTrait()
    
    
    
def main():
    from physipy import m, s, K
    
    first_trait = QuantityTrait(5*m)
    print(first_trait)
    print(first_trait.default_value)
    second = QuantityTrait(2*s)
    print(second)
    print(second.default_value)
    print(type(second.default_value))

    #third = QuantityTrait(5)
    #print(repr(third))
    #print(repr(third.default_value))
    #print(type(third.default_value))
    #print(repr(third.min))
    #print(repr(third.max))
    
    fourth = QuantityTrait(5*m, min=0*m)
    print(repr(fourth))
    print(repr(fourth.default_value))
    print(repr(type(fourth.default_value)))
    print(repr(fourth.min))
    print(repr(type(fourth.min)))
    print(repr(fourth.max))
    print(repr(type(fourth.max)))
    
    #fifth = QuantityTrait(5*m, min=0*s)
    
    sixth = QuantityTrait(3*m, max=1*m)
    print(sixth)
    
    flo = traitlets.Float(3, max=1)
    print(repr(flo.default_value))
    print(repr(flo.min))
    print(repr(flo.max))
    

if __name__ == "__main__":
    main()