# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example wrapping uncertainties : add the unit to `nominal_value` and `std_dev`

# %% [markdown]
# Numpy's integration is a great example of how physipy can wrap any kind of numerical values, but it  integrated in the source code of physipy so it's a bit cheating.
# Let's see how physipy deals with a non-integrated numerical-like package : uncertaintie. By "non-integrated" I mean that no source code of physipy makes any explicit reference to uncertainties, so we only rely on the general wrapping interface of physipy.

# %%
from physipy import m, s, K, Quantity, Dimension
import uncertainties as u

# create a pure uncertainties instance
x = u.ufloat(0.20, 0.01)  # x = 0.20+/-0.01
print(x)
print(x**2)
print(type(x))

# %% [markdown]
# If we create a quantity version by mulitplying by 1 meter, the returned value is of Quantity type:

# %%
# now let's create a quanity version of x
xq = x*m
print(xq)
print(xq**2, type(xq**2))
print(xq+2*m, type(xq+2*m))
print(xq.value)

# %% [markdown]
# That's a pretty neat result __that didn't need any additional code__.
# Now uncertainties instance have a `nominal_value` and `std_dev` attributes.

# %%
# Creation must be done this way and not by "x*m" because "x*m" 
# will multiply the uncerainties variable by 1, and turn it into a
# AffineScalarFunc instance, which is not hashable and breaks my 
# register_property_backend that relies on dict lookup
xq = Quantity(x, Dimension("m")) 

# %%
print(x.nominal_value)
print(x.std_dev)

# %% [markdown]
# In physipy, if an attribute doesn't exist in the quantity instance, the lookup falls back on the backend value, ie on the uncertainties variable, so by default we get the same result on `xq` (note that we don't get auto-completion either for the same reason):

# %%
print(xq.nominal_value)
print(xq.std_dev)

# %% [markdown]
# It would be great that `xq.nominal_value` actually prints `0.2 m`, not loosing the unit and making it explicit that the nominal value is actually 0.2 meters. To do that, we can add a property back to specify what we want `xq.nominal_value` to return : a property back is a dictionnary with key the name of the attribute, and as value the corresponding method to get the wanted result.
#
# For the nominal_value and standard deviation, we just want to add back the unit and make the variable a Quantity, so we multiply by the corresponding SI unit:

# %%
from physipy.quantity.quantity import register_property_backend
uncertainties_property_backend_interface = {
    # res is the backend result of the attribute lookup, and q the wrapping quantity
    "nominal_value":lambda q, res: q._SI_unitary_quantity*res,
    "std_dev":lambda q, res: q._SI_unitary_quantity*res,
}

register_property_backend(type(xq.value), 
                         uncertainties_property_backend_interface)

# %% [markdown]
# With this property back interface registered we get the desired result for `print(xq.nominal_value)`:

# %%
print(xq.nominal_value)

# %% [markdown]
# # Why the duck type approach doesn't work

# %% [markdown]
# Another approach to do this would be to create a new class like this

# %%
from physipy import m, s, K, Quantity, Dimension
import uncertainties as u

# create a pure uncertainties instance
x = u.ufloat(0.20, 0.01)  # x = 0.20+/-0.01

class UWrappedQuantity(Quantity):
    @property
    def nominal_value(self):
        return self.value.nominal_value * self._SI_unitary_quantity
    @property
    def std_dev(self):
        return self.value.std_dev * self._SI_unitary_quantity

xq2 = UWrappedQuantity(x, Dimension("m"))
#print(xq2)
print(type(xq2), xq2, xq2.nominal_value)

# %% [markdown]
# But with this definition, newly created instance will be of type `Quantity`, not `UWrappedQuantity`, loosing again the unit on `nominal_value`

# %%
print(xq2+2*m, type(xq2+2*m), (xq2+2*m).value, (xq2+2*m)._SI_unitary_quantity)
print((xq2+2*m).nominal_value)

# %% [markdown]
# # Wrapping mcerp

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
print(type((x*m).value))

# %%

# %%

# %% [markdown]
# By design, physipy can wrap pretty well most numerical-like packages, like numpy or uncertainties.
#

# %%

# %% [markdown]
# A FAIRE !!!!!!
# Ajouter Mul dans MyFraction !!!!!!!!!

# %%

# %%

# %% [markdown]
# # Interface

# %% [markdown]
# The simplest way to use physipy with specific objects, like fractions or your own class is to create quantities that wrap your value : here `Quantity` wraps the custom value as its `value` attribute.

# %% [markdown]
# A simple example : 

# %%
import fractions
from physipy import Quantity, Dimension, m

# %%
length = Quantity(fractions.Fraction(3, 26), Dimension("m"))
print(length)

# %% [markdown]
# Then when doing calculation, physipy deals with everything for you :

# %%
print(length + 2*m)


# %% [markdown]
# # More complex objects

# %% [markdown]
# Now say we want to customize the basic Fraction object, by overloading its str method : 

# %%
class MyFraction(fractions.Fraction):
    def __str__(self):
        return f"[[[{self.numerator}/{self.denominator}]]]"

    def __mul__(self, other):
        print(type(other))
        if not isinstance(other, fractions.Fraction):
            other = fractions.Fraction(other)
        print(type(other))
        fres = self*other
        return fraction.Fraction(fres)
    __rmul__ = __mul__


# %%
my_length = MyFraction(3, 26)
print(my_length)
print(my_length*my_length)

# %% [markdown]
# Now what happens when we create a quantity with this value

# %%
my_length_q = my_length*m
print(my_length_q)


# %% [markdown]
# Notice that we lost the custom str : that's because the value is not a `MyFraction` instance but `fractions.Fraction` : it was lost in the multiplication process. Indeed, since `my_length*m` falls back on the `__mul__` of Quantity, that uses the `__mul__` method of the instance, which is `fractions.Fraction.__mul__`, which returns a `fractions.Fraction` instance, not a `MyFraction` instance.
#
# So we would like physipy to return a `MyFraction` when computing multiplication with a Quantity objet.

# %% [markdown]
# Now we can't expect each user to rewrite its custom Fraction class to be compatible with Quantity, so we do the opposite : Quantity will wrap the custom class with an interface class.

# %%
class QuantityWrappedMyFraction(Quantity):
    def __str__(self):
        print("toto")
        str_myfraction = str(self.value)
        return str_myfraction + "-"+self.dimension.str_SI_unit()

from physipy.quantity.quantity import register_value_backend
register_value_backend(
    # here we use the class of the base value
    fractions.Fraction, 
    # here the rewritten version
    QuantityWrappedMyFraction)

# %%
my_length_q = my_length*m
print(my_length_q)

# %%

# %%
my_length_q.value

# %%
import mcerp
from physipy import m, Quantity, Dimension

# %%
type(x1)

# %%
x1 = Quantity(mcerp.N(24, 1), Dimension("m"))

# %%
x2 = mcerp.N(2, 1)

# %%
from physipy.quantity.quantity import register_property_backend, register_value_backend
register_value_backend(type(x1))

# %%
x1*m

# %%
