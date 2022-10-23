# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Numeric backend interface

# %% [markdown]
# ## Example wrapping uncertainties : add the unit to `nominal_value` and `std_dev`

# %% [markdown]
# Numpy's integration is a great example of how physipy can wrap any kind of numerical values, but this integration is written in the source code of physipy so it's a bit cheating.  
# Let's see how physipy deals with a non-integrated numerical-like package : uncertainties. By "non-integrated" I mean that no source code of physipy makes any explicit reference to uncertainties, so we only rely on the general wrapping interface of physipy.

# %%
from physipy import m, s, K, Quantity, Dimension
import uncertainties.umath as um
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
print(xq)                    # a Quantity
print(xq**2, type(xq**2))    # a Quantity
print(xq+2*m, type(xq+2*m))  # a Quantity
print(xq.value)              # an uncertainties value
print(m*x == x*m)            # True

# %% [markdown]
# That's a pretty neat result __that didn't need any additional code__.  
# Now going a bit further, uncertainties instance have a `nominal_value` and `std_dev` attributes.

# %%
# Creation must be done this way and not by "x*m" because "x*m" 
# will multiply the uncerainties variable by 1, and turn it into a
# AffineScalarFunc instance, which is not hashable and breaks my 
# register_property_backend that relies on dict lookup
#x = u.ufloat(0.20, 0.01)  # x = 0.20+/-0.01
xq = Quantity(x, Dimension("m")) # xq = x *m

# %%
print(x.nominal_value)
print(x.std_dev)

# %% [markdown]
# In physipy, if an attribute doesn't exist in the quantity instance, the lookup falls back on the backend value, ie on the uncertainties variable, so by default we get the same result on `xq` (note that we don't get auto-completion either for the same reason):

# %%
print(xq.nominal_value) # 0.2
print(xq.std_dev)       # 0.1

# %% [markdown]
# It would be great that `xq.nominal_value` actually prints `0.2 m`, not loosing the unit and making it explicit that the nominal value is actually 0.2 meters. To do that, we can add a property back to specify what we want `xq.nominal_value` to return : a property backend is a dictionnary with key the name of the attribute, and as value the corresponding method to get the wanted result.
#
# For the nominal_value and standard deviation, we just want to add back the unit and make the variable a Quantity, so we multiply by the corresponding SI unit:

# %%
type(xq.value)

# %%
import uncertainties as uc
from physipy.quantity.quantity import register_property_backend

uncertainties_property_backend_interface = {
    # res is the backend result of the attribute lookup, and q the wrapping quantity
    "nominal_value":lambda q, res: q._SI_unitary_quantity*res,
    "std_dev":lambda q, res: q._SI_unitary_quantity*res,
    "n":lambda q, res: q._SI_unitary_quantity*res,
    "s":lambda q, res: q._SI_unitary_quantity*res,
}

print("Registering uncertainties")
register_property_backend(uc.core.Variable, 
                         uncertainties_property_backend_interface)

# %% [markdown]
# With this property back interface registered we get the desired result for `print(xq.nominal_value)`:

# %%
print(type(xq.value))
print(xq.nominal_value) # 0.2 m, instead of just 0.2 previously
print(xq.std_dev)       # 0.1 m, instead of just 0.1 previously

# %%
yq = 2*xq 

# %%
type(list(yq.derivatives.keys())[0])

# %%
from physipy.math import decorator_angle_or_dimless_to_dimless
from physipy import rad

# %%
sin = decorator_angle_or_dimless_to_dimless(um.sin)
print(sin(x*rad))
print(um.sin(x))

# %%
    

# %% [markdown]
# ## Why the duck type approach doesn't work

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
# ## Interface

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
print(length**2)
print(length.dimension)
print(length.is_mass())
print(length.sum())


# %% [markdown]
# # More complex objects

# %% [markdown]
# Now say we want to customize the basic Fraction object, by overloading its str method : 

# %%
class MyFraction(fractions.Fraction):
    def __str__(self):
        return f"[[[{self.numerator}/{self.denominator}]]]"


# %%
my_length = MyFraction(3, 26)
print(my_length)
print(my_length*my_length)
print(Quantity(MyFraction(3, 26), Dimension("L")))
print(MyFraction(3, 26)*m)


# %% [markdown]
# Notice that we lost the custom str : that's because the value is not a `MyFraction` instance but `fractions.Fraction` : it was lost in the multiplication process. Indeed, since `my_length*m` falls back on the `__mul__` of Quantity, that uses the `__mul__` method of the instance, which is `fractions.Fraction.__mul__`, which returns a `fractions.Fraction` instance, not a `MyFraction` instance.
#
# So we would like physipy to return a `MyFraction` when computing multiplication with a Quantity objet.

# %% [markdown]
# Now we can't expect each user to rewrite its custom Fraction class to be compatible with Quantity, so we do the opposite : Quantity will wrap the custom class with an interface class.

# %%
class QuantityWrappedMyFraction(Quantity):
    def __str__(self):
        str_myfraction = str(self.value)
        return "QuantityWrappedMyFraction : " + str_myfraction + "-"+self.dimension.str_SI_unit()

from physipy.quantity.quantity import register_value_backend
register_value_backend(
    # here we use the class of the base value
    fractions.Fraction, 
    # here the rewritten version
    QuantityWrappedMyFraction)

# %%
print(my_length)
print(my_length*my_length)
print(Quantity(MyFraction(3, 26), Dimension("L")))
print(MyFraction(3, 26)*m)

# %% [markdown]
# # Requirements for easy backend supports

# %% [markdown]
#  - The class must be hashable : for the dict lookup on type, we need the class as key, so they need to be hashable

# %%
