from quantity import Quantity, Dimension, m, s, kg, A, cd

# Unités utiles
mm = m/1000
mm.symbole = "mm"

mum = mm/1000
mum.symbole = "mum"

nm = mum/1000
nm.symbole = "nm"

km = 1000*m
km.symbole = "km"

ms = 0.001*s
ms.symbole = "ms"

Hz = 1/s
Hz.symbole = "Hz"

N = kg*m/s**2
N.symbole = "N"

Pa = N/m**2
Pa.symbole = "Pa"

J = kg*(m/s)**2
J.symbole = "J"

W = J/s
W.symbole = "W"

C = A*s
C.symbole = "C"

#deg = np.pi*rad/180
#deg.symbole = "deg"

h = 3600*s
h.symbole = "h"

ph = Quantity(1,Dimension(None),symbol="ph")
murad = Quantity(10**-6,Dimension("rad"),symbol="murad")

cy = Quantity(1,Dimension(None),symboly="cy")

#mrad = 0.001*rad
#mrad.symbole = "mrad"

#msr = 0.001*sr
#msr.symbole = "msr"