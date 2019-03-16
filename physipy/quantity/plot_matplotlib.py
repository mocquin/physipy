
"""
TODO : 
 - [ ] : one-function fast plot
 - [ ] : multiline plot, with axis-unit checking
 - [ ] : dimension-checking for axis
 - [ ] : favunit axis-labelling


Ideas/Ressources : 
 - astropy overloads matplotlib : http://docs.astropy.org/en/stable/units/quantity.html#plotting-quantities
 - https://pint.readthedocs.io/en/latest/plotting.html
 - https://matplotlib.org/gallery/units/units_scatter.html

"""


import sys
sys.path.insert(0,'/Users/mocquin/Documents/CLE/Optique/Python/JUPYTER/MYLIB10/MODULES/quantity-master')


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import sympy
import numpy as np
from quantity import Quantity, m, sr, SI_units_derived, units, make_quantity, interp, linspace, K, vectorize, kg, s, quantify, Dimension

km = SI_units_derived["km"]
mm = SI_units_derived["mm"]
W = units["W"]

style.use("seaborn-notebook")

#%matplotlib
#%matplotlib inline
#%matplotlib notebook

# Figure
plt.rcParams["figure.dpi"]=200
plt.rcParams["figure.figsize"]=(10,6)
plt.rcParams["font.size"]=10
# Ticks mineurs
plt.rcParams["xtick.minor.visible"]=True
plt.rcParams["ytick.minor.visible"]=True
plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
plt.rcParams["axes.labelsize"]="medium"
plt.rcParams["axes.titlesize"]="medium"
# Legende
plt.rcParams["legend.fontsize"]=12
# Grilles
plt.rcParams["grid.alpha"]=0.75
plt.rcParams["grid.linestyle"]="--"
plt.rcParams["grid.color"]="grey"
plt.rcParams["axes.grid"]=True
plt.rcParams["axes.grid.which"]="both"
#print(mplpp.rcParams)
#mplpp.rcParams['axes.spines.top']=False
#mplpp.rcParams['axes.spines.right']=False
#mplpp.grid(b=True,which='minor',alpha=0.5)
#mplpp.grid(b=True,which='major',alpha=1)
#mplpp.legend(loc="upper right",fontsize=10,frameon=True)
#mplpp.grid(b=True, which='major', color='k', linestyle='-')
#mplpp.grid(b=True, which='minor', color='k', alpha=0.5)


#def grouped(iterable, n):
#    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
#    return zip(*[iter(iterable)]*n)
#
#
#def read_args_plot(*args):
#    new_args = []
#    for x_, y_ in grouped(args,2):
#        X_valeur = x_.value
#        Y_valeur = y_.value
#        str_X = str(x_)
#        str_Y = str(y_)
#        if str_X == "":
#            expr_X = "s.u"
#        else:
#            expr_X = sympy.latex(sympy.sympify(str_X))            
#        if str_Y == "":
#            expr_Y = "s.u"
#        else:
#            expr_Y = sympy.latex(sympy.sympify(str_Y))
#        new_args = new_args + [X_valeur,Y_valeur]        
#    return expr_X, expr_Y, new_args
#
#
#def list_args(*args):
#    liste_arg_float = []
#    for el in args:
#        if isinstance(el,Quantity):
#            el_float = el/el.favunit
#        else:
#            el_float = el
#        liste_arg_float = liste_arg_float + [el_float]
#    return liste_arg_float
#
## Plot
#def plot(*args,**kwargs):
#    
#    expr_X, expr_Y, *new_args = read_args_plot(*args)
#
#    # Ajout des unités sur les labels des axes    
#    xlabel_ = "$\ " + str(expr_X) + "$"
#    ylabel_ = "$\ " + str(expr_Y) + "$"
#    plt.xlabel(xlabel_)
#    plt.ylabel(ylabel_)
#
#    return plt.plot(*new_args,**kwargs)
#
#x = np.array([1,2,3])*sr
#x.favunit = sr
#y = np.array([2,3,2])*m
#y.favunit = m
#z = np.array([1,2,3])*m
#z.favunit = mm
#a = np.array([3,3,4])*W
#a.favunit = m*W


class qaxes(matplotlib.)



def change_favunit(axis, favunit):
    pass
    

def add_SI_unit_label(axis):
    """add label to axis corresponding to its dimension (if has one)"""
    if hasattr(axis, 'dimension'):
        l = axis.set_label(str(axis.str_SI_unit()))
        return l

def check_or_add_axis_dimension(axis, obj):
    """Check or add axis a dimension attribute, corresponding to obj dimension.
    
    If obj is not a quantity, it is converted to a quantity with dimension None.
    """
    if not (isinstance(axis, matplotlib.axis.XAxis) or isinstance(axis, matplotlib.axis.YAxis)):
        raise TypeError("axis must be a matplotlib.axis.XAxis or YAxis, but is :" + str(type(axis)))
    
    q_obj = quantify(obj)
    if hasattr(axis, 'dimension'):
        if not axis.dimension == q_obj.dimension:
            raise ValueError("Axis and obj must have same dimension, here axis has " + str(axis.dimension) + " and obj has " + str(q_obj.dimension))
    else:
        axis.dimension = q_obj.dimension

#plt.plot(); ax = plt.gca();
#check_or_add_axis_dimension(ax.xaxis, 0.6*m)
#check_or_add_axis_dimension(ax.xaxis, 1*m)
#check_or_add_axis_dimension(ax.xaxis, 1) # this should raise error
#check_or_add_axis_dimension(ax.xaxis, 1*kg)

def add_axis_favunit(axis, q, favunit):
    
    if not hasattr(axis, "ratio_favunit"):
        ratio_favunit = make_quantity(q/favunit)
        axis.ratio_favunit = ratio_favunit
        axis.favunit = favunit
        dim_SI = ratio_favunit.dimension
        if dim_SI == Dimension(None):
            axis.favq = favunit
        else:
            axis.favq = favunit * Quantity(1, dim_SI)
        q.favunit = favunit
        plt.plot(q._compute_value(), q._compute_value(), "r*")
        plt.xlabel(q._compute_complement_value())
    else:
        raise ValueError("axis already has favunit")
        
def update_axis_value(axis, new_favunit):
    if not hasattr(axis, "favunit"):
        raise ValueError("Axis must already have ratio_favunit")
    ax = axis.axes
    for line in ax.lines:
        old_value = line.get_xdata()
        old_q = old_value * axis.favunit
        axis.favunit = new_favunit
        old_q.favunit = new_favunit
        x = old_q._compute_value()
        line.set_xdata(old_q._compute_value())
        ax.set_xlabel(old_q._compute_complement_value())
        # update bounds
        ax.relim()
        ax.autoscale_view()  
        plt.legend()


plt.plot(); ax = plt.gca();
add_axis_favunit(ax.xaxis, 2*m, mm)
#add_axis_favunit(ax.xaxis, 2*m, kg)
#demi_mmkg = 0.5*mm*kg
#demi_mmkg.symbol = "0.5*mm*kg"
#add_axis_favunit(ax.xaxis, 2*m, demi_mmkg)
update_axis_value(ax.xaxis, km)
update_axis_value(ax.xaxis, mm)



def q_plot(x, y, *args, **kwargs):

    ax = plt.gca()
    
    # dimension checking
    for axis, q in zip([ax.xaxis, ax.yaxis], [x, y]):
        check_or_add_axis_dimension(axis, q)
        
        
        # Ajout d'un label si dimension
        # label_dimension(axis)


    x_q = make_quantity(x)
    y_q = make_quantity(y)
    l = plt.plot(x_q.value, y_q.value, *args, **kwargs)

    return l


def f_plot(f, xstart, xstop, *args, **kwargs):
    """Plot function between start and stop, with quantity handling.
    
    Uses xstart.favunit if available."""
    q_xstart = make_quantity(xstart)
    q_xstop = make_quantity(xstop)
    ech_x = linspace(q_xstart, q_xstop)
    # keeping track of possible favunit in x
    ech_x.favunit = q_xstart.favunit
    fvec = vectorize(f)
    y = fvec(ech_x)
    line_2D = q_plot(ech_x, y, *args, **kwargs)
    return line_2D





# def show(f, xstart, xstop, *args, **kwargs):
#    p = f_plot(f, xstart=xstart, xstop=xstop, *args, **kwargs)
#    plt.show()
#    return p



#








def main():
    
    
    sca = 1
    vec = np.array([0.9, 1, 1.1])
    #plt.plot(sca, vec)
    #plt.plot(sca, sca, "*")
    #plt.plot(vec, vec, "-")
    


    a = 1
    b = 1*W
    b.favunit = W
    bb = 2*W
    bb.favunit = W
    bbb = 1.5*W
    bbb.favunit = W
    bbbb = 0.75*W
    bbbb.favunit = W
    bbbbb = 0.35*W
    bbbbb.favunit = 0.5*W
    bbbbb.favunit.symbol = "0.5*W"
    c = 2*mm
    cc = 1*mm
    cc.favunit =kg
    ccc = 1.5*mm
    ccc.favunit = kg
    cccc = 1.75*mm
    cccc.favunit = mm*kg
    ccccc = 1.25*mm
    ccccc.favunit = kg
    km = 1000*m
    km.symbol = "km"
    d = 1*W
    dd = d*2
    dd.favunit = kg/s
    ddd = d*3
    ddd.favunit = W
    
    #q_plot(b, c, "r*", label="2 mm sans favunit")
    #q_plot(bb, cc, "b*", label="1 mm avec favunit mm")
    #q_plot(bbb, ccc, "y*", label='1.5mm en 1.5W, favunit kg')
    #q_plot(bbbb, cccc, "g*", label="1.75mm en 0W, favunit mm*kg")
    #q_plot(bbbbb, ccccc, "*c")
    #plt.legend()
    #a = plt.gca()
    #b = change_favunit(a, "y", mm)
    #c = change_favunit(a, "y", kg)
    
    #print(a.get_ydata())
    
    
    
    
    # arr = np.array([1,2,3])
    # av = arr
    # bv = b*arr
    # cv = c*arr
    # dv = d*arr
    # 
    # sca = [a, b, c, d]
    # vec = [av, bv, cv, dv]
    # 
    # for x, y in zip(sca, sca):
    #     plt.plot(x, y)
    # for x, y in zip(vec, vec):
    #     plt.plot(x,y)
    
    # def func_q(T_C):
    #     return T_C + 273.12*K
    # 
    # def func_q2(T_C):
    #     return T_C + 273.12*K +20*K
    # 
    # def func_pasq(T):
    #     return T + 273.1
    # f_plot(func_q, 200*K, 300*K)
    # f_plot(func_pasq, 200, 300)
    # q = f_plot(func_q2, 200*K, 300*K)

if __name__ == "__main__":
    main()



#def f_plot(f, x0=0, xmin=-10, xmax=10,  *args, **kwargs):
#    x_value = np.arange(x0, x0+10)
#    func_vec = vectorize(f)
#    y = func_vec(x_value)
#    p = q_plot(x_value,y)
#    return p


#show(func_q, 233*K, 500*K)
#q_plot(x,y)
#a = q_plot(z,a)
#plt.xlim(-1, 10)
#f_plot(func_q)
