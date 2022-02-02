
class ObservableQuantityDescriptor():
    
    def __init__(self, deps=[]):
        self.deps = deps
        
    def __set_name__(self, owner, name):
        # self.R
        self.public_name = name
        # actually refers to self._R_w
        self.private_name = '_' + name + "_observable_proxy_descriptor"
    
    def __set__(self, obj, qvalue):
        # if not ObservableQuantity exists already, where value is a quantity
        if not hasattr(obj, self.private_name):
            #print("setting value")
            setattr(obj, self.private_name, qvalue)
        # if a ObservableQuantity is there, overwrite it
        else:
            if not qvalue is getattr(obj, self.private_name):
                old = getattr(obj, self.private_name)
                new = qvalue
                change = {"old":old, "new":new}
                setattr(obj, self.private_name, new)
                for dep in self.deps:
                    getattr(obj, "compute_"+dep)(change)
            else:
                pass
        if hasattr(obj, "_observables_dict"):
            if self.public_name in obj._observables_dict:
                return
            else:
                obj._observables_dict[self.public_name] = getattr(obj, self.private_name)
        else:
            # create a list of the observables
            setattr(obj, "_observables_dict", {})
            obj._observables_dict[self.public_name] = getattr(obj, self.private_name)

            
        
    def __get__(self, obj, objtype=None):
        if hasattr(obj, self.private_name):
            # get the ObservableQuantity instance, so basically a Quantity
            value = getattr(obj, self.private_name)
            return value
        # if it doesn't exist yet
        else:
            # first compute to set the value
            # the compute_toto() call should set a .toto value
            getattr(obj, "compute_"+self.public_name)({})
            # return the newly set value .toto
            return getattr(obj, self.public_name)

        
if __name__ == "__main__":

    ms = units["ms"]

    class RC():

        # to check the list of available observables : rc._observables_dict
        R = ObservableQuantityDescriptor(["tau"])
        C = ObservableQuantityDescriptor(["tau"])
        tau = ObservableQuantityDescriptor()

        def __init__(self, R, C):
            self.R = R
            self.C = C

        def compute_tau(self, change):
            self.tau = self.R * self.C
            self.tau.favunit = ms


    rc = RC(1*ohm, 1*F)
    print("First getting")
    print(rc.tau)

    print("second")
    rc.R = 2*ohm
    print(rc.C, rc.R, rc.tau)
    rc._observables_dict