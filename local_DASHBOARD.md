
About wrapping numpy functions :
https://odlgroup.github.io/odl/guide/numpy_guide.html
https://gist.github.com/shoyer/36b84ab064f027df318c0b823558de24
https://docs.scipy.org/doc/numpy/release.html#array-ufunc-added
http://www.numpy.org/neps/nep-0013-ufunc-overrides.html
http://www.astro.utoronto.ca/%7Emhvk/numpy-doc/neps/ufunc-overrides.html#proposed-interface
https://pint.readthedocs.io/en/0.9/numpy.html#comments
 
From physics : 
 - some packages relies on hand-parsing string or regex to parse fractions ? (could allow not relying on sympy)
 - use lambda and map to add/sub dicts of dimension power
 - Dimension object (PhysicalUnit) have a scale_factor and a offset attribute, that are multiplied as Dimensions are multiplie (and power are added), with conversion_factor/tuple method
 - add rpow method (must be dimensionless)
 - try to import uncertainties
 
 - allow to fix the default precision output of an instance
 - allow to fix an output format (as string) of an instance
 
From astropy comparison : 
- display about :
 - add a pretty display with latex
 - change display of powered units : m**2 to m2 ?
- other
 - allow conversion in different unit system ?
 - astropy converts value to float ?
 - should .to() be inplace ?  
 - declare a favunit for SI_units_derived ? 
 - keep symbols when * or / quantities ?
 - [X] : add imperial units from astropy. See units.
 - add cgs units from astropy ? See units.
 - add astrophys units from astropy ? See units.
 - find other common unit with same dimension ?
 - [X] : add SI units in units
 - add a string parser constructor ?
 - add a decorator for checking dimension
 - add a parser to favunit for functions output
 - Add method to simply return value in other systems
 - Deal with powers with fractions if necessary (utils)

From pint :
 - Allow adding units from a text file
 - Remplacer isinstance(quantity) par isinstance(self.__class__)
 - To() et ito()
 - Faire des property ?
 - Essayer de s’affranchir de numpy ?
 - Ajouter une propriété dimensionnalty qui renvoi les length
 - Ajouter un dict de dimneisonnaltiy classique comme acceleration, qui peut être consulté pour afficher la diemnsionnalty de façon plus sympa
 - Différencier in et to, l’un qui change juste la favunit, l’autre qui vérifier en plus que la favunit est de même dimension que l’objet
 - Prévoir un nom ? (m = Quantity(1, Dimension(« L »), name= ‘meter’))
 - Pouvoir spécifier un specifier par défaut ; Quantity.default_spec = « :s »
 - Utiliser __array_wrap__ pour overloader les fonctions numpy (see pint)
 - Check and refuse any « ^ » notation in string parsing ? only python power ** ? what about sympy ?
 - Method check(“dimensionality”) pour verifier à la main la dimension
 - Decorateur pour check dimension (accept Dimension object, Quantity, dimensionnality)
 - Ajouter methode plus_minus pour ajouter une incertitude
 - Able to define the units and base system in a file
 
From unum:
 - Quantify is a staticmethod, and defined as a decorator outside the class
 - Formatter is defined as a class, set_formet and reset_format as classmethod
 - Add a copy method with flags to include repr formats.
 - Add Fraction support for values
 - Store all quantities through their symbol, and check for Name conflict ?
 - Clean imports with del
 - Store quantities in a dict-like object ?
 - Use of Unicode superscripts ? display only, copy-paste problems ?
 
From magnitude : 
 - create a dictionnary tied to Quantity class, containing all the quantities ? units ? 
 - should Dimension be stored within the Quantity class (Quantity(1, kg=1, m=1)) ?
 - allow creating quantities with value and str of symbol of other quantity ?
 - init create oprec and oformat to None. They can be set later, and str and repr rely on the module value of oprec and oformat
 
From quantities :
 - Should Dimension allow addition (returning the same) ?
 - Copy ? hash ?
 - Dimension dict can drop unit if dimension is 0, or add a new unit (when multiplicating for ex)
 - Allow comparing dimension (??)
 - All numpy function and their corresponding checks are stored in a dict
 - Uses eval

Set constants value hard value
Add a handler to check if scipy is available, and update constants values verbosily
Add matplotlib compatibility
créer des entry points pour permettre la conversion de quantités en ligne de coommande
If a result dimension is equal to one of the 22 SI derived units, make it its favunit.
 
