from physipy import *
from physipy.qwidgets.qipywidgets import QuantityText
import traitlets


class QTrait(traitlets.TraitType):
    info_text = 'a quantity'

    def validate(self, obj, value):
        if isinstance(value, Quantity):
            return value
        self.error(obj, value)


class QuantityWithWidget(Quantity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        w = QuantityText(Quantity(self.value, self.dimension))
        self.w = w


def widgetify(q):
    q = quantify(q)
    return QuantityWithWidget(q.value, q.dimension)


class TraitedQuantity(traitlets.HasTraits):
    value = QTrait()

    def __init__(self, q):
        self.widget = QuantityText(q)
        self.qvalue = q

        traitlets.link(
            (self.widget, "value"),
            (self, "qvalue"),
        )

        # def _update_q(change):
        #    q = change.new
        #    self.value = q
        # self.widget.observe(_update_q, names="value")

    # @traitlets.observe("value")
    # def _update_w(self, change):
    #    q = change.new
    #    self.widget.value = q
    # self.value.observe(_update_w, names="value")

    def __repr__(self):
        return repr(self.value)


Q = TraitedQuantity(3 * m)
