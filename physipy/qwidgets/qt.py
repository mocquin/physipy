from physipy import units
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QMainWindow
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QApplication, QVBoxLayout, QLabel, QMainWindow
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSpacerItem, \
    QVBoxLayout, QWidget, QLineEdit, QComboBox
import PyQt5.QtWidgets
import PyQt5.QtCore as QtCore


from physipy import quantify, Quantity


class QuantityQtSlider(QWidget):
    """
    QSlider are integer slider so we have to handle the floating point conversion ourselve.
    QSlider  :   0  -----------  N     : raw_value
    Quantity : qmin ----------- qmax   : public_value
    """

    def __init__(self, qminimum, qmaximum, value=None,
                 descr="Quantity", favunit=None, parent=None):
        super(QuantityQtSlider, self).__init__(parent=parent)
        self.setAutoFillBackground(True)
        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.Color("#6d6875")) # 6d6875
        # self.setPalette(p)
        self.setStyleSheet('background-color: #e36414;')
        self.descr = descr

        from physipy import units
        from numpy import pi
        self.units = units
        self.context = {**units, "pi": pi}

        # Horizontal Box
        self.horizontalLayout = QHBoxLayout(self)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        # Label for value
        self.numlabel = QLabel(self)
        self.numlabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.numlabel.setMinimumWidth(100)
        self.numlabel.setStyleSheet("background-color: #fb8b24")
        # self.numlabel.setMargin(0)
        # Label for description
        self.desclabel = QLabel(self)
        self.desclabel.setText(descr)
        self.desclabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.desclabel.setMinimumWidth(20)
        self.desclabel.setStyleSheet(
            "background-color: #9a031e")  # ;margin:0px")
        # QLineEdit
        self.qlineedit = QLineEdit()
        self.qlineedit.setMinimumWidth(20)
        # ComboBox
        self.cb = QComboBox()
        units_strings = [k for k in units.keys()]
        self.cb.addItems([u_str for u_str, u_q in self.units.items()
                         if qminimum.dimension == u_q.dimension])
        self.cb.currentIndexChanged.connect(self.selectionchange)

        # Raw slider
        self.qtslider = QSlider(self)

        # Add widgets
        self.horizontalLayout.addWidget(self.desclabel)
        self.horizontalLayout.addWidget(self.qtslider)
        self.horizontalLayout.addWidget(self.numlabel)
        self.horizontalLayout.addWidget(self.qlineedit)
        self.horizontalLayout.addWidget(self.cb)

        # Customize slider
        self.N = 1000000
        self.qtslider.setMaximum(self.N)
        self.qtslider.setOrientation(Qt.Horizontal)
        self.qtslider.setFixedWidth(200)
        self.qtslider.valueChanged.connect(self.setLabelValue)
        self.qtslider.setStyleSheet("background-color: #5f0f40")

        self.resize(self.sizeHint())

        # Init
        qminimum = quantify(qminimum)
        qmaximum = quantify(qmaximum)

        if not qminimum.dimension == qmaximum.dimension:
            raise ValueError

        if value is not None:
            value = quantify(value)
            if not qminimum.dimension == value.dimension:
                raise ValueError
        else:
            value = qminimum

        # Save min and max quantity
        self.qminimum = qminimum
        self.qmaximum = qmaximum
        self.dimension = qminimum.dimension

        # favunit
        if favunit is None:
            # we fall back on the passed quantity's favunit
            # (that could be None also)
            self.favunit = value._pick_smart_favunit()
        else:
            self.favunit = favunit
        self.value = value
        self.value.favunit = self.favunit

        # Programmatically sets the text
        self.qlineedit.setText(self.text_value)
        self.qtslider.setValue(self.public_to_raw(value))
        # self.setLabelValue(self.qtslider.value())

        self.qlineedit.returnPressed.connect(self.parse_qlineedit)

    def selectionchange(self):
        unit_str = self.cb.currentText()
        favunit = self.units[unit_str]
        self.favunit = favunit
        self.value.favunit = favunit
        # Programmatically sets the text
        self.qlineedit.setText(self.text_value)
        self.qtslider.setValue(self.public_to_raw(self.value))
        self.setLabelValue(self.public_to_raw(self.value))

        # self.setLabelValue(self.qtslider.value)

    def parse_qlineedit(self):
        text = self.qlineedit.text()  # Retrieves text in the field
        try:
            expression = text.replace(" ", "*")  # to parse "5 m" into "5*m"
            old_favunit = self.favunit
            # TODO : use a Lexer/Parser to allow
            # only valid mathematical text
            res = eval(expression, self.context)
            res = quantify(res)

            # update quantity value
            self.value = res
            self.favunit = old_favunit
            # see above, TODO find a way to link those 2
            self.value.favunit = self.favunit

            # update display_value
            # done by slider signal
            # update slider value
            self.qtslider.setValue(self.public_to_raw(res))

        except BaseException:
            # if anything fails, do nothing
            # self.text.value = str(self.value) # self.value.favunit is used
            # here
            pass

    def raw_to_public(self, raw_value):
        try:
            public_value = self.qminimum + float(raw_value) * (self.qmaximum - self.qminimum) / (
                self.qtslider.maximum() - self.qtslider.minimum())
            return public_value
        except BaseException:
            pass

    def public_to_raw(self, public_value):
        try:
            raw_value = (public_value - self.qminimum) / (self.qmaximum - \
                         self.qminimum) * (self.qtslider.maximum() - self.qtslider.minimum())
            return raw_value
        except BaseException:
            pass

    def setLabelValue(self, qt_value):
        self.value = self.raw_to_public(qt_value)
        self.value.favunit = self.favunit
        self.numlabel.setText(self.text_value)
        self.qlineedit.setText(self.text_value)

    @property
    def text_value(self):
        return "{0:.5f}".format(self.value)


class ParamSet(QWidget):

    def __init__(self, dict_model):
        super(ParamSet, self).__init__()

        self.dict_model = dict_model
        self.layout = QVBoxLayout()
        self.sliders = []
        for k, v in self.dict_model.items():
            qmin = v["min"]
            qmax = v["max"]
            val = v["value"]

            qs = QuantityQtSlider(qmin, qmax, value=val, descr=k)
            self.sliders.append(qs)

        for qs in self.sliders:
            self.layout.addWidget(qs)
        self.setLayout(self.layout)
