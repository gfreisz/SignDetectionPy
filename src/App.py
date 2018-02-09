import sys

from PyQt4 import QtGui
from Views.FormMain import form_main

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    windows = form_main.form_main()
    windows.show()
    sys.exit(app.exec_())
