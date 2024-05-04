import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit

class OutputTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.output_text_edit = OutputTextEdit()
        self.setCentralWidget(self.output_text_edit)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
