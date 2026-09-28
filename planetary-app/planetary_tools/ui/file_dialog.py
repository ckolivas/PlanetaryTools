"""Shared file picker with favourites above a separate recent-directory list."""
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QFileDialog as QtFileDialog, QFrame, QLabel, QListView, QListWidget,
    QListWidgetItem, QSplitter, QStyle, QVBoxLayout, QWidget,
)

from planetary_tools.ui import recent_files


class QFileDialog(QtFileDialog):
    def __init__(self, parent=None, caption="", directory="", filter=""):
        super().__init__(parent)
        self.setOption(self.Option.DontUseNativeDialog)
        self.setWindowTitle(caption)
        if filter:
            self.setNameFilter(filter)
        if directory:
            if Path(directory).is_dir():
                self.setDirectory(directory)
            else:
                self.setDirectory(str(Path(directory).parent))
                self.selectFile(Path(directory).name)

        favourites = recent_files._settings().value("fileDialog/favourites")
        if favourites is not None:
            if isinstance(favourites, str):
                favourites = [favourites]
            self.setSidebarUrls([QUrl(url) for url in favourites])
        self._add_recent_sidebar()
        self.finished.connect(self._remember_selection)

    def _add_recent_sidebar(self):
        # Keep Qt's own favourites view (including drag/drop and its context
        # menu). Qt has no public separator API for setSidebarUrls, so wrap the
        # existing widget instead of adding non-directory rows to its model.
        sidebar = self.findChild(QListView, "sidebar")
        if sidebar is None or not isinstance(sidebar.parentWidget(), QSplitter):
            return
        splitter = sidebar.parentWidget()
        panel = QWidget()
        panel.setObjectName("directorySidebar")
        splitter.replaceWidget(splitter.indexOf(sidebar), panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Favourites"))
        layout.addWidget(sidebar, 1)
        separator = QFrame()
        separator.setObjectName("recentDirectoriesSeparator")
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        layout.addWidget(QLabel("Recent directories"))
        self.recent_directories = QListWidget()
        self.recent_directories.setObjectName("recentDirectories")
        self.recent_directories.setAccessibleName("Recent directories")
        self.recent_directories.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        for directory in recent_files.list_recent_directories():
            item = QListWidgetItem(icon, Path(directory).name or directory)
            item.setData(Qt.ItemDataRole.UserRole, directory)
            item.setToolTip(directory)
            self.recent_directories.addItem(item)
        self.recent_directories.itemClicked.connect(self._visit_recent)
        self.recent_directories.itemActivated.connect(self._visit_recent)
        layout.addWidget(self.recent_directories, 1)

    def _visit_recent(self, item):
        directory = item.data(Qt.ItemDataRole.UserRole)
        if Path(directory).is_dir():
            self.setDirectory(directory)

    def _remember_selection(self, result):
        settings = recent_files._settings()
        settings.setValue("fileDialog/favourites", [url.toString() for url in self.sidebarUrls()])
        if result == self.DialogCode.Accepted:
            for path in reversed(self.selectedFiles()):
                directory = Path(path) if self.fileMode() == self.FileMode.Directory else Path(path).parent
                recent_files.remember_directory(directory)
        settings.sync()

    @classmethod
    def _choose(cls, parent, caption, directory, filter, initialFilter, options, mode, accept):
        dialog = cls(parent, caption, directory, filter)
        dialog.setOptions(options | cls.Option.DontUseNativeDialog)
        dialog.setFileMode(mode)
        dialog.setAcceptMode(accept)
        if initialFilter:
            dialog.selectNameFilter(initialFilter)
        try:
            if dialog.exec() == cls.DialogCode.Accepted:
                return dialog.selectedFiles(), dialog.selectedNameFilter()
            return [], ""
        finally:
            dialog.deleteLater()

    @classmethod
    def getOpenFileName(cls, parent=None, caption="", directory="", filter="", initialFilter="", options=QtFileDialog.Option(0)):
        paths, selected = cls._choose(parent, caption, directory, filter, initialFilter, options,
                                      cls.FileMode.ExistingFile, cls.AcceptMode.AcceptOpen)
        return (paths[0] if paths else ""), selected

    @classmethod
    def getOpenFileNames(cls, parent=None, caption="", directory="", filter="", initialFilter="", options=QtFileDialog.Option(0)):
        return cls._choose(parent, caption, directory, filter, initialFilter, options,
                           cls.FileMode.ExistingFiles, cls.AcceptMode.AcceptOpen)

    @classmethod
    def getSaveFileName(cls, parent=None, caption="", directory="", filter="", initialFilter="", options=QtFileDialog.Option(0)):
        paths, selected = cls._choose(parent, caption, directory, filter, initialFilter, options,
                                      cls.FileMode.AnyFile, cls.AcceptMode.AcceptSave)
        return (paths[0] if paths else ""), selected

    @classmethod
    def getExistingDirectory(cls, parent=None, caption="", directory="", options=QtFileDialog.Option.ShowDirsOnly):
        paths, _ = cls._choose(parent, caption, directory, "", "", options,
                               cls.FileMode.Directory, cls.AcceptMode.AcceptOpen)
        return paths[0] if paths else ""
