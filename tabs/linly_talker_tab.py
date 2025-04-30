import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                               QComboBox, QMessageBox)

from ui_components import VideoPlayer


class LinlyTalkerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 视频文件夹
        self.video_folder = QLineEdit("videos")
        self.layout.addWidget(QLabel("视频文件夹"))
        self.layout.addWidget(self.video_folder)

        # AI配音方式
        self.talker_method = QComboBox()
        self.talker_method.addItems(['Wav2Lip', 'Wav2Lipv2', 'SadTalker'])
        self.layout.addWidget(QLabel("AI配音方式"))
        self.layout.addWidget(self.talker_method)

        # 施工中提示
        construction_label = QLabel("施工中，请静候佳音 可参考 https://github.com/Kedreamix/Linly-Talker")
        construction_label.setOpenExternalLinks(True)
        self.layout.addWidget(construction_label)

        # 状态显示
        self.status_label = QLabel("功能开发中")
        self.layout.addWidget(QLabel("合成状态:"))
        self.layout.addWidget(self.status_label)

        # 视频播放器
        self.video_player = VideoPlayer("合成视频")
        self.layout.addWidget(self.video_player)

        self.setLayout(self.layout)