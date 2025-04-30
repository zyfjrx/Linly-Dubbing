import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtCore import Qt

# 请确保导入正确模块
try:
    # 导入自定义控件文件
    from ui_components import (CustomSlider, FloatSlider, RadioButtonGroup,
                               AudioSelector, VideoPlayer)

    # 导入各个功能标签页
    from tabs.full_auto_tab import FullAutoTab
    from tabs.settings_tab import SettingsTab
    from tabs.download_tab import DownloadTab
    from tabs.demucs_tab import DemucsTab
    from tabs.asr_tab import ASRTab
    from tabs.translation_tab import TranslationTab
    from tabs.tts_tab import TTSTab
    from tabs.video_tab import SynthesizeVideoTab
    from tabs.linly_talker_tab import LinlyTalkerTab

    # 尝试导入实际的功能模块
    try:
        from tools.step000_video_downloader import download_from_url
        from tools.step010_demucs_vr import separate_all_audio_under_folder
        from tools.step020_asr import transcribe_all_audio_under_folder
        from tools.step030_translation import translate_all_transcript_under_folder
        from tools.step040_tts import generate_all_wavs_under_folder
        from tools.step050_synthesize_video import synthesize_all_video_under_folder
        from tools.do_everything import do_everything
        from tools.utils import SUPPORT_VOICE
    except ImportError as e:
        print(f"警告: 无法导入一些工具模块: {e}")
        # 定义临时的支持语音列表
        SUPPORT_VOICE = ['zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural',
                         'en-US-JennyNeural', 'ja-JP-NanamiNeural']

except ImportError as e:
    print(f"错误: 初始化应用程序失败: {e}")
    sys.exit(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("智能视频多语言AI配音/翻译工具 - Linly-Dubbing")
        self.resize(1024, 768)

        # 创建选项卡
        self.tab_widget = QTabWidget()

        # 创建标签页实例
        self.full_auto_tab = FullAutoTab()
        self.settings_tab = SettingsTab()

        # 连接配置页面的配置变更信号到一键自动化页面
        self.settings_tab.config_changed.connect(self.full_auto_tab.update_config)

        # 添加各个选项卡
        self.tab_widget.addTab(self.full_auto_tab, "一键自动化 One-Click")
        self.tab_widget.addTab(self.settings_tab, "配置页面 Settings")
        self.tab_widget.addTab(DownloadTab(), "自动下载视频")
        self.tab_widget.addTab(DemucsTab(), "人声分离")
        self.tab_widget.addTab(ASRTab(), "AI智能语音识别")
        self.tab_widget.addTab(TranslationTab(), "字幕翻译")
        self.tab_widget.addTab(TTSTab(), "AI语音合成")
        self.tab_widget.addTab(SynthesizeVideoTab(), "视频合成")
        self.tab_widget.addTab(LinlyTalkerTab(), "Linly-Talker 对口型（开发中）")

        # 设置中央窗口部件
        self.setCentralWidget(self.tab_widget)


def main():
    # 设置高DPI缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    # 创建主窗口
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()