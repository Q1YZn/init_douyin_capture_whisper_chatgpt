from __future__ import annotations

import os
import sys
from datetime import datetime

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import DesktopSettings
from .db import ClientJobRepository
from .f2_runtime import F2RuntimeReport
from .live_capture import MonitorEvent
from .i18n import TranslationCatalog, normalize_language
from .models import (
    DesktopAnalysisRequest,
    MonitorProfileRecord,
    WorkflowSnapshot,
    WorkflowStage,
    WorkflowStatus,
)
from .notifications import MemoryPressureWatcher, RobotNotificationConfig, RobotNotifier
from .services import DesktopWorkflowService, MonitorProfileService
from .ui_tokens import build_app_stylesheet, get_tokens


OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
DEEPSEEK_PROVIDER = "deepseek"

ANALYSIS_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    OPENAI_COMPATIBLE_PROVIDER: {
        "base_url": "http://127.0.0.1:8317/v1",
        "model_id": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
    },
    DEEPSEEK_PROVIDER: {
        "base_url": "https://api.deepseek.com",
        "model_id": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}


class AnalysisWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, workflow: DesktopWorkflowService, request: DesktopAnalysisRequest) -> None:
        super().__init__()
        self.workflow = workflow
        self.request = request

    def run(self) -> None:
        try:
            result = self.workflow.run_analysis(self.request)
            self.finished_ok.emit(
                {
                    "job_id": result.job_id,
                    "summary_path": str(result.summary_path),
                    "timeline_path": str(result.timeline_path),
                    "report_url": result.report_url,
                    "uploaded": result.uploaded,
                    "asr_model_reference": result.asr_model_reference or "",
                    "asr_device": result.asr_device or "",
                    "asr_compute_type": result.asr_compute_type or "",
                    "video_size_bytes": result.video_size_bytes,
                    "analysis_tier": result.analysis_tier or "",
                    "payment_required": result.payment_required,
                    "asset_storage": result.asset_storage or "",
                    "asset_uri": result.asset_uri or "",
                    "offload_required": result.offload_required,
                    "summary_excerpt": result.summary_excerpt or "",
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class CaptureWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, workflow: DesktopWorkflowService, room_id_or_url: str) -> None:
        super().__init__()
        self.workflow = workflow
        self.room_id_or_url = room_id_or_url

    def stop(self) -> None:
        self.workflow.stop_capture()

    def run(self) -> None:
        try:
            request = self.workflow.build_capture_request(self.room_id_or_url)
            result = self.workflow.run_capture(request)
            self.finished_ok.emit(
                {
                    "room_id": result.room_id,
                    "replay_path": str(result.replay_path) if result.replay_path else "",
                    "occupancy_path": str(result.occupancy_path),
                    "danmaku_path": str(result.danmaku_path),
                    "metadata_path": str(result.metadata_path),
                    "workspace_dir": str(result.workspace_dir),
                    "stopped_by_user": result.stopped_by_user,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class MonitorWorker(QThread):
    event_emitted = Signal(dict)
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(self, workflow: DesktopWorkflowService, profile_urls: list[str]) -> None:
        super().__init__()
        self.workflow = workflow
        self.profile_urls = profile_urls

    def stop(self) -> None:
        self.workflow.stop_monitor()

    def _emit_event(self, event: MonitorEvent) -> None:
        self.event_emitted.emit(
            {
                "event_type": event.event_type,
                "profile_url": event.profile_url,
                "message": event.message,
                "nickname": event.nickname or "",
                "room_id": event.room_id or "",
                "webcast_id": event.webcast_id or "",
                "replay_path": str(event.replay_path) if event.replay_path else "",
                "workspace_dir": str(event.workspace_dir) if event.workspace_dir else "",
                "severity": event.severity.value if event.severity else "",
                "stage": event.stage.value if event.stage else "",
                "status": event.status.value if event.status else "",
            }
        )

    def run(self) -> None:
        try:
            request = self.workflow.build_monitor_request(self.profile_urls)
            self.workflow.run_monitor(request, event_callback=self._emit_event)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    MONITOR_URL_ROLE = Qt.ItemDataRole.UserRole
    OUTPUT_KEY_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self, app_mode: str = "all", *, auto_start_monitor: bool | None = None) -> None:
        super().__init__()
        self.app_mode = app_mode
        self.auto_start_monitor = (app_mode == "capture") if auto_start_monitor is None else auto_start_monitor
        settings = DesktopSettings.from_env()
        repo = ClientJobRepository(settings.sqlite_path)
        self.repo = repo
        self.workflow = DesktopWorkflowService(settings, repo)
        self.monitor_profiles = MonitorProfileService(repo)
        self.worker: AnalysisWorker | None = None
        self.capture_worker: CaptureWorker | None = None
        self.monitor_worker: MonitorWorker | None = None
        self.f2_report = self.workflow.probe_f2_runtime()
        self.current_language = self._load_language_preference()
        self.current_theme_mode = self._load_theme_mode_preference()
        self.current_analysis_provider = self._load_analysis_provider_preference()
        self.catalog = TranslationCatalog(self.current_language)
        self.tokens = get_tokens(self._resolve_color_scheme())
        self.robot_notifier = RobotNotifier()
        self.memory_watcher = MemoryPressureWatcher()
        self.memory_timer = QTimer(self)
        self.memory_timer.setInterval(60_000)
        self.memory_timer.timeout.connect(self._check_memory_pressure)
        self.current_snapshot = WorkflowSnapshot(
            title=self._tr("ready").rstrip("."),
            stage=WorkflowStage.IDLE,
            status=WorkflowStatus.IDLE,
            message=self._tr("ready"),
            progress=0,
            mode_label=self._mode_label(WorkflowStage.IDLE),
        )
        self.setStyleSheet(build_app_stylesheet(self.tokens))
        self._build_ui()
        self._set_mono_fonts()
        self._set_asr_runtime_info(self.workflow.resolve_asr_model_reference(), self._tr("ready"), "")
        self._set_f2_runtime_info(self.f2_report)
        self._apply_snapshot(self.current_snapshot, push_timeline=False)
        self._load_monitor_profiles()
        self._load_runtime_warnings()
        app = QApplication.instance()
        if app is not None:
            app.styleHints().colorSchemeChanged.connect(self._on_system_color_scheme_changed)
        if self._supports_capture_mode():
            self.memory_timer.start()
        QTimer.singleShot(0, self._maybe_prompt_start_saved_monitor)

    def _build_ui(self) -> None:
        self.setWindowTitle(self._window_title_text())
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        root.addWidget(self._build_header())
        root.addWidget(self._build_notice_banner())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.resize(1440, 920)

    def _tr(self, key: str, **kwargs: object) -> str:
        return self.catalog.text(key, **kwargs)

    def _lang_text(self, zh_text: str, en_text: str) -> str:
        return zh_text if self.current_language == "zh_CN" else en_text

    def _supports_capture_mode(self) -> bool:
        return self.app_mode in {"all", "capture"}

    def _supports_analysis_mode(self) -> bool:
        return self.app_mode in {"all", "analyzer"}

    def _window_title_text(self) -> str:
        if self.app_mode == "capture":
            return self._lang_text("抖音录播采集端", "Douyin Capture Client")
        if self.app_mode == "analyzer":
            return self._lang_text("抖音视频分析端", "Douyin Analyzer Client")
        return self._tr("app_title")

    def _window_subtitle_text(self) -> str:
        if self.app_mode == "capture":
            return self._lang_text(
                "无人值守监控直播并落地录播、人数与弹幕文件。",
                "Unattended live monitoring with replay, occupancy, and danmaku capture.",
            )
        if self.app_mode == "analyzer":
            return self._lang_text(
                "导入视频与结构化文件，透传到兼容 OpenAI / DeepSeek 的分析模型。",
                "Import video and structured files, then pass through to OpenAI-compatible or DeepSeek analysis.",
            )
        return self._tr("app_subtitle")

    def _load_language_preference(self) -> str:
        saved = self.repo.get_setting("ui_language")
        if saved:
            return normalize_language(str(saved))
        return "zh_CN"

    @staticmethod
    def _normalize_theme_mode(value: str | None) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"light", "dark", "system"}:
            return normalized
        return "light"

    def _load_theme_mode_preference(self) -> str:
        saved = self.repo.get_setting("ui_theme_mode")
        return self._normalize_theme_mode(saved)

    def _load_analysis_provider_preference(self) -> str:
        saved = str(self.repo.get_setting("analysis_provider") or "").strip().lower()
        if saved in ANALYSIS_PROVIDER_PRESETS:
            return saved
        return DEEPSEEK_PROVIDER if self.app_mode == "analyzer" else OPENAI_COMPATIBLE_PROVIDER

    def _resolve_system_color_scheme(self) -> str:
        app = QApplication.instance()
        if app is None:
            return "light"
        return "dark" if app.styleHints().colorScheme() == Qt.ColorScheme.Dark else "light"

    def _resolve_color_scheme(self) -> str:
        if self.current_theme_mode == "dark":
            return "dark"
        if self.current_theme_mode == "system":
            return self._resolve_system_color_scheme()
        return "light"

    def _mode_label(self, stage: WorkflowStage) -> str:
        mapping = {
            WorkflowStage.IDLE: "mode_idle",
            WorkflowStage.ANALYSIS: "mode_analysis",
            WorkflowStage.CAPTURE: "mode_capture",
            WorkflowStage.MONITOR: "mode_monitor",
            WorkflowStage.COMPLETED: "mode_idle",
        }
        return self._tr(mapping.get(stage, "mode_idle"))

    def _apply_theme(self) -> None:
        self.tokens = get_tokens(self._resolve_color_scheme())
        self.setStyleSheet(build_app_stylesheet(self.tokens))
        if hasattr(self, "notice_label"):
            self._set_notice(self.current_snapshot.message, self.current_snapshot.status)

    def _on_system_color_scheme_changed(self, *_args: object) -> None:
        if self.current_theme_mode != "system":
            return
        self._apply_theme()

    def _on_language_changed(self, index: int) -> None:
        if index < 0:
            return
        value = str(self.language_combo.itemData(index) or "")
        normalized = normalize_language(value)
        if normalized == self.current_language:
            return
        self.current_language = normalized
        self.catalog = TranslationCatalog(self.current_language)
        self.repo.set_setting("ui_language", normalized)
        self._refresh_translations()

    def _on_theme_mode_changed(self, index: int) -> None:
        if index < 0:
            return
        value = str(self.theme_combo.itemData(index) or "")
        normalized = self._normalize_theme_mode(value)
        if normalized == self.current_theme_mode:
            return
        self.current_theme_mode = normalized
        self.repo.set_setting("ui_theme_mode", normalized)
        self._apply_theme()

    def _analysis_preset(self, provider: str) -> dict[str, str]:
        preset = ANALYSIS_PROVIDER_PRESETS.get(provider, ANALYSIS_PROVIDER_PRESETS[OPENAI_COMPATIBLE_PROVIDER]).copy()
        env_key = preset.get("api_key_env", "")
        if env_key:
            preset["api_key"] = os.getenv(env_key, "")
        if provider == DEEPSEEK_PROVIDER:
            preset["model_id"] = os.getenv("DEEPSEEK_MODEL_ID", preset["model_id"])
            preset["base_url"] = os.getenv("DEEPSEEK_BASE_URL", preset["base_url"])
        else:
            preset["base_url"] = os.getenv("OPENAI_BASE_URL", preset["base_url"])
            preset["model_id"] = os.getenv("OPENAI_CHAT_MODEL_ID", preset["model_id"])
        return preset

    def _analysis_provider_display(self, provider: str) -> str:
        if provider == DEEPSEEK_PROVIDER:
            return self._lang_text("DeepSeek 直连", "DeepSeek Direct")
        return self._lang_text("OpenAI 兼容代理", "OpenAI-Compatible Proxy")

    def _notification_provider_display(self, provider: str) -> str:
        mapping = {
            "none": self._lang_text("不发送", "Disabled"),
            "feishu": self._lang_text("飞书机器人", "Feishu Robot"),
            "dingtalk": self._lang_text("钉钉机器人", "DingTalk Robot"),
        }
        return mapping.get(provider, provider)

    def _load_notification_config(self) -> RobotNotificationConfig:
        provider = str(self.repo.get_setting("capture_notification_provider", "none") or "none").strip().lower()
        if provider not in {"none", "feishu", "dingtalk"}:
            provider = "none"
        webhook_url = str(self.repo.get_setting("capture_notification_webhook", "") or "").strip()
        threshold_raw = self.repo.get_setting("capture_memory_threshold_mb", 2048)
        cooldown_raw = self.repo.get_setting("capture_notification_cooldown_minutes", 30)
        try:
            threshold = max(int(threshold_raw), 256)
        except (TypeError, ValueError):
            threshold = 2048
        try:
            cooldown = max(int(cooldown_raw), 1)
        except (TypeError, ValueError):
            cooldown = 30
        return RobotNotificationConfig(
            provider=provider,
            webhook_url=webhook_url,
            memory_threshold_mb=threshold,
            cooldown_minutes=cooldown,
        )

    def _save_notification_config(self) -> RobotNotificationConfig:
        config = self._current_notification_config()
        self.repo.set_setting("capture_notification_provider", config.provider)
        self.repo.set_setting("capture_notification_webhook", config.webhook_url)
        self.repo.set_setting("capture_memory_threshold_mb", config.memory_threshold_mb)
        self.repo.set_setting("capture_notification_cooldown_minutes", config.cooldown_minutes)
        return config

    def _current_notification_config(self) -> RobotNotificationConfig:
        if not hasattr(self, "notification_provider_combo"):
            return RobotNotificationConfig()
        return RobotNotificationConfig(
            provider=str(self.notification_provider_combo.currentData() or "none"),
            webhook_url=self.notification_webhook_input.text().strip(),
            memory_threshold_mb=int(self.memory_threshold_spin.value()),
            cooldown_minutes=int(self.notification_cooldown_spin.value()),
        )

    def _load_analysis_provider_settings(self) -> None:
        provider = self.current_analysis_provider
        preset = self._analysis_preset(provider)
        base_url = str(self.repo.get_setting("analysis_base_url") or preset["base_url"]).strip()
        api_key = str(self.repo.get_setting("analysis_api_key") or preset.get("api_key", "")).strip()
        model_id = str(self.repo.get_setting("analysis_model_id") or preset["model_id"]).strip()
        self.analysis_provider_combo.blockSignals(True)
        self.analysis_provider_combo.setCurrentIndex(max(self.analysis_provider_combo.findData(provider), 0))
        self.analysis_provider_combo.blockSignals(False)
        self.analysis_base_url_input.setText(base_url)
        self.analysis_api_key_input.setText(api_key)
        self.analysis_model_input.setText(model_id)

    def _save_analysis_provider_settings(self) -> None:
        if not hasattr(self, "analysis_provider_combo"):
            return
        provider = str(self.analysis_provider_combo.currentData() or OPENAI_COMPATIBLE_PROVIDER)
        self.current_analysis_provider = provider
        self.repo.set_setting("analysis_provider", provider)
        self.repo.set_setting("analysis_base_url", self.analysis_base_url_input.text().strip())
        self.repo.set_setting("analysis_api_key", self.analysis_api_key_input.text().strip())
        self.repo.set_setting("analysis_model_id", self.analysis_model_input.text().strip())

    def _on_analysis_provider_changed(self, index: int) -> None:
        if index < 0:
            return
        provider = str(self.analysis_provider_combo.itemData(index) or OPENAI_COMPATIBLE_PROVIDER)
        preset = self._analysis_preset(provider)
        self.current_analysis_provider = provider
        self.analysis_base_url_input.setText(preset["base_url"])
        self.analysis_api_key_input.setText(preset.get("api_key", ""))
        self.analysis_model_input.setText(preset["model_id"])
        self._save_analysis_provider_settings()

    def _refresh_analysis_provider_labels(self) -> None:
        self.analysis_provider_label.setText(self._lang_text("分析模型", "Analysis Provider"))
        self.analysis_provider_combo.blockSignals(True)
        self.analysis_provider_combo.setItemText(
            0,
            self._analysis_provider_display(OPENAI_COMPATIBLE_PROVIDER),
        )
        self.analysis_provider_combo.setItemText(
            1,
            self._analysis_provider_display(DEEPSEEK_PROVIDER),
        )
        current = self.analysis_provider_combo.currentData()
        if current:
            self.analysis_provider_combo.setCurrentIndex(max(self.analysis_provider_combo.findData(current), 0))
        self.analysis_provider_combo.blockSignals(False)
        self.analysis_base_url_label.setText(self._lang_text("接口地址", "Base URL"))
        self.analysis_api_key_label.setText(self._lang_text("接口密钥", "API Key"))
        self.analysis_model_label.setText(self._lang_text("模型名称", "Model ID"))

    def _refresh_notification_labels(self) -> None:
        self.notification_provider_label.setText(self._lang_text("机器人通知", "Robot Notification"))
        self.notification_provider_combo.blockSignals(True)
        self.notification_provider_combo.setItemText(0, self._notification_provider_display("none"))
        self.notification_provider_combo.setItemText(1, self._notification_provider_display("feishu"))
        self.notification_provider_combo.setItemText(2, self._notification_provider_display("dingtalk"))
        current = self.notification_provider_combo.currentData()
        if current:
            self.notification_provider_combo.setCurrentIndex(max(self.notification_provider_combo.findData(current), 0))
        self.notification_provider_combo.blockSignals(False)
        self.notification_webhook_label.setText(self._lang_text("Webhook 地址", "Webhook URL"))
        self.memory_threshold_label.setText(self._lang_text("低内存阈值(MB)", "Low Memory Threshold (MB)"))
        self.notification_cooldown_label.setText(self._lang_text("告警冷却(分钟)", "Alert Cooldown (min)"))
        self.notification_save_button.setText(self._lang_text("保存告警配置", "Save Alert Settings"))

    def _refresh_translations(self) -> None:
        self.setWindowTitle(self._window_title_text())
        self.header_title.setText(self._window_title_text())
        self.header_subtitle.setText(self._window_subtitle_text())
        self.language_label.setText(self._tr("language_label"))
        self.theme_label.setText(self._tr("theme_mode"))
        current = self.language_combo.currentData()
        self.language_combo.blockSignals(True)
        self.language_combo.setItemText(0, self._tr("language_zh"))
        self.language_combo.setItemText(1, self._tr("language_en"))
        if current:
            self.language_combo.setCurrentIndex(max(self.language_combo.findData(current), 0))
        self.language_combo.blockSignals(False)
        current_theme = self.theme_combo.currentData()
        self.theme_combo.blockSignals(True)
        self.theme_combo.setItemText(0, self._tr("theme_light"))
        self.theme_combo.setItemText(1, self._tr("theme_dark"))
        self.theme_combo.setItemText(2, self._tr("theme_system"))
        if current_theme:
            self.theme_combo.setCurrentIndex(max(self.theme_combo.findData(current_theme), 0))
        self.theme_combo.blockSignals(False)
        if hasattr(self, "analysis_provider_label"):
            self._refresh_analysis_provider_labels()
        if hasattr(self, "notification_provider_label"):
            self._refresh_notification_labels()
        self.status_card_eyebrows["mode"].setText(self._tr("status_mode"))
        self.status_card_eyebrows["last_task"].setText(self._tr("status_last_task"))
        self.status_card_eyebrows["asr"].setText(self._tr("status_asr"))
        self.status_card_eyebrows["f2"].setText(self._tr("status_f2"))
        self.capture_group.setTitle(self._tr("capture_group"))
        if hasattr(self, "monitor_group"):
            self.monitor_group.setTitle(self._tr("monitor_group"))
        self.workflow_group.setTitle(self._tr("workflow_group"))
        self.outputs_group.setTitle(self._tr("outputs_group"))
        self.summary_group.setTitle(self._tr("summary_group"))
        self.log_group.setTitle(self._tr("log_group"))
        self.live_target_label.setText(self._tr("live_target"))
        self.replay_label.setText(self._tr("replay_label"))
        self.occupancy_label.setText(self._tr("occupancy_label"))
        self.danmaku_label.setText(self._tr("danmaku_label"))
        self.room_input.setPlaceholderText(self._tr("live_target_placeholder"))
        self.monitor_url_input.setPlaceholderText(self._tr("monitor_url_placeholder"))
        self.capture_hint.setText(self._tr("capture_hint"))
        self.capture_button.setText(self._tr("capture_button"))
        self.stop_capture_button.setText(self._tr("stop_capture_button"))
        self.run_button.setText(self._tr("run_analysis_button"))
        if hasattr(self, "add_monitor_button"):
            self.add_monitor_button.setText(self._tr("add_profile_button"))
            self.remove_monitor_button.setText(self._tr("remove_selected_button"))
            self.monitor_button.setText(self._tr("monitor_start_button"))
            self.save_monitor_button.setText(self._tr("monitor_save_button"))
            self.stop_monitor_button.setText(self._tr("monitor_stop_button"))
        self.replay_browse_button.setText(self._tr("browse_button"))
        self.occupancy_browse_button.setText(self._tr("browse_button"))
        self.danmaku_browse_button.setText(self._tr("browse_button"))
        if hasattr(self, "monitor_tree"):
            self.monitor_tree.setHeaderLabels(
                [
                    self._tr("monitor_headers_enabled"),
                    self._tr("monitor_headers_url"),
                    self._tr("monitor_headers_status"),
                    self._tr("monitor_headers_last_live"),
                    self._tr("monitor_headers_last_capture"),
                ]
            )
        self.outputs_tree.setHeaderLabels([self._tr("outputs_header_artifact"), self._tr("outputs_header_value")])
        if hasattr(self, "monitor_tree"):
            for column in range(self.monitor_tree.columnCount()):
                self.monitor_tree.resizeColumnToContents(column)
        self.outputs_tree.resizeColumnToContents(0)
        self.summary_text.setPlaceholderText(self._tr("summary_placeholder"))
        self._set_asr_runtime_info(
            getattr(self, "_last_asr_model_reference", self.workflow.resolve_asr_model_reference()),
            getattr(self, "_last_asr_device", self._tr("ready")),
            getattr(self, "_last_asr_compute_type", ""),
        )
        self._set_f2_runtime_info(self.f2_report)
        if (
            self.current_snapshot.stage == WorkflowStage.IDLE
            and self.current_snapshot.status == WorkflowStatus.IDLE
            and self.current_snapshot.progress == 0
        ):
            self.current_snapshot = WorkflowSnapshot(
                title=self._tr("ready").rstrip("."),
                stage=WorkflowStage.IDLE,
                status=WorkflowStatus.IDLE,
                message=self._tr("ready"),
                progress=0,
                mode_label=self._mode_label(WorkflowStage.IDLE),
            )
        self.status_cards["mode"].setText(self._mode_label(self.current_snapshot.stage))
        self.status_cards["last_task"].setText(
            f"{self.current_snapshot.title}\n"
            f"{self._format_status_text(self.current_snapshot.stage, self.current_snapshot.status)}"
        )
        self._set_notice(self.current_snapshot.message, self.current_snapshot.status)

    def _build_header(self) -> QWidget:
        card = self._build_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        top_row = QHBoxLayout()
        title_box = QVBoxLayout()
        self.header_title = QLabel(self._window_title_text())
        self.header_title.setProperty("role", "title")
        self.header_subtitle = QLabel(self._window_subtitle_text())
        self.header_subtitle.setProperty("role", "muted")
        title_box.addWidget(self.header_title)
        title_box.addWidget(self.header_subtitle)
        top_row.addLayout(title_box, 1)
        top_row.addStretch(1)
        language_box = QVBoxLayout()
        language_box.setSpacing(6)
        self.language_label = QLabel(self._tr("language_label"))
        self.language_label.setProperty("role", "eyebrow")
        self.language_combo = QComboBox()
        self.language_combo.addItem(self._tr("language_zh"), "zh_CN")
        self.language_combo.addItem(self._tr("language_en"), "en_US")
        self.language_combo.setCurrentIndex(max(self.language_combo.findData(self.current_language), 0))
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        language_box.addWidget(self.language_label)
        language_box.addWidget(self.language_combo)
        top_row.addLayout(language_box)
        theme_box = QVBoxLayout()
        theme_box.setSpacing(6)
        self.theme_label = QLabel(self._tr("theme_mode"))
        self.theme_label.setProperty("role", "eyebrow")
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(self._tr("theme_light"), "light")
        self.theme_combo.addItem(self._tr("theme_dark"), "dark")
        self.theme_combo.addItem(self._tr("theme_system"), "system")
        self.theme_combo.setCurrentIndex(max(self.theme_combo.findData(self.current_theme_mode), 0))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_mode_changed)
        theme_box.addWidget(self.theme_label)
        theme_box.addWidget(self.theme_combo)
        top_row.addSpacing(12)
        top_row.addLayout(theme_box)
        layout.addLayout(top_row)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self.status_cards: dict[str, QLabel] = {}
        self.status_card_eyebrows: dict[str, QLabel] = {}
        for key, eyebrow in (
            ("mode", self._tr("status_mode")),
            ("last_task", self._tr("status_last_task")),
            ("asr", self._tr("status_asr")),
            ("f2", self._tr("status_f2")),
        ):
            frame = self._build_status_card(key, eyebrow)
            value = frame.findChild(QLabel, f"{key}_value")
            if value is not None:
                self.status_cards[key] = value
            eyebrow_label = frame.findChild(QLabel, f"{key}_eyebrow")
            if eyebrow_label is not None:
                self.status_card_eyebrows[key] = eyebrow_label
            cards_row.addWidget(frame, 1)
        layout.addLayout(cards_row)
        return card

    def _build_notice_banner(self) -> QWidget:
        card = self._build_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        self.notice_label = QLabel(self._tr("ready"))
        self.notice_label.setWordWrap(True)
        layout.addWidget(self.notice_label)
        return card

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._build_capture_group())
        if self._supports_capture_mode():
            layout.addWidget(self._build_monitor_group(), 1)
        else:
            layout.addStretch(1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._build_workflow_group())
        layout.addWidget(self._build_outputs_group())
        layout.addWidget(self._build_summary_group())
        layout.addWidget(self._build_log_group(), 1)
        return panel

    def _build_capture_group(self) -> QWidget:
        group = QGroupBox(self._tr("capture_group"))
        self.capture_group = group
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.room_input = QLineEdit()
        self.room_input.setPlaceholderText(self._tr("live_target_placeholder"))
        self.live_target_label = QLabel(self._tr("live_target"))
        form.addRow(self.live_target_label, self.room_input)
        self.replay_label = QLabel(self._tr("replay_label"))
        self.replay_input, self.replay_browse_button = self._create_picker_row()
        form.addRow(self.replay_label, self._wrap_picker_row(self.replay_input, self.replay_browse_button))
        self.occupancy_label = QLabel(self._tr("occupancy_label"))
        self.occupancy_input, self.occupancy_browse_button = self._create_picker_row()
        form.addRow(self.occupancy_label, self._wrap_picker_row(self.occupancy_input, self.occupancy_browse_button))
        self.danmaku_label = QLabel(self._tr("danmaku_label"))
        self.danmaku_input, self.danmaku_browse_button = self._create_picker_row()
        form.addRow(self.danmaku_label, self._wrap_picker_row(self.danmaku_input, self.danmaku_browse_button))
        layout.addLayout(form)

        if not self._supports_analysis_mode():
            for field in (self.replay_input, self.occupancy_input, self.danmaku_input):
                field.setReadOnly(True)
            self.replay_browse_button.hide()
            self.occupancy_browse_button.hide()
            self.danmaku_browse_button.hide()

        self.capture_hint = QLabel(self._tr("capture_hint"))
        self.capture_hint.setProperty("role", "muted")
        self.capture_hint.setWordWrap(True)
        layout.addWidget(self.capture_hint)

        if self._supports_analysis_mode():
            layout.addWidget(self._build_analysis_provider_group())

        actions = QGridLayout()
        actions.setHorizontalSpacing(10)
        actions.setVerticalSpacing(10)
        self.capture_button = QPushButton(self._tr("capture_button"))
        self.capture_button.clicked.connect(self._capture_live_data)
        self.stop_capture_button = QPushButton(self._tr("stop_capture_button"))
        self.stop_capture_button.setProperty("variant", "ghost")
        self.stop_capture_button.clicked.connect(self._stop_capture)
        self.stop_capture_button.setEnabled(False)
        self.run_button = QPushButton(self._tr("run_analysis_button"))
        self.run_button.clicked.connect(self._run_analysis)
        if self._supports_capture_mode():
            actions.addWidget(self.capture_button, 0, 0)
            actions.addWidget(self.stop_capture_button, 0, 1)
        else:
            self.capture_button.hide()
            self.stop_capture_button.hide()
        if self._supports_analysis_mode():
            analysis_row = 1 if self._supports_capture_mode() else 0
            actions.addWidget(self.run_button, analysis_row, 0, 1, 2)
        else:
            self.run_button.hide()
        layout.addLayout(actions)
        return group

    def _build_monitor_group(self) -> QWidget:
        group = QGroupBox(self._tr("monitor_group"))
        self.monitor_group = group
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        row = QHBoxLayout()
        self.monitor_url_input = QLineEdit()
        self.monitor_url_input.setPlaceholderText(self._tr("monitor_url_placeholder"))
        self.add_monitor_button = QPushButton(self._tr("add_profile_button"))
        self.add_monitor_button.clicked.connect(self._add_monitor_profile)
        self.remove_monitor_button = QPushButton(self._tr("remove_selected_button"))
        self.remove_monitor_button.setProperty("variant", "ghost")
        self.remove_monitor_button.clicked.connect(self._remove_selected_monitor_profiles)
        row.addWidget(self.monitor_url_input, 1)
        row.addWidget(self.add_monitor_button)
        row.addWidget(self.remove_monitor_button)
        layout.addLayout(row)

        self.monitor_tree = QTreeWidget()
        self.monitor_tree.setColumnCount(5)
        self.monitor_tree.setHeaderLabels(
            [
                self._tr("monitor_headers_enabled"),
                self._tr("monitor_headers_url"),
                self._tr("monitor_headers_status"),
                self._tr("monitor_headers_last_live"),
                self._tr("monitor_headers_last_capture"),
            ]
        )
        self.monitor_tree.setRootIsDecorated(False)
        self.monitor_tree.setAlternatingRowColors(True)
        self.monitor_tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.monitor_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.monitor_tree.setMinimumHeight(250)
        self.monitor_tree.itemChanged.connect(self._on_monitor_item_changed)
        layout.addWidget(self.monitor_tree, 1)

        footer = QHBoxLayout()
        self.monitor_button = QPushButton(self._tr("monitor_start_button"))
        self.monitor_button.clicked.connect(self._start_monitor)
        self.save_monitor_button = QPushButton(self._tr("monitor_save_button"))
        self.save_monitor_button.setProperty("variant", "ghost")
        self.save_monitor_button.clicked.connect(self._save_monitor_profiles)
        self.stop_monitor_button = QPushButton(self._tr("monitor_stop_button"))
        self.stop_monitor_button.setProperty("variant", "ghost")
        self.stop_monitor_button.clicked.connect(self._stop_monitor)
        self.stop_monitor_button.setEnabled(False)
        footer.addWidget(self.monitor_button)
        footer.addWidget(self.save_monitor_button)
        footer.addWidget(self.stop_monitor_button)
        layout.addLayout(footer)
        layout.addWidget(self._build_notification_group())
        return group

    def _build_analysis_provider_group(self) -> QWidget:
        box = self._build_card()
        layout = QFormLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.analysis_provider_label = QLabel()
        self.analysis_provider_combo = QComboBox()
        self.analysis_provider_combo.addItem("", OPENAI_COMPATIBLE_PROVIDER)
        self.analysis_provider_combo.addItem("", DEEPSEEK_PROVIDER)
        self.analysis_provider_combo.currentIndexChanged.connect(self._on_analysis_provider_changed)
        self.analysis_base_url_label = QLabel()
        self.analysis_base_url_input = QLineEdit()
        self.analysis_api_key_label = QLabel()
        self.analysis_api_key_input = QLineEdit()
        self.analysis_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.analysis_model_label = QLabel()
        self.analysis_model_input = QLineEdit()
        layout.addRow(self.analysis_provider_label, self.analysis_provider_combo)
        layout.addRow(self.analysis_base_url_label, self.analysis_base_url_input)
        layout.addRow(self.analysis_api_key_label, self.analysis_api_key_input)
        layout.addRow(self.analysis_model_label, self.analysis_model_input)
        self._refresh_analysis_provider_labels()
        self._load_analysis_provider_settings()
        return box

    def _build_notification_group(self) -> QWidget:
        box = self._build_card()
        layout = QFormLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.notification_provider_label = QLabel()
        self.notification_provider_combo = QComboBox()
        self.notification_provider_combo.addItem("", "none")
        self.notification_provider_combo.addItem("", "feishu")
        self.notification_provider_combo.addItem("", "dingtalk")
        self.notification_webhook_label = QLabel()
        self.notification_webhook_input = QLineEdit()
        self.memory_threshold_label = QLabel()
        self.memory_threshold_spin = QSpinBox()
        self.memory_threshold_spin.setRange(256, 131072)
        self.memory_threshold_spin.setSingleStep(256)
        self.notification_cooldown_label = QLabel()
        self.notification_cooldown_spin = QSpinBox()
        self.notification_cooldown_spin.setRange(1, 1440)
        self.notification_cooldown_spin.setSingleStep(5)
        self.notification_save_button = QPushButton()
        self.notification_save_button.setProperty("variant", "ghost")
        self.notification_save_button.clicked.connect(self._on_save_notification_settings)
        layout.addRow(self.notification_provider_label, self.notification_provider_combo)
        layout.addRow(self.notification_webhook_label, self.notification_webhook_input)
        layout.addRow(self.memory_threshold_label, self.memory_threshold_spin)
        layout.addRow(self.notification_cooldown_label, self.notification_cooldown_spin)
        layout.addRow(QWidget(), self.notification_save_button)
        config = self._load_notification_config()
        self.notification_provider_combo.setCurrentIndex(max(self.notification_provider_combo.findData(config.provider), 0))
        self.notification_webhook_input.setText(config.webhook_url)
        self.memory_threshold_spin.setValue(config.memory_threshold_mb)
        self.notification_cooldown_spin.setValue(config.cooldown_minutes)
        self._refresh_notification_labels()
        return box

    def _build_workflow_group(self) -> QWidget:
        group = QGroupBox(self._tr("workflow_group"))
        self.workflow_group = group
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.phase_list = QListWidget()
        self.phase_list.setMinimumHeight(180)
        layout.addWidget(self.phase_list)
        return group

    def _build_outputs_group(self) -> QWidget:
        group = QGroupBox(self._tr("outputs_group"))
        self.outputs_group = group
        layout = QVBoxLayout(group)
        self.outputs_tree = QTreeWidget()
        self.outputs_tree.setColumnCount(2)
        self.outputs_tree.setHeaderLabels([self._tr("outputs_header_artifact"), self._tr("outputs_header_value")])
        self.outputs_tree.setRootIsDecorated(False)
        self.outputs_tree.setMinimumHeight(160)
        layout.addWidget(self.outputs_tree)
        return group

    def _build_summary_group(self) -> QWidget:
        group = QGroupBox(self._tr("summary_group"))
        self.summary_group = group
        layout = QVBoxLayout(group)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText(self._tr("summary_placeholder"))
        self.summary_text.setMinimumHeight(140)
        layout.addWidget(self.summary_text)
        return group

    def _build_log_group(self) -> QWidget:
        group = QGroupBox(self._tr("log_group"))
        self.log_group = group
        layout = QVBoxLayout(group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        return group

    def _build_card(self) -> QFrame:
        frame = QFrame()
        frame.setProperty("card", True)
        return frame

    def _build_status_card(self, key: str, eyebrow_text: str) -> QWidget:
        frame = self._build_card()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        eyebrow = QLabel(eyebrow_text)
        eyebrow.setObjectName(f"{key}_eyebrow")
        eyebrow.setProperty("role", "eyebrow")
        value = QLabel("-")
        value.setObjectName(f"{key}_value")
        value.setWordWrap(True)
        layout.addWidget(eyebrow)
        layout.addWidget(value)
        return frame

    def _create_picker_row(self) -> tuple[QLineEdit, QPushButton]:
        line_edit = QLineEdit()
        button = QPushButton(self._tr("browse_button"))
        button.setProperty("variant", "ghost")
        button.clicked.connect(lambda: self._browse_file(line_edit))
        return line_edit, button

    def _wrap_picker_row(self, line_edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row

    def _set_mono_fonts(self) -> None:
        mono = QFont("Geist Mono")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.summary_text.setFont(mono)
        self.log.setFont(mono)
        self.outputs_tree.setFont(mono)
        self.phase_list.setFont(mono)

    def _browse_file(self, line_edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self._tr("browse_dialog_title"))
        if path:
            line_edit.setText(path)

    def _set_asr_runtime_info(self, model_reference: str, device: str, compute_type: str) -> None:
        self._last_asr_model_reference = model_reference
        self._last_asr_device = device
        self._last_asr_compute_type = compute_type
        asr_text = model_reference or "-"
        device_text = device or "-"
        if compute_type and compute_type not in {"auto", "unknown"}:
            device_text = f"{device_text} ({compute_type})"
        self.status_cards["asr"].setText(f"{asr_text}\n{device_text}")

    def _set_f2_runtime_info(self, report: F2RuntimeReport) -> None:
        status = self._tr("f2_status_ready") if report.ready else self._tr("f2_status_review")
        methods = ", ".join(name for name, ok in report.method_support.items() if ok)
        self.status_cards["f2"].setText(f"f2 {report.version} | {status}\n{methods or self._tr('f2_status_no_methods')}")

    def _load_runtime_warnings(self) -> None:
        if not self.f2_report.warnings:
            return
        self._set_notice(" | ".join(self.f2_report.warnings), WorkflowStatus.WARNING)
        for warning in self.f2_report.warnings:
            self._append_log(self._tr("runtime_warning", warning=warning))

    def _set_notice(self, message: str, status: WorkflowStatus) -> None:
        color = {
            WorkflowStatus.SUCCESS: self.tokens.success,
            WorkflowStatus.WARNING: self.tokens.warning,
            WorkflowStatus.ERROR: self.tokens.danger,
            WorkflowStatus.RUNNING: self.tokens.accent,
            WorkflowStatus.STOPPED: self.tokens.warning,
        }.get(status, self.tokens.muted_text)
        self.notice_label.setText(message)
        self.notice_label.setStyleSheet(
            f"background: {self.tokens.surface}; color: {color}; border-radius: 10px; padding: 6px 2px;"
        )

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.appendPlainText(f"[{timestamp}] {message}")

    def _format_status_text(self, stage: WorkflowStage, status: WorkflowStatus) -> str:
        return f"{stage.value.upper()} · {status.value.upper()}"

    def _format_status_text(self, stage: WorkflowStage, status: WorkflowStatus) -> str:
        return f"{stage.value.upper()} | {status.value.upper()}"

    def _apply_snapshot(self, snapshot: WorkflowSnapshot, *, push_timeline: bool = True) -> None:
        self.current_snapshot = snapshot
        self.progress.setValue(max(0, min(snapshot.progress, 100)))
        self.status_cards["mode"].setText(snapshot.mode_label)
        self.status_cards["last_task"].setText(f"{snapshot.title}\n{self._format_status_text(snapshot.stage, snapshot.status)}")
        self._set_notice(snapshot.message, snapshot.status)
        if snapshot.outputs:
            self._render_outputs(snapshot.outputs)
        if snapshot.summary_lines:
            self.summary_text.setPlainText("\n".join(snapshot.summary_lines))
        if push_timeline:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.phase_list.insertItem(0, f"[{timestamp}] {snapshot.title} | {snapshot.stage.value} | {snapshot.status.value} | {snapshot.message}")
        self._sync_buttons()

    def _render_outputs(self, outputs: dict[str, str]) -> None:
        self.outputs_tree.clear()
        for key, value in outputs.items():
            item = QTreeWidgetItem([key, value or "-"])
            item.setData(0, self.OUTPUT_KEY_ROLE, key)
            self.outputs_tree.addTopLevelItem(item)
        self.outputs_tree.resizeColumnToContents(0)

    @staticmethod
    def _fmt_optional(text: str | None) -> str:
        return text or "-"

    def _record_from_item(self, item: QTreeWidgetItem) -> MonitorProfileRecord:
        profile_url = str(item.data(1, self.MONITOR_URL_ROLE) or item.text(1)).strip()
        return MonitorProfileRecord(
            profile_url=profile_url,
            enabled=item.checkState(0) == Qt.CheckState.Checked,
            last_status=None if item.text(2) in {"", "-"} else item.text(2),
            last_live_at=None if item.text(3) in {"", "-"} else item.text(3),
            last_capture_dir=None if item.text(4) in {"", "-"} else item.text(4),
            last_room_id=None,
        )

    def _item_for_profile(self, profile_url: str) -> QTreeWidgetItem | None:
        for index in range(self.monitor_tree.topLevelItemCount()):
            item = self.monitor_tree.topLevelItem(index)
            if str(item.data(1, self.MONITOR_URL_ROLE) or item.text(1)).strip() == profile_url.strip():
                return item
        return None

    def _build_monitor_item(self, profile: MonitorProfileRecord) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(0, Qt.CheckState.Checked if profile.enabled else Qt.CheckState.Unchecked)
        item.setText(1, profile.profile_url)
        item.setData(1, self.MONITOR_URL_ROLE, profile.profile_url)
        item.setText(2, self._fmt_optional(profile.last_status))
        item.setText(3, self._fmt_optional(profile.last_live_at))
        item.setText(4, self._fmt_optional(profile.last_capture_dir))
        return item

    def _collect_monitor_profiles(self) -> list[MonitorProfileRecord]:
        records: list[MonitorProfileRecord] = []
        for index in range(self.monitor_tree.topLevelItemCount()):
            records.append(self._record_from_item(self.monitor_tree.topLevelItem(index)))
        return records

    def _replace_monitor_tree(self, profiles: list[MonitorProfileRecord]) -> None:
        self.monitor_tree.blockSignals(True)
        self.monitor_tree.clear()
        for profile in profiles:
            self.monitor_tree.addTopLevelItem(self._build_monitor_item(profile))
        self.monitor_tree.blockSignals(False)
        for column in range(5):
            self.monitor_tree.resizeColumnToContents(column)

    def _upsert_monitor_item(self, profile: MonitorProfileRecord) -> None:
        item = self._item_for_profile(profile.profile_url)
        if item is None:
            self.monitor_tree.addTopLevelItem(self._build_monitor_item(profile))
            return
        self.monitor_tree.blockSignals(True)
        item.setCheckState(0, Qt.CheckState.Checked if profile.enabled else Qt.CheckState.Unchecked)
        item.setText(2, self._fmt_optional(profile.last_status))
        item.setText(3, self._fmt_optional(profile.last_live_at))
        item.setText(4, self._fmt_optional(profile.last_capture_dir))
        self.monitor_tree.blockSignals(False)

    def _load_monitor_profiles(self) -> None:
        if not hasattr(self, "monitor_tree"):
            return
        profiles = self.monitor_profiles.list_profiles()
        if profiles:
            self._replace_monitor_tree(profiles)
            self._append_log(self._tr("loaded_monitor_profiles_log", count=len(profiles)))

    def _save_monitor_profiles(self) -> None:
        profiles = self._collect_monitor_profiles()
        self.monitor_profiles.save_profiles(profiles)
        self._set_notice(self._tr("saved_monitor_profiles", count=len(profiles)), WorkflowStatus.SUCCESS)
        self._append_log(self._tr("saved_monitor_profiles_log", count=len(profiles)))

    def _on_save_notification_settings(self) -> None:
        config = self._save_notification_config()
        self._append_log(
            self._lang_text(
                f"已保存机器人告警配置: {self._notification_provider_display(config.provider)}",
                f"Saved alert settings: {self._notification_provider_display(config.provider)}",
            )
        )
        self._set_notice(
            self._lang_text("告警配置已保存。", "Alert settings saved."),
            WorkflowStatus.SUCCESS,
        )

    def _check_memory_pressure(self) -> None:
        if not self._supports_capture_mode():
            return
        if not (self.monitor_worker and self.monitor_worker.isRunning()):
            return
        config = self._load_notification_config()
        if not config.enabled():
            return
        should_alert, available_mb = self.memory_watcher.should_alert(config)
        if not should_alert or available_mb is None:
            return
        message = self._lang_text(
            f"可用内存降到 {available_mb} MB，已低于阈值 {config.memory_threshold_mb} MB。",
            f"Available memory dropped to {available_mb} MB, below the threshold of {config.memory_threshold_mb} MB.",
        )
        self._append_log(f"[health] {message}")
        self._set_notice(message, WorkflowStatus.WARNING)
        self._send_robot_notification(
            self._lang_text("采集端低内存告警", "Capture Client Low Memory Alert"),
            message,
        )

    def _send_robot_notification(self, title: str, body: str) -> None:
        config = self._load_notification_config()
        if not config.enabled():
            return
        try:
            self.robot_notifier.send_text(config, title, body)
            self._append_log(
                self._lang_text(
                    f"机器人告警已发送: {title}",
                    f"Robot alert sent: {title}",
                )
            )
        except Exception as exc:
            self._append_log(
                self._lang_text(
                    f"机器人告警发送失败: {exc}",
                    f"Robot alert failed: {exc}",
                )
            )

    def _maybe_prompt_start_saved_monitor(self) -> None:
        if not self._supports_capture_mode():
            return
        enabled_count = len([profile for profile in self.monitor_profiles.enabled_profiles() if profile.profile_url.strip()])
        if enabled_count == 0:
            return
        if self.auto_start_monitor:
            self._append_log(
                self._lang_text(
                    f"检测到 {enabled_count} 个已启用监控，已自动启动。",
                    f"Detected {enabled_count} enabled monitor profiles and started automatically.",
                )
            )
            self._start_monitor()
            return
        answer = QMessageBox.question(
            self,
            self._tr("saved_monitor_list_title"),
            self._tr("saved_monitor_list_body", count=enabled_count),
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._start_monitor()

    def _add_monitor_profile(self) -> None:
        profile_url = self.monitor_url_input.text().strip()
        if not profile_url:
            self._set_notice(self._tr("enter_profile_url"), WorkflowStatus.WARNING)
            return
        if self._item_for_profile(profile_url) is not None:
            self._set_notice(self._tr("duplicate_profile"), WorkflowStatus.WARNING)
            return
        try:
            profile = self.monitor_profiles.add_profile(profile_url)
        except ValueError as exc:
            self._set_notice(str(exc), WorkflowStatus.WARNING)
            return
        self._upsert_monitor_item(profile)
        self.monitor_url_input.clear()
        self._append_log(self._tr("added_monitor_profile", url=profile.profile_url))
        self._set_notice(self._tr("profile_added"), WorkflowStatus.SUCCESS)

    def _remove_selected_monitor_profiles(self) -> None:
        selected = self.monitor_tree.selectedItems()
        if not selected:
            self._set_notice(self._tr("select_monitor_rows"), WorkflowStatus.WARNING)
            return
        profile_urls = [str(item.data(1, self.MONITOR_URL_ROLE) or item.text(1)).strip() for item in selected]
        removed = self.monitor_profiles.remove_profiles(profile_urls)
        for item in selected:
            self.monitor_tree.takeTopLevelItem(self.monitor_tree.indexOfTopLevelItem(item))
        self._append_log(self._tr("removed_monitor_profiles_log", count=removed))
        self._set_notice(self._tr("removed_monitor_profiles", count=removed), WorkflowStatus.SUCCESS)

    def _on_monitor_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        profile_url = str(item.data(1, self.MONITOR_URL_ROLE) or item.text(1)).strip()
        if profile_url:
            self.monitor_profiles.set_enabled(profile_url, item.checkState(0) == Qt.CheckState.Checked)

    def _run_analysis(self) -> None:
        self._save_analysis_provider_settings()
        try:
            request = self.workflow.build_analysis_request(
                replay_path=self.replay_input.text(),
                occupancy_path=self.occupancy_input.text(),
                danmaku_path=self.danmaku_input.text(),
                room_id=self.room_input.text(),
                analysis_provider=str(self.analysis_provider_combo.currentData() or OPENAI_COMPATIBLE_PROVIDER)
                if hasattr(self, "analysis_provider_combo")
                else OPENAI_COMPATIBLE_PROVIDER,
                analysis_base_url=self.analysis_base_url_input.text() if hasattr(self, "analysis_base_url_input") else "",
                analysis_api_key=self.analysis_api_key_input.text() if hasattr(self, "analysis_api_key_input") else "",
                analysis_model_id=self.analysis_model_input.text() if hasattr(self, "analysis_model_input") else "",
            )
        except ValueError as exc:
            self._set_notice(str(exc), WorkflowStatus.WARNING)
            return

        self._append_log(self._tr("analysis_start_log"))
        self._set_asr_runtime_info(self.workflow.resolve_asr_model_reference(), self._tr("running"), "")
        self.worker = AnalysisWorker(self.workflow, request)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("analysis_snapshot_title"),
                stage=WorkflowStage.ANALYSIS,
                status=WorkflowStatus.RUNNING,
                message=self._tr("analysis_snapshot_message"),
                progress=15,
                mode_label=self._mode_label(WorkflowStage.ANALYSIS),
                summary_lines=[
                    self._tr("analysis_summary_started"),
                    self._tr("analysis_summary_local"),
                ],
            )
        )

    def _capture_live_data(self) -> None:
        target = self.room_input.text().strip()
        if not target:
            self._set_notice(self._tr("capture_missing_target"), WorkflowStatus.WARNING)
            return
        self._append_log(self._tr("capture_start_log", target=target))
        self.capture_worker = CaptureWorker(self.workflow, target)
        self.capture_worker.finished_ok.connect(self._on_capture_finished)
        self.capture_worker.failed.connect(self._on_capture_failed)
        self.capture_worker.start()
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("capture_snapshot_title"),
                stage=WorkflowStage.CAPTURE,
                status=WorkflowStatus.RUNNING,
                message=self._tr("capture_snapshot_message", target=target),
                progress=8,
                mode_label=self._mode_label(WorkflowStage.CAPTURE),
            )
        )

    def _stop_capture(self) -> None:
        if self.capture_worker and self.capture_worker.isRunning():
            self._append_log(self._tr("stop_capture_log"))
            self.capture_worker.stop()
            self._apply_snapshot(
                WorkflowSnapshot(
                    title=self._tr("stop_capture_title"),
                    stage=WorkflowStage.CAPTURE,
                    status=WorkflowStatus.STOPPED,
                    message=self._tr("stop_capture_message"),
                    progress=max(self.progress.value(), 10),
                    mode_label=self._mode_label(WorkflowStage.CAPTURE),
                )
            )

    def _start_monitor(self) -> None:
        self._save_monitor_profiles()
        self._save_notification_config()
        enabled_profiles = self.monitor_profiles.enabled_profiles()
        if not enabled_profiles:
            self._set_notice(self._tr("enable_profile_first"), WorkflowStatus.WARNING)
            return

        self._append_log(self._tr("monitor_start_log", count=len(enabled_profiles)))
        self.monitor_worker = MonitorWorker(self.workflow, [profile.profile_url for profile in enabled_profiles])
        self.monitor_worker.event_emitted.connect(self._on_monitor_event)
        self.monitor_worker.failed.connect(self._on_monitor_failed)
        self.monitor_worker.finished_ok.connect(self._on_monitor_finished)
        self.monitor_worker.start()
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("monitor_start_title"),
                stage=WorkflowStage.MONITOR,
                status=WorkflowStatus.RUNNING,
                message=self._tr("monitor_start_message", count=len(enabled_profiles)),
                progress=max(self.progress.value(), 12),
                mode_label=self._mode_label(WorkflowStage.MONITOR),
            )
        )

    def _stop_monitor(self) -> None:
        if self.monitor_worker and self.monitor_worker.isRunning():
            self._append_log(self._tr("stop_monitor_log"))
            self.monitor_worker.stop()
            self._apply_snapshot(
                WorkflowSnapshot(
                    title=self._tr("stop_monitor_title"),
                    stage=WorkflowStage.MONITOR,
                    status=WorkflowStatus.STOPPED,
                    message=self._tr("stop_monitor_message"),
                    progress=self.progress.value(),
                    mode_label=self._mode_label(WorkflowStage.MONITOR),
                )
            )

    def _on_finished(self, payload: dict) -> None:
        self._set_asr_runtime_info(
            payload.get("asr_model_reference", ""),
            payload.get("asr_device", ""),
            payload.get("asr_compute_type", ""),
        )
        self._append_log(self._tr("analysis_finished_log", job_id=payload["job_id"]))
        self._append_log(self._tr("summary_log", path=payload["summary_path"]))
        self._append_log(self._tr("timeline_log", path=payload["timeline_path"]))
        if payload.get("report_url"):
            self._append_log(self._tr("cloud_report_log", url=payload["report_url"]))
        summary_lines = [
            f"{self._tr('job_id')}: {payload['job_id']}",
            f"{self._tr('tier')}: {payload.get('analysis_tier') or self._tr('standard_tier')}",
            f"{self._tr('uploaded')}: {payload.get('uploaded')}",
            f"{self._tr('payment_required')}: {payload.get('payment_required')}",
            f"{self._tr('offload_required')}: {payload.get('offload_required')}",
            payload.get("summary_excerpt") or self._tr("summary_unavailable"),
        ]
        outputs = {
            "summary_path": payload["summary_path"],
            "timeline_path": payload["timeline_path"],
            "report_url": payload.get("report_url") or "-",
            "asset_storage": payload.get("asset_storage") or "inline",
            "video_size_bytes": str(payload.get("video_size_bytes") or "-"),
        }
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("analysis_finished_title"),
                stage=WorkflowStage.COMPLETED,
                status=WorkflowStatus.SUCCESS,
                message=self._tr("analysis_finished_message"),
                progress=100,
                mode_label=self._mode_label(WorkflowStage.IDLE),
                outputs=outputs,
                summary_lines=summary_lines,
            )
        )

    def _on_failed(self, message: str) -> None:
        self._append_log(self._tr("analysis_failed_log", message=message))
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("analysis_failed_title"),
                stage=WorkflowStage.ANALYSIS,
                status=WorkflowStatus.ERROR,
                message=message,
                progress=0,
                mode_label=self._mode_label(WorkflowStage.IDLE),
                summary_lines=[self._tr("analysis_failed_short"), message],
            )
        )
        if self.app_mode != "capture":
            QMessageBox.critical(self, self._tr("analysis_failed_title"), message)

    def _on_capture_finished(self, payload: dict) -> None:
        replay_path = payload.get("replay_path") or ""
        occupancy_path = payload.get("occupancy_path") or ""
        danmaku_path = payload.get("danmaku_path") or ""
        workspace_dir = payload.get("workspace_dir") or ""

        self.replay_input.setText(replay_path)
        self.occupancy_input.setText(occupancy_path)
        self.danmaku_input.setText(danmaku_path)

        stopped = bool(payload.get("stopped_by_user"))
        status_text = self._tr("capture_stopped_by_user") if stopped else self._tr("capture_finished")
        self._append_log(self._tr("capture_finished_log", status=status_text, room_id=payload.get("room_id")))
        self._append_log(self._tr("workspace_log", path=workspace_dir or "-"))
        snapshot_status = WorkflowStatus.STOPPED if stopped else WorkflowStatus.SUCCESS
        summary_lines = [
            self._tr("room_id_line", value=payload.get("room_id") or "-"),
            self._tr("replay_line", value=replay_path or "-"),
            self._tr("occupancy_line", value=occupancy_path or "-"),
            self._tr("danmaku_line", value=danmaku_path or "-"),
        ]
        self._apply_snapshot(
            WorkflowSnapshot(
                title=status_text,
                stage=WorkflowStage.CAPTURE,
                status=snapshot_status,
                message=self._tr("capture_files_ready") if not stopped else self._tr("capture_partial_preserved"),
                progress=35,
                mode_label=self._mode_label(WorkflowStage.IDLE),
                outputs={
                    "workspace_dir": workspace_dir or "-",
                    "replay_path": replay_path or "-",
                    "occupancy_path": occupancy_path or "-",
                    "danmaku_path": danmaku_path or "-",
                    "metadata_path": payload.get("metadata_path") or "-",
                },
                summary_lines=summary_lines,
            )
        )
        if not stopped:
            self._send_robot_notification(
                self._lang_text("录播采集完成", "Replay Capture Completed"),
                self._lang_text(
                    f"房间 {payload.get('room_id') or '-'} 采集完成。\n工作目录: {workspace_dir or '-'}",
                    f"Capture for room {payload.get('room_id') or '-'} completed.\nWorkspace: {workspace_dir or '-'}",
                ),
            )

    def _on_capture_failed(self, message: str) -> None:
        self._append_log(self._tr("capture_failed_log", message=message))
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("capture_failed_title"),
                stage=WorkflowStage.CAPTURE,
                status=WorkflowStatus.ERROR,
                message=message,
                progress=0,
                mode_label=self._mode_label(WorkflowStage.IDLE),
                summary_lines=[self._tr("capture_failed_short"), message],
            )
        )
        self._send_robot_notification(
            self._lang_text("录播采集失败", "Replay Capture Failed"),
            message,
        )
        if self.app_mode != "capture":
            QMessageBox.critical(self, self._tr("capture_failed_title"), message)

    def _on_monitor_event(self, payload: dict) -> None:
        message = str(payload.get("message", "")).strip()
        profile_url = str(payload.get("profile_url", "")).strip()
        prefix = f"[monitor] {profile_url}: " if profile_url else "[monitor] "
        self._append_log(prefix + message)

        updated = self.monitor_profiles.apply_monitor_event(payload)
        if updated is not None:
            self._upsert_monitor_item(updated)

        replay_path = str(payload.get("replay_path", "")).strip()
        workspace_dir = str(payload.get("workspace_dir", "")).strip()
        if replay_path:
            self.replay_input.setText(replay_path)
        stage = WorkflowStage(payload.get("stage") or WorkflowStage.MONITOR.value)
        status = WorkflowStatus(payload.get("status") or WorkflowStatus.IDLE.value)

        outputs = {}
        if replay_path:
            outputs["replay_path"] = replay_path
        if workspace_dir:
            outputs["workspace_dir"] = workspace_dir
        room_id = str(payload.get("room_id", "")).strip()
        if room_id:
            outputs["room_id"] = room_id

        self._apply_snapshot(
            WorkflowSnapshot(
                title=str(payload.get("event_type") or "monitor_event").replace("_", " ").title(),
                stage=stage,
                status=status,
                message=message or self._tr("monitor_event_default"),
                progress=max(self.progress.value(), 20 if status == WorkflowStatus.RUNNING else self.progress.value()),
                mode_label=self._mode_label(WorkflowStage.MONITOR)
                if self.monitor_worker and self.monitor_worker.isRunning()
                else self._mode_label(WorkflowStage.IDLE),
                outputs=outputs or self.current_snapshot.outputs,
                summary_lines=[
                    self._tr("profile_line", value=profile_url or "-"),
                    self._tr("status_line", value=payload.get("status") or "-"),
                    self._tr("severity_line", value=payload.get("severity") or "-"),
                    message or self._tr("no_message"),
                ],
            )
        )

    def _on_monitor_failed(self, message: str) -> None:
        self._append_log(self._tr("monitor_failed_log", message=message))
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("monitor_failed_title"),
                stage=WorkflowStage.MONITOR,
                status=WorkflowStatus.ERROR,
                message=message,
                progress=self.progress.value(),
                mode_label=self._mode_label(WorkflowStage.IDLE),
                summary_lines=[self._tr("monitor_failed_short"), message],
            )
        )
        self._send_robot_notification(
            self._lang_text("监控循环异常", "Monitor Loop Error"),
            message,
        )
        if self.app_mode != "capture":
            QMessageBox.critical(self, self._tr("monitor_failed_title"), message)

    def _on_monitor_finished(self) -> None:
        self._append_log(self._tr("monitor_stopped_log"))
        self._apply_snapshot(
            WorkflowSnapshot(
                title=self._tr("monitor_stopped_title"),
                stage=WorkflowStage.MONITOR,
                status=WorkflowStatus.STOPPED,
                message=self._tr("monitor_stopped_message"),
                progress=self.progress.value(),
                mode_label=self._mode_label(WorkflowStage.IDLE),
            )
        )

    def _sync_buttons(self) -> None:
        capture_running = bool(self.capture_worker and self.capture_worker.isRunning())
        analysis_running = bool(self.worker and self.worker.isRunning())
        monitor_running = bool(self.monitor_worker and self.monitor_worker.isRunning())
        if hasattr(self, "capture_button"):
            self.capture_button.setEnabled(not capture_running and not analysis_running)
        if hasattr(self, "stop_capture_button"):
            self.stop_capture_button.setEnabled(capture_running)
        if hasattr(self, "run_button"):
            self.run_button.setEnabled(not analysis_running and not capture_running)
        if hasattr(self, "monitor_button"):
            self.monitor_button.setEnabled(not monitor_running and not capture_running)
        if hasattr(self, "stop_monitor_button"):
            self.stop_monitor_button.setEnabled(monitor_running)


def main(app_mode: str = "all", *, auto_start_monitor: bool | None = None) -> int:
    app = QApplication(sys.argv)
    window = MainWindow(app_mode=app_mode, auto_start_monitor=auto_start_monitor)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
