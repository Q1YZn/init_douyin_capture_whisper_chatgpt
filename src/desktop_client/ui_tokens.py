from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UITokens:
    bg: str
    bg_alt: str
    surface: str
    surface_alt: str
    border: str
    border_strong: str
    text: str
    muted_text: str
    accent: str
    accent_hover: str
    accent_soft: str
    accent_text: str
    success: str
    warning: str
    danger: str
    selection: str
    radius: int = 14


LIGHT_TOKENS = UITokens(
    bg="#F2EEE6",
    bg_alt="#E5DDD0",
    surface="#FBF7F0",
    surface_alt="#E9E2D6",
    border="#D6CCBD",
    border_strong="#B9AD9A",
    text="#21313A",
    muted_text="#6D7A84",
    accent="#D06F45",
    accent_hover="#BA633D",
    accent_soft="#F3D7C8",
    accent_text="#FFF7F2",
    success="#2B7A78",
    warning="#B7791F",
    danger="#B55447",
    selection="#D5E4E3",
)


DARK_TOKENS = UITokens(
    bg="#1B1F24",
    bg_alt="#252C33",
    surface="#232A31",
    surface_alt="#303942",
    border="#3A4650",
    border_strong="#566575",
    text="#E7E2D8",
    muted_text="#A9B3BC",
    accent="#E2875C",
    accent_hover="#F09A72",
    accent_soft="#5A392D",
    accent_text="#FFF5EF",
    success="#4DAAA4",
    warning="#D39A4B",
    danger="#D9786A",
    selection="#364651",
)


def get_tokens(color_scheme: str) -> UITokens:
    return DARK_TOKENS if color_scheme == "dark" else LIGHT_TOKENS


def build_app_stylesheet(tokens: UITokens) -> str:
    t = tokens
    return f"""
    QWidget {{
        background: {t.bg};
        color: {t.text};
        font-family: "Geist Sans", "Segoe UI", "Microsoft YaHei UI", sans-serif;
        font-size: 13px;
    }}
    QMainWindow {{
        background: {t.bg};
    }}
    QLabel[role="eyebrow"] {{
        color: {t.muted_text};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    QLabel[role="title"] {{
        font-size: 26px;
        font-weight: 700;
    }}
    QLabel[role="muted"] {{
        color: {t.muted_text};
    }}
    QFrame[card="true"], QGroupBox {{
        background: {t.surface};
        border: 1px solid {t.border};
        border-radius: {t.radius}px;
    }}
    QGroupBox {{
        margin-top: 14px;
        padding: 16px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        left: 14px;
        top: -8px;
        padding: 0 6px;
        color: {t.muted_text};
        background: {t.bg};
    }}
    QLineEdit, QPlainTextEdit, QTreeWidget, QListWidget, QComboBox {{
        background: {t.surface};
        border: 1px solid {t.border};
        border-radius: 10px;
        padding: 8px;
        selection-background-color: {t.selection};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTreeWidget:focus, QListWidget:focus, QComboBox:focus {{
        border: 1px solid {t.border_strong};
    }}
    QPushButton {{
        background: {t.accent};
        color: {t.accent_text};
        border: none;
        border-radius: 10px;
        padding: 10px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {t.accent_hover};
    }}
    QPushButton:disabled {{
        background: {t.surface_alt};
        color: {t.muted_text};
    }}
    QPushButton[variant="ghost"] {{
        background: {t.surface};
        color: {t.text};
        border: 1px solid {t.border};
    }}
    QPushButton[variant="ghost"]:hover {{
        background: {t.surface_alt};
    }}
    QProgressBar {{
        background: {t.surface_alt};
        border: 1px solid {t.border};
        border-radius: 9px;
        text-align: center;
        min-height: 18px;
        color: {t.text};
    }}
    QProgressBar::chunk {{
        background: {t.accent};
        border-radius: 8px;
    }}
    QHeaderView::section {{
        background: {t.bg_alt};
        color: {t.muted_text};
        border: none;
        border-bottom: 1px solid {t.border};
        padding: 8px;
        font-weight: 600;
    }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background: {t.selection};
        color: {t.text};
    }}
    """
