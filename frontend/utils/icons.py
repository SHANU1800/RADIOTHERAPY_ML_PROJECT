"""
SVG icons for the frontend (replacing emojis).
"""
from typing import Optional

def get_svg_icon(icon_name: str, size: int = 24, color: str = "currentColor") -> str:
    """
    Get SVG icon as HTML string.
    
    Args:
        icon_name: Name of the icon
        size: Size in pixels
        color: Color (default: currentColor)
        
    Returns:
        SVG HTML string
    """
    icons = {
        "lungs": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3C8.5 3 6 5.5 6 9c0 2.5 1.5 4.5 3 6l3 3 3-3c1.5-1.5 3-3.5 3-6 0-3.5-2.5-6-6-6z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M9 12c-1.5 0-3-1-3-2.5S7.5 7 9 7" stroke="{color}" stroke-width="2" stroke-linecap="round"/><path d="M15 12c1.5 0 3-1 3-2.5S16.5 7 15 7" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "upload": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="17 8 12 3 7 8" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><line x1="12" y1="3" x2="12" y2="15" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="11" cy="11" r="8" stroke="{color}" stroke-width="2"/><path d="m21 21-4.35-4.35" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><line x1="18" y1="20" x2="18" y2="10" stroke="{color}" stroke-width="2" stroke-linecap="round"/><line x1="12" y1="20" x2="12" y2="4" stroke="{color}" stroke-width="2" stroke-linecap="round"/><line x1="6" y1="20" x2="6" y2="14" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "package": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="3.27 6.96 12 12.01 20.73 6.96" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><line x1="12" y1="22.08" x2="12" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "robot": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="8" width="18" height="12" rx="2" stroke="{color}" stroke-width="2"/><path d="M12 8V6a2 2 0 0 0-2-2H8" stroke="{color}" stroke-width="2" stroke-linecap="round"/><path d="M12 8V6a2 2 0 0 1 2-2h2" stroke="{color}" stroke-width="2" stroke-linecap="round"/><circle cx="8" cy="14" r="1" fill="{color}"/><circle cx="16" cy="14" r="1" fill="{color}"/><path d="M9 18h6" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><polyline points="20 6 9 17 4 12" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "x": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><line x1="18" y1="6" x2="6" y2="18" stroke="{color}" stroke-width="2" stroke-linecap="round"/><line x1="6" y1="6" x2="18" y2="18" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "alert": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><line x1="12" y1="9" x2="12" y2="13" stroke="{color}" stroke-width="2" stroke-linecap="round"/><line x1="12" y1="17" x2="12.01" y2="17" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "info": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="2"/><line x1="12" y1="16" x2="12" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/><line x1="12" y1="8" x2="12.01" y2="8" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>',
        "message": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "file": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="13 2 13 9 20 9" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    }
    return icons.get(icon_name, "")

def icon_html(icon_name: str, size: int = 24, color: str = "currentColor") -> str:
    """Get icon as HTML string for use in markdown."""
    return get_svg_icon(icon_name, size, color)
