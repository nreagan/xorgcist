"""xorgcist — Linux GUI for X11 multi-display and input configuration.

Single file. Layout: imports → constants → dataclasses → parse → compute →
emit → validate → state regen → UI callbacks → UI render → UI build → main.

Design: one `State` dataclass holds everything (semantic + render-transient
+ derived). Pure functions transform state into text. Callbacks mutate state
and call `redraw(state)` (or `redraw_light` for high-frequency events).
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import dearpygui.dearpygui as dpg
except OSError as _e:
    if "GLIBC" in str(_e):
        sys.stderr.write(
            "xorgcist: installed dearpygui needs a newer glibc than this system has.\n"
            f"  detail: {_e}\n"
            "  fix:    pip install --force-reinstall 'dearpygui<=1.9.0'\n"
            "          (RHEL8 ships glibc 2.28; dearpygui 1.9.1+ requires 2.29+)\n"
        )
        sys.exit(1)
    raise


# ============================== CONSTANTS ==============================

BASE_VIEWPORT_W, BASE_VIEWPORT_H = 1160, 720
BASE_FORM_W = 360
CHROME_H_ESTIMATE = 180
PREVIEW_BOTTOM_RESERVE = 200

UI_SCALE_MIN, UI_SCALE_MAX = 0.6, 3.0
SCALE_EPSILON = 0.001
REFRESH_TOLERANCE_HZ = 0.01
SNAP_THRESHOLD_FB = 20
SCREEN_GROUP_MAX = 15

ROTATIONS = ("normal", "left", "right", "inverted")
COLOR_DEPTHS = (16, 24, 30)

W_NUM, W_DROP, W_DROP_WIDE = 120, 160, 180
COLOR_DIM = (160, 160, 160, 255)
COLOR_HEAD = (140, 180, 220, 255)

T_CANVAS = "canvas"
T_FORM = "form"
T_INPUT_LIST = "input_list"
T_TOUCH_LIST = "touch_list"
T_CONFIG = "config_preview"
T_RUNTIME = "runtime_preview"
T_VALIDATION = "validation_log"
T_STATUS = "status_bar"
T_HEADER = "header_counts"
T_CANVAS_HANDLERS = "canvas_handlers"

SCREEN_PALETTE = (
    (60, 80, 130, 255), (70, 120, 70, 255), (140, 70, 70, 255),
    (130, 100, 50, 255), (110, 70, 130, 255), (60, 130, 130, 255),
    (140, 100, 130, 255), (90, 90, 90, 255),
)
IDENTITY_MATRIX = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# ============================== MODEL ==============================

@dataclass
class Output:
    name: str
    connected: bool
    width: int = 0
    height: int = 0
    x: int = 0
    y: int = 0
    primary: bool = False
    screen_id: int = 0
    available_modes: list[tuple[int, int, float]] = field(default_factory=list)
    enabled: bool = True
    rotation: str = "normal"
    refresh_rate: float = 60.0
    color_depth: int = 24
    scale: float = 1.0
    mirror_of: Optional[str] = None


@dataclass
class GPU:
    busid: str
    vendor: str
    driver: str
    name: str


@dataclass
class InputDevice:
    id: int
    name: str
    type: str  # keyboard / pointer / touchscreen / tablet / touchpad


@dataclass
class TouchscreenMapping:
    device_id: int
    device_name: str
    target_output: Optional[str] = None
    enabled: bool = True


@dataclass
class XorgSection:
    kind: str
    identifier: str = ""
    options: dict[str, str] = field(default_factory=dict)
    raw: str = ""


@dataclass
class XorgConf:
    sections: list[XorgSection] = field(default_factory=list)


@dataclass
class ValidationError:
    severity: str
    message: str


@dataclass
class State:
    """Everything: semantic, transient UI, and derived. One source of truth.
    `regenerate()` updates the derived fields after each mutation."""
    # Semantic
    gpus: list[GPU] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    inputs: list[InputDevice] = field(default_factory=list)
    touchscreens: list[TouchscreenMapping] = field(default_factory=list)
    # Derived (recomputed by regenerate)
    framebuffer_width: int = 0
    framebuffer_height: int = 0
    generated_config: str = ""
    generated_runtime: list[str] = field(default_factory=list)
    validation_errors: list[ValidationError] = field(default_factory=list)
    # Transient UI
    selected_output: Optional[str] = None
    dragging_output: Optional[str] = None
    drag_anchor_fb: tuple[int, int] = (0, 0)
    canvas_scale: float = 1.0
    canvas_off_x: float = 0.0
    canvas_off_y: float = 0.0
    ui_scale: float = 1.0
    status_message: str = "Ready."
    env_session: str = "?"
    env_wayland: bool = False
    env_display: str = "?"


# ============================== DETECT + PARSE ==============================

def run_cmd(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            print(f"xorgcist: {cmd[0]} exit {r.returncode}: {r.stderr.strip()}", file=sys.stderr)
        return r.stdout
    except FileNotFoundError:
        print(f"xorgcist: {cmd[0]} not found in PATH", file=sys.stderr)
        return ""
    except subprocess.SubprocessError as e:
        print(f"xorgcist: {cmd[0]} failed: {e}", file=sys.stderr)
        return ""


def parse_xrandr(text: str) -> list[Output]:
    outputs: list[Output] = []
    current: Optional[Output] = None
    header_re = re.compile(
        r'^(\S+)\s+(connected|disconnected)\s*(primary)?'
        r'\s*(?:(\d+)x(\d+)\+(\d+)\+(\d+))?'
    )
    mode_re = re.compile(r'^\s+(\d+)x(\d+).*?(\*current)?\s*(\+preferred)?\s*$')
    v_re = re.compile(r'^\s+v:.*?clock\s+([\d.]+)\s*Hz')
    pending_mode: Optional[tuple[int, int]] = None
    pending_current = False
    for line in text.splitlines():
        if line and not line[0].isspace():
            m = header_re.match(line)
            if m:
                if current is not None:
                    outputs.append(current)
                # rotation: keyword outside parenthesized supported-rotations list
                cleaned = re.sub(r'\([^)]*\)', '', line)
                rot = "normal"
                for kw in ("left", "right", "inverted"):
                    if re.search(rf'(?<![\w-]){kw}(?![\w-])', cleaned):
                        rot = kw
                        break
                current = Output(
                    name=m.group(1),
                    connected=(m.group(2) == "connected"),
                    primary=(m.group(3) == "primary"),
                    width=int(m.group(4)) if m.group(4) else 0,
                    height=int(m.group(5)) if m.group(5) else 0,
                    x=int(m.group(6)) if m.group(6) else 0,
                    y=int(m.group(7)) if m.group(7) else 0,
                    rotation=rot,
                )
                pending_mode, pending_current = None, False
            continue
        if current is None:
            continue
        mm = mode_re.match(line)
        if mm:
            pending_mode = (int(mm.group(1)), int(mm.group(2)))
            pending_current = bool(mm.group(3))
            continue
        if pending_mode and (rm := v_re.match(line)):
            try:
                refresh = float(rm.group(1))
            except ValueError:
                refresh = 60.0
            entry = (pending_mode[0], pending_mode[1], refresh)
            if entry not in current.available_modes:
                current.available_modes.append(entry)
            if pending_current:
                current.refresh_rate = refresh
            pending_mode, pending_current = None, False
    if current is not None:
        outputs.append(current)
    return outputs


def parse_lspci_gpus(text: str) -> list[GPU]:
    """Parse `lspci -D` and sort NVIDIA > AMD > Intel > Unknown so that
    `state.gpus[0]` is the discrete GPU on hybrid laptops."""
    gpus: list[GPU] = []
    pci_re = re.compile(
        r'^([\da-f]{4}):([\da-f]{2}):([\da-f]{2})\.(\d)\s+(.+?):\s+(.+?)$',
        re.IGNORECASE,
    )
    for line in text.splitlines():
        if not any(k in line for k in ("VGA", "3D controller", "Display controller")):
            continue
        m = pci_re.match(line)
        if not m:
            continue
        _, bus, dev, func, _, desc = m.groups()
        busid = f"PCI:{int(bus, 16)}:{int(dev, 16)}:{int(func)}"
        if "NVIDIA" in desc:
            vendor, driver = "NVIDIA", "nvidia"
        elif any(k in desc for k in ("AMD", "ATI", "Advanced Micro")):
            vendor, driver = "AMD", "amdgpu"
        elif "Intel" in desc:
            vendor, driver = "Intel", "modesetting"
        else:
            vendor, driver = "Unknown", "modesetting"
        gpus.append(GPU(busid=busid, vendor=vendor, driver=driver, name=desc.strip()))
    priority = {"NVIDIA": 0, "AMD": 1, "Intel": 2, "Unknown": 3}
    gpus.sort(key=lambda g: priority.get(g.vendor, 99))
    return gpus


def parse_xinput_list(text: str) -> list[InputDevice]:
    devs: list[InputDevice] = []
    detail_re = re.compile(r'(?:↳)\s+(.+?)\s+id=(\d+)\s+\[(slave|master)\s+(\w+)')
    for line in text.splitlines():
        m = detail_re.search(line)
        if not m or m.group(3) == "master":
            continue
        name, kind = m.group(1).strip(), m.group(4)
        lname = name.lower()
        # Classify: touchpad checks before generic touch (Synaptics names
        # contain both "Touch" and "Touchpad"); wacom-with-pen is a tablet
        # (otherwise it'd fall through to "touchscreen" via generic touch).
        if "touchpad" in lname or "trackpad" in lname:
            type_ = "touchpad"
        elif "wacom" in lname and ("pen" in lname or "stylus" in lname):
            type_ = "tablet"
        elif "tablet" in lname:
            type_ = "tablet"
        elif re.search(r'\btouch(?:screen|\s*panel)\b', lname):
            type_ = "touchscreen"
        elif "touch" in lname and kind == "pointer":
            type_ = "touchscreen"
        else:
            type_ = kind
        devs.append(InputDevice(id=int(m.group(2)), name=name, type=type_))
    return devs


def parse_xorg_conf(text: str) -> XorgConf:
    conf = XorgConf()
    lines = text.splitlines()
    sec_re = re.compile(r'^\s*Section\s+"(\w+)"', re.IGNORECASE)
    end_re = re.compile(r'^\s*EndSection', re.IGNORECASE)
    id_re = re.compile(r'^\s*Identifier\s+"(.+?)"', re.IGNORECASE)
    opt_re = re.compile(r'^\s*Option\s+"(.+?)"\s+"([^"]*)"', re.IGNORECASE)
    busid_re = re.compile(r'^\s*BusID\s+"(.+?)"', re.IGNORECASE)
    driver_re = re.compile(r'^\s*Driver\s+"(.+?)"', re.IGNORECASE)
    i = 0
    while i < len(lines):
        m = sec_re.match(lines[i])
        if not m:
            i += 1
            continue
        kind, identifier, options, body = m.group(1), "", {}, []
        i += 1
        while i < len(lines) and not end_re.match(lines[i]):
            body.append(lines[i])
            line = lines[i]
            if (im := id_re.match(line)):
                identifier = im.group(1)
            if (om := opt_re.match(line)):
                options[om.group(1)] = om.group(2)
            if (bm := busid_re.match(line)):
                options["BusID"] = bm.group(1)
            if (dm := driver_re.match(line)):
                options["Driver"] = dm.group(1)
            i += 1
        conf.sections.append(XorgSection(kind=kind, identifier=identifier,
                                         options=options, raw="\n".join(body)))
        i += 1
    return conf


# ============================== COMPUTE ==============================

def effective_dimensions(o: Output) -> tuple[int, int]:
    return (o.height, o.width) if o.rotation in ("left", "right") else (o.width, o.height)


def is_active(o: Output) -> bool:
    return o.connected and o.enabled and o.width > 0


def compute_framebuffer_extent(outputs: list[Output]) -> tuple[int, int]:
    actives = [o for o in outputs if is_active(o)]
    if not actives:
        return (0, 0)
    w = h = 0
    for o in actives:
        ew, eh = effective_dimensions(o)
        w = max(w, o.x + ew)
        h = max(h, o.y + eh)
    return (w, h)


def compute_screen_framebuffer(outputs: list[Output], screen_id: int) -> tuple[int, int, int, int]:
    in_screen = [o for o in outputs if is_active(o) and o.screen_id == screen_id]
    if not in_screen:
        return (0, 0, 0, 0)
    xs, ys, x2s, y2s = [], [], [], []
    for o in in_screen:
        ew, eh = effective_dimensions(o)
        xs.append(o.x); ys.append(o.y)
        x2s.append(o.x + ew); y2s.append(o.y + eh)
    return (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))


def rotation_matrix(rotation: str) -> tuple[float, ...]:
    """Standard xinput rotation matrices (matches X.Org / Arch / Ubuntu docs)."""
    if rotation == "left":
        return (0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    if rotation == "right":
        return (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    if rotation == "inverted":
        return (-1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0)
    return IDENTITY_MATRIX


def matrix_multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = a
    b0, b1, b2, b3, b4, b5, b6, b7, b8 = b
    return (
        a0*b0+a1*b3+a2*b6, a0*b1+a1*b4+a2*b7, a0*b2+a1*b5+a2*b8,
        a3*b0+a4*b3+a5*b6, a3*b1+a4*b4+a5*b7, a3*b2+a4*b5+a5*b8,
        a6*b0+a7*b3+a8*b6, a6*b1+a7*b4+a8*b7, a6*b2+a7*b5+a8*b8,
    )


def derive_touchscreen_matrix(outputs: list[Output], target_name: str) -> tuple[float, ...]:
    """Compose region (output's rectangle within its X screen's framebuffer)
    × rotation. xinput's `Coordinate Transformation Matrix` expects this
    9-float row-major form; the matrix is applied per device event.

    Takes a list of outputs (not a State) so callers can pass either the raw
    state.outputs (for UI display) or the vendor-aware-collapsed `eff` copy
    (for runtime emit on non-NVIDIA, where all outputs share one X screen)."""
    target = next((o for o in outputs if o.name == target_name), None)
    if target is None:
        return IDENTITY_MATRIX
    smin_x, smin_y, sw, sh = compute_screen_framebuffer(outputs, target.screen_id)
    if sw <= 0 or sh <= 0:
        return rotation_matrix(target.rotation)
    eff_w, eff_h = effective_dimensions(target)
    region = (eff_w / sw, 0.0, (target.x - smin_x) / sw,
              0.0, eff_h / sh, (target.y - smin_y) / sh,
              0.0, 0.0, 1.0)
    return matrix_multiply(region, rotation_matrix(target.rotation))


def find_output_at(outputs: list[Output], x: int, y: int) -> Optional[Output]:
    for o in outputs:
        if not is_active(o):
            continue
        ew, eh = effective_dimensions(o)
        if o.x <= x < o.x + ew and o.y <= y < o.y + eh:
            return o
    return None


def runtime_screen_index(outputs: list[Output], screen_id: int) -> int:
    groups = sorted({o.screen_id for o in outputs if is_active(o)})
    return groups.index(screen_id) if screen_id in groups else screen_id


def snap_output_to_neighbors(outputs: list[Output], dragged: Output,
                              threshold: int = SNAP_THRESHOLD_FB) -> None:
    others = [o for o in outputs if is_active(o) and o.name != dragged.name]
    if not others:
        return
    dw, dh = effective_dimensions(dragged)
    xs, ys = [], []
    for o in others:
        ow, oh = effective_dimensions(o)
        xs += [o.x, o.x + ow]
        ys += [o.y, o.y + oh]
    # For each axis: collect every (distance, snapped-coord) candidate within
    # threshold, then pick the closest. Picking "first hit" was order-dependent
    # and could snap to a farther edge when two neighbors were both in range.
    cand_x = [(abs(dragged.x - tx), tx) for tx in xs]
    cand_x += [(abs((dragged.x + dw) - tx), tx - dw) for tx in xs]
    cand_x = [c for c in cand_x if c[0] <= threshold]
    if cand_x:
        dragged.x = min(cand_x)[1]
    cand_y = [(abs(dragged.y - ty), ty) for ty in ys]
    cand_y += [(abs((dragged.y + dh) - ty), ty - dh) for ty in ys]
    cand_y = [c for c in cand_y if c[0] <= threshold]
    if cand_y:
        dragged.y = min(cand_y)[1]


def normalize_origin(outputs: list[Output]) -> None:
    actives = [o for o in outputs if is_active(o)]
    if not actives:
        return
    min_x, min_y = min(o.x for o in actives), min(o.y for o in actives)
    if min_x or min_y:
        for o in actives:
            o.x -= min_x
            o.y -= min_y


def apply_mirroring(outputs: list[Output]) -> None:
    """Resolve mirror_of chains in-place. Cycles leave positions unchanged."""
    by_name = {o.name: o for o in outputs}
    for o in outputs:
        if not o.mirror_of or o.mirror_of == o.name:
            continue
        seen, cur = set(), o
        while cur.mirror_of and cur.name not in seen and cur.mirror_of != cur.name:
            seen.add(cur.name)
            nxt = by_name.get(cur.mirror_of)
            if nxt is None:
                break
            if not nxt.mirror_of:
                o.x, o.y = nxt.x, nxt.y
                o.screen_id = nxt.screen_id
                break
            cur = nxt


def enforce_single_primary(outputs: list[Output]) -> None:
    flagged = [o for o in outputs if o.primary]
    if len(flagged) <= 1:
        return
    actives = [o for o in flagged if is_active(o)]
    keeper = actives[0] if actives else flagged[0]
    for o in flagged:
        if o is not keeper:
            o.primary = False


# ============================== EMIT ==============================

def emit_xorg_conf(state: State) -> str:
    """Emit the full xorg.conf snippet. Vendor-aware:

    * **NVIDIA driver**: emits a Monitor section per X screen, Device sections
      with the documented `Screen N` directive (only when multi-X-screen,
      since both Devices share the same BusID), Screen sections with a direct
      `Monitor "MonitorN"` directive plus the NVIDIA-specific `MetaModes`
      Option to position multiple physical outputs within each X screen.
      Verified against NVIDIA's `configmultxscreens.html` for driver versions
      460 (2021), 470 (2022), and 535 (2024) — syntax stable across them all.
    * **Other drivers (amdgpu / modesetting / intel / radeon)**: collapses
      everything to a single minimal Device + Screen section. There is no
      `Monitor` directive (the driver autodetects via EDID), no MetaModes
      (NVIDIA-specific; ignored elsewhere), no `Screen N` (NVIDIA-specific
      multi-X-screen routing). All layout/rotation/positioning happens in
      the runtime xrandr script — that's how real-world AMD/Intel multi-
      display setups work, and the Gentoo wiki confirms minimal xorg.conf
      is the norm for amdgpu.

    Resolves mirrors and single-primary on a copy so the user's typed values
    aren't mutated."""
    eff = [dataclasses.replace(o, available_modes=list(o.available_modes))
           for o in state.outputs]
    apply_mirroring(eff)
    enforce_single_primary(eff)

    header = '# Generated by xorgcist. Review before installing.\n'
    if not any(is_active(o) for o in eff):
        return header

    gpu = state.gpus[0] if state.gpus else GPU(busid="", vendor="?",
                                                driver="modesetting", name="?")
    is_nvidia = gpu.driver == "nvidia"

    if is_nvidia:
        return _emit_nvidia_conf(eff, gpu, header)
    return _emit_generic_conf(eff, gpu, header)


def _emit_nvidia_conf(eff: list[Output], gpu: GPU, header: str) -> str:
    """NVIDIA-specific xorg.conf emit. Shape matches NVIDIA's documented
    multi-X-screen example: per-screen Monitor + Device + Screen sections,
    Devices share BusID and use the `Screen N` directive, Screens use the
    `Monitor "MonitorN"` directive plus `Option "MetaModes"` for positioning
    multiple displays within each X screen."""
    screens: dict[int, list[Output]] = {}
    for o in eff:
        if is_active(o):
            screens.setdefault(o.screen_id, []).append(o)
    sorted_groups = sorted(screens.keys())
    multi = len(sorted_groups) > 1

    parts = [header]
    # One Monitor section per X screen, named "MonitorN". NVIDIA's example
    # uses these as placeholders — driver still autodetects modes via EDID.
    for runtime_idx, _ in enumerate(sorted_groups):
        parts.append(
            f'Section "Monitor"\n'
            f'    Identifier  "Monitor{runtime_idx}"\n'
            f'EndSection\n'
        )
    # Device + Screen per X screen.
    for runtime_idx, group in enumerate(sorted_groups):
        busid_line = f'    BusID       "{gpu.busid}"\n' if gpu.busid else ""
        # `Screen N` only when multi-X-screen (without it, two Devices sharing
        # a BusID would be ambiguous). Single-screen NVIDIA omits it.
        screen_n_line = f'    Screen      {runtime_idx}\n' if multi else ""
        parts.append(
            f'Section "Device"\n'
            f'    Identifier  "Device{runtime_idx}"\n'
            f'    Driver      "{gpu.driver}"\n'
            f'{busid_line}'
            f'{screen_n_line}'
            f'EndSection\n'
        )
        outs_in_screen = screens[group]
        min_x = min(o.x for o in outs_in_screen)
        min_y = min(o.y for o in outs_in_screen)
        depth = max(o.color_depth for o in outs_in_screen)
        metamodes = []
        for o in outs_in_screen:
            mode = f"{o.width}x{o.height}"
            if o.refresh_rate > 0:
                mode += f"_{int(round(o.refresh_rate))}"
            entry = f"{o.name}: {mode}+{o.x - min_x}+{o.y - min_y}"
            if o.rotation != "normal":
                entry += f" {{Rotation={o.rotation}}}"
            metamodes.append(entry)
        screen_lines = [
            f'Section "Screen"',
            f'    Identifier  "Screen{runtime_idx}"',
            f'    Device      "Device{runtime_idx}"',
            f'    Monitor     "Monitor{runtime_idx}"',
        ]
        if metamodes:
            screen_lines.append(
                f'    Option      "MetaModes" "{", ".join(metamodes)}"'
            )
        screen_lines += [
            '    SubSection  "Display"',
            f'        Depth       {depth}',
            '    EndSubSection',
            'EndSection\n',
        ]
        parts.append("\n".join(screen_lines))

    parts.append(_emit_server_layout(eff, sorted_groups))
    return "\n".join(parts)


def _emit_generic_conf(eff: list[Output], gpu: GPU, header: str) -> str:
    """Minimal xorg.conf for non-NVIDIA drivers (amdgpu/modesetting/intel/...).
    These drivers don't use MetaModes or NVIDIA's multi-X-screen pattern;
    the runtime xrandr script handles all layout, positioning, and rotation.
    User's per-output `screen_id` grouping is ignored here — multi-X-screen
    is genuinely an NVIDIA-only practical feature."""
    parts = [header]
    busid_line = f'    BusID       "{gpu.busid}"\n' if gpu.busid else ""
    parts.append(
        f'Section "Device"\n'
        f'    Identifier  "Device0"\n'
        f'    Driver      "{gpu.driver}"\n'
        f'{busid_line}'
        f'EndSection\n'
    )
    parts.append(
        f'Section "Screen"\n'
        f'    Identifier  "Screen0"\n'
        f'    Device      "Device0"\n'
        f'EndSection\n'
    )
    parts.append(
        f'Section "ServerLayout"\n'
        f'    Identifier  "Layout0"\n'
        f'    Screen      0 "Screen0" 0 0\n'
        f'EndSection\n'
    )
    return "\n".join(parts)


def _emit_server_layout(eff: list[Output], sorted_groups: list[int]) -> str:
    """ServerLayout with relative position tokens between Screens. Xorg
    silently zeroes the absolute-coords form for any but the first Screen,
    so each subsequent Screen is anchored against the closest already-placed
    one via RightOf / LeftOf / Above / Below."""
    boxes = [compute_screen_framebuffer(eff, g) for g in sorted_groups]
    layout = ['Section "ServerLayout"', '    Identifier  "Layout0"',
              f'    Screen      0 "Screen0" 0 0']
    placed = [0]
    for i in range(1, len(sorted_groups)):
        def dsq(j):
            ax, ay, aw, ah = boxes[j]
            bx, by, bw, bh = boxes[i]
            dx = (bx + bw / 2) - (ax + aw / 2)
            dy = (by + bh / 2) - (ay + ah / 2)
            return dx * dx + dy * dy
        anchor = min(placed, key=dsq)
        ax, ay, aw, ah = boxes[anchor]
        bx, by, bw, bh = boxes[i]
        dx = (bx + bw / 2) - (ax + aw / 2)
        dy = (by + bh / 2) - (ay + ah / 2)
        if abs(dx) >= abs(dy):
            tok = "RightOf" if dx >= 0 else "LeftOf"
        else:
            tok = "Below" if dy >= 0 else "Above"
        layout.append(f'    Screen      {i} "Screen{i}" {tok} "Screen{anchor}"')
        placed.append(i)
    layout.append('EndSection\n')
    return "\n".join(layout)


DISPLAY_SETUP_LINE = (
    '_X="${DISPLAY:-:0}"; _X="${_X%.*}"   '
    '# X server from $DISPLAY (handles :N, :N.M, host:N.M, unset)'
)


def emit_runtime_commands(state: State) -> list[str]:
    """xrandr layout commands + xinput touchscreen calibration. Use this
    script in autostart / .xinitrc / a systemd user unit.

    Multi-X-screen scripts derive the X server number from $DISPLAY at
    runtime (not at emit time) so the same script works regardless of login
    order — second-user sessions get :1, third get :2, etc., and the script
    follows. xinput commands are server-wide and use whatever $DISPLAY is set
    to, no prefix needed."""
    eff = [dataclasses.replace(o, available_modes=list(o.available_modes))
           for o in state.outputs]
    apply_mirroring(eff)
    enforce_single_primary(eff)
    by_name = {o.name: o for o in eff}

    # Multi-X-screen is NVIDIA-only in xorgcist (NVIDIA's `Screen N` + shared
    # BusID + MetaModes pattern). For other drivers (amdgpu/modesetting/intel/
    # radeon) the xorg.conf is collapsed to a single Screen and xrandr does
    # all positioning at runtime — the script must match. Real multi-X-screen
    # on those drivers uses `Option "ZaphodHeads"` which is a different
    # mechanism, out of scope here.
    gpu = state.gpus[0] if state.gpus else GPU(busid="", vendor="?",
                                                driver="modesetting", name="?")
    is_nvidia = gpu.driver == "nvidia"
    if is_nvidia:
        sorted_groups = sorted({o.screen_id for o in eff if is_active(o)})
    else:
        # Treat every active output as if it were on a single X screen.
        # We use the *minimum* user-set screen_id as the canonical one so
        # `compute_screen_framebuffer` is called with a key that exists in eff.
        active_ids = {o.screen_id for o in eff if is_active(o)}
        sorted_groups = [min(active_ids)] if active_ids else []
        for o in eff:
            o.screen_id = sorted_groups[0] if sorted_groups else 0
    multi = len(sorted_groups) > 1
    cmds: list[str] = []
    if multi:
        cmds.append(DISPLAY_SETUP_LINE)

    # Active outputs per screen, mirrors after non-mirrors so --same-as targets
    # are configured first.
    for rt_idx, group in enumerate(sorted_groups):
        prefix = f'DISPLAY="$_X.{rt_idx}" ' if multi else ''
        smin_x, smin_y, _, _ = compute_screen_framebuffer(eff, group)
        in_screen = [o for o in eff if is_active(o) and o.screen_id == group]
        in_screen.sort(key=lambda o: 1 if o.mirror_of and o.mirror_of != o.name else 0)
        for o in in_screen:
            parts = [f'{prefix}xrandr --output {o.name}',
                     f'--mode {o.width}x{o.height}']
            # Only emit --rate when (w, h, refresh) matches a real EDID modeline
            target = (o.width, o.height)
            if any((mw, mh) == target and abs(mr - o.refresh_rate) < REFRESH_TOLERANCE_HZ
                   for mw, mh, mr in o.available_modes):
                parts.append(f'--rate {o.refresh_rate:.2f}')
            # Mirror only if the target is a real, active output; otherwise
            # `xrandr --same-as <missing>` errors and breaks the rest of the
            # script under `set -e`. Fall through to absolute --pos.
            mirror_target = (by_name.get(o.mirror_of)
                             if o.mirror_of and o.mirror_of != o.name else None)
            if mirror_target is not None and is_active(mirror_target):
                parts.append(f'--same-as {o.mirror_of}')
            else:
                parts.append(f'--pos {o.x - smin_x}x{o.y - smin_y}')
            if o.rotation != "normal":
                parts.append(f'--rotate {o.rotation}')
            if o.primary:
                parts.append('--primary')
            if abs(o.scale - 1.0) > SCALE_EPSILON:
                parts.append(f'--scale {o.scale:.3f}x{o.scale:.3f}')
            cmds.append(" ".join(parts))

    # Connected-but-disabled outputs: --off, routed to a real runtime screen
    for o in eff:
        if not o.connected or o.enabled:
            continue
        if o.screen_id in sorted_groups:
            rt_idx = sorted_groups.index(o.screen_id)
        elif sorted_groups:
            rt_idx = 0
        else:
            rt_idx = -1
        prefix = f'DISPLAY="$_X.{rt_idx}" ' if multi and rt_idx >= 0 else ''
        cmds.append(f'{prefix}xrandr --output {o.name} --off')

    # Touchscreens. xinput operates server-wide and inherits $DISPLAY from
    # the user's session. Calibration is just enable + set the Coordinate
    # Transformation Matrix — the matrix is screen-relative to the device's
    # current master pointer, which on real-world setups lives where the
    # physical touch panel is. (Multi-pointer X via `xinput create-master /
    # reattach` is a separate feature for multi-user touch tables; not what
    # touchscreen calibration needs.)
    for ts in state.touchscreens:
        if not ts.enabled:
            cmds.append(f'xinput disable {ts.device_id}')
            continue
        if ts.target_output is None:
            continue
        target = by_name.get(ts.target_output)
        if target is None or not is_active(target):
            continue
        cmds.append(f'xinput enable {ts.device_id}')
        # Pass `eff` (which has been collapsed for non-NVIDIA) so the matrix
        # is computed against the same screen layout the runtime script will
        # actually be applied against.
        m = derive_touchscreen_matrix(eff, ts.target_output)
        cmds.append(
            f'xinput set-prop {ts.device_id} "Coordinate Transformation Matrix" '
            + " ".join(f"{v:.6f}" for v in m)
        )
    return cmds


# ============================== VALIDATE ==============================

def validate_xorg_conf(text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    conf = parse_xorg_conf(text)
    if not [s for s in conf.sections if s.kind == "ServerLayout"]:
        errors.append(ValidationError("error", 'Missing required Section "ServerLayout"'))
    seen: set[tuple[str, str]] = set()
    for s in conf.sections:
        if not s.identifier:
            continue
        key = (s.kind, s.identifier)
        if key in seen:
            errors.append(ValidationError("error",
                                          f'Duplicate {s.kind} identifier "{s.identifier}"'))
        seen.add(key)
    screen_ids = {s.identifier for s in conf.sections if s.kind == "Screen"}
    layout_re = re.compile(r'Screen\s+\d+\s+"(.+?)"')
    for s in conf.sections:
        if s.kind != "ServerLayout":
            continue
        for line in s.raw.splitlines():
            m = layout_re.search(line)
            if m and m.group(1) not in screen_ids:
                errors.append(ValidationError(
                    "error", f'ServerLayout references non-existent Screen "{m.group(1)}"'))
    busid_re = re.compile(r'^PCI:(?:[\da-f]+@)?[\da-f]+:[\da-f]+:[\da-f]+$', re.IGNORECASE)
    for s in conf.sections:
        if s.kind == "Device":
            busid = s.options.get("BusID", "")
            if busid and not busid_re.match(busid):
                errors.append(ValidationError(
                    "warning", f'Device "{s.identifier}" has unusual BusID "{busid}"'))
    return errors


# ============================== STATE INIT + REGEN ==============================

def init_state() -> State:
    s = State(
        outputs=parse_xrandr(run_cmd(["xrandr", "--verbose"])),
        gpus=parse_lspci_gpus(run_cmd(["lspci", "-D"])),
        inputs=parse_xinput_list(run_cmd(["xinput", "list", "--short"])),
    )
    s.touchscreens = [
        TouchscreenMapping(device_id=d.id, device_name=d.name)
        for d in s.inputs if d.type == "touchscreen"
    ]
    s.env_session = os.environ.get("XDG_SESSION_TYPE", "?")
    s.env_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    s.env_display = os.environ.get("DISPLAY", "?")
    return s


def init_demo_state() -> State:
    s = State(
        gpus=[GPU(busid="PCI:1:0:0", vendor="NVIDIA", driver="nvidia",
                  name="NVIDIA RTX 4090")],
        outputs=[
            Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0,
                   primary=True, screen_id=0, refresh_rate=144.0,
                   available_modes=[(2560, 1440, 144.0), (1920, 1080, 60.0)]),
            Output("DP-2", connected=True, width=1920, height=1080, x=2560, y=360,
                   screen_id=0, refresh_rate=60.0,
                   available_modes=[(1920, 1080, 60.0), (1280, 720, 60.0)]),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=4480, y=0,
                   screen_id=1, rotation="left", refresh_rate=60.0,
                   available_modes=[(1920, 1080, 60.0)]),
            Output("HDMI-2", connected=True, width=1280, height=720,
                   screen_id=0, enabled=False,
                   available_modes=[(1280, 720, 60.0)]),
        ],
        inputs=[
            InputDevice(id=10, name="DELL Mouse", type="pointer"),
            InputDevice(id=11, name="AT Translated Set 2 keyboard", type="keyboard"),
            InputDevice(id=15, name="ELO TouchSystems Touchscreen 2700", type="touchscreen"),
        ],
        touchscreens=[
            TouchscreenMapping(device_id=15, device_name="ELO TouchSystems Touchscreen 2700"),
        ],
    )
    return s


def regenerate(s: State) -> None:
    s.framebuffer_width, s.framebuffer_height = compute_framebuffer_extent(s.outputs)
    s.generated_config = emit_xorg_conf(s)
    s.generated_runtime = emit_runtime_commands(s)
    s.validation_errors = validate_xorg_conf(s.generated_config)


# ============================== UI ==============================
# Callbacks are minimal: mutate state, regenerate, redraw. No cb_/action_ split.
# `redraw_full` rebuilds everything; `redraw_canvas_only` is used during drag
# and per-keystroke value typing so the focused widget isn't destroyed.

def redraw_full(s: State) -> None:
    regenerate(s)
    redraw_canvas(s)
    redraw_form(s)
    redraw_inputs(s)
    redraw_touch(s)
    redraw_preview(s)
    redraw_header(s)
    redraw_status(s)


def redraw_canvas_only(s: State) -> None:
    """Used during drag and per-keystroke input. Rebuilds canvas + preview
    but leaves form/list panes alone so a focused input field survives."""
    regenerate(s)
    redraw_canvas(s)
    redraw_preview(s)
    redraw_header(s)


def redraw_status(s: State) -> None:
    if dpg.does_item_exist(T_STATUS):
        dpg.set_value(T_STATUS, s.status_message)


def redraw_header(s: State) -> None:
    if not dpg.does_item_exist(T_HEADER):
        return
    connected = sum(1 for o in s.outputs if o.connected)
    active = sum(1 for o in s.outputs if is_active(o))
    dpg.set_value(T_HEADER,
                  f"GPUs: {len(s.gpus)}   Outputs: {connected} ({active} active)   "
                  f"Inputs: {len(s.inputs)}   Touchscreens: {len(s.touchscreens)}")


def redraw_preview(s: State) -> None:
    if dpg.does_item_exist(T_CONFIG):
        dpg.set_value(T_CONFIG, s.generated_config)
    if dpg.does_item_exist(T_RUNTIME):
        dpg.set_value(T_RUNTIME, "\n".join(s.generated_runtime))
    if dpg.does_item_exist(T_VALIDATION):
        if s.validation_errors:
            dpg.set_value(T_VALIDATION, "\n".join(
                f"[{e.severity:<7}] {e.message}" for e in s.validation_errors))
        else:
            dpg.set_value(T_VALIDATION, "(no validation issues)")


def redraw_canvas(s: State) -> None:
    if not dpg.does_item_exist(T_CANVAS):
        return
    dpg.delete_item(T_CANVAS, children_only=True)
    cw = dpg.get_item_width(T_CANVAS) or 800
    ch = dpg.get_item_height(T_CANVAS) or 500
    fb_w, fb_h = s.framebuffer_width, s.framebuffer_height
    if fb_w == 0 or fb_h == 0:
        s.canvas_scale, s.canvas_off_x, s.canvas_off_y = 1.0, 0.0, 0.0
        actives = sum(1 for o in s.outputs if is_active(o))
        dpg.draw_text((20, 20), "Nothing to draw — no active positioned outputs.",
                      color=(255, 180, 180, 255), size=18, parent=T_CANVAS)
        dpg.draw_text((20, 50),
                      f"xrandr returned {len(s.outputs)} output(s); {actives} active.",
                      color=(200, 200, 200, 255), size=14, parent=T_CANVAS)
        if not s.outputs and s.env_wayland:
            dpg.draw_text((20, 90),
                          f"XDG_SESSION_TYPE={s.env_session}   WAYLAND={s.env_wayland}   DISPLAY={s.env_display}",
                          color=(180, 180, 180, 255), size=12, parent=T_CANVAS)
            dpg.draw_text((20, 110),
                          "On Wayland — xrandr can't see real outputs. "
                          "Run with --demo or boot into a real X11 session.",
                          color=(220, 200, 160, 255), size=12, parent=T_CANVAS)
        elif s.outputs:
            for i, o in enumerate(s.outputs[:10]):
                flags = []
                if not o.connected: flags.append("disconnected")
                if not o.enabled: flags.append("disabled")
                if o.connected and o.width == 0: flags.append("no-mode")
                dpg.draw_text(
                    (20, 90 + i * 18),
                    f"  • {o.name}: {', '.join(flags) or 'ok'}, {o.width}x{o.height}+{o.x}+{o.y}",
                    color=(180, 180, 180, 255), size=12, parent=T_CANVAS,
                )
        return
    scale = min(cw / fb_w, ch / fb_h) * 0.9
    s.canvas_scale = scale
    s.canvas_off_x = (cw - fb_w * scale) / 2
    s.canvas_off_y = (ch - fb_h * scale) / 2
    label_size = max(10, int(14 * s.ui_scale))
    detail_size = max(8, int(12 * s.ui_scale))
    # Draw the selected output last so its rectangle (and label) sit on top
    # of any neighbors that visually overlap or abut it.
    draw_order = sorted(s.outputs, key=lambda o: 1 if o.name == s.selected_output else 0)
    for o in draw_order:
        if not is_active(o):
            continue
        ew, eh = effective_dimensions(o)
        x1 = s.canvas_off_x + o.x * scale
        y1 = s.canvas_off_y + o.y * scale
        x2 = x1 + ew * scale
        y2 = y1 + eh * scale
        selected = (o.name == s.selected_output)
        base = SCREEN_PALETTE[o.screen_id % len(SCREEN_PALETTE)]
        if selected:
            outline = (255, 220, 110, 255)
            fill = tuple(min(c + 40, 255) for c in base[:3]) + (255,)
        else:
            outline = (210, 210, 210, 255)
            fill = base
        rt_idx = runtime_screen_index(s.outputs, o.screen_id)
        dpg.draw_rectangle((x1, y1), (x2, y2), color=outline, fill=fill,
                           thickness=2 if selected else 1, parent=T_CANVAS)
        tags = []
        if o.primary: tags.append("PRIMARY")
        if o.rotation != "normal": tags.append(o.rotation.upper())
        if o.mirror_of: tags.append(f"MIRROR→{o.mirror_of}")
        if abs(o.scale - 1.0) > SCALE_EPSILON: tags.append(f"×{o.scale:.2f}")
        tag_str = "  [" + " ".join(tags) + "]" if tags else ""
        dpg.draw_text((x1 + 6, y1 + 6),
                      f"{o.name}  (Group {o.screen_id} → :0.{rt_idx}){tag_str}",
                      color=(255, 255, 255, 255), size=label_size, parent=T_CANVAS)
        dpg.draw_text((x1 + 6, y1 + 6 + label_size + 2),
                      f"{ew}x{eh}  @{int(round(o.refresh_rate))}Hz  d{o.color_depth}",
                      color=(180, 180, 180, 255), size=detail_size, parent=T_CANVAS)


def redraw_form(s: State) -> None:
    if not dpg.does_item_exist(T_FORM):
        return
    dpg.delete_item(T_FORM, children_only=True)
    # Per-output item-handler registries from the previous redraw — they live
    # globally (not under T_FORM) so children_only=True doesn't drop them.
    # Iterate by alias prefix rather than by current outputs so handlers for
    # outputs removed since the last redraw (e.g., after Reload) get cleaned up.
    for tag in list(dpg.get_aliases()):
        if tag.startswith("form_handler_"):
            dpg.delete_item(tag)
    for o in s.outputs:
        if not o.connected:
            continue
        rt_idx = runtime_screen_index(s.outputs, o.screen_id)
        # add_collapsing_header doesn't accept a `callback` kwarg, so we bind
        # an item-handler registry with a clicked handler. One registry per
        # header so each carries its own user_data (the output name).
        h_tag = f"form_handler_{o.name}"
        with dpg.item_handler_registry(tag=h_tag):
            dpg.add_item_clicked_handler(callback=cb_select_output,
                                          user_data=(s, o.name))
        header_tag = f"form_header_{o.name}"
        with dpg.collapsing_header(
            label=f"{o.name}{' [DISABLED]' if not o.enabled else ''}",
            parent=T_FORM,
            default_open=(o.name == s.selected_output),
            tag=header_tag,
        ):
            dpg.add_checkbox(label="Enabled", default_value=o.enabled,
                             callback=cb_attr, user_data=(s, o.name, "enabled", bool))
            dpg.add_checkbox(label="Primary", default_value=o.primary,
                             callback=cb_primary, user_data=(s, o.name))
            dpg.add_separator()
            dpg.add_text("Position", color=COLOR_HEAD)
            dpg.add_input_int(label="X", default_value=o.x, width=W_NUM,
                              callback=cb_attr_light, user_data=(s, o.name, "x", int))
            dpg.add_input_int(label="Y", default_value=o.y, width=W_NUM,
                              callback=cb_attr_light, user_data=(s, o.name, "y", int))
            dpg.add_input_int(label=f"Screen Group  →  :0.{rt_idx}",
                              default_value=o.screen_id, width=W_NUM,
                              min_value=0, max_value=SCREEN_GROUP_MAX,
                              callback=cb_attr_light, user_data=(s, o.name, "screen_id", int))
            dpg.add_separator()
            dpg.add_text("Mode", color=COLOR_HEAD)
            mode_items = [f"{w}x{h} @ {r:.1f}Hz" for w, h, r in o.available_modes]
            current_mode = f"{o.width}x{o.height} @ {o.refresh_rate:.1f}Hz"
            if current_mode not in mode_items:
                mode_items.insert(0, current_mode)
            dpg.add_combo(items=mode_items, label="Mode", default_value=current_mode,
                          width=W_DROP, callback=cb_mode, user_data=(s, o.name))
            dpg.add_combo(items=[str(d) for d in COLOR_DEPTHS], label="Color depth",
                          default_value=str(o.color_depth), width=W_NUM,
                          callback=cb_attr, user_data=(s, o.name, "color_depth", int))
            dpg.add_separator()
            dpg.add_text("Orientation / scale", color=COLOR_HEAD)
            dpg.add_combo(items=list(ROTATIONS), label="Rotation",
                          default_value=o.rotation, width=W_DROP,
                          callback=cb_attr, user_data=(s, o.name, "rotation", str))
            dpg.add_input_float(label="Scale", default_value=o.scale, width=W_NUM,
                                format="%.2f", step=0.05, min_value=0.5, max_value=4.0,
                                callback=cb_attr_light, user_data=(s, o.name, "scale", float))
            dpg.add_separator()
            dpg.add_text("Mirror", color=COLOR_HEAD)
            mirror_items = ["(none)"] + [oo.name for oo in s.outputs
                                         if oo.connected and oo.name != o.name]
            mirror_default = o.mirror_of if o.mirror_of in mirror_items else "(none)"
            dpg.add_combo(items=mirror_items, label="Mirror of",
                          default_value=mirror_default, width=W_DROP,
                          callback=cb_mirror, user_data=(s, o.name))
            dpg.add_text(f"Native: {o.width}x{o.height}", color=COLOR_DIM)
            dpg.add_text(f"{len(o.available_modes)} modes available", color=COLOR_DIM)
        dpg.bind_item_handler_registry(header_tag, h_tag)


def redraw_inputs(s: State) -> None:
    if not dpg.does_item_exist(T_INPUT_LIST):
        return
    dpg.delete_item(T_INPUT_LIST, children_only=True)
    for d in s.inputs:
        dpg.add_text(f"[{d.type:<11}] {d.name}  (id={d.id})", parent=T_INPUT_LIST)


def redraw_touch(s: State) -> None:
    if not dpg.does_item_exist(T_TOUCH_LIST):
        return
    dpg.delete_item(T_TOUCH_LIST, children_only=True)
    if not s.touchscreens:
        dpg.add_text("No touchscreens detected.", color=COLOR_DIM, parent=T_TOUCH_LIST)
        return
    output_names = ["(unassigned)"] + [o.name for o in s.outputs if o.connected]
    for ts in s.touchscreens:
        with dpg.group(parent=T_TOUCH_LIST):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", default_value=ts.enabled,
                                 callback=cb_touch_enabled, user_data=(s, ts.device_id))
                dpg.add_text(f"{ts.device_name}  (id={ts.device_id})")
            with dpg.group(horizontal=True):
                dpg.add_text("    Map to output:")
                current = ts.target_output if ts.target_output else "(unassigned)"
                dpg.add_combo(items=output_names, default_value=current,
                              width=W_DROP_WIDE, callback=cb_touch_target,
                              user_data=(s, ts.device_id))
            if ts.target_output and ts.enabled:
                # UI display: use raw outputs (matches what the user sees in
                # the layout canvas, not the collapsed-for-runtime view).
                m = derive_touchscreen_matrix(s.outputs, ts.target_output)
                dpg.add_text(
                    f"    [{m[0]:.4f} {m[1]:.4f} {m[2]:.4f}]\n"
                    f"    [{m[3]:.4f} {m[4]:.4f} {m[5]:.4f}]\n"
                    f"    [{m[6]:.4f} {m[7]:.4f} {m[8]:.4f}]",
                    color=(180, 180, 220, 255),
                )
            dpg.add_separator()


# Callbacks. `cb_attr` triggers full redraw (combo / checkbox; user clicked once
# so no focus to lose). `cb_attr_light` triggers canvas-only redraw (input fields
# fire per-keystroke; full rebuild would destroy the focused widget).

def cb_select_output(sender, app_data, user_data):
    """Header click on the right pane → mark this output active so its
    rectangle draws on top of the canvas. Sidebar order is preserved.
    Clicking the already-active header is a no-op so Dear PyGui's natural
    collapse/expand still works."""
    s, name = user_data
    if s.selected_output == name:
        return
    s.selected_output = name
    redraw_full(s)


def cb_attr(sender, value, user_data):
    s, name, attr, conv = user_data
    for o in s.outputs:
        if o.name == name:
            setattr(o, attr, conv(value))
            s.selected_output = name
    redraw_full(s)


def cb_attr_light(sender, value, user_data):
    s, name, attr, conv = user_data
    for o in s.outputs:
        if o.name == name:
            setattr(o, attr, conv(value))
            s.selected_output = name
    redraw_canvas_only(s)


def cb_primary(sender, value, user_data):
    s, name = user_data
    for o in s.outputs:
        o.primary = (o.name == name) if value else (o.primary and o.name != name)
    redraw_full(s)


def cb_mode(sender, value, user_data):
    s, name = user_data
    m = re.match(r'^\s*(\d+)x(\d+)(?:\s*@\s*([\d.]+)\s*Hz)?', value)
    if not m:
        return
    w, h = int(m.group(1)), int(m.group(2))
    refresh = float(m.group(3)) if m.group(3) else None
    for o in s.outputs:
        if o.name == name:
            o.width, o.height = w, h
            if refresh is not None:
                o.refresh_rate = refresh
            s.selected_output = name
    redraw_full(s)


def cb_mirror(sender, value, user_data):
    s, name = user_data
    target = None if value == "(none)" else value
    for o in s.outputs:
        if o.name == name:
            o.mirror_of = target
    redraw_full(s)


def cb_touch_target(sender, value, user_data):
    s, dev_id = user_data
    target = None if value == "(unassigned)" else value
    for ts in s.touchscreens:
        if ts.device_id == dev_id:
            ts.target_output = target
    redraw_full(s)


def cb_touch_enabled(sender, value, user_data):
    s, dev_id = user_data
    for ts in s.touchscreens:
        if ts.device_id == dev_id:
            ts.enabled = bool(value)
    redraw_full(s)


def cb_canvas_clicked(sender, app_data, user_data):
    s: State = user_data
    if not dpg.does_item_exist(T_CANVAS) or not dpg.is_item_hovered(T_CANVAS):
        return
    mx, my = dpg.get_mouse_pos(local=False)
    rx, ry = dpg.get_item_rect_min(T_CANVAS)
    if s.canvas_scale <= 0:
        return
    fx = (mx - rx - s.canvas_off_x) / s.canvas_scale
    fy = (my - ry - s.canvas_off_y) / s.canvas_scale
    hit = find_output_at(s.outputs, int(fx), int(fy))
    s.selected_output = hit.name if hit else None
    if hit:
        s.dragging_output = hit.name
        s.drag_anchor_fb = (int(fx) - hit.x, int(fy) - hit.y)
    redraw_canvas_only(s)


def cb_canvas_mouse_move(sender, app_data, user_data):
    s: State = user_data
    if s.dragging_output is None:
        return
    if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        s.dragging_output = None
        return
    if s.canvas_scale <= 0:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    rx, ry = dpg.get_item_rect_min(T_CANVAS)
    fx = (mx - rx - s.canvas_off_x) / s.canvas_scale
    fy = (my - ry - s.canvas_off_y) / s.canvas_scale
    out = next((o for o in s.outputs if o.name == s.dragging_output), None)
    if out is None:
        return
    eff_w, eff_h = effective_dimensions(out)
    fb_w = s.framebuffer_width or eff_w
    fb_h = s.framebuffer_height or eff_h
    new_x = max(-eff_w, min(fb_w, int(fx) - s.drag_anchor_fb[0]))
    new_y = max(-eff_h, min(fb_h, int(fy) - s.drag_anchor_fb[1]))
    if new_x != out.x or new_y != out.y:
        out.x, out.y = new_x, new_y
        normalize_origin(s.outputs)
        redraw_canvas_only(s)


def cb_canvas_mouse_up(sender, app_data, user_data):
    s: State = user_data
    if s.dragging_output is None:
        return
    out = next((o for o in s.outputs if o.name == s.dragging_output), None)
    if out is not None:
        snap_output_to_neighbors(s.outputs, out)
        normalize_origin(s.outputs)
    s.dragging_output = None
    redraw_full(s)  # full because snap may have moved enough to need preview update


def cb_viewport_resize(s: State):
    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    if vw <= 0 or vh <= 0:
        return
    s.ui_scale = max(UI_SCALE_MIN, min(UI_SCALE_MAX, vh / BASE_VIEWPORT_H))
    dpg.set_global_font_scale(s.ui_scale)
    form_w = int(BASE_FORM_W * s.ui_scale)
    canvas_w = max(200, vw - form_w - 40)
    canvas_h = max(200, vh - int(CHROME_H_ESTIMATE * s.ui_scale))
    if dpg.does_item_exist(T_CANVAS):
        dpg.configure_item(T_CANVAS, width=canvas_w, height=canvas_h)
    if dpg.does_item_exist(T_FORM):
        dpg.configure_item(T_FORM, width=form_w, height=canvas_h)
    redraw_canvas(s)


# ============================== BUILD UI ==============================

def build_ui(s: State) -> None:
    dpg.create_context()
    dpg.create_viewport(title="xorgcist", width=BASE_VIEWPORT_W, height=BASE_VIEWPORT_H)

    def save_config_cb(sender, app_data):
        path = app_data["file_path_name"]
        Path(path).write_text(s.generated_config)
        s.status_message = f"Saved {len(s.generated_config)} bytes → {path}"
        redraw_status(s)

    def save_runtime_cb(sender, app_data):
        path = app_data["file_path_name"]
        body = ("#!/bin/sh\n# Generated by xorgcist — apply at session start.\n"
                + "\n".join(s.generated_runtime) + "\n")
        Path(path).write_text(body)
        try:
            Path(path).chmod(0o755)
        except OSError:
            pass
        s.status_message = f"Saved runtime script → {path}"
        redraw_status(s)

    with dpg.file_dialog(directory_selector=False, show=False, callback=save_config_cb,
                         tag="save_config_dialog", width=700, height=420,
                         default_filename="10-xorgcist.conf"):
        dpg.add_file_extension(".conf")
        dpg.add_file_extension(".*")
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_runtime_cb,
                         tag="save_runtime_dialog", width=700, height=420,
                         default_filename="apply-xorgcist.sh"):
        dpg.add_file_extension(".sh")
        dpg.add_file_extension(".*")

    with dpg.item_handler_registry(tag=T_CANVAS_HANDLERS):
        dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left,
                                     callback=cb_canvas_clicked, user_data=s)
    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=cb_canvas_mouse_move, user_data=s)
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                      callback=cb_canvas_mouse_up, user_data=s)

    def reload_cb():
        fresh = init_state()
        s.outputs = fresh.outputs
        s.gpus = fresh.gpus
        s.inputs = fresh.inputs
        s.touchscreens = fresh.touchscreens
        s.selected_output = None
        s.dragging_output = None
        s.drag_anchor_fb = (0, 0)
        s.status_message = "Reloaded state from system"
        redraw_full(s)

    def copy_config():
        try:
            dpg.set_clipboard_text(s.generated_config)
            s.status_message = "Config copied to clipboard"
        except Exception as e:
            s.status_message = f"Clipboard failed: {e}"
        redraw_status(s)

    def copy_runtime():
        try:
            dpg.set_clipboard_text("\n".join(s.generated_runtime))
            s.status_message = "Runtime commands copied to clipboard"
        except Exception as e:
            s.status_message = f"Clipboard failed: {e}"
        redraw_status(s)

    with dpg.window(tag="primary_window"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reload from system", callback=reload_cb)
            dpg.add_text("|")
            dpg.add_text("", tag=T_HEADER)
        dpg.add_separator()
        with dpg.tab_bar():
            with dpg.tab(label="Display Layout"):
                dpg.add_text(
                    "Drag displays to position them. Outputs sharing a Screen Group go on one X "
                    "screen; different Groups become separate X screens. Color = grouping.",
                    color=COLOR_DIM, wrap=1100,
                )
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    with dpg.drawlist(width=800, height=500, tag=T_CANVAS):
                        pass
                    with dpg.child_window(width=BASE_FORM_W, height=500, tag=T_FORM):
                        pass
            with dpg.tab(label="Input Routing"):
                dpg.add_text("Detected input devices (informational).", color=COLOR_DIM)
                dpg.add_separator()
                with dpg.child_window(tag=T_INPUT_LIST, height=-1):
                    pass
            with dpg.tab(label="Touchscreen Mapping"):
                dpg.add_text(
                    "Map each touchscreen to its physical output. The Coordinate "
                    "Transformation Matrix is derived from your layout + rotation; "
                    "applied via `xinput set-prop`.",
                    color=COLOR_DIM, wrap=900,
                )
                dpg.add_separator()
                with dpg.child_window(tag=T_TOUCH_LIST, height=-1):
                    pass
            with dpg.tab(label="Generated Config"):
                with dpg.tab_bar():
                    with dpg.tab(label="xorg.conf"):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Save...",
                                           callback=lambda: dpg.show_item("save_config_dialog"))
                            dpg.add_button(label="Copy", callback=copy_config)
                        dpg.add_input_text(tag=T_CONFIG, multiline=True, readonly=True,
                                           width=-1, height=-PREVIEW_BOTTOM_RESERVE)
                        dpg.add_text("Validation:")
                        dpg.add_input_text(tag=T_VALIDATION, multiline=True, readonly=True,
                                           width=-1, height=-1)
                    with dpg.tab(label="Runtime commands (xrandr + xinput)"):
                        dpg.add_text(
                            "Apply at session start (autostart, .xinitrc, systemd user unit).",
                            color=COLOR_DIM, wrap=900,
                        )
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Save script...",
                                           callback=lambda: dpg.show_item("save_runtime_dialog"))
                            dpg.add_button(label="Copy", callback=copy_runtime)
                        dpg.add_input_text(tag=T_RUNTIME, multiline=True, readonly=True,
                                           width=-1, height=-1)
        dpg.add_separator()
        dpg.add_input_text(tag=T_STATUS, default_value="Ready.", readonly=True, width=-1)

    dpg.bind_item_handler_registry(T_CANVAS, T_CANVAS_HANDLERS)

    dpg.setup_dearpygui()
    dpg.set_primary_window("primary_window", True)
    dpg.set_viewport_resize_callback(lambda: cb_viewport_resize(s))
    dpg.show_viewport()
    cb_viewport_resize(s)
    redraw_full(s)
    dpg.start_dearpygui()
    dpg.destroy_context()


# ============================== MAIN ==============================

def main() -> int:
    if "--demo" in sys.argv:
        s = init_demo_state()
        s.env_session = os.environ.get("XDG_SESSION_TYPE", "?")
        s.env_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        s.env_display = os.environ.get("DISPLAY", "?")
        print("xorgcist: --demo mode (synthetic data).", file=sys.stderr)
    else:
        s = init_state()
        if not s.outputs:
            print(
                "xorgcist: no outputs detected. Likely Wayland.\n"
                "         Try `python3 xorgcist.py --demo` or boot into an X11 session.",
                file=sys.stderr,
            )
    build_ui(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
