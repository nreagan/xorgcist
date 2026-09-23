#!/usr/bin/env python3
"""xorgcist: arrange NVIDIA X11 displays, then generate an xorg.conf and
touchscreen calibration commands. It only shows and saves text; nothing on the
system is changed. Needs Python 3.6+ and tkinter (RHEL8: dnf install python3-tkinter).
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------- constants

ROTATIONS = ["normal", "left", "right", "inverted"]

# Undoes a display rotation in the touch panel's normalized 0..1 space.
# Row-major 3x3, same layout as xinput's "Coordinate Transformation Matrix".
ROTATION_MATRIX = {
    "normal": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    "left": (0, -1, 1, 1, 0, 0, 0, 0, 1),
    "right": (0, 1, 0, -1, 0, 1, 0, 0, 1),
    "inverted": (-1, 0, 1, 0, -1, 1, 0, 0, 1),
}

ASPECTS = [(16, 9), (16, 10), (4, 3), (5, 4), (21, 9), (32, 9), (3, 2), (1, 1)]

SCREEN_COLORS = ["#a9c8f5", "#f7c99b", "#b3e0a6", "#f5b1b1",
                 "#d2bdf2", "#f2e59b", "#a8e3dd", "#e3c7ad"]

# ---------------------------------------------------------------- detect

OUTPUT_RE = re.compile(r"^(\S+) (connected|disconnected)([^(]*)")
GEOMETRY_RE = re.compile(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)")
MODE_RE = re.compile(r"^\s+(\d+)x(\d+)(\S*)\s+(.*)$")
SCREEN_RE = re.compile(r"^Screen \d+:.* current (\d+) x (\d+)")
XINPUT_RE = re.compile(r"^\W*(.+?)\s+id=(\d+)\s+\[slave\s+pointer")


def run(cmd, display=None):
    env = dict(os.environ, DISPLAY=display) if display else None
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=10, env=env)
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else ""


def parse_xrandr(text):
    """One `xrandr --screen N` listing -> (connected displays, screen width)."""
    displays, width, d = [], 0, None
    for line in text.splitlines():
        m = SCREEN_RE.match(line)
        if m:
            width = int(m.group(1))
            continue
        m = OUTPUT_RE.match(line)
        if m:
            d = None
            if m.group(2) == "connected":
                head = m.group(3)
                g = GEOMETRY_RE.search(head)
                d = {"name": m.group(1), "modes": {}, "size": None, "rate": None,
                     "preferred": None, "enabled": bool(g), "screen": 0,
                     "rotation": next((w for w in head.split() if w in ROTATIONS), "normal"),
                     "x": int(g.group(3)) if g else 0, "y": int(g.group(4)) if g else 0}
                displays.append(d)
            continue
        m = MODE_RE.match(line)
        if d is None or not m or "i" in m.group(3):
            continue
        size = (int(m.group(1)), int(m.group(2)))
        rates = d["modes"].setdefault(size, [])
        rate = None
        for tok in m.group(4).split():
            num = re.match(r"\d+(?:\.\d+)?", tok)
            if num:
                rate = float(num.group())
                if rate not in rates:
                    rates.append(rate)
            if rate is None:
                continue
            if "*" in tok:
                d["size"], d["rate"] = size, rate
            if "+" in tok and d["preferred"] is None:
                d["preferred"] = (size, rate)

    result = []
    for d in displays:
        modes = {s: r for s, r in d["modes"].items() if r}
        if not modes:
            continue
        first = next(iter(modes))
        fallback = d.pop("preferred") or (first, modes[first][0])
        d["modes"] = modes
        if d["size"] is None:
            d["size"], d["rate"] = fallback
        result.append(d)
    return result, width


def parse_xinput(text):
    """`xinput list` -> ["11  ELAN Touchscreen", ...], touchscreens first."""
    devices = []
    for line in text.splitlines():
        m = XINPUT_RE.match(line)
        if m and "XTEST" not in m.group(1):
            devices.append("%s  %s" % (m.group(2), m.group(1)))
    return sorted(devices, key=lambda d: "touch" not in d.lower())


def device_id(text):
    m = re.match(r"\s*(\d+)", text)
    return m.group(1) if m else None


def pci_to_busid(pci):
    """lspci-style hex "0000:65:00.0" -> xorg.conf decimal "PCI:101:0:0"."""
    domain, bus, devfn = pci.split(":")
    dev, fn = devfn.split(".")
    domain, bus, dev, fn = [int(v, 16) for v in (domain, bus, dev, fn)]
    if domain:
        return "PCI:%d@%d:%d:%d" % (bus, domain, dev, fn)
    return "PCI:%d:%d:%d" % (bus, dev, fn)


def nvidia_gpus():
    try:
        names = sorted(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return []
    return [pci_to_busid(n) for n in names]


def detect(ctrl_display=None):
    listings = []
    for n in range(16):
        text = run(["xrandr", "--screen", str(n), "--query"], ctrl_display)
        if not text:
            break
        listings.append(text)
    xinput_text, gpus = run(["xinput", "list"], ctrl_display), nvidia_gpus()

    # xrandr positions are per X screen; lay the X screens out left to right.
    displays, origin = [], 0
    for screen, text in enumerate(listings):
        found, width = parse_xrandr(text)
        for d in found:
            d["screen"] = screen
            d["x"] += origin
        displays += found
        origin += width
    right = max([d["x"] + footprint(d)[0] for d in displays if d["enabled"]] or [0])
    for d in displays:
        if not d["enabled"]:
            d["x"], d["y"] = right, 0
            right += footprint(d)[0]

    state = {"displays": displays, "gpus": gpus, "screen_gpu": [],
             "devices": parse_xinput(xinput_text), "touch": [],
             "selected": 0 if displays else None}
    compact_screens(state)

    target = ctrl_display or os.environ.get("DISPLAY", "")
    if not displays:
        status = ("xrandr found no connected displays on X display %s. "
                  "Check the display number and X authorization (see README)." % (target or "(unset)"))
    else:
        status = "X display %s: %d display(s) on %d X screen(s). NVIDIA GPU: %s." % (
            target, len(displays), len(listings),
            ", ".join(gpus) or "none found in /proc/driver/nvidia/gpus, so BusID is left out")
    return state, status

# ---------------------------------------------------------------- compute


def footprint(d):
    w, h = d["size"]
    return (h, w) if d["rotation"] in ("left", "right") else (w, h)


def bbox(displays):
    return (min(d["x"] for d in displays), min(d["y"] for d in displays),
            max(d["x"] + footprint(d)[0] for d in displays),
            max(d["y"] + footprint(d)[1] for d in displays))


def enabled(state):
    return [d for d in state["displays"] if d["enabled"]]


def screen_count(state):
    return len({d["screen"] for d in enabled(state)})


def aspect_label(w, h):
    for a, b in ASPECTS:
        if abs(w / h - a / b) < 0.03 * a / b:
            return "%d:%d" % (a, b)
    g = math.gcd(w, h)
    return "%d:%d" % (w // g, h // g)


def mode_name(d):
    # NVIDIA names pool modes "WxH_R" (R = rounded refresh); plain "WxH" is
    # the driver's best mode at that size, so only name the rate if there's a choice.
    w, h = d["size"]
    if len(d["modes"][d["size"]]) > 1:
        return "%dx%d_%d" % (w, h, int(d["rate"] + 0.5))
    return "%dx%d" % (w, h)


def compact_screens(state):
    """Renumber X screens 0..n-1 (X requires consecutive numbers). A disabled
    display whose screen disappeared gets an out-of-range number, meaning
    "new screen" when it is enabled again."""
    used = sorted({d["screen"] for d in enabled(state)})
    default = state["gpus"][0] if state["gpus"] else ""
    old = state["screen_gpu"]
    state["screen_gpu"] = [old[s] if s < len(old) else default for s in used]
    remap = {s: i for i, s in enumerate(used)}
    for d in state["displays"]:
        d["screen"] = remap.get(d["screen"], len(state["displays"]))


def matmul3(a, b):
    return tuple(sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3))
                 for r in range(3) for c in range(3))


def touch_matrix(shown, d):
    # X scales absolute input to the bounding box of *all* X screens
    # (dix/getevents.c scale_to_desktop), so "total" is the whole desktop.
    x0, y0, x1, y1 = bbox(shown)
    w, h = footprint(d)
    tw, th = x1 - x0, y1 - y0
    scale = (w / tw, 0, (d["x"] - x0) / tw,
             0, h / th, (d["y"] - y0) / th,
             0, 0, 1)
    return matmul3(scale, ROTATION_MATRIX[d["rotation"]])


def fmt_num(v):
    s = ("%.6f" % v).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def unreachable_screens(state):
    """X screens the pointer can't get to: X only crosses shared edges."""
    shown = enabled(state)
    n = screen_count(state)
    rects = [bbox([d for d in shown if d["screen"] == s]) for s in range(n)]

    def edge_connected(a, b):
        xo = min(a[2], b[2]) - max(a[0], b[0])
        yo = min(a[3], b[3]) - max(a[1], b[1])
        return (xo >= 0 and yo > 0) or (xo > 0 and yo >= 0)

    reached, todo = {0}, [0]
    while todo and n:
        s = todo.pop()
        for t in range(n):
            if t not in reached and edge_connected(rects[s], rects[t]):
                reached.add(t)
                todo.append(t)
    return [s for s in range(n) if s not in reached]

# ---------------------------------------------------------------- emit


def emit_xorg_conf(state):
    shown = enabled(state)
    if not shown:
        return "# No displays enabled.\n"
    x0, y0, _, _ = bbox(shown)
    gpus = state["screen_gpu"]
    layout = ['Section "ServerLayout"', '    Identifier     "Layout0"']
    sections = []
    for s, gpu in enumerate(gpus):
        ds = sorted((d for d in shown if d["screen"] == s), key=lambda d: (d["x"], d["y"]))
        sx, sy, ex, ey = bbox(ds)
        shared_gpu = gpus.count(gpu) > 1
        layout.append('    Screen      %d  "Screen%d" %d %d' % (s, s, sx - x0, sy - y0))

        sections += ["", 'Section "Device"',
                     '    Identifier     "Device%d"' % s,
                     '    Driver         "nvidia"']
        if gpu:
            sections.append('    BusID          "%s"' % gpu)
        if shared_gpu:
            sections.append("    Screen          %d" % gpus[:s].count(gpu))
        sections.append("EndSection")

        metamodes = ", ".join(
            "%s: %s +%d+%d%s" % (d["name"], mode_name(d), d["x"] - sx, d["y"] - sy,
                                 "" if d["rotation"] == "normal"
                                 else " {Rotation=%s}" % d["rotation"])
            for d in ds)
        sections += ["", 'Section "Screen"',
                     '    Identifier     "Screen%d"' % s,
                     '    Device         "Device%d"' % s,
                     "    DefaultDepth    24"]
        if shared_gpu:
            sections.append('    Option         "UseDisplayDevice" "%s"'
                            % ", ".join(d["name"] for d in ds))
        sections += ['    Option         "MetaModes" "%s"' % metamodes,
                     '    SubSection     "Display"',
                     "        Depth       24",
                     "        Virtual     %d %d" % (ex - sx, ey - sy),
                     "    EndSubSection",
                     "EndSection"]
    layout.append("EndSection")
    return "\n".join(["# Generated by xorgcist (NVIDIA driver).", ""] + layout + sections) + "\n"


def emit_touch_script(state):
    shown = enabled(state)
    lines = ["#!/bin/sh",
             "# Generated by xorgcist. Run inside the X session, e.g. from session autostart.",
             "# xinput settings are lost when the device is re-plugged or X restarts."]
    rows = [t for t in state["touch"] if device_id(t["device"])]
    if not shown or not rows:
        return "\n".join(lines + ["", "# No touchscreens added."]) + "\n"
    x0, y0, x1, y1 = bbox(shown)
    lines.append("# Desktop spanning all X screens: %dx%d" % (x1 - x0, y1 - y0))
    lines.append("# xinput IDs can change when input devices are re-plugged or added; see: xinput list")
    if screen_count(state) > 1:
        lines.append("# Multiple X screens: the pointer only crosses shared screen edges.")
    for t in rows:
        d = state["displays"][t["display"]]
        name = " ".join(t["device"].split())
        lines.append("")
        if not d["enabled"]:
            lines.append("# %s -> %s skipped: that display is disabled." % (name, d["name"]))
            continue
        w, h = footprint(d)
        lines.append("# %s -> %s (%dx%d%+d%+d%s)" % (
            name, d["name"], w, h, d["x"] - x0, d["y"] - y0,
            "" if d["rotation"] == "normal" else ", rotated " + d["rotation"]))
        lines.append('xinput set-prop %s --type=float "Coordinate Transformation Matrix" %s' % (
            device_id(t["device"]), " ".join(fmt_num(v) for v in touch_matrix(shown, d))))
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------- UI


def display_label(d):
    return d["name"] + ("" if d["enabled"] else " (off)")


def selected(state):
    return state["displays"][state["selected"]]


def fit_view(bb, cw, ch, pad=30):
    x0, y0, x1, y1 = bb
    scale = max(min((cw - 2 * pad) / max(x1 - x0, 1),
                    (ch - 2 * pad) / max(y1 - y0, 1)), 0.001)
    return (scale,
            (cw - (x1 - x0) * scale) / 2 - x0 * scale,
            (ch - (y1 - y0) * scale) / 2 - y0 * scale)


def canvas_order(state):
    """Enabled display indexes in drawing order, the selected one on top."""
    sel = state["selected"]
    order = [i for i, d in enumerate(state["displays"]) if d["enabled"] and i != sel]
    if sel is not None and state["displays"][sel]["enabled"]:
        order.append(sel)
    return order


def display_at(state, ui, cx, cy):
    scale, ox, oy = ui["view"]
    x, y = (cx - ox) / scale, (cy - oy) / scale
    for i in reversed(canvas_order(state)):
        d = state["displays"][i]
        w, h = footprint(d)
        if d["x"] <= x < d["x"] + w and d["y"] <= y < d["y"] + h:
            return i
    return None


def draw_canvas(state, ui):
    c = ui["canvas"]
    c.delete("all")
    cw, ch = c.winfo_width(), c.winfo_height()
    shown = enabled(state)
    if not shown:
        c.create_text(cw / 2, ch / 2, fill="#666",
                      text="No displays enabled." if state["displays"] else "No displays detected.")
        return
    if ui["drag"] is None:
        ui["view"] = fit_view(bbox(shown), cw, ch)
    scale, ox, oy = ui["view"]

    def box(x0, y0, x1, y1):
        return ox + x0 * scale, oy + y0 * scale, ox + x1 * scale, oy + y1 * scale

    font = "TkSmallCaptionFont"
    n = screen_count(state)
    legend_x = 8
    for s in range(n if n > 1 else 0):
        b = box(*bbox([d for d in shown if d["screen"] == s]))
        c.create_rectangle(b[0] - 3, b[1] - 3, b[2] + 3, b[3] + 3, outline="#777", dash=(4, 3))
        c.create_rectangle(legend_x, 8, legend_x + 12, 20, outline="#666",
                           fill=SCREEN_COLORS[s % len(SCREEN_COLORS)])
        label = c.create_text(legend_x + 16, 14, anchor="w", text="X screen %d" % s,
                              font=font, fill="#333")
        legend_x = c.bbox(label)[2] + 14
    for i in canvas_order(state):
        d = state["displays"][i]
        w, h = footprint(d)
        sel = i == state["selected"]
        b = box(d["x"], d["y"], d["x"] + w, d["y"] + h)
        c.create_rectangle(*b, fill=SCREEN_COLORS[d["screen"] % len(SCREEN_COLORS)],
                           outline="#111" if sel else "#666", width=3 if sel else 1)
        label = "%s\n%dx%d @ %.2f Hz\n%+d%+d" % (d["name"], d["size"][0], d["size"][1],
                                                d["rate"], d["x"], d["y"])
        if d["rotation"] != "normal":
            label += "\nrotated " + d["rotation"]
        c.create_text((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, text=label, justify="center",
                      font=font, width=max(b[2] - b[0] - 6, 1))


def set_entry(entry, value):
    entry.delete(0, "end")
    entry.insert(0, str(value))


def set_text(widget, text):
    if widget.get("1.0", "end-1c") == text:
        return
    top = widget.yview()[0]
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.configure(state="disabled")
    widget.yview_moveto(top)


def sync_panel(state, ui, skip_xy):
    ui["display"]["values"] = [display_label(d) for d in state["displays"]]
    fields = [ui[k] for k in ("screen", "res", "rate", "rot", "x", "y")]
    if state["selected"] is None:
        for w in fields + [ui["enabled_cb"]]:
            w.state(["disabled"])
        return
    d = selected(state)
    ui["display"].current(state["selected"])
    ui["enabled"].set(1 if d["enabled"] else 0)
    ui["screen"]["values"] = [str(s) for s in range(screen_count(state))] + ["New"]
    ui["screen"].set(str(d["screen"]) if d["enabled"] else "")
    sizes = list(d["modes"])
    ui["res"]["values"] = ["%dx%d  (%s)" % (w, h, aspect_label(w, h)) for w, h in sizes]
    ui["res"].current(sizes.index(d["size"]))
    rates = d["modes"][d["size"]]
    ui["rate"]["values"] = ["%.2f Hz" % r for r in rates]
    ui["rate"].current(rates.index(d["rate"]))
    ui["rot"].set(d["rotation"])
    if not skip_xy:
        set_entry(ui["x"], d["x"])
        set_entry(ui["y"], d["y"])
    for w in fields:
        w.state(["!disabled"] if d["enabled"] else ["disabled"])


def sync_gpu_rows(state, ui):
    if len(state["gpus"]) < 2:
        return
    frame, widgets = ui["gpu_rows"], ui["gpu_widgets"]
    if len(widgets) != len(state["screen_gpu"]):
        for w in frame.winfo_children():
            w.destroy()
        del widgets[:]
        for s in range(len(state["screen_gpu"])):
            ttk.Label(frame, text="X screen %d" % s).grid(row=s, column=0, sticky="w", padx=(0, 6))
            cb = ttk.Combobox(frame, state="readonly", values=state["gpus"], width=18)
            cb.grid(row=s, column=1, sticky="ew", pady=1)
            cb.bind("<<ComboboxSelected>>", lambda e, s=s: on_gpu(state, ui, s))
            widgets.append(cb)
    for cb, gpu in zip(widgets, state["screen_gpu"]):
        cb.set(gpu)


def build_touch_rows(state, ui):
    frame = ui["touch_rows"]
    for w in frame.winfo_children():
        w.destroy()
    del ui["touch_widgets"][:]
    for i, t in enumerate(state["touch"]):
        dev = ttk.Combobox(frame, values=state["devices"], width=26)
        dev.set(t["device"])
        disp = ttk.Combobox(frame, state="readonly", width=12)
        dev.grid(row=i, column=0, sticky="ew", pady=1)
        ttk.Label(frame, text="->").grid(row=i, column=1, padx=3)
        disp.grid(row=i, column=2, sticky="ew")
        ttk.Button(frame, text="x", width=2,
                   command=lambda i=i: on_touch_remove(state, ui, i)).grid(row=i, column=3, padx=(3, 0))
        dev.bind("<<ComboboxSelected>>", lambda e, i=i: on_touch_device(state, ui, i))
        dev.bind("<KeyRelease>", lambda e, i=i: on_touch_device(state, ui, i))
        disp.bind("<<ComboboxSelected>>", lambda e, i=i: on_touch_display(state, ui, i))
        ui["touch_widgets"].append((dev, disp))


def sync_touch_rows(state, ui):
    labels = [display_label(d) for d in state["displays"]]
    for (dev, disp), t in zip(ui["touch_widgets"], state["touch"]):
        disp["values"] = labels
        disp.current(t["display"])


def refresh(state, ui, skip_xy=False):
    draw_canvas(state, ui)
    sync_panel(state, ui, skip_xy)
    sync_gpu_rows(state, ui)
    sync_touch_rows(state, ui)
    lost = unreachable_screens(state)
    ui["warning"].configure(text="" if not lost else
                            "The mouse and touch input only cross shared edges between X screens, "
                            "so they can't reach X screen %s." % ", ".join(str(s) for s in lost))
    set_text(ui["conf_text"], emit_xorg_conf(state))
    set_text(ui["touch_text"], emit_touch_script(state))

# ---------------------------------------------------------------- handlers


def on_press(state, ui, e):
    ui["canvas"].focus_set()
    i = display_at(state, ui, e.x, e.y)
    if i is None:
        return
    d = state["displays"][i]
    scale, ox, oy = ui["view"]
    state["selected"] = i
    ui["drag"] = (i, (e.x - ox) / scale - d["x"], (e.y - oy) / scale - d["y"])
    refresh(state, ui)


def on_motion(state, ui, e):
    if ui["drag"] is None:
        return
    i, dx, dy = ui["drag"]
    scale, ox, oy = ui["view"]
    d = state["displays"][i]
    d["x"] = int(round((e.x - ox) / scale - dx))
    d["y"] = int(round((e.y - oy) / scale - dy))
    refresh(state, ui)


def on_release(state, ui):
    if ui["drag"] is not None:
        ui["drag"] = None
        refresh(state, ui)


def on_pick_display(state, ui):
    state["selected"] = ui["display"].current()
    refresh(state, ui)


def on_enabled(state, ui):
    d = selected(state)
    d["enabled"] = bool(ui["enabled"].get())
    if d["enabled"]:
        others = {o["screen"] for o in enabled(state) if o is not d}
        d["screen"] = min(d["screen"], len(others))
    compact_screens(state)
    refresh(state, ui)


def on_screen(state, ui):
    value = ui["screen"].get()
    selected(state)["screen"] = screen_count(state) if value == "New" else int(value)
    compact_screens(state)
    refresh(state, ui)


def on_resolution(state, ui):
    d = selected(state)
    d["size"] = list(d["modes"])[ui["res"].current()]
    d["rate"] = min(d["modes"][d["size"]], key=lambda r: abs(r - d["rate"]))
    refresh(state, ui)


def on_rate(state, ui):
    d = selected(state)
    d["rate"] = d["modes"][d["size"]][ui["rate"].current()]
    refresh(state, ui)


def on_rotation(state, ui):
    selected(state)["rotation"] = ui["rot"].get()
    refresh(state, ui)


def on_position(state, ui, key):
    try:
        value = int(ui[key].get())
    except ValueError:
        return
    selected(state)[key] = value
    refresh(state, ui, skip_xy=True)


def on_gpu(state, ui, s):
    state["screen_gpu"][s] = ui["gpu_widgets"][s].get()
    refresh(state, ui)


def on_touch_add(state, ui):
    if not state["displays"]:
        return
    state["touch"].append({"device": state["devices"][0] if state["devices"] else "",
                           "display": state["selected"] or 0})
    build_touch_rows(state, ui)
    refresh(state, ui)


def on_touch_remove(state, ui, i):
    del state["touch"][i]
    build_touch_rows(state, ui)
    refresh(state, ui)


def on_touch_device(state, ui, i):
    state["touch"][i]["device"] = ui["touch_widgets"][i][0].get()
    refresh(state, ui)


def on_touch_display(state, ui, i):
    state["touch"][i]["display"] = ui["touch_widgets"][i][1].current()
    refresh(state, ui)


def on_copy(ui, text):
    ui["root"].clipboard_clear()
    ui["root"].clipboard_append(text)
    ui["status"].configure(text="Copied to clipboard.")


def on_save(ui, text, filename, executable):
    path = filedialog.asksaveasfilename(parent=ui["root"], initialfile=filename)
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        if executable:
            os.chmod(path, 0o755)
    except OSError as e:
        messagebox.showerror("Save failed", str(e), parent=ui["root"])
        return
    ui["status"].configure(text="Saved " + path)

# ---------------------------------------------------------------- layout


def combo(parent, row, label, on_select):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
    cb = ttk.Combobox(parent, state="readonly", width=20)
    cb.grid(row=row, column=1, sticky="ew", pady=2)
    cb.bind("<<ComboboxSelected>>", on_select)
    return cb


def output_tab(book, title, on_copy_click, on_save_click):
    frame = ttk.Frame(book, padding=4)
    book.add(frame, text=title)
    buttons = ttk.Frame(frame)
    buttons.pack(side="bottom", fill="x", pady=(4, 0))
    ttk.Button(buttons, text="Save as...", command=on_save_click).pack(side="right")
    ttk.Button(buttons, text="Copy", command=on_copy_click).pack(side="right", padx=4)
    text = tk.Text(frame, height=14, font="TkFixedFont", wrap="none", state="disabled")
    yscroll = ttk.Scrollbar(frame, command=text.yview)
    xscroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    xscroll.pack(side="bottom", fill="x")
    yscroll.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    return text


def build_ui(root, state, status):
    ui = {"root": root, "view": (1.0, 0.0, 0.0), "drag": None,
          "gpu_widgets": [], "touch_widgets": []}
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="both", expand=True)
    canvas = tk.Canvas(top, width=700, height=360, background="#fafafa",
                       highlightthickness=1, highlightbackground="#c8c8c8")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.bind("<ButtonPress-1>", lambda e: on_press(state, ui, e))
    canvas.bind("<B1-Motion>", lambda e: on_motion(state, ui, e))
    canvas.bind("<ButtonRelease-1>", lambda e: on_release(state, ui))
    canvas.bind("<Configure>", lambda e: draw_canvas(state, ui))
    ui["canvas"] = canvas

    side = ttk.Frame(top, padding=(10, 0, 0, 0))
    side.pack(side="left", fill="y")
    props = ttk.LabelFrame(side, text="Display", padding=6)
    props.pack(fill="x")
    props.columnconfigure(1, weight=1)
    ui["display"] = combo(props, 0, "Output", lambda e: on_pick_display(state, ui))
    ui["enabled"] = tk.IntVar()
    ui["enabled_cb"] = ttk.Checkbutton(props, text="Enabled", variable=ui["enabled"],
                                       command=lambda: on_enabled(state, ui))
    ui["enabled_cb"].grid(row=1, column=1, sticky="w", pady=2)
    ui["screen"] = combo(props, 2, "X screen", lambda e: on_screen(state, ui))
    ui["res"] = combo(props, 3, "Resolution", lambda e: on_resolution(state, ui))
    ui["rate"] = combo(props, 4, "Refresh", lambda e: on_rate(state, ui))
    ui["rot"] = combo(props, 5, "Rotation", lambda e: on_rotation(state, ui))
    ui["rot"]["values"] = ROTATIONS
    ttk.Label(props, text="Position").grid(row=6, column=0, sticky="w")
    pos = ttk.Frame(props)
    pos.grid(row=6, column=1, sticky="w", pady=2)
    for key in ("x", "y"):
        ttk.Label(pos, text=key.upper()).pack(side="left")
        ui[key] = ttk.Entry(pos, width=7)
        ui[key].pack(side="left", padx=(2, 8))
        ui[key].bind("<KeyRelease>", lambda e, k=key: on_position(state, ui, k))
        ui[key].bind("<FocusOut>", lambda e: refresh(state, ui))

    ui["gpu_rows"] = ttk.LabelFrame(side, text="GPU per X screen", padding=6)
    if len(state["gpus"]) > 1:
        ui["gpu_rows"].pack(fill="x", pady=(8, 0))

    touch = ttk.LabelFrame(side, text="Touchscreens", padding=6)
    touch.pack(fill="x", pady=(8, 0))
    ui["touch_rows"] = ttk.Frame(touch)
    ui["touch_rows"].pack(fill="x")
    ttk.Button(touch, text="Add touchscreen",
               command=lambda: on_touch_add(state, ui)).pack(anchor="w", pady=(4, 0))

    ui["warning"] = ttk.Label(root, foreground="#b00020", padding=(8, 0))
    ui["warning"].pack(fill="x")
    book = ttk.Notebook(root, padding=(8, 4, 8, 0))
    book.pack(fill="both", expand=True)
    ui["conf_text"] = output_tab(
        book, "xorg.conf",
        lambda: on_copy(ui, emit_xorg_conf(state)),
        lambda: on_save(ui, emit_xorg_conf(state), "xorg.conf", False))
    ui["touch_text"] = output_tab(
        book, "Touchscreen commands",
        lambda: on_copy(ui, emit_touch_script(state)),
        lambda: on_save(ui, emit_touch_script(state), "touchscreen.sh", True))
    ui["status"] = ttk.Label(root, text=status, padding=(8, 2, 8, 6))
    ui["status"].pack(fill="x")
    return ui


def main():
    parser = argparse.ArgumentParser(
        description="Arrange NVIDIA X11 displays and generate an xorg.conf plus "
                    "touchscreen xinput commands. Nothing on the system is changed.")
    parser.add_argument("-c", "--ctrl-display", metavar="DISPLAY",
                        help="read displays and input devices from this X display, "
                             "like nvidia-settings -c; the window still opens on $DISPLAY")
    args = parser.parse_args()
    state, status = detect(args.ctrl_display)
    try:
        root = tk.Tk()
    except tk.TclError as e:
        sys.exit("xorgcist: cannot open a window: %s" % e)
    root.title("xorgcist")
    ui = build_ui(root, state, status)
    refresh(state, ui)
    root.mainloop()


if __name__ == "__main__":
    main()
