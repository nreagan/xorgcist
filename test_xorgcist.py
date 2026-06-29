"""Tests for xorgcist's pure functions: parse / compute / emit / validate.

Run with:  python3 -m unittest test_xorgcist

UI is intentionally not tested.
"""

import sys
import unittest
from unittest import mock

# Stub out dearpygui so the module imports cleanly without it installed.
sys.modules.setdefault("dearpygui", mock.MagicMock())
sys.modules.setdefault("dearpygui.dearpygui", mock.MagicMock())

from xorgcist import (  # noqa: E402
    GPU,
    IDENTITY_MATRIX,
    InputDevice,
    Output,
    State,
    TouchscreenMapping,
    ValidationError,
    apply_mirroring,
    compute_framebuffer_extent,
    compute_screen_framebuffer,
    derive_touchscreen_matrix,
    effective_dimensions,
    emit_runtime_commands,
    emit_xorg_conf,
    enforce_single_primary,
    find_output_at,
    init_demo_state,
    is_active,
    matrix_multiply,
    normalize_origin,
    parse_lspci_gpus,
    parse_xinput_list,
    parse_xorg_conf,
    parse_xrandr,
    regenerate,
    rotation_matrix,
    runtime_screen_index,
    snap_output_to_neighbors,
    validate_xorg_conf,
)


# ============================== FIXTURES ==============================

SAMPLE_XRANDR = """\
Screen 0: minimum 8 x 8, current 5760 x 1080, maximum 32767 x 32767
DP-1 connected primary 1920x1080+0+0 (0x123) normal (normal left inverted right x axis y axis) 510mm x 287mm
\t1920x1080 (0x123) 148.50MHz +HSync +VSync *current +preferred
\t        h: width  1920 start 2008 end 2052 total 2200 skew    0 clock  67.50KHz
\t        v: height 1080 start 1083 end 1088 total 1125           clock  60.00Hz
\t1280x720 (0x140) 74.25MHz +HSync +VSync
\t        h: width  1280 start 1390 end 1430 total 1650 skew    0 clock  45.00KHz
\t        v: height  720 start  725 end  730 total  750           clock  60.00Hz
DP-2 connected 1920x1080+1920+0 (0x124) normal (normal left inverted right x axis y axis) 510mm x 287mm
\t1920x1080 (0x124) 148.50MHz +HSync +VSync *current +preferred
\t        h: width  1920 start 2008 end 2052 total 2200 skew    0 clock  67.50KHz
\t        v: height 1080 start 1083 end 1088 total 1125           clock  60.00Hz
DP-3 connected 1920x1080+3840+0 (0x125) normal 510mm x 287mm
\t1920x1080 (0x125) 148.50MHz +HSync +VSync *current
\t        h: width  1920 start 2008 end 2052 total 2200 skew    0 clock  67.50KHz
\t        v: height 1080 start 1083 end 1088 total 1125           clock  60.00Hz
HDMI-1 disconnected (normal left inverted right x axis y axis)
"""

SAMPLE_LSPCI = """\
0000:00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 770 (rev 04)
0000:01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)
0000:02:00.0 Audio device: Realtek Semiconductor Co., Ltd. ALC1220
"""

SAMPLE_XINPUT = """\
⎡ Virtual core pointer                          \tid=2\t[master pointer  (3)]
⎜   ↳ DELL Mouse                           \tid=10\t[slave  pointer  (2)]
⎜   ↳ Wacom Bamboo Pen                     \tid=12\t[slave  pointer  (2)]
⎜   ↳ ELO TouchSystems Touchscreen 2700    \tid=15\t[slave  pointer  (2)]
⎣ Virtual core keyboard                         \tid=3\t[master keyboard (2)]
    ↳ AT Translated Set 2 keyboard             \tid=11\t[slave  keyboard (3)]
"""

SAMPLE_XORG_CONF = """\
Section "Device"
    Identifier  "Device0"
    Driver      "nvidia"
    BusID       "PCI:1:0:0"
EndSection

Section "Screen"
    Identifier  "Screen0"
    Device      "Device0"
EndSection

Section "ServerLayout"
    Identifier  "Layout0"
    Screen      0 "Screen0" 0 0
EndSection
"""


# ============================== PARSE ==============================

class TestParseXrandr(unittest.TestCase):
    def test_parses_three_connected_outputs(self):
        outs = parse_xrandr(SAMPLE_XRANDR)
        self.assertEqual(len([o for o in outs if o.connected]), 3)

    def test_first_output_is_primary(self):
        dp1 = next(o for o in parse_xrandr(SAMPLE_XRANDR) if o.name == "DP-1")
        self.assertTrue(dp1.primary)
        self.assertEqual((dp1.width, dp1.height, dp1.x, dp1.y), (1920, 1080, 0, 0))

    def test_positions_parsed(self):
        by_name = {o.name: o for o in parse_xrandr(SAMPLE_XRANDR)}
        self.assertEqual(by_name["DP-1"].x, 0)
        self.assertEqual(by_name["DP-2"].x, 1920)
        self.assertEqual(by_name["DP-3"].x, 3840)

    def test_disconnected_outputs_kept(self):
        hdmi = next(o for o in parse_xrandr(SAMPLE_XRANDR) if o.name == "HDMI-1")
        self.assertFalse(hdmi.connected)

    def test_modes_with_refresh(self):
        dp1 = next(o for o in parse_xrandr(SAMPLE_XRANDR) if o.name == "DP-1")
        self.assertIn((1920, 1080, 60.0), dp1.available_modes)
        self.assertIn((1280, 720, 60.0), dp1.available_modes)

    def test_h_lines_not_parsed_as_modes(self):
        dp1 = next(o for o in parse_xrandr(SAMPLE_XRANDR) if o.name == "DP-1")
        for w, h, _ in dp1.available_modes:
            self.assertNotEqual((w, h), (1920, 2008))

    def test_rotation_extracted_from_header(self):
        sample = (
            "DP-1 connected 1920x1080+0+0 left (0x4d) "
            "(normal left inverted right x axis y axis)\n"
            "\t1920x1080 (0x4d) *current\n"
            "\t        v: height 1080 start ... clock 60.00Hz\n"
        )
        outs = parse_xrandr(sample)
        self.assertEqual(outs[0].rotation, "left")

    def test_normal_rotation_default(self):
        outs = parse_xrandr(SAMPLE_XRANDR)
        self.assertEqual(outs[0].rotation, "normal")

    def test_refresh_extracted_from_current_mode(self):
        sample = (
            "DP-1 connected 1920x1080+0+0\n"
            "\t1920x1080 (0x4d) 148.50MHz *current\n"
            "\t        v: height 1080 start ... clock 144.00Hz\n"
        )
        outs = parse_xrandr(sample)
        self.assertAlmostEqual(outs[0].refresh_rate, 144.0)

    def test_hz_with_optional_space(self):
        # Some xrandr builds emit "60.00 Hz" with a space.
        sample = (
            "DP-1 connected 1920x1080+0+0\n"
            "\t1920x1080 *current\n"
            "\t        v: height ... clock 60.00 Hz\n"
        )
        outs = parse_xrandr(sample)
        self.assertAlmostEqual(outs[0].refresh_rate, 60.0)

    def test_connected_no_resolution_block(self):
        outs = parse_xrandr("DP-1 connected (normal left inverted right x axis y axis)\n")
        self.assertEqual(len(outs), 1)
        self.assertTrue(outs[0].connected)
        self.assertEqual(outs[0].width, 0)


class TestParseLspci(unittest.TestCase):
    def test_finds_gpus(self):
        gpus = parse_lspci_gpus(SAMPLE_LSPCI)
        self.assertEqual(len(gpus), 2)

    def test_busid_format(self):
        nvidia = next(g for g in parse_lspci_gpus(SAMPLE_LSPCI) if g.vendor == "NVIDIA")
        self.assertEqual(nvidia.busid, "PCI:1:0:0")
        self.assertEqual(nvidia.driver, "nvidia")

    def test_nvidia_sorts_first_over_intel(self):
        gpus = parse_lspci_gpus(SAMPLE_LSPCI)
        self.assertEqual(gpus[0].vendor, "NVIDIA")
        self.assertEqual(gpus[1].vendor, "Intel")

    def test_amd_sorts_before_intel(self):
        sample = (
            "0000:00:02.0 VGA compatible controller: Intel Corporation UHD\n"
            "0000:03:00.0 VGA compatible controller: Advanced Micro Devices Radeon\n"
        )
        gpus = parse_lspci_gpus(sample)
        self.assertEqual(gpus[0].vendor, "AMD")


class TestParseXinput(unittest.TestCase):
    def test_finds_slave_devices(self):
        names = [d.name for d in parse_xinput_list(SAMPLE_XINPUT)]
        self.assertIn("DELL Mouse", names)
        self.assertIn("ELO TouchSystems Touchscreen 2700", names)

    def test_skips_master_devices(self):
        names = [d.name for d in parse_xinput_list(SAMPLE_XINPUT)]
        self.assertNotIn("Virtual core pointer", names)

    def test_classifies_touchscreen(self):
        elo = next(d for d in parse_xinput_list(SAMPLE_XINPUT) if "ELO" in d.name)
        self.assertEqual(elo.type, "touchscreen")

    def test_classifies_wacom_pen_as_tablet(self):
        wacom = next(d for d in parse_xinput_list(SAMPLE_XINPUT) if "Wacom" in d.name)
        self.assertEqual(wacom.type, "tablet")

    def test_classifies_touchpad_not_touchscreen(self):
        sample = "⎜   ↳ SynPS/2 Synaptics TouchPad   id=14   [slave  pointer  (2)]\n"
        devs = parse_xinput_list(sample)
        self.assertEqual(devs[0].type, "touchpad")

    def test_classifies_apple_trackpad_as_touchpad(self):
        sample = "⎜   ↳ Apple Inc. Magic Trackpad 2   id=20   [slave  pointer  (2)]\n"
        devs = parse_xinput_list(sample)
        self.assertEqual(devs[0].type, "touchpad")


class TestParseXorgConf(unittest.TestCase):
    def test_parses_all_sections(self):
        kinds = [s.kind for s in parse_xorg_conf(SAMPLE_XORG_CONF).sections]
        self.assertEqual(sorted(kinds), ["Device", "Screen", "ServerLayout"])

    def test_parses_busid_and_driver(self):
        device = next(s for s in parse_xorg_conf(SAMPLE_XORG_CONF).sections
                      if s.kind == "Device")
        self.assertEqual(device.options.get("BusID"), "PCI:1:0:0")
        self.assertEqual(device.options.get("Driver"), "nvidia")

    def test_parses_empty_string_option_value(self):
        # `Option "X" ""` is legal xorg syntax (clear an inherited option).
        # Earlier opt_re used `(.+?)` which required ≥1 char and silently
        # dropped these lines on round-trip.
        text = (
            'Section "Monitor"\n'
            '    Identifier "M0"\n'
            '    Option "EmptyVal" ""\n'
            '    Option "Real" "yes"\n'
            'EndSection\n'
        )
        sec = parse_xorg_conf(text).sections[0]
        self.assertEqual(sec.options.get("EmptyVal"), "")
        self.assertEqual(sec.options.get("Real"), "yes")


# ============================== COMPUTE ==============================

class TestEffectiveDimensions(unittest.TestCase):
    def test_normal_unchanged(self):
        o = Output("DP-1", connected=True, width=1920, height=1080)
        self.assertEqual(effective_dimensions(o), (1920, 1080))

    def test_left_swaps(self):
        o = Output("DP-1", connected=True, width=1920, height=1080, rotation="left")
        self.assertEqual(effective_dimensions(o), (1080, 1920))

    def test_right_swaps(self):
        o = Output("DP-1", connected=True, width=1920, height=1080, rotation="right")
        self.assertEqual(effective_dimensions(o), (1080, 1920))

    def test_inverted_unchanged(self):
        o = Output("DP-1", connected=True, width=1920, height=1080, rotation="inverted")
        self.assertEqual(effective_dimensions(o), (1920, 1080))


class TestRotationMatrix(unittest.TestCase):
    def test_normal_is_identity(self):
        self.assertEqual(rotation_matrix("normal"), IDENTITY_MATRIX)

    def test_left_corner_full_set(self):
        # Standard X.Org: left maps (x,y) → (1-y, x)
        m = rotation_matrix("left")
        def apply(rx, ry):
            return (m[0]*rx + m[1]*ry + m[2], m[3]*rx + m[4]*ry + m[5])
        self.assertEqual(tuple(round(v, 3) for v in apply(0, 0)), (1.0, 0.0))
        self.assertEqual(tuple(round(v, 3) for v in apply(1, 0)), (1.0, 1.0))
        self.assertEqual(tuple(round(v, 3) for v in apply(1, 1)), (0.0, 1.0))
        self.assertEqual(tuple(round(v, 3) for v in apply(0, 1)), (0.0, 0.0))

    def test_right_corner_mapping(self):
        m = rotation_matrix("right")
        def apply(rx, ry):
            return (m[0]*rx + m[1]*ry + m[2], m[3]*rx + m[4]*ry + m[5])
        self.assertEqual(tuple(round(v, 3) for v in apply(0, 0)), (0.0, 1.0))


class TestMatrixMultiply(unittest.TestCase):
    def test_identity_left(self):
        a = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        self.assertEqual(matrix_multiply(IDENTITY_MATRIX, a), a)

    def test_identity_right(self):
        a = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        self.assertEqual(matrix_multiply(a, IDENTITY_MATRIX), a)


class TestIsActive(unittest.TestCase):
    def test_disabled_inactive(self):
        self.assertFalse(is_active(
            Output("DP-1", connected=True, width=1920, height=1080, enabled=False)))

    def test_disconnected_inactive(self):
        self.assertFalse(is_active(Output("DP-1", connected=False)))

    def test_no_mode_inactive(self):
        self.assertFalse(is_active(Output("DP-1", connected=True)))

    def test_normal_active(self):
        self.assertTrue(is_active(Output("DP-1", connected=True, width=1920, height=1080)))


class TestFramebufferExtent(unittest.TestCase):
    def test_horizontal(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, y=0),
        ]
        self.assertEqual(compute_framebuffer_extent(outs), (3840, 1080))

    def test_with_disabled(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0),
            Output("HDMI-1", connected=False),
        ]
        self.assertEqual(compute_framebuffer_extent(outs), (1920, 1080))

    def test_with_rotated(self):
        outs = [
            Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=2560, y=0,
                   rotation="left"),
        ]
        w, h = compute_framebuffer_extent(outs)
        self.assertEqual(w, 2560 + 1080)
        self.assertEqual(h, 1920)

    def test_empty(self):
        self.assertEqual(compute_framebuffer_extent([]), (0, 0))


class TestScreenFramebuffer(unittest.TestCase):
    def test_one_screen_two_outputs(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, y=0, screen_id=0),
        ]
        self.assertEqual(compute_screen_framebuffer(outs, 0), (0, 0, 3840, 1080))

    def test_isolates_one_screen(self):
        outs = [
            Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=4480, y=360, screen_id=1),
        ]
        self.assertEqual(compute_screen_framebuffer(outs, 1), (4480, 360, 1920, 1080))


class TestFindOutputAt(unittest.TestCase):
    def test_basic(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, y=0),
        ]
        self.assertEqual(find_output_at(outs, 100, 100).name, "DP-1")
        self.assertEqual(find_output_at(outs, 2500, 100).name, "DP-2")
        self.assertIsNone(find_output_at(outs, 5000, 100))

    def test_skips_disabled(self):
        outs = [Output("DP-1", connected=True, width=1920, height=1080,
                       enabled=False)]
        self.assertIsNone(find_output_at(outs, 100, 100))

    def test_uses_effective_dimensions(self):
        outs = [Output("DP-1", connected=True, width=1920, height=1080,
                       rotation="left")]
        self.assertIsNone(find_output_at(outs, 1500, 100))
        self.assertEqual(find_output_at(outs, 500, 100).name, "DP-1")


class TestRuntimeScreenIndex(unittest.TestCase):
    def test_remaps_noncontiguous(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, screen_id=5),
        ]
        self.assertEqual(runtime_screen_index(outs, 0), 0)
        self.assertEqual(runtime_screen_index(outs, 5), 1)

    def test_unknown_returns_label(self):
        outs = [Output("DP-1", connected=True, width=1920, height=1080, screen_id=0)]
        self.assertEqual(runtime_screen_index(outs, 99), 99)


class TestSnapAndNormalize(unittest.TestCase):
    def test_snap_right_edge(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1910, y=0),
        ]
        snap_output_to_neighbors(outs, outs[1])
        self.assertEqual(outs[1].x, 1920)

    def test_snap_top_edges(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, y=15),
        ]
        snap_output_to_neighbors(outs, outs[1])
        self.assertEqual(outs[1].y, 0)

    def test_snap_picks_closest_when_two_neighbors_in_range(self):
        # When two neighbor edges are both within snap threshold, the
        # earlier emitter picked the FIRST hit in iteration order. With two
        # equally-valid snap targets, that meant the snap could land on the
        # farther edge depending on how `others` happened to be ordered.
        # Now snap to the closest edge regardless of order.
        # A's left edge at x=120 (dist from dragged.x=128 → 8).
        # B's left edge at x=125 (dist from dragged.x=128 → 3, closer).
        a = Output("A", connected=True, width=100, height=100, x=120, y=0)
        b = Output("B", connected=True, width=100, height=100, x=125, y=0)
        d = Output("D", connected=True, width=100, height=100, x=128, y=0)
        snap_output_to_neighbors([a, b, d], d, threshold=10)
        self.assertEqual(d.x, 125)  # B (closer), not A (first in list)

    def test_normalize_origin(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=-100, y=-50),
            Output("DP-2", connected=True, width=1920, height=1080, x=1820, y=-50),
        ]
        normalize_origin(outs)
        self.assertEqual((outs[0].x, outs[0].y), (0, 0))
        self.assertEqual(outs[1].x, 1920)


class TestApplyMirroring(unittest.TestCase):
    def test_basic_mirror_copies_position(self):
        outs = [
            Output("DP-1", connected=True, width=2560, height=1440, x=100, y=50,
                   screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=9999, y=9999,
                   screen_id=1, mirror_of="DP-1"),
        ]
        apply_mirroring(outs)
        dp2 = outs[1]
        self.assertEqual((dp2.x, dp2.y), (100, 50))
        # Mirror copies position + screen_id but NOT dimensions or rotation
        self.assertEqual((dp2.width, dp2.height), (1920, 1080))
        self.assertEqual(dp2.screen_id, 0)

    def test_chain_resolves_to_root(self):
        outs = [
            Output("A", connected=True, width=1920, height=1080, x=999, y=999, mirror_of="B"),
            Output("B", connected=True, width=1920, height=1080, x=888, y=888, mirror_of="C"),
            Output("C", connected=True, width=1920, height=1080, x=100, y=200),
        ]
        apply_mirroring(outs)
        a = next(o for o in outs if o.name == "A")
        self.assertEqual((a.x, a.y), (100, 200))

    def test_two_node_cycle_safe(self):
        outs = [
            Output("A", connected=True, width=1920, height=1080, x=10, y=20, mirror_of="B"),
            Output("B", connected=True, width=1920, height=1080, x=30, y=40, mirror_of="A"),
        ]
        apply_mirroring(outs)
        self.assertEqual((outs[0].x, outs[0].y), (10, 20))
        self.assertEqual((outs[1].x, outs[1].y), (30, 40))

    def test_self_mirror_ignored(self):
        outs = [
            Output("A", connected=True, width=1920, height=1080, x=42, y=43, mirror_of="A"),
        ]
        apply_mirroring(outs)
        self.assertEqual((outs[0].x, outs[0].y), (42, 43))

    def test_missing_target_safe(self):
        outs = [
            Output("A", connected=True, width=1920, height=1080, x=42, y=43, mirror_of="GHOST"),
        ]
        apply_mirroring(outs)
        self.assertEqual((outs[0].x, outs[0].y), (42, 43))


class TestEnforceSinglePrimary(unittest.TestCase):
    def test_active_beats_disabled(self):
        outs = [
            Output("disabled-pri", connected=True, width=1920, height=1080,
                   primary=True, enabled=False),
            Output("active-pri", connected=True, width=1920, height=1080,
                   primary=True, enabled=True),
        ]
        enforce_single_primary(outs)
        self.assertTrue(next(o for o in outs if o.name == "active-pri").primary)
        self.assertFalse(next(o for o in outs if o.name == "disabled-pri").primary)

    def test_only_first_kept_when_all_active(self):
        outs = [
            Output("DP-1", connected=True, width=1920, height=1080, primary=True),
            Output("DP-2", connected=True, width=1920, height=1080, primary=True),
        ]
        enforce_single_primary(outs)
        primaries = [o for o in outs if o.primary]
        self.assertEqual(len(primaries), 1)
        self.assertEqual(primaries[0].name, "DP-1")


class TestTouchscreenMatrix(unittest.TestCase):
    def test_full_screen_no_rotation_is_identity(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, screen_id=0)]
        m = derive_touchscreen_matrix(s.outputs, "DP-1")
        self.assertEqual(m, IDENTITY_MATRIX)

    def test_right_half_screen(self):
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, y=0, screen_id=0),
        ]
        m = derive_touchscreen_matrix(s.outputs, "DP-2")
        self.assertAlmostEqual(m[0], 0.5)
        self.assertAlmostEqual(m[2], 0.5)

    def test_per_screen_framebuffer_isolation(self):
        # HDMI-1 alone on screen 1 — matrix should be identity (full screen),
        # NOT scaled by the global multi-screen extent.
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=4480, y=360, screen_id=1),
        ]
        m = derive_touchscreen_matrix(s.outputs, "HDMI-1")
        self.assertAlmostEqual(m[0], 1.0)
        self.assertAlmostEqual(m[2], 0.0)

    def test_rotation_only_for_alone_rotated_output(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080,
                            screen_id=0, rotation="left")]
        m = derive_touchscreen_matrix(s.outputs, "DP-1")
        self.assertEqual(m, rotation_matrix("left"))


# ============================== EMIT ==============================

class TestEmitXorgConf(unittest.TestCase):
    def _basic_state(self):
        s = State()
        s.gpus = [GPU(busid="PCI:1:0:0", vendor="NVIDIA", driver="nvidia", name="RTX")]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, y=0, screen_id=0),
        ]
        return s

    def test_full_emit(self):
        text = emit_xorg_conf(self._basic_state())
        self.assertIn('Section "Device"', text)
        self.assertIn('Section "Screen"', text)
        self.assertIn('Section "ServerLayout"', text)
        self.assertIn("DP-1: 1920x1080+0+0", text)
        self.assertIn("DP-2: 1920x1080+1920+0", text)
        self.assertNotIn("1920x1080_60", text)

    @staticmethod
    def _nvidia_gpu():
        # Helper: many tests below specifically exercise the NVIDIA emit path
        # (multi-X-screen, MetaModes, Monitor sections). Without an explicit
        # NVIDIA GPU the State defaults to driver="modesetting" and we hit the
        # generic minimal path instead, which doesn't have those features.
        return GPU(busid="PCI:1:0:0", vendor="NVIDIA", driver="nvidia", name="RTX")

    def test_per_screen_metamodes_normalized(self):
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("HDMI-1", connected=True, width=1920, height=1080, x=3840, y=0, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn("HDMI-1: 1920x1080+0+0", text)

    def test_separate_screens(self):
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn('Identifier  "Screen0"', text)
        self.assertIn('Identifier  "Screen1"', text)

    def test_remaps_noncontiguous_screen_ids(self):
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, screen_id=5),
        ]
        text = emit_xorg_conf(s)
        self.assertNotIn('Identifier  "Screen5"', text)
        self.assertIn('Identifier  "Screen0"', text)
        self.assertIn('Identifier  "Screen1"', text)

    def test_serverlayout_uses_relative_below(self):
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=0, y=2000, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn('Below "Screen0"', text)

    def test_rotation_in_metamodes(self):
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, rotation="left")]
        text = emit_xorg_conf(s)
        self.assertIn("{Rotation=left}", text)

    def test_nvidia_binds_display_devices_per_screen(self):
        # Multi-X-screen NVIDIA configs need each Screen to claim only its own
        # physical display device(s); otherwise the first Screen can consume
        # heads that later Screens need in order to initialize.
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-0.8", connected=True, width=2560, height=1440, screen_id=0),
            Output("DP-2.8", connected=True, width=2560, height=1440, x=2560, screen_id=0),
            Output("HDMI-0", connected=True, width=1024, height=768, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn('Option      "UseDisplayDevice" "DP-0.8, DP-2.8"', text)
        self.assertIn('Option      "UseDisplayDevice" "HDMI-0"', text)

    def test_no_primary_in_monitor(self):
        # Primary is silently ignored by all drivers in Monitor section;
        # primary takes effect via runtime xrandr --primary instead.
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, primary=True)]
        text = emit_xorg_conf(s)
        self.assertNotIn('"Primary"', text)

    def test_color_depth_uses_max(self):
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, color_depth=24),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, color_depth=30),
        ]
        text = emit_xorg_conf(s)
        self.assertIn("Depth       30", text)

    def test_disabled_outputs_excluded(self):
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, enabled=False),
        ]
        text = emit_xorg_conf(s)
        self.assertNotIn("DP-2:", text)

    def test_nvidia_emits_monitor_directive_not_monitor_x_option(self):
        # NVIDIA driver README (verified against versions 460/470/535) shows
        # `Monitor "MonitorN"` directive in Screen sections, NOT an
        # `Option "monitor-<port>"` line. Earlier code emitted the latter
        # (which is undocumented and silently ignored).
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [Output("DP-0.8", connected=True, width=1920, height=1080)]
        text = emit_xorg_conf(s)
        self.assertIn('Monitor     "Monitor0"', text)
        self.assertNotIn('Option      "monitor-', text)
        self.assertNotIn('Option      "Monitor-', text)

    def test_generic_driver_emits_minimal_config(self):
        # AMD/Intel/modesetting xorg.conf is a stub: minimal Device + Screen +
        # ServerLayout. No MetaModes (NVIDIA-specific). No Monitor sections
        # (driver autodetects via EDID). No `Screen N` (NVIDIA multi-X-screen
        # routing). The runtime xrandr script handles all layout.
        s = State()
        s.gpus = [GPU(busid="PCI:3:0:0", vendor="AMD", driver="amdgpu", name="RX")]
        s.outputs = [
            Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=2560, y=0, screen_id=0),
        ]
        text = emit_xorg_conf(s)
        self.assertIn('Driver      "amdgpu"', text)
        self.assertIn('BusID       "PCI:3:0:0"', text)
        self.assertNotIn('MetaModes', text)
        self.assertNotIn('Section "Monitor"', text)
        self.assertNotIn('Screen      0\n', text)  # the NVIDIA-specific Device-Screen N line

    def test_generic_driver_collapses_multi_screen_user_grouping(self):
        # Even if the user grouped outputs into multiple screen_ids, non-NVIDIA
        # collapses to a single X screen — multi-X-screen on AMD/Intel needs
        # ZaphodHeads which we don't emit. User's runtime xrandr script handles
        # positioning all outputs in the single screen's framebuffer.
        s = State()
        s.gpus = [GPU(busid="PCI:3:0:0", vendor="AMD", driver="amdgpu", name="RX")]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn('Identifier  "Screen0"', text)
        self.assertNotIn('Identifier  "Screen1"', text)


class TestEmitRuntimeCommands(unittest.TestCase):
    def test_basic_xrandr(self):
        s = State()
        s.outputs = [
            Output("DP-A", connected=True, width=1920, height=1080,
                   available_modes=[(1920, 1080, 60.0)]),
            Output("DP-1", connected=True, width=1920, height=1080, x=1920,
                   refresh_rate=144.0,
                   available_modes=[(1920, 1080, 144.0)]),
        ]
        cmds = emit_runtime_commands(s)
        dp1 = next(c for c in cmds if "--output DP-1" in c)
        self.assertIn("--mode 1920x1080", dp1)
        self.assertIn("--pos 1920x0", dp1)
        self.assertIn("--rate 144.00", dp1)

    def test_rate_omitted_when_no_matching_modeline(self):
        # If user picked a refresh that isn't in available_modes, omit --rate
        # so xrandr picks the driver default rather than erroring on a
        # nonexistent refresh.
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080,
                            refresh_rate=240.0,
                            available_modes=[(1920, 1080, 60.0)])]
        cmds = emit_runtime_commands(s)
        self.assertNotIn("--rate", cmds[0])

    @staticmethod
    def _nvidia_gpu():
        # Multi-X-screen runtime indirection is NVIDIA-only (matches xorg.conf
        # emit which only emits separate Screens for NVIDIA). State() defaults
        # to driver="modesetting", which collapses to a single screen.
        return GPU(busid="PCI:1:0:0", vendor="NVIDIA", driver="nvidia", name="RTX")

    def test_multi_screen_uses_display_prefix_with_runtime_derivation(self):
        # Multi-X-screen scripts derive the X server number from $DISPLAY at
        # script-runtime via shell parameter expansion (handles RHEL8 :1,
        # second-user logins :2, multi-seat, etc.) so the same script works
        # regardless of login order. The setup line is the first command.
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=4480, screen_id=1),
        ]
        cmds = emit_runtime_commands(s)
        self.assertTrue(cmds[0].startswith('_X='))
        self.assertIn('${DISPLAY:-:0}', cmds[0])
        self.assertIn('${_X%.*}', cmds[0])
        dp1 = next(c for c in cmds if "--output DP-1" in c)
        hdmi = next(c for c in cmds if "--output HDMI-1" in c)
        self.assertTrue(dp1.startswith('DISPLAY="$_X.0" '))
        self.assertTrue(hdmi.startswith('DISPLAY="$_X.1" '))

    def test_generic_driver_collapses_runtime_to_single_screen(self):
        # On non-NVIDIA, even if the user grouped outputs into multiple
        # screen_ids, the runtime emit collapses to a single X screen — no
        # _X= setup line, no DISPLAY prefix. xrandr against the user's
        # current $DISPLAY handles all positioning in one framebuffer.
        s = State()
        s.gpus = [GPU(busid="PCI:3:0:0", vendor="AMD", driver="amdgpu", name="RX")]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=1920, screen_id=1),
        ]
        cmds = emit_runtime_commands(s)
        self.assertFalse(any(c.startswith('_X=') for c in cmds))
        for c in cmds:
            self.assertFalse(c.startswith('DISPLAY="$_X'))

    def test_single_screen_omits_setup_line_and_display_prefix(self):
        # Single-screen scripts have no DISPLAY indirection — the user's
        # session $DISPLAY already points to the right place.
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, screen_id=0)]
        cmds = emit_runtime_commands(s)
        self.assertFalse(any(c.startswith('_X=') for c in cmds))
        for c in cmds:
            self.assertFalse(c.startswith("DISPLAY="))

    def test_disabled_emits_off(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, enabled=False)]
        cmds = emit_runtime_commands(s)
        self.assertTrue(any("--output DP-1 --off" in c for c in cmds))

    def test_disabled_on_phantom_screen_routes_to_runtime_zero(self):
        # Bug: disabled output on its OWN screen_id used to address an
        # X screen the conf never created. Now routed to runtime screen 0.
        # Tested under NVIDIA so the multi-X-screen DISPLAY indirection path
        # is actually exercised (under generic driver everything collapses).
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080,
                   screen_id=1, enabled=False),
        ]
        cmds = emit_runtime_commands(s)
        for c in cmds:
            self.assertNotIn('DISPLAY="$_X.1"', c)
        self.assertTrue(any("--output DP-2 --off" in c for c in cmds))

    def test_rotation_in_xrandr(self):
        s = State()
        s.outputs = [Output("HDMI-1", connected=True, width=1920, height=1080,
                            rotation="left")]
        cmds = emit_runtime_commands(s)
        self.assertIn("--rotate left", cmds[0])

    def test_primary_flag(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, primary=True)]
        cmds = emit_runtime_commands(s)
        self.assertIn("--primary", cmds[0])

    def test_mirror_uses_same_as_emitted_after_target(self):
        s = State()
        s.outputs = [
            Output("A-mirror", connected=True, width=1920, height=1080,
                   mirror_of="Z-target", screen_id=0),
            Output("Z-target", connected=True, width=1920, height=1080, screen_id=0),
        ]
        cmds = emit_runtime_commands(s)
        target_idx = next(i for i, c in enumerate(cmds) if "--output Z-target" in c)
        mirror_idx = next(i for i, c in enumerate(cmds) if "--output A-mirror" in c)
        self.assertLess(target_idx, mirror_idx)
        mirror = cmds[mirror_idx]
        self.assertIn("--same-as Z-target", mirror)

    def test_self_mirror_falls_back_to_pos(self):
        s = State()
        s.outputs = [Output("A", connected=True, width=1920, height=1080,
                            x=0, y=0, mirror_of="A")]
        cmds = emit_runtime_commands(s)
        a = next(c for c in cmds if "--output A" in c)
        self.assertNotIn("--same-as A", a)
        self.assertIn("--pos 0x0", a)

    def test_dangling_mirror_falls_back_to_pos(self):
        # If mirror_of points to an output that doesn't exist (or is inactive),
        # `xrandr --same-as <missing>` errors and breaks the rest of the script
        # under `set -e`. Fall through to absolute positioning.
        s = State()
        s.outputs = [
            Output("ANCHOR", connected=True, width=1920, height=1080,
                   x=0, y=0, screen_id=0),
            Output("DP-1", connected=True, width=1920, height=1080,
                   x=1920, y=200, screen_id=0, mirror_of="GHOST"),
        ]
        cmds = emit_runtime_commands(s)
        line = next(c for c in cmds if "--output DP-1" in c)
        self.assertNotIn("--same-as GHOST", line)
        self.assertIn("--pos 1920x200", line)

    def test_mirror_to_disabled_target_falls_back_to_pos(self):
        # A real (existing) target output that's been disabled is also
        # not a valid --same-as target.
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080,
                   x=100, y=200, mirror_of="DP-2"),
            Output("DP-2", connected=True, width=1920, height=1080,
                   x=0, y=0, enabled=False),
        ]
        cmds = emit_runtime_commands(s)
        line = next(c for c in cmds if "--output DP-1 " in c)
        self.assertNotIn("--same-as DP-2", line)
        self.assertIn("--pos", line)

    def test_scale_in_xrandr(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=2560, height=1440, scale=1.5)]
        cmds = emit_runtime_commands(s)
        self.assertIn("--scale 1.500x1.500", cmds[0])

    def test_disabled_touchscreen_emits_xinput_disable(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080)]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="DP-1", enabled=False)]
        cmds = emit_runtime_commands(s)
        self.assertTrue(any(c == 'xinput disable ELO' for c in cmds))
        self.assertFalse(any("set-prop" in c for c in cmds))

    def test_enabled_touchscreen_emits_enable_then_setprop(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080)]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="DP-1")]
        cmds = emit_runtime_commands(s)
        xinput = [c for c in cmds if c.startswith("xinput")]
        self.assertEqual(xinput[0], 'xinput enable ELO')
        self.assertTrue(xinput[1].startswith('xinput set-prop ELO'))

    def test_runtime_shell_quotes_device_and_output_names(self):
        s = State()
        s.outputs = [
            Output("DP weird", connected=True, width=1920, height=1080,
                   available_modes=[(1920, 1080, 60.0)]),
            Output("Mirror Out", connected=True, width=1920, height=1080,
                   mirror_of="DP weird"),
        ]
        s.touchscreens = [
            TouchscreenMapping(device_id=15, device_name='Elo "quoted" touch',
                               target_output="DP weird")
        ]
        cmds = emit_runtime_commands(s)
        self.assertTrue(any("xrandr --output 'DP weird'" in c for c in cmds))
        self.assertTrue(any("--same-as 'DP weird'" in c for c in cmds))
        self.assertTrue(any(c == 'xinput enable \'Elo "quoted" touch\'' for c in cmds))
        self.assertTrue(any("'Coordinate Transformation Matrix'" in c for c in cmds))

    def test_xinput_lines_have_no_display_prefix(self):
        # xinput enable / set-prop / disable are server-wide and inherit
        # $DISPLAY from the user's session. A DISPLAY prefix would just be a
        # second source of truth for which X server to talk to.
        # Set up under NVIDIA + multi-X-screen so the DISPLAY-prefix path is
        # actually being exercised on xrandr lines, making the contrast
        # between xrandr (prefixed) and xinput (bare) meaningful.
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=1920, screen_id=1),
        ]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="HDMI-1")]
        cmds = emit_runtime_commands(s)
        for c in cmds:
            if "xinput" in c:
                self.assertFalse(c.startswith("DISPLAY="))

    def test_no_mpx_master_juggling_emitted(self):
        # Earlier the emitter spat out `xinput create-master` + `reattach` for
        # touchscreens on non-default X screens. That's MPX (Multi-Pointer X),
        # which is a separate feature for multi-user touch tables, not what
        # touchscreen calibration needs. Real-world setups (incl. NASA-style
        # multi-display NVIDIA configs) calibrate with just enable + set-prop.
        s = State()
        s.gpus = [self._nvidia_gpu()]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=1920, screen_id=1),
        ]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="HDMI-1")]
        cmds = emit_runtime_commands(s)
        self.assertFalse(any("create-master" in c for c in cmds))
        self.assertFalse(any("reattach" in c for c in cmds))

    def test_runtime_script_independent_of_state_env_display(self):
        # state.env_display is captured for diagnostic display in the UI but
        # MUST NOT be baked into the emitted script — login order varies, and
        # the script needs to work whether the user is :0, :1, or :47 today.
        # Tested under NVIDIA + multi-X-screen because that's the path where
        # DISPLAY indirection actually matters; on generic driver the script
        # has no DISPLAY references to test.
        def make_state(env):
            s = State()
            s.gpus = [self._nvidia_gpu()]
            s.env_display = env
            s.outputs = [
                Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
                Output("HDMI-1", connected=True, width=1920, height=1080, x=1920, screen_id=1),
            ]
            return s
        cmds_zero = emit_runtime_commands(make_state(":0"))
        cmds_one = emit_runtime_commands(make_state(":1"))
        cmds_arb = emit_runtime_commands(make_state(":47.2"))
        self.assertEqual(cmds_zero, cmds_one)
        self.assertEqual(cmds_one, cmds_arb)

    def test_stale_touchscreen_target_skipped(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080)]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="REMOVED")]
        cmds = emit_runtime_commands(s)
        self.assertFalse(any("set-prop" in c for c in cmds))


# ============================== VALIDATE ==============================

class TestValidate(unittest.TestCase):
    def test_missing_serverlayout_is_error(self):
        text = 'Section "Device"\n    Identifier "Device0"\nEndSection\n'
        errors = validate_xorg_conf(text)
        self.assertTrue(any("ServerLayout" in e.message for e in errors))

    def test_duplicate_identifier_is_error(self):
        text = (
            'Section "Device"\n    Identifier  "Device0"\nEndSection\n'
            'Section "Device"\n    Identifier  "Device0"\nEndSection\n'
            'Section "ServerLayout"\n    Identifier  "Layout0"\nEndSection\n'
        )
        errors = validate_xorg_conf(text)
        self.assertTrue(any("Duplicate" in e.message for e in errors))

    def test_dangling_screen_reference_is_error(self):
        text = (
            'Section "ServerLayout"\n    Identifier  "Layout0"\n'
            '    Screen      0 "ScreenDoesNotExist" 0 0\n'
            'EndSection\n'
        )
        errors = validate_xorg_conf(text)
        self.assertTrue(any("non-existent Screen" in e.message for e in errors))

    def test_clean_config_no_errors(self):
        errors = [e for e in validate_xorg_conf(SAMPLE_XORG_CONF) if e.severity == "error"]
        self.assertEqual(errors, [])

    def test_hex_busid_accepted(self):
        text = (
            'Section "Device"\n    Identifier  "Device0"\n'
            '    BusID       "PCI:0a:00:0"\nEndSection\n'
            'Section "ServerLayout"\n    Identifier  "Layout0"\nEndSection\n'
        )
        errors = validate_xorg_conf(text)
        self.assertFalse(any("BusID" in e.message for e in errors))

    def test_pci_domain_busid_accepted(self):
        text = (
            'Section "Device"\n    Identifier  "Device0"\n'
            '    BusID       "PCI:0000@01:00:0"\nEndSection\n'
            'Section "ServerLayout"\n    Identifier  "Layout0"\nEndSection\n'
        )
        errors = validate_xorg_conf(text)
        self.assertFalse(any("BusID" in e.message for e in errors))


# ============================== GOLDEN FIXTURES ==============================
# These pin the exact text emitted for a comprehensive synthetic state. They
# catch unintended changes to formatting / ordering / whitespace that
# per-feature assertions would miss. They are NOT proof of correctness — only
# regression markers. If you intentionally change emitter output, regenerate
# the strings below by running this state through `regenerate()` and verify
# the result loads in real Xorg before committing.

GOLDEN_XORG_CONF = """\
# Generated by xorgcist. Review before installing.

Section "Monitor"
    Identifier  "Monitor0"
EndSection

Section "Monitor"
    Identifier  "Monitor1"
EndSection

Section "Device"
    Identifier  "Device0"
    Driver      "nvidia"
    BusID       "PCI:1:0:0"
    Screen      0
EndSection

Section "Screen"
    Identifier  "Screen0"
    Device      "Device0"
    Monitor     "Monitor0"
    Option      "UseDisplayDevice" "DP-1, DP-2"
    Option      "MetaModes" "DP-1: 2560x1440+0+0, DP-2: 1920x1080+2560+360"
    SubSection  "Display"
        Depth       24
    EndSubSection
EndSection

Section "Device"
    Identifier  "Device1"
    Driver      "nvidia"
    BusID       "PCI:1:0:0"
    Screen      1
EndSection

Section "Screen"
    Identifier  "Screen1"
    Device      "Device1"
    Monitor     "Monitor1"
    Option      "UseDisplayDevice" "HDMI-1"
    Option      "MetaModes" "HDMI-1: 1920x1080+0+0 {Rotation=left}"
    SubSection  "Display"
        Depth       24
    EndSubSection
EndSection

Section "ServerLayout"
    Identifier  "Layout0"
    Screen      0 "Screen0" 0 0
    Screen      1 "Screen1" RightOf "Screen0"
EndSection
"""

GOLDEN_RUNTIME = [
    '_X="${DISPLAY:-:0}"; _X="${_X%.*}"   # X server from $DISPLAY (handles :N, :N.M, host:N.M, unset)',
    'DISPLAY="$_X.0" xrandr --output DP-1 --mode 2560x1440 --rate 144.00 --pos 0x0 --primary',
    'DISPLAY="$_X.0" xrandr --output DP-2 --mode 1920x1080 --rate 60.00 --pos 2560x360',
    'DISPLAY="$_X.1" xrandr --output HDMI-1 --mode 1920x1080 --rate 60.00 --pos 0x0 --rotate left',
    'xinput enable ELO',
    "xinput set-prop ELO 'Coordinate Transformation Matrix' "
    '0.000000 -1.000000 1.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000',
]


def _golden_state():
    """Synthetic state exercising: multi-X-screen, NVIDIA driver, primary,
    rotation, RHEL8-style env_display=':1', touchscreen on non-default screen."""
    s = State()
    s.env_display = ":1"
    s.gpus = [GPU(busid="PCI:1:0:0", vendor="NVIDIA", driver="nvidia", name="RTX 4090")]
    s.outputs = [
        Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0,
               primary=True, screen_id=0, refresh_rate=144.0,
               available_modes=[(2560, 1440, 144.0)]),
        Output("DP-2", connected=True, width=1920, height=1080, x=2560, y=360,
               screen_id=0, refresh_rate=60.0,
               available_modes=[(1920, 1080, 60.0)]),
        Output("HDMI-1", connected=True, width=1920, height=1080, x=4480, y=0,
               screen_id=1, rotation="left", refresh_rate=60.0,
               available_modes=[(1920, 1080, 60.0)]),
    ]
    s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                         target_output="HDMI-1")]
    return s


class TestGoldenFixtures(unittest.TestCase):
    def test_golden_xorg_conf_byte_equal(self):
        # If this fails after an intentional emitter change: regenerate
        # GOLDEN_XORG_CONF, validate the new output in real Xorg, and update.
        self.assertEqual(emit_xorg_conf(_golden_state()), GOLDEN_XORG_CONF)

    def test_golden_runtime_byte_equal(self):
        self.assertEqual(emit_runtime_commands(_golden_state()), GOLDEN_RUNTIME)


# ============================== INTEGRATION ==============================

class TestRegenerateIntegration(unittest.TestCase):
    def test_demo_state_regenerates_clean(self):
        s = init_demo_state()
        regenerate(s)
        errors = [e for e in s.validation_errors if e.severity == "error"]
        self.assertEqual(errors, [])
        self.assertIn("Section", s.generated_config)
        self.assertTrue(any("xrandr" in c for c in s.generated_runtime))

    def test_state_outputs_not_mutated_by_emit(self):
        # User-typed mirror should not lose original position when un-mirrored.
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=500, y=200,
                   mirror_of="DP-1"),
        ]
        regenerate(s)
        dp2 = next(o for o in s.outputs if o.name == "DP-2")
        # Original position survives
        self.assertEqual((dp2.x, dp2.y), (500, 200))


if __name__ == "__main__":
    unittest.main()
