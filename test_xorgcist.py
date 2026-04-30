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
        m = derive_touchscreen_matrix(s, "DP-1")
        self.assertEqual(m, IDENTITY_MATRIX)

    def test_right_half_screen(self):
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, y=0, screen_id=0),
        ]
        m = derive_touchscreen_matrix(s, "DP-2")
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
        m = derive_touchscreen_matrix(s, "HDMI-1")
        self.assertAlmostEqual(m[0], 1.0)
        self.assertAlmostEqual(m[2], 0.0)

    def test_rotation_only_for_alone_rotated_output(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080,
                            screen_id=0, rotation="left")]
        m = derive_touchscreen_matrix(s, "DP-1")
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
        self.assertIn("DP-1: 1920x1080_60+0+0", text)
        self.assertIn("DP-2: 1920x1080_60+1920+0", text)

    def test_per_screen_metamodes_normalized(self):
        s = State()
        s.outputs = [
            Output("HDMI-1", connected=True, width=1920, height=1080, x=3840, y=0, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn("HDMI-1: 1920x1080_60+0+0", text)

    def test_separate_screens(self):
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn('Identifier  "Screen0"', text)
        self.assertIn('Identifier  "Screen1"', text)

    def test_remaps_noncontiguous_screen_ids(self):
        s = State()
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
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, x=0, y=0, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080, x=0, y=2000, screen_id=1),
        ]
        text = emit_xorg_conf(s)
        self.assertIn('Below "Screen0"', text)

    def test_rotation_emitted_in_monitor(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, rotation="left")]
        text = emit_xorg_conf(s)
        self.assertIn('Option      "Rotate" "left"', text)

    def test_rotation_in_metamodes(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, rotation="left")]
        text = emit_xorg_conf(s)
        self.assertIn("{Rotation=left}", text)

    def test_no_primary_in_monitor(self):
        # Primary is silently ignored by all drivers in Monitor section;
        # primary takes effect via runtime xrandr --primary instead.
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, primary=True)]
        text = emit_xorg_conf(s)
        self.assertNotIn('"Primary"', text)

    def test_color_depth_uses_max(self):
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, color_depth=24),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, color_depth=30),
        ]
        text = emit_xorg_conf(s)
        self.assertIn("Depth       30", text)

    def test_disabled_outputs_excluded(self):
        s = State()
        s.gpus = [GPU(busid="PCI:1:0:0", vendor="NVIDIA", driver="nvidia", name="X")]
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080),
            Output("DP-2", connected=True, width=1920, height=1080, x=1920, enabled=False),
        ]
        text = emit_xorg_conf(s)
        self.assertNotIn("DP-2:", text)


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

    def test_multi_screen_uses_display_prefix(self):
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=2560, height=1440, x=0, y=0, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=4480, screen_id=1),
        ]
        cmds = emit_runtime_commands(s)
        dp1 = next(c for c in cmds if "--output DP-1" in c)
        hdmi = next(c for c in cmds if "--output HDMI-1" in c)
        self.assertTrue(dp1.startswith("DISPLAY=:0.0 "))
        self.assertTrue(hdmi.startswith("DISPLAY=:0.1 "))

    def test_disabled_emits_off(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080, enabled=False)]
        cmds = emit_runtime_commands(s)
        self.assertTrue(any("--output DP-1 --off" in c for c in cmds))

    def test_disabled_on_phantom_screen_routes_to_runtime_zero(self):
        # Bug: disabled output on its OWN screen_id used to address an
        # X screen the conf never created. Now routed to runtime screen 0.
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("DP-2", connected=True, width=1920, height=1080,
                   screen_id=1, enabled=False),
        ]
        cmds = emit_runtime_commands(s)
        for c in cmds:
            self.assertNotIn("DISPLAY=:0.1 ", c)
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
        self.assertTrue(any(c == "xinput disable 15" for c in cmds))
        self.assertFalse(any("set-prop" in c for c in cmds))

    def test_enabled_touchscreen_emits_enable_then_setprop(self):
        s = State()
        s.outputs = [Output("DP-1", connected=True, width=1920, height=1080)]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="DP-1")]
        cmds = emit_runtime_commands(s)
        xinput = [c for c in cmds if c.startswith("xinput")]
        self.assertEqual(xinput[0], "xinput enable 15")
        self.assertTrue(xinput[1].startswith("xinput set-prop 15"))

    def test_xinput_no_display_prefix(self):
        # xinput devices are server-global; DISPLAY prefix would be cosmetic.
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=1920, screen_id=1),
        ]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="HDMI-1")]
        cmds = emit_runtime_commands(s)
        for c in cmds:
            if c.startswith("xinput"):
                self.assertFalse(c.startswith("DISPLAY="))

    def test_multi_screen_emits_master_pointer_note(self):
        s = State()
        s.outputs = [
            Output("DP-1", connected=True, width=1920, height=1080, screen_id=0),
            Output("HDMI-1", connected=True, width=1920, height=1080, x=1920, screen_id=1),
        ]
        s.touchscreens = [TouchscreenMapping(device_id=15, device_name="ELO",
                                             target_output="HDMI-1")]
        cmds = emit_runtime_commands(s)
        self.assertTrue(any("master pointer" in c.lower() for c in cmds))

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
