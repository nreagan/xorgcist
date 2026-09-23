# xorgcist

A small GUI for laying out NVIDIA displays on X11. Drag displays into place (or type exact positions), choose resolution, refresh rate, rotation and X screen, and xorgcist generates:

- an `xorg.conf` for the NVIDIA driver, and
- `xinput` commands that map each touchscreen onto its display.

It never changes your system. You review the text, then copy it or save it wherever you like.

![xorgcist screenshot](screenshot.png)

## Run

```sh
sudo dnf install python3-tkinter     # Debian/Ubuntu: sudo apt install python3-tk
python3 xorgcist.py                  # inside the X session you want to configure
```

Needs Python 3.6 or newer (stock RHEL8 `python3` works) and tkinter. It reads `xrandr`, `xinput` and `/proc/driver/nvidia/gpus`. There are no other dependencies.

### From another machine

Like `nvidia-settings -c`, the `-c` / `--ctrl-display` option reads displays and input devices from one X display, while the window opens on `$DISPLAY`. Over SSH X forwarding:

```sh
ssh -Y you@target
ls /tmp/.X11-unix/                              # X0, X1, ... = running X displays (RHEL8 desktop sessions are often :1)
xauth merge /run/user/`id -u`/gdm/Xauthority    # authorize this SSH session for your GDM desktop session
python3 xorgcist.py -c :1
```

You must be logged in at the target's console as the same user, because that Xauthority file belongs to the logged-in user. If X was started with `startx`, the cookie is already in `~/.Xauthority`, so skip the merge. The GPU list and saved files stay on the target, which is where they're needed.

## Using it

- **Move a display:** drag it on the canvas, or type X/Y. Positions are absolute desktop pixels. There is no snapping.
- **Output fields:** Enabled, X screen (an existing one or `New`), Resolution (with aspect ratio), Refresh, Rotation, Primary display (one per X screen) and Force full composition pipeline (NVIDIA's tearing fix).
- **Touchscreens:** click *Add touchscreen*, pick the input device (listed as `xinput` ID and name, or type an ID), then pick the display it should cover.
- **Save or copy:** the bottom panes update live. Use *Copy* or *Save as...* on either one.

## What gets generated

**xorg.conf.** One `Device` and one `Screen` per X screen, placed with absolute coordinates in `ServerLayout`.
- Each screen's `MetaModes` lists its displays with mode, offset and rotation, e.g. `DP-0: 2560x1440_144 +0+0 {Rotation=left, ForceFullCompositionPipeline=On}`.
- The primary display is set with `Option "nvidiaXineramaInfoOrder" "DP-0"`, which is what nvidia-settings writes for "Make this the primary display".
- `Virtual` fixes each X screen's size to the bounding box of its displays.
- When several X screens share one GPU, each `Device` gets `Screen N` and each `Screen` gets `UseDisplayDevice`, so every X screen drives exactly the displays you assigned.

To install: `sudo cp xorg.conf /etc/X11/xorg.conf`, then restart X. If it doesn't come up as expected, run `grep -E '\((EE|WW)\)' /var/log/Xorg.0.log`. With rootless X the log is `~/.local/share/xorg/Xorg.0.log`.

**Touchscreen commands.** A `sh` script with one `xinput set-prop ... "Coordinate Transformation Matrix"` line per touchscreen, calculated as on the [Arch wiki](https://wiki.archlinux.org/title/Calibrating_Touchscreen) with the display's rotation included. Run it inside the X session, for example from session autostart. X forgets the setting when the device is re-plugged or X restarts.

Devices are referenced by their `xinput list` ID, so two identical touchscreens can be told apart. Working out which ID belongs to which physical screen is up to you. X assigns the IDs in a fixed order when it starts, so they stay the same across reboots as long as the same devices are plugged into the same ports. They can change after re-plugging a device or adding another input device.

## NVIDIA and X11 details

- **Mode names.** NVIDIA calls a mode `WxH_R`, with R the refresh rate rounded to a whole number. Plain `WxH` means "the driver's best mode at that size", so xorgcist only includes the rate when a resolution offers more than one. The README says these names are built "approximately" and can gain a suffix (`_60_0`). If the log says a mode wasn't found, add `Option "ModeDebug" "true"` to the `Device` section. The X log will then list the exact names.
- **Touch matrix size.** X scales touch input to the bounding box of *all* X screens (xserver `scale_to_desktop`), so the matrix is always relative to the whole desktop.
- **Moving between X screens.** The pointer, including touch, only crosses between X screens that share an edge, one screen per event. xorgcist warns when an X screen can't be reached.
- **BusID** is decimal (`PCI:bus@domain:device:function`), while `lspci` prints hex. xorgcist converts it.

## References

- NVIDIA driver README for your exact version (`cat /proc/driver/nvidia/version`): `https://download.nvidia.com/XFree86/Linux-x86_64/<version>/README/README.txt`. Relevant sections:
  - "Configuring Multiple Display Devices on One X Screen" (MetaModes)
  - "Configuring Multiple X Screens on One Card"
  - Appendix B (X config options)
  - Appendix C (display device names)
- `man xorg.conf`: `ServerLayout`, `Device` / `Screen`, `TransformationMatrix`.

## License

MIT, see [LICENSE](LICENSE).
