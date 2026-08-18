"""Phase 2 checks: the EditController (GUI-free) and the widget painting
mechanics (canvas -> source px -> layer), including left-paint vs pan routing.

The controller tests always run; the widget tests need Tk and are skipped if no
display is available.

Run: py -3 _test_edit_widget.py
"""
import types
import numpy as np
import manual_edits as me


def _teardown(w):
    """Destroy a Tk root and reset the default-root/image state so the next
    GUI test in this process starts from a clean interpreter (avoids stale
    'pyimageN' handles leaking across roots)."""
    try:
        w.destroy()
    except Exception:
        pass
    try:
        import tkinter, gc
        tkinter._default_root = None
        gc.collect()
    except Exception:
        pass


# -- EditController (no GUI) -------------------------------------------------
def test_controller_paint_stroke():
    L = me.EditLayer(200, 200)
    changes = []
    c = me.EditController(L, on_change=lambda: changes.append(1))
    c.channel = "alpha"; c.brush = 6; c.alpha_value = 180
    c.begin(50, 50); c.end()
    assert L.alpha_mask[50, 50] and L.alpha[50, 50] == 180
    assert changes                                   # on_change fired


def test_controller_interpolates_drag():
    L = me.EditLayer(200, 200)
    c = me.EditController(L)
    c.channel = "alpha"; c.brush = 6; c.alpha_value = 255
    c.begin(40, 100); c.to(120, 100); c.end()        # fast horizontal drag
    # The gap between endpoints must be filled (no holes).
    assert L.alpha_mask[100, 40] and L.alpha_mask[100, 120]
    assert L.alpha_mask[100, 80], "drag interpolation left a gap"


def test_controller_eraser_and_undo():
    L = me.EditLayer(100, 100)
    c = me.EditController(L)
    c.channel = "rgb"; c.brush = 10; c.rgb_value = (10, 20, 30)
    c.begin(50, 50); c.end()
    assert L.rgb_mask[50, 50]
    c.tool = "eraser"
    c.begin(50, 50); c.end()
    assert not L.rgb_mask[50, 50]                    # erased -> reverts
    c.undo()                                          # undo the erase
    assert L.rgb_mask[50, 50] and tuple(L.rgb[50, 50]) == (10, 20, 30)


def test_controller_picker_is_noop_paint():
    L = me.EditLayer(80, 80)
    c = me.EditController(L)
    c.tool = "picker"
    c.begin(10, 10); c.to(20, 20); c.end()
    assert L.is_empty()                               # picker paints nothing


# -- Widget painting mechanics (Tk) ------------------------------------------
def test_widget_paint_and_pan():
    try:
        import customtkinter as ctk
        import main
        from PIL import Image
        root = ctk.CTk(); root.withdraw()
    except Exception as e:
        print("SKIP widget tests (no GUI):", e)
        return
    try:
        pane = main.ZoomPanPreview(root)
        pane.canvas.winfo_width = lambda: 400
        pane.canvas.winfo_height = lambda: 300
        # 1024x512 base downscaled from a 2048x1024 source (scale = 2).
        pane.set_source_scale(2.0)
        pane.set_image(Image.new("RGB", (1024, 512), (80, 80, 80)))

        layer = me.EditLayer(2048, 1024)
        ctrl = me.EditController(layer)
        ctrl.channel = "alpha"; ctrl.brush = 10; ctrl.alpha_value = 222
        pane.on_paint_begin = ctrl.begin
        pane.on_paint_to = ctrl.to
        pane.on_paint_end = ctrl.end
        pane.set_edit_mode(True, 10)

        E = lambda x, y: types.SimpleNamespace(x=x, y=y, state=0)
        sp = pane._canvas_to_source_px(200, 150)      # canvas centre -> source px
        assert sp is not None
        # A left click in edit mode must paint the layer at those SOURCE coords.
        pane._on_left_press(E(200, 150))
        pane._on_left_release(E(200, 150))
        sx, sy = int(sp[0]), int(sp[1])
        assert layer.alpha_mask[sy, sx] and layer.alpha[sy, sx] == 222
        assert pane._brush_cursor_id is not None       # brush cursor drawn

        # Left-drag in edit mode does NOT pan.
        ox, oy = pane._offset_x, pane._offset_y
        pane._on_left_press(E(120, 120))
        pane._on_left_drag(E(160, 120))
        pane._on_left_release(E(160, 120))
        assert (pane._offset_x, pane._offset_y) == (ox, oy), "painting panned the view"

        # Right-drag ALWAYS pans, even while editing.
        pane._on_press(E(200, 150)); pane._on_drag(E(150, 150)); pane._on_release(E(150, 150))
        assert pane._offset_x != ox, "right-drag didn't pan in edit mode"

        # With edit mode OFF, left-drag pans.
        pane.set_edit_mode(False)
        ox2 = pane._offset_x
        pane._on_left_press(E(200, 150)); pane._on_left_drag(E(240, 150)); pane._on_left_release(E(240, 150))
        assert pane._offset_x != ox2, "left-drag didn't pan when not editing"
    finally:
        _teardown(root)


def test_app_integration():
    """Full TuneTab path: load -> alpha-only enables editing -> paint -> picker
    -> undo -> normal view disables. Skipped if no display."""
    try:
        import customtkinter as ctk
        import main
        from PIL import Image
        app = main.AlphaToolsApp()
        app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP app integration (no GUI):", e)
        return
    try:
        t = app.tune
        t.source_image = Image.new("RGBA", (300, 200), (100, 120, 140, 255))
        t.source_path = "synthetic.png"
        t.preview_source = t._make_preview_source(t.source_image)
        t._init_edit_layer()
        assert t.edit is not None and "_manual_edits" in t.params
        # View modes must NOT enable editing on their own.
        t._show_alpha_only.set(True); t._on_view_mode_changed("alpha")
        assert not t.after_view._edit_active
        # Explicit toggle turns it on; channel is independent of the view.
        t._set_edit_enabled(True)
        assert t.after_view._edit_active
        t.edit.channel = "alpha"; t.edit.brush = 8; t.edit.alpha_value = 200
        t._edit_begin(150, 100); t._edit_end()
        assert t.edit.layer.alpha_mask[100, 150]
        # Picker tool is one-shot -> reverts to pencil.
        t.last_after = t.source_image
        t.edit.tool = "picker"; t._edit_begin(150, 100)
        assert t.edit.tool == "pencil" and t.edit.alpha_value == 255
        t._edit_undo()
        assert t.edit.layer.is_empty()
        t._set_edit_enabled(False)
        assert not t.after_view._edit_active
    finally:
        _teardown(app)


def test_alpha_edit_in_texture_view():
    """Alpha can be painted while the texture-only view is active (channel is
    decoupled from the view mode). Skipped if no display."""
    try:
        import customtkinter as ctk
        import main
        from PIL import Image
        app = main.AlphaToolsApp(); app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP alpha-in-texture-view (no GUI):", e)
        return
    try:
        t = app.tune
        t.source_image = Image.new("RGBA", (200, 200), (100, 120, 140, 255))
        t.source_path = None
        t.preview_source = t._make_preview_source(t.source_image)
        t._init_edit_layer()
        t._show_texture_only.set(True); t._on_view_mode_changed("texture")
        t._set_edit_enabled(True)
        t.edit.channel = "alpha"; t.edit.brush = 6; t.edit.alpha_value = 0
        assert t.after_view._edit_active
        t._edit_begin(100, 100); t._edit_end()
        assert t.edit.layer.alpha_mask[100, 100] and t.edit.layer.alpha[100, 100] == 0
    finally:
        _teardown(app)


def test_load_edits_onto_new_image():
    """Save a sidecar from one image, then load it onto a different-size image
    via _load_edits (file dialog + confirm monkeypatched). Skipped if no GUI."""
    try:
        import os
        import customtkinter as ctk
        import main
        import manual_edits as me3
        from PIL import Image
        app = main.AlphaToolsApp(); app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP load-edits (no GUI):", e)
        return
    try:
        t = app.tune
        # Make + save an edit layer for a 100x100 image.
        t.source_image = Image.new("RGBA", (100, 100), (10, 20, 30, 255))
        t.source_path = "_le_a.png"
        t.preview_source = t._make_preview_source(t.source_image)
        t._init_edit_layer()
        t._set_edit_enabled(True)
        t.edit.channel = "alpha"; t.edit.brush = 6; t.edit.alpha_value = 200
        t._edit_begin(40, 40); t._edit_end()
        sidecar = str(me3.EditLayer.sidecar_path("_le_a.png"))
        t._save_edits()
        assert os.path.exists(sidecar)

        # Switch to a different-size image with NO matching sidecar.
        t.source_image = Image.new("RGBA", (80, 80), (0, 0, 0, 255))
        t.source_path = "_le_b.png"
        t.preview_source = t._make_preview_source(t.source_image)
        t._init_edit_layer()
        assert t.edit.layer.is_empty()               # nothing auto-loaded

        # Load the saved sidecar onto it (monkeypatch the dialogs).
        main.filedialog.askopenfilename = lambda **k: sidecar
        main.messagebox.askyesno = lambda *a, **k: True
        t._load_edits()
        assert (t.edit.layer.w, t.edit.layer.h) == (80, 80)   # sized to current img
        assert t.edit.layer.alpha_mask[40, 40] and t.edit.layer.alpha[40, 40] == 200
        assert t.params["_manual_edits"] is t.edit.layer
        os.remove(sidecar)
    finally:
        _teardown(app)


def test_granulate_tool_via_panel():
    """Selecting Granulate locks the channel to RGB, enables the Variation
    slider, and painting speckles the source colour. Skipped if no display."""
    try:
        import numpy as _np
        import customtkinter as ctk
        import main
        from PIL import Image
        app = main.AlphaToolsApp(); app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP granulate tool (no GUI):", e)
        return
    try:
        t = app.tune
        p = t.manual_edits_panel
        t.source_image = Image.new("RGBA", (80, 80), (100, 100, 100, 255))
        t.source_path = None
        t.preview_source = t._make_preview_source(t.source_image)
        t._init_edit_layer()
        assert t.edit.source_rgb is not None            # base for jitter
        t._set_edit_enabled(True)
        p._tool_var.set("Granulate"); p._push()
        assert t.edit.tool == "granulate" and t.edit.channel == "rgb"
        assert p._chan_var.get() == "RGB"               # channel locked to RGB
        assert str(p._gran_slider.cget("state")) == "normal"
        p._gran_var.set(20); p._on_gran(20)
        assert t.edit.granulate_amount == 20
        t.edit.brush = 10
        t._edit_begin(40, 40); t._edit_end()
        reg = t.edit.layer.rgb[36:44, 36:44]
        assert t.edit.layer.rgb_mask[40, 40]            # painted
        assert reg.std() > 0                            # speckled, not flat
        assert int(reg.min()) >= 80 and int(reg.max()) <= 120   # around 100 +/-20
    finally:
        _teardown(app)


def test_ctrl_zoom_center():
    """Ctrl +/- zoom about the viewport centre: a centred image stays centred;
    a panned view scales its offset so the centre is stable; mirrors to the
    linked pane; clamps at the limits. Skipped if no display."""
    try:
        import customtkinter as ctk
        import main
        from PIL import Image
        root = ctk.CTk(); root.withdraw()
    except Exception as e:
        print("SKIP ctrl-zoom (no GUI):", e)
        return
    try:
        a = main.ZoomPanPreview(root)
        b = main.ZoomPanPreview(root)
        a.link_with(b)
        for p in (a, b):
            p.canvas.winfo_width = lambda: 400
            p.canvas.winfo_height = lambda: 300
            p.set_image(Image.new("RGB", (200, 200), (50, 50, 50)))

        # Centred (offset 0): zoom in keeps it centred.
        a._zoom, a._offset_x, a._offset_y = 1.0, 0, 0
        a._zoom_by(1.25)
        assert abs(a._zoom - 1.25) < 1e-6 and a._offset_x == 0 and a._offset_y == 0

        # Panned: the offset scales with the zoom (viewport centre stays put).
        a._zoom, a._offset_x, a._offset_y = 2.0, 40, 20
        a._zoom_by(1.25)
        assert abs(a._zoom - 2.5) < 1e-6
        assert abs(a._offset_x - 50) < 1e-6 and abs(a._offset_y - 25) < 1e-6
        # Mirrored to the linked pane.
        assert abs(b._zoom - a._zoom) < 1e-6 and abs(b._offset_x - a._offset_x) < 1e-6

        # Zoom out scales the other way.
        a._zoom, a._offset_x, a._offset_y = 2.0, 40, 0
        a._zoom_by(1 / 1.25)
        assert abs(a._zoom - 1.6) < 1e-6 and abs(a._offset_x - 32) < 1e-6

        # Clamp at the max — offset unchanged when the zoom can't grow.
        a._zoom, a._offset_x = 40.0, 10
        a._zoom_by(1.25)
        assert a._zoom == 40.0 and a._offset_x == 10
    finally:
        _teardown(root)


def test_ctrl_click_picks():
    """Ctrl+left-click samples a value (no stroke), regardless of the tool."""
    try:
        import customtkinter as ctk
        import main
        from PIL import Image
        root = ctk.CTk(); root.withdraw()
    except Exception as e:
        print("SKIP ctrl-click pick (no GUI):", e)
        return
    try:
        pane = main.ZoomPanPreview(root)
        pane.canvas.winfo_width = lambda: 400
        pane.canvas.winfo_height = lambda: 300
        pane.set_source_scale(1.0)
        pane.set_image(Image.new("RGB", (300, 200), (0, 0, 0)))
        picked, painted = [], []
        pane.on_paint_pick = lambda sx, sy: picked.append((sx, sy))
        pane.on_paint_begin = lambda sx, sy: painted.append((sx, sy))
        pane.set_edit_mode(True, 6)
        E = lambda x, y, state=0: types.SimpleNamespace(x=x, y=y, state=state)
        pane._on_left_press(E(200, 150, state=0x0004))    # Ctrl held
        pane._on_left_release(E(200, 150, state=0x0004))
        assert picked and not painted and not pane._left_paint
        # Without Ctrl, the same click paints.
        pane._on_left_press(E(200, 150, state=0))
        assert painted
    finally:
        _teardown(root)


def test_panel_integration():
    """ManualEditsPanel drives the controller: tool/brush/alpha/colour push,
    picker reflect, sidecar save, stop editing. Skipped if no display."""
    try:
        import os
        import customtkinter as ctk
        import main
        import manual_edits as me2
        from PIL import Image
        app = main.AlphaToolsApp()
        app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP panel integration (no GUI):", e)
        return
    try:
        t = app.tune
        p = t.manual_edits_panel
        assert isinstance(p, main.ManualEditsPanel)
        t.source_image = Image.new("RGBA", (300, 200), (30, 60, 90, 255))
        t.source_path = None
        t.preview_source = t._make_preview_source(t.source_image)
        t._init_edit_layer()
        # Enable via the panel toggle (NOT the view mode); channel via selector.
        p._enable_var.set(True); t._set_edit_enabled(True)
        assert t.after_view._edit_active
        p._chan_var.set("Alpha"); p._push()
        assert t.edit.channel == "alpha"

        p._size_var.set(4); p._bump_size(1)
        p._alpha_var.set(180); p._on_alpha(180)
        p._tool_var.set("Eraser"); p._push()
        assert (t.edit.brush, t.edit.alpha_value, t.edit.tool) == (5, 180, "eraser")

        p._chan_var.set("RGB"); p._rgb_hex = "#00FF80"; p._push()
        assert t.edit.channel == "rgb" and t.edit.rgb_value == (0, 255, 128)

        # Picker reflects into the panel and reverts the tool.
        t.last_after = t.source_image
        p._chan_var.set("Alpha"); p._push()
        t.edit.tool = "picker"; t._edit_begin(150, 100)
        assert p._tool_var.get() == "Pencil" and p._alpha_var.get() == 255

        # Save edits -> sidecar.
        p._tool_var.set("Pencil"); p._push()
        t._edit_begin(150, 100); t._edit_end()
        t.source_path = "_sc_test.png"
        t._save_edits()
        sc = str(me2.EditLayer.sidecar_path("_sc_test.png"))
        assert os.path.exists(sc)
        os.remove(sc)

        # Turning the toggle off disables editing.
        p._enable_var.set(False); t._set_edit_enabled(False)
        assert not t.after_view._edit_active
    finally:
        _teardown(app)


# GUI tests each create + destroy a Tk root; Tk doesn't cleanly support several
# roots in one process (stale image handles), so each runs in its own process.
_GUI_TESTS = {
    "test_widget_paint_and_pan", "test_app_integration",
    "test_alpha_edit_in_texture_view", "test_ctrl_click_picks",
    "test_panel_integration", "test_load_edits_onto_new_image",
    "test_ctrl_zoom_center", "test_granulate_tool_via_panel",
}

if __name__ == "__main__":
    import sys
    import subprocess
    names = sorted(k for k in globals() if k.startswith("test_"))
    if len(sys.argv) > 1:                      # child: run one named test
        globals()[sys.argv[1]]()
        print(f"PASS  {sys.argv[1]}")
        sys.exit(0)
    passed = 0
    for name in names:
        if name in _GUI_TESTS:                 # isolate in a fresh interpreter
            r = subprocess.run([sys.executable, __file__, name],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL  {name}\n{r.stdout}{r.stderr}")
                sys.exit(1)
            print(r.stdout.strip() or f"PASS  {name}")
        else:
            globals()[name]()
            print(f"PASS  {name}")
        passed += 1
    print(f"\nAll {passed} edit-widget tests passed.")
