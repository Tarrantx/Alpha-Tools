"""Phase 2 checks: the EditController (GUI-free) and the widget painting
mechanics (canvas -> source px -> layer), including left-paint vs pan routing.

The controller tests always run; the widget tests need Tk and are skipped if no
display is available.

Run: py -3 _test_edit_widget.py
"""
import types
import numpy as np
import manual_edits as me


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

        E = lambda x, y: types.SimpleNamespace(x=x, y=y)
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
        root.destroy()


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
        t._show_alpha_only.set(True); t._update_edit_mode()
        assert t.after_view._edit_active and t.edit.channel == "alpha"
        t.edit.brush = 8; t.edit.alpha_value = 200
        t._edit_begin(150, 100); t._edit_end()
        assert t.edit.layer.alpha_mask[100, 150]
        t.last_after = t.source_image
        t.edit.tool = "picker"; t._edit_begin(150, 100)
        assert t.edit.tool == "pencil" and t.edit.alpha_value == 255
        t._edit_undo()
        assert t.edit.layer.is_empty()
        t._show_alpha_only.set(False); t._update_edit_mode()
        assert not t.after_view._edit_active
    finally:
        app.destroy()


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
        t._show_alpha_only.set(True); t._on_view_mode_changed("alpha")
        assert t.after_view._edit_active and t.edit.channel == "alpha"

        p._size_var.set(4); p._bump_size(1)
        p._alpha_var.set(180); p._on_alpha(180)
        p._tool_var.set("Eraser"); p._push()
        assert (t.edit.brush, t.edit.alpha_value, t.edit.tool) == (5, 180, "eraser")

        t._show_texture_only.set(True); t._show_alpha_only.set(False)
        t._on_view_mode_changed("texture")
        p._rgb_hex = "#00FF80"; p._push()
        assert t.edit.channel == "rgb" and t.edit.rgb_value == (0, 255, 128)

        # Picker reflects into the panel.
        t.last_after = t.source_image
        t._show_alpha_only.set(True); t._show_texture_only.set(False)
        t._on_view_mode_changed("alpha")
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

        t._stop_editing()
        assert not t.after_view._edit_active
    finally:
        app.destroy()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} edit-widget tests passed.")
