# collect.py — Interactive click-collection tool for the NOVA project.
# Uses Gradio to present images one-by-one and record user clicks
# (in order of decreasing importance) into a JSON file.

import argparse  # Command-line argument parsing
import json      # JSON serialization / deserialization
import os        # Filesystem operations (paths, file existence, etc.)

import cv2       # OpenCV — image reading, drawing, color conversion
import gradio as gr  # Gradio — web-based UI framework


# Minimum and maximum number of clicks the user must/can place per image
MIN_CLICKS = 3
MAX_CLICKS = 6

# Accepted image file extensions for the input directory
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(images_dir):
    """Return a sorted list of image filenames found in *images_dir*."""
    if not os.path.isdir(images_dir):
        return []
    # Only keep files whose lowercase extension matches IMAGE_EXTENSIONS
    return sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(IMAGE_EXTENSIONS))


def load_data(out_path):
    """Load previously saved annotations from a JSON file, if it exists."""
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {}


def save_data(data, out_path):
    """Atomically save *data* to *out_path* via a temporary file."""
    # Write to a .tmp file first, then atomically replace the target
    # to avoid data corruption if the process is interrupted mid-write.
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, out_path)


def draw_clicks(image_rgb, clicks):
    """Draw numbered circles on *image_rgb* at each click position."""
    annotated = image_rgb.copy()
    for i, (x, y) in enumerate(clicks, start=1):
        # Black outer circle (slightly larger) for contrast
        cv2.circle(annotated, (x, y), 14, (0, 0, 0), -1)
        # Orange inner circle
        cv2.circle(annotated, (x, y), 12, (255, 80, 40), -1)
        # White number label centered on the circle
        cv2.putText(annotated, str(i), (x - 6, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


# =====================================================================
# Session — keeps track of the annotation state for the current run
# =====================================================================

class Session:
    def __init__(self, images_dir, out_path):
        self.images_dir = images_dir   # Folder containing the images
        self.out_path = out_path       # Path to the output JSON file
        self.images = list_images(images_dir)  # All image filenames
        self.data = load_data(out_path)        # Already-saved annotations
        self.index = 0                 # Index of the current image
        self.clicks = []               # Click coordinates for the current image
        self.redoing = False           # True when re-annotating a previously done image
        self._skip_done()              # Advance index past images that are already annotated

    def _skip_done(self):
        """Move self.index forward to the first image that has no annotation yet."""
        while self.index < len(self.images) and self.images[self.index] in self.data:
            self.index += 1

    @property
    def finished(self):
        """Return True when every image has been annotated (and we are not in redo mode)."""
        return self.index >= len(self.images) and not self.redoing

    @property
    def name(self):
        """Return the filename of the current image, or None if finished."""
        if self.redoing:
            return self.images[self.index]
        return None if self.finished else self.images[self.index]

    def current_image(self):
        """Load and return the current image as an RGB NumPy array."""
        if self.name is None:
            return None
        path = os.path.join(self.images_dir, self.name)
        img = cv2.imread(path)
        # OpenCV loads images in BGR; convert to RGB for Gradio display
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def add_click(self, x, y):
        """Record a new click if the maximum has not been reached yet."""
        if len(self.clicks) < MAX_CLICKS:
            self.clicks.append((int(x), int(y)))

    def undo(self):
        """Remove the last recorded click (if any)."""
        if self.clicks:
            self.clicks.pop()

    def annotated_images(self):
        """Return a list of image filenames that already have annotations."""
        return [img for img in self.images if img in self.data]

    def redo(self, image_name):
        """Switch to redo mode: re-annotate *image_name* from scratch."""
        if image_name not in self.images:
            return
        self.index = self.images.index(image_name)
        self.clicks = []
        self.redoing = True

    def reset_all(self):
        """Delete every annotation and start over from the first image."""
        self.data = {}
        save_data(self.data, self.out_path)
        self.index = 0
        self.clicks = []
        self.redoing = False

    def save_and_next(self):
        """Save the current clicks and advance to the next image.

        Returns (ok, message).  ok is False if minimum clicks not met.
        """
        if len(self.clicks) < MIN_CLICKS:
            return False, f"At least {MIN_CLICKS} clicks are required (you have {len(self.clicks)})."
        # Store clicks as a list of dicts with x, y, and order
        self.data[self.name] = [
            {"x": x, "y": y, "order": i + 1}
            for i, (x, y) in enumerate(self.clicks)
        ]
        save_data(self.data, self.out_path)
        self.clicks = []
        if self.redoing:
            # Finished redo: go back to the first undone image (or end)
            self.redoing = False
            self.index = 0
            self._skip_done()
        else:
            self.index += 1
            self._skip_done()
        return True, "Saved."

    def status(self):
        """Return a human-readable status string for the current state."""
        if self.redoing:
            return (f"Redoing: {self.name} --- "
                    f"clicks: {len(self.clicks)}/{MAX_CLICKS} (minimum {MIN_CLICKS})")
        if self.finished:
            return f"Done. {len(self.data)} images annotated in total."
        return (f"{self.index + 1}/{len(self.images)} --- {self.name} --- "
                f"clicks: {len(self.clicks)}/{MAX_CLICKS} (minimum {MIN_CLICKS})")


# ===================== UI helpers =====================

def _home_msg(session, out_path):
    """Build the Markdown string shown on the home/dashboard panel."""
    total = len(session.images)
    done = len(session.annotated_images())
    remaining = total - done

    lines = ["## 📋 Click Collection --- Control Panel\n"]
    lines.append(f"**{done}/{total}** images annotated.")
    if remaining > 0:
        lines.append(f" **{remaining}** images remaining.\n")
    else:
        lines.append(f" ✅ All images have been annotated!\n")
    lines.append(f"\nData is saved to `{out_path}`.\n")
    return "".join(lines)


# All UI outputs: image, home_panel, home_msg, home_continue_btn,
#   redo_dd, home_redo_btn, home_reset_btn,
#   edit_row, undo_btn, next_btn, stop_btn
_ALL_KEYS = 11  # Total number of UI components returned by layout functions

def _home_outputs(session, out_path):
    """Return Gradio update tuples to display the home screen."""
    annotated = session.annotated_images()
    has_remaining = not session.finished
    return (
        gr.update(value=None, visible=False),                         # image — hide the editor
        gr.update(visible=True),                                      # home_panel — show home
        gr.update(value=_home_msg(session, out_path)),                # home_msg — refresh text
        gr.update(visible=has_remaining,                              # home_continue_btn
                  value="▶️ Continue collection" if annotated else "▶️ Start collection"),
        gr.update(choices=session.images,                             # redo_dd — dropdown choices
                  value=session.images[0] if session.images else None,
                  visible=True),
        gr.update(visible=True),                                      # home_redo_btn
        gr.update(visible=bool(annotated)),                           # home_reset_btn
        gr.update(visible=False),                                     # edit_row — hide editing
        gr.update(visible=False),                                     # undo_btn
        gr.update(visible=False),                                     # next_btn
        gr.update(visible=False),                                     # stop_btn
    )

def _editing_outputs(session, show_stop=True):
    """Return Gradio update tuples to display the editing/annotation screen.

    *show_stop=False* hides the 'Save and exit' button (used during redo).
    """
    img = session.current_image()
    return (
        gr.update(value=img, visible=True),                           # image — show the editor
        gr.update(visible=False),                                     # home_panel — hide home
        gr.update(),                                                  # home_msg — unchanged
        gr.update(),                                                  # home_continue_btn
        gr.update(),                                                  # redo_dd
        gr.update(),                                                  # home_redo_btn
        gr.update(),                                                  # home_reset_btn
        gr.update(visible=True),                                      # edit_row — show editing
        gr.update(visible=True),                                      # undo_btn
        gr.update(visible=True),                                      # next_btn
        gr.update(visible=show_stop),                                 # stop_btn
    )


# =====================================================================
# build_interface — constructs the full Gradio UI
# =====================================================================

def build_interface(images_dir, out_path):
    """Create and return the Gradio Blocks interface."""
    session = Session(images_dir, out_path)
    annotated = session.annotated_images()

    with gr.Blocks(title="NOVA --- Click Collection") as demo:
        # Header instructions
        gr.Markdown(
            "# Click Collection\n"
            "Click on the image in order of decreasing importance "
            f"(from most to least important), {MIN_CLICKS}-{MAX_CLICKS} clicks, then **Next**."
        )
        # Hidden textbox used only as a dummy second output for image.select
        status = gr.Textbox(visible=False)

        # ============= HOME PANEL =============
        with gr.Column(visible=True) as home_panel:
            home_msg = gr.Markdown(value=_home_msg(session, out_path))
            # Button label depends on whether any images are already annotated
            continue_label = "▶️ Continue collection" if annotated else "▶️ Start collection"
            home_continue_btn = gr.Button(
                continue_label, variant="primary",
                visible=not session.finished
            )
            with gr.Row():
                # Dropdown to pick a specific image (for redo or direct annotation)
                redo_dd = gr.Dropdown(
                    label="Image to annotate",
                    choices=session.images,
                    value=session.images[0] if session.images else None,
                    interactive=True,
                )
                home_redo_btn = gr.Button("🔄 Do single", variant="secondary")
            # "Redo all" button — only visible when there are existing annotations
            home_reset_btn = gr.Button("🗑️ Redo all", variant="stop",
                                        visible=bool(annotated))

        # ============= EDITING VIEW =============
        # Image display area (hidden until the user starts annotating)
        image = gr.Image(type="numpy", interactive=False, height=500,
                         visible=False)
        # Editing action buttons (hidden until annotation starts)
        with gr.Row(visible=False) as edit_row:
            undo_btn = gr.Button("Undo last click")
            next_btn = gr.Button("Next", variant="primary")
            stop_btn = gr.Button("⏹ Save and exit", variant="stop")

        # ============= CALLBACKS =============
        # Collect all outputs in the same order used by _home_outputs / _editing_outputs
        all_outputs = [
            image, home_panel, home_msg, home_continue_btn,
            redo_dd, home_redo_btn, home_reset_btn,
            edit_row, undo_btn, next_btn, stop_btn,
        ]

        def on_continue():
            """Start or resume the sequential annotation flow."""
            if session.finished:
                return _home_outputs(session, out_path)
            return _editing_outputs(session)

        def on_redo(selected_image):
            """Re-annotate a single image chosen from the dropdown."""
            if not selected_image:
                return tuple(gr.update() for _ in range(_ALL_KEYS))
            session.redo(selected_image)
            # Hide the "Save and exit" button during single-image redo
            return _editing_outputs(session, show_stop=False)

        def on_reset():
            """Erase all annotations and start from the first image."""
            session.reset_all()
            return _editing_outputs(session)

        def on_click(evt: gr.SelectData):
            """Handle a click on the displayed image."""
            if session.finished or session.current_image() is None:
                return gr.update(), gr.update()
            # Record click coordinates and redraw the annotated image
            session.add_click(evt.index[0], evt.index[1])
            return draw_clicks(session.current_image(), session.clicks), gr.update()

        def on_undo():
            """Remove the last click and refresh the image."""
            session.undo()
            img = session.current_image()
            if img is None:
                return gr.update(), gr.update()
            # If there are remaining clicks, redraw them; otherwise show the clean image
            return (draw_clicks(img, session.clicks) if session.clicks else img,
                    gr.update())

        def on_next():
            """Save the current annotation and move to the next image."""
            was_redoing = session.redoing
            ok, msg = session.save_and_next()
            # If save succeeded and there's nothing left (or redo finished), go home
            if ok and (session.finished or was_redoing):
                return _home_outputs(session, out_path)
            # If save failed (too few clicks), keep the current image visible
            if not ok:
                img = session.current_image()
                return (
                    gr.update(value=draw_clicks(img, session.clicks) if session.clicks else img),
                    *[gr.update() for _ in range(_ALL_KEYS - 1)],
                )
            # Otherwise, show the next image
            return _editing_outputs(session)

        def on_stop():
            """Save current clicks (if minimum met) and return to the home screen."""
            if len(session.clicks) >= MIN_CLICKS:
                session.save_and_next()
            return _home_outputs(session, out_path)

        # --- Wire callbacks to UI components ---
        home_continue_btn.click(on_continue, outputs=all_outputs)
        home_redo_btn.click(on_redo, inputs=[redo_dd], outputs=all_outputs)
        home_reset_btn.click(on_reset, outputs=all_outputs)

        image.select(on_click, outputs=[image, status])
        undo_btn.click(on_undo, outputs=[image, status])
        next_btn.click(on_next, outputs=all_outputs)
        stop_btn.click(on_stop, outputs=all_outputs)

    return demo


# =====================================================================
# Entry point — parse CLI arguments and launch the Gradio app
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click collection for NOVA.")
    parser.add_argument("--images", default="img")   # Folder containing the images
    parser.add_argument("--out", default="clicks.json")  # Output JSON file
    args = parser.parse_args()

    # Abort if the images folder is empty or does not exist
    if not list_images(args.images):
        raise SystemExit(f"No images found in '{args.images}'.")

    # Build and launch the Gradio interface
    build_interface(args.images, args.out).launch()
