"""Tkinter-based desktop application for CCTV age and gender analysis."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Dict, Optional

import cv2
from PIL import Image, ImageTk

from .detector import AGE_LABELS, AgeGenderDetector


class CCTVApp(tk.Tk):
    """Main Tkinter application window."""

    def __init__(self) -> None:
        super().__init__()
        self._init_style()
        self.title("CCTV Age & Gender Analyzer")
        self.geometry("1280x720")
        try:
            self.state("zoomed")
        except tk.TclError:
            self.attributes("-zoomed", True)

        self._detector = AgeGenderDetector()
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_job: Optional[str] = None
        self._loop_lock = threading.Lock()
        self._running = False

        self._selected_video: Optional[Path] = None
        self._status_var = tk.StringVar(value="Status: Idle")
        self._file_var = tk.StringVar(value="No video selected")
        self._stats: Dict[str, int] = {}
        self._age_counts: Dict[str, int] = {label: 0 for label in AGE_LABELS}

        self._build_layout()
        self._update_stats_labels()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_style(self) -> None:
        style = ttk.Style()
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabelframe", padding=12)
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 11))
        style.configure("Primary.TButton", font=("Segoe UI", 11))
        style.configure("StatsValue.TLabel", font=("Segoe UI", 12, "bold"))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=20)
        root.grid(row=0, column=0, sticky="NSEW")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)

        header = ttk.Label(
            root, text="Age & Gender Video Analyzer", style="Header.TLabel"
        )
        header.grid(row=0, column=0, columnspan=2, sticky="W")

        controls_frame = ttk.Frame(root, padding=(0, 12, 0, 12))
        controls_frame.grid(row=1, column=0, columnspan=2, sticky="WE")
        controls_frame.columnconfigure(0, weight=1)

        upload_button = ttk.Button(
            controls_frame,
            text="Choose video…",
            style="Primary.TButton",
            command=self._on_choose_video,
        )
        upload_button.grid(row=0, column=0, sticky="W")

        self._stop_button = ttk.Button(
            controls_frame,
            text="Stop",
            command=self._stop_stream,
            style="Primary.TButton",
            state=tk.DISABLED,
        )
        self._stop_button.grid(row=0, column=1, padx=(12, 0))

        file_label = ttk.Label(
            controls_frame, textvariable=self._file_var, font=("Segoe UI", 11)
        )
        file_label.grid(row=1, column=0, columnspan=2, sticky="W", pady=(8, 0))

        # Video Display
        video_frame = ttk.Labelframe(
            root, text="Live Preview", style="Section.TLabelframe"
        )
        video_frame.grid(row=2, column=0, sticky="NSEW", padx=(0, 16))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self._video_label = ttk.Label(video_frame, anchor="center")
        self._video_label.grid(row=0, column=0, sticky="NSEW")

        # Dashboard
        dashboard = ttk.Labelframe(root, text="Dashboard", style="Section.TLabelframe")
        dashboard.grid(row=2, column=1, sticky="NSEW")
        dashboard.columnconfigure(0, weight=1)

        self._status_display = ttk.Label(
            dashboard, textvariable=self._status_var, style="Status.TLabel"
        )
        self._status_display.grid(row=0, column=0, sticky="W", pady=(0, 8))

        summary_frame = ttk.Frame(dashboard)
        summary_frame.grid(row=1, column=0, sticky="WE", pady=(0, 12))
        summary_frame.columnconfigure(0, weight=1)
        self._summary_labels: Dict[str, ttk.Label] = {}

        summary_items = {
            "frames": "Frames processed",
            "faces": "Faces detected",
            "male": "Male detections",
            "female": "Female detections",
        }
        for idx, (key, title) in enumerate(summary_items.items()):
            container = ttk.Frame(summary_frame, padding=8)
            container.grid(row=idx, column=0, sticky="WE", pady=2)
            ttk.Label(container, text=title).grid(row=0, column=0, sticky="W")
            value_label = ttk.Label(container, text="0", style="StatsValue.TLabel")
            value_label.grid(row=0, column=1, sticky="E")
            self._summary_labels[key] = value_label

        age_frame = ttk.Frame(dashboard)
        age_frame.grid(row=2, column=0, sticky="NSEW")
        ttk.Label(age_frame, text="Detections by age group").grid(
            row=0, column=0, sticky="W", pady=(0, 6)
        )

        self._age_tree = ttk.Treeview(
            age_frame,
            columns=("age", "count"),
            show="headings",
            height=len(AGE_LABELS),
        )
        self._age_tree.heading("age", text="Age group")
        self._age_tree.heading("count", text="Detections")
        self._age_tree.column("age", width=140, anchor="w")
        self._age_tree.column("count", width=100, anchor="center")
        self._age_tree.grid(row=1, column=0, sticky="NSEW")

        scrollbar = ttk.Scrollbar(
            age_frame, orient="vertical", command=self._age_tree.yview
        )
        self._age_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky="NS")

        for age in AGE_LABELS:
            self._age_tree.insert("", "end", iid=age, values=(age, 0))

        self._status_var.set("Status: Idle")

    def _on_choose_video(self) -> None:
        if self._running:
            messagebox.showinfo(
                "Streaming active",
                "Stop the current analysis before uploading a new video.",
            )
            return

        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ),
        )
        if not file_path:
            return

        self._selected_video = Path(file_path)
        self._file_var.set(f"Selected: {self._selected_video.name}")
        self._start_video_stream()

    def _stop_stream(self) -> None:
        with self._loop_lock:
            self._running = False
            if self._frame_job is not None:
                self.after_cancel(self._frame_job)
                self._frame_job = None

            if self._capture is not None:
                self._capture.release()
                self._capture = None

        self._stop_button.configure(state=tk.DISABLED)
        self._video_label.configure(image="")
        self._video_label.image = None
        self._status_var.set("Status: Idle")
        self._stats.clear()
        self._age_counts = {label: 0 for label in AGE_LABELS}
        self._update_stats_labels()

    def _start_video_stream(self) -> None:
        if self._selected_video is None:
            messagebox.showwarning("No video", "Please choose a video file first.")
            return

        capture = cv2.VideoCapture(str(self._selected_video))
        if not capture.isOpened():
            messagebox.showerror(
                "Failed to open", "Unable to read the selected video file."
            )
            capture.release()
            return

        with self._loop_lock:
            if self._capture is not None:
                self._capture.release()
            self._capture = capture
            self._running = True
            self._stats = {"frames": 0, "faces": 0, "male": 0, "female": 0}
            self._age_counts = {label: 0 for label in AGE_LABELS}

        self._stop_button.configure(state=tk.NORMAL)
        self._status_var.set("Status: Processing video…")
        self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        with self._loop_lock:
            if not self._running:
                return
            self._frame_job = self.after(10, self._process_frame)

    def _process_frame(self) -> None:
        with self._loop_lock:
            if not self._running or self._capture is None:
                return

            ret, frame = self._capture.read()
            if not ret:
                self._status_var.set("Status: Completed analysis.")
                self._stop_stream()
                return

        annotated, results = self._detector.analyze_frame(frame)
        self._update_stats(results)
        faces_detected = len(results)
        self._status_var.set(
            f"Status: Processing — {faces_detected} face(s) detected in current frame"
        )

        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=image)
        self._video_label.configure(image=imgtk)
        self._video_label.image = imgtk

        self._schedule_next_frame()

    def _update_stats(self, results) -> None:
        self._stats["frames"] += 1
        self._stats["faces"] += len(results)
        for result in results:
            key = result.gender.lower()
            if key in self._stats:
                self._stats[key] += 1
            if result.age in self._age_counts:
                self._age_counts[result.age] += 1

        self._update_stats_labels()

    def _update_stats_labels(self) -> None:
        for key, label in self._summary_labels.items():
            value = self._stats.get(key, 0)
            label.configure(text=f"{value}")

        for age, count in self._age_counts.items():
            if age in self._age_tree.get_children():
                self._age_tree.set(age, column="count", value=count)

    def _on_close(self) -> None:
        self._stop_stream()
        self.destroy()


def run() -> None:
    """Entrypoint used by external callers."""
    app = CCTVApp()
    app.mainloop()


if __name__ == "__main__":
    run()
