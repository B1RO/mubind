"""Read-only home status checks that never delay drawing or navigating the menu."""

import threading


class HomeReadiness:
    def __init__(self, load_config, component_status, checkpoint_line, labels):
        self.load_config = load_config
        self.component_status = component_status
        self.checkpoint_line = checkpoint_line
        self.labels = labels
        self.checkpoint = "checkpoint  checking…"
        self.status = "checking…"
        self.detail = ""
        self.ready = False
        self.title = [("class:text", "Setup"), ("class:status class:muted", " · checking…")]
        self.stopped = threading.Event()
        self.thread = None

    def instruction(self):
        detail = f"\n  {self.detail}" if self.detail else ""
        return f"  {self.checkpoint}{detail}\n"

    def render(self):
        self.title[:] = [
            ("class:ready" if self.ready else "class:text", "Setup"),
            ("class:status class:muted", " · " + self.status),
        ]

    def attach(self, application):
        def start(_):
            if self.thread is None and not self.stopped.is_set():
                self.thread = threading.Thread(
                    target=self.check, args=(application.invalidate,), daemon=True,
                    name="mubind-home-readiness",
                )
                self.thread.start()

        application.after_render.add_handler(start)

    def check(self, invalidate):
        def update():
            if not self.stopped.is_set():
                invalidate()

        try:
            config = self.load_config()
            if self.stopped.is_set():
                return
            self.checkpoint = self.checkpoint_line()
            statuses = []
            for index, component in enumerate(config.components):
                if self.stopped.is_set():
                    return
                self.status = f"checking {index + 1}/{len(config.components)}…"
                self.detail = "Checking " + self.labels[component] + "…"
                update()
                statuses.append(self.component_status(component, config))
            if self.stopped.is_set():
                return
            self.ready = bool(statuses) and all(status == "Ready" for status in statuses)
            self.status = "ready" if self.ready else "needs attention"
            self.detail = ""
        except Exception as exc:
            self.status = "check failed"
            self.detail = "Setup needs attention · " + str(exc)
        update()

    def close(self):
        # Do not join a slow filesystem read on the UI thread. A check already
        # underway may finish, but no next check or late redraw can run.
        self.stopped.set()
