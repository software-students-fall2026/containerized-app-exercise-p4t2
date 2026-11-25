""" blink detection and tracking logic"""

from collections import deque

THRESHOLD_RATIO = 0.75  # how far below the baseline the ear needs to fall before we call the eyes closed
MIN_BLINK_SECONDS = 0.05
MAX_BLINK_SECONDS = 0.40

ROLLING_WINDOW_SECONDS = 30.0  # how much recent history sets the baseline
WARMUP_SECONDS = 2.0  # use DEFAULT_THRESHOLD until this much data exists
DEFAULT_THRESHOLD = 0.20  # a typical threshold, used only during warmup
BASELINE_PERCENTILE = 0.75  # used to determine what ear value is considered as 'open'

class BlinkDetector:
    """
    Turns a stream of EAR values into blinks.
    The "eyes closed" threshold follows the person's recent EAR
    """

    def __init__(self):
        self.window = deque()  # (time, ear) pairs
        self.baseline = None
        self.threshold = DEFAULT_THRESHOLD
        self.closed_since = None

    def update(self, ear, now):
        """
        feed one sample. ear is None if no face was found.
        returns eye_state, blink_time
        """
        if ear is None:
            self.closed_since = None
            return "no face", None
        self._update_threshold(ear, now)
        return self._detect_blink(ear, now)

    def _update_threshold(self, ear, now):
        """add the sample, drop old ones, recompute baseline and threshold."""
        self.window.append((now, ear))
        while now - self.window[0][0] > ROLLING_WINDOW_SECONDS:
            self.window.popleft()

        span = now - self.window[0][0]
        if span < WARMUP_SECONDS or len(self.window) < 10:
            self.threshold = DEFAULT_THRESHOLD
            return

        values = sorted(value for _, value in self.window)
        self.baseline = values[int(BASELINE_PERCENTILE * (len(values) - 1))]
        self.threshold = self.baseline * THRESHOLD_RATIO

    def _detect_blink(self, ear, now):
        """
        a closed-to-open transition of the right length counts as a blink
        return eye_state, blink_time
        blink_the time the eyes closed if a blink just finished, otherwise None.
        """
        
        if ear < self.threshold:
            # eyes close
            if self.closed_since is None:
                # eyes just closed
                self.closed_since = now
            return "close", None

        # eyes open
        if self.closed_since is None:
            # wasn't closed last call
            return "open", None

        # transition from close to open
        started = self.closed_since
        self.closed_since = None
        if MIN_BLINK_SECONDS <= now - started <= MAX_BLINK_SECONDS:
            # closed period is accepted
            return "blink", started
        return "open", None