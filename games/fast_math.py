"""
games/fast_math.py

A fully deterministic, self-contained "Fast Math" mini-game engine.

No external services (Groq, LLMs, network calls, etc.) are used anywhere
in this file. All question generation and answer checking is done with
plain Python arithmetic and regex parsing, so it is safe to run on a
low-latency request/response loop with an ESP32 client.

Public interface (used by the server layer):
    game = FastMathGame()
    game.start()                     -> dict
    game.next_question()             -> dict
    game.process_input(user_input)   -> dict

Response shapes (NEW structured protocol):

    Question (first or mid-game, no prior answer to report):
        {
            "type": "game_fast_math",
            "equation": "2 + 3 * 4",
            "timer": 5,
            "emotion": "THINKING_E",
            "round": 1,
            "total_rounds": 5,
            "score": 0,
            "feedback": None
        }

    Question served right after grading a previous answer:
        same shape as above, but "feedback" is a short string describing
        whether the previous answer was correct/wrong/timed out.

    Game over:
        {
            "type": "game_fast_math_over",
            "emotion": "HAPPY",
            "score": 4,
            "total_rounds": 5,
            "feedback": "Correct! 2 + 3 * 4 = 14. Nice reflexes!"
        }

"type" is the field the ESP32/server should switch on. "equation" and
"timer" are guaranteed present on every "game_fast_math" payload, per
spec. The extra fields (round/total_rounds/score/feedback) are additive
and safe to ignore on the client if you don't want them yet.
"""

import random
import re
import time
from typing import Dict, Optional, Union


class FastMathGame:
    """
    A simple BODMAS ("order of operations") speed-math quiz game.

    Flow:
        1. start()           -> resets state, returns first question.
        2. next_question()   -> generates a new equation, starts the
                                 5-second timer, returns the question.
        3. process_input(x)  -> grades the previous question against
                                 the 5-second limit, then either returns
                                 the next question (with feedback) or a
                                 game-over payload.
    """

    # ---- Tunable constants -------------------------------------------------
    TIME_LIMIT_SECONDS: float = 5.0
    TOTAL_ROUNDS: int = 5
    MIN_OPERAND: int = 1
    MAX_OPERAND: int = 12
    # '+' , '-' , '*' only -> guarantees integer results and gives a real
    # BODMAS/order-of-operations puzzle whenever '*' is mixed with '+'/'-'.
    OPERATORS = ("+", "-", "*")

    # Small tolerance so normal server/network scheduling jitter right at
    # the boundary doesn't unfairly time out an answer that was actually
    # submitted in time. Kept tiny so the 5-second feel stays intact.
    TIMEOUT_GRACE_SECONDS: float = 0.15

    # How many times we'll retry generation to avoid repeating the exact
    # previous equation before giving up and accepting a repeat (keeps
    # question generation O(1)-ish and fully deterministic in the worst case).
    MAX_DUPLICATE_RETRIES: int = 6

    VALID_EMOTIONS = {"NORMAL", "EXCITED", "HAPPY", "SAD", "SHOCKED", "THINKING_E"}

    TYPE_QUESTION = "game_fast_math"
    TYPE_GAME_OVER = "game_fast_math_over"

    def __init__(self) -> None:
        self.score: int = 0
        self.round_number: int = 0
        self.current_question_text: Optional[str] = None
        self.current_answer: Optional[int] = None
        self.question_start_time: Optional[float] = None
        self.game_active: bool = False
        # Tracks the last served equation text so next_question() can avoid
        # immediately repeating it. Reset on every start().
        self._last_question_text: Optional[str] = None

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def start(self) -> Dict[str, Union[str, int, float, None]]:
        """Reset all game state and serve the first question."""
        self.score = 0
        self.round_number = 0
        self.current_question_text = None
        self.current_answer = None
        self.question_start_time = None
        self.game_active = True
        self._last_question_text = None

        return self.next_question(feedback=None)

    def next_question(
        self, feedback: Optional[str] = None
    ) -> Dict[str, Union[str, int, float, None]]:
        """
        Generate a new deterministic BODMAS equation, arm the 5-second
        server-side timer, and return the structured question payload.

        `feedback` (optional) is stitched in when this call follows a
        graded answer, so the client can show "Correct! ... Next up: ...".
        """
        expression, answer = self._generate_equation(self.round_number)

        self.current_question_text = expression
        self.current_answer = answer
        self._last_question_text = expression
        self.question_start_time = time.time()
        self.game_active = True

        return self._make_question_response(
            emotion="THINKING_E",
            equation=expression,
            feedback=feedback,
        )

    def process_input(
        self, user_input: str
    ) -> Dict[str, Union[str, int, float, None]]:
        """
        Grade the user's answer to the currently active question.

        Checks (in order):
            1. Is there even an active question / timer running?
            2. Did the 5-second limit expire?  -> timeout
            3. Could a number be parsed from user_input? -> invalid/wrong
            4. Is the parsed number correct?    -> correct/wrong

        Then advances the round counter and either serves the next
        question (with feedback attached) or ends the game.
        """
        if not self.game_active or self.question_start_time is None:
            # Defensive fallback: nothing is in progress, so restart cleanly.
            return self.start()

        # Guard against a clock going backwards (NTP step, ESP32 rollover
        # quirks, etc.) so elapsed time never comes out negative.
        elapsed = max(0.0, time.time() - self.question_start_time)
        correct_answer = self.current_answer
        question_text = self.current_question_text

        if elapsed > (self.TIME_LIMIT_SECONDS + self.TIMEOUT_GRACE_SECONDS):
            feedback = f"Too slow! {question_text} = {correct_answer}."
            emotion = "SHOCKED"
        else:
            parsed_value = self._extract_number(user_input)

            if parsed_value is None:
                feedback = f"No number there — {question_text} = {correct_answer}."
                emotion = "SAD"
            elif parsed_value == correct_answer:
                self.score += 1
                feedback = f"Correct! {question_text} = {correct_answer}. Nice reflexes!"
                emotion = "EXCITED" if elapsed < self.TIME_LIMIT_SECONDS / 2 else "HAPPY"
            else:
                feedback = f"Not quite — {question_text} = {correct_answer}, not {parsed_value}."
                emotion = "SAD"

        self.round_number += 1

        # Close out the current question regardless of outcome.
        self.question_start_time = None

        if self.round_number >= self.TOTAL_ROUNDS:
            self.game_active = False
            final_emotion = "HAPPY" if self.score >= (self.TOTAL_ROUNDS // 2 + 1) else "NORMAL"
            return self._make_game_over_response(final_emotion, feedback)

        # Not done yet -> next question, with feedback about the last one.
        return self.next_question(feedback=feedback)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _difficulty_for_round(self, round_index: int) -> "tuple[int, int, tuple]":
        """
        Return (min_operand, max_operand, operator_pool) for the given
        zero-based round index, ramping up gradually so round 1 is an
        easy warm-up and later rounds are the full difficulty — while
        staying comfortably within mental-math range (operands never
        exceed MAX_OPERAND, so a round never takes longer to solve than
        the original fixed-difficulty version did).
        """
        # Operand ceiling grows a little each round, capped at MAX_OPERAND.
        step = round_index * 2
        max_operand = min(self.MAX_OPERAND, 6 + step)
        min_operand = self.MIN_OPERAND

        # Early rounds lean on '+'/'-' so the very first question isn't
        # already a multiplication puzzle; '*' becomes fully available
        # (and thus more likely) from round 2 onward.
        if round_index == 0:
            operators = ("+", "-")
        else:
            operators = self.OPERATORS

        return min_operand, max_operand, operators

    def _generate_equation(self, round_index: int) -> "tuple[str, int]":
        """
        Build a 3-operand, 2-operator BODMAS expression using only
        +, -, * so the correct answer is always a clean integer.
        Difficulty (operand range / operator pool) ramps with
        `round_index`. Retries a few times to avoid immediately
        repeating the previous question.

        Returns (expression_string, correct_integer_answer).
        """
        min_operand, max_operand, operators = self._difficulty_for_round(round_index)

        expression = None
        answer = None

        for _ in range(self.MAX_DUPLICATE_RETRIES):
            a = random.randint(min_operand, max_operand)
            b = random.randint(min_operand, max_operand)
            c = random.randint(min_operand, max_operand)
            op1 = random.choice(operators)
            op2 = random.choice(operators)

            candidate = f"{a} {op1} {b} {op2} {c}"

            if candidate != self._last_question_text:
                expression = candidate
                # Evaluate with standard operator precedence ourselves
                # (no eval()), since '*' must be applied before '+'/'-'
                # for a proper BODMAS test.
                answer = self._evaluate(a, op1, b, op2, c)
                break

        if expression is None:
            # All retries happened to collide (extremely unlikely) —
            # fall back to whatever we last generated rather than loop
            # forever, keeping generation deterministic and bounded.
            a = random.randint(min_operand, max_operand)
            b = random.randint(min_operand, max_operand)
            c = random.randint(min_operand, max_operand)
            op1 = random.choice(operators)
            op2 = random.choice(operators)
            expression = f"{a} {op1} {b} {op2} {c}"
            answer = self._evaluate(a, op1, b, op2, c)

        return expression, answer

    @staticmethod
    def _evaluate(a: int, op1: str, b: int, op2: str, c: int) -> int:
        """
        Manually apply operator precedence for a `a <op1> b <op2> c`
        expression built only from +, -, * (no division, no eval()).
        """

        def apply(x: int, op: str, y: int) -> int:
            if op == "+":
                return x + y
            if op == "-":
                return x - y
            if op == "*":
                return x * y
            raise ValueError(f"Unsupported operator: {op}")

        if op2 == "*":
            # a <op1> (b * c)
            return apply(a, op1, b * c)
        if op1 == "*":
            # (a * b) <op2> c
            return apply(a * b, op2, c)
        # Neither is '*': plain left-to-right is fine (+/- are equal precedence).
        return apply(apply(a, op1, b), op2, c)

    @staticmethod
    def _extract_number(user_input) -> Optional[int]:
        """
        Pull the first integer (optionally negative) out of a free-text
        user reply, e.g. "umm i think its 11!" -> 11.
        Returns None if no number could be found, and never raises —
        malformed, missing, or non-string input (None, bytes, numbers,
        objects with a broken __str__) is handled defensively.
        """
        if user_input is None:
            return None

        if not isinstance(user_input, str):
            try:
                if isinstance(user_input, (bytes, bytearray)):
                    user_input = user_input.decode("utf-8", errors="ignore")
                else:
                    user_input = str(user_input)
            except Exception:
                return None

        if not user_input:
            return None

        # Cap the scan length so a pathologically huge payload can't cost
        # meaningful regex time on a low-latency embedded round-trip.
        text = user_input[:64]

        try:
            match = re.search(r"-?\d+", text)
        except Exception:
            return None

        if not match:
            return None

        try:
            return int(match.group())
        except (ValueError, OverflowError):
            return None

    def _safe_emotion(self, emotion: str) -> str:
        return emotion if emotion in self.VALID_EMOTIONS else "NORMAL"

    def _make_question_response(
        self, emotion: str, equation: str, feedback: Optional[str]
    ) -> Dict[str, Union[str, int, float, None]]:
        """Build the structured game_fast_math question payload."""
        return {
            "type": self.TYPE_QUESTION,
            "equation": equation,
            "timer": int(self.TIME_LIMIT_SECONDS),
            "emotion": self._safe_emotion(emotion),
            "round": self.round_number + 1,
            "total_rounds": self.TOTAL_ROUNDS,
            "score": self.score,
            "feedback": feedback,
        }

    def _make_game_over_response(
        self, emotion: str, feedback: str
    ) -> Dict[str, Union[str, int, float, None]]:
        """Build the structured game-over payload."""
        return {
            "type": self.TYPE_GAME_OVER,
            "emotion": self._safe_emotion(emotion),
            "score": self.score,
            "total_rounds": self.TOTAL_ROUNDS,
            "feedback": feedback,
        }
