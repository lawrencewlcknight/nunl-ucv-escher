"""Small synchronous HTTP clients with conservative mutation recovery.

GETs use bounded backoff. GTO POSTs follow the provider's engine-busy retry
policy; lost responses and timeouts are not blindly replayed.
"""
from functools import wraps
from http.client import HTTPException
import json
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler, HTTPSHandler

from .game import DEFAULT_ACTION_MENU
from .remote import ProtocolError, check_successor, parse_gtow, parse_slumbot


class APIError(RuntimeError):
    pass


class AmbiguousOutcome(APIError):
    """A mutation may have succeeded. Stop unless an authoritative read resolves it."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # In particular, never forward a researcher key to a redirected origin.
        return None


class JSONTransport:
    post_attempts = 1
    retryable_post_statuses = ()
    deadline = None

    def __init__(self, base_url, *, headers=None, timeout=30, min_interval=.5,
                 get_attempts=3, sleep=time.sleep, clock=time.monotonic):
        if timeout <= 0 or min_interval < 0 or not 1 <= get_attempts <= 5:
            raise ValueError("Invalid HTTP timing/retry settings")
        self.base_url = base_url.rstrip("/")
        # Identify the research client: GTO Wizard's edge filter rejects the
        # default Python-urllib signature before API-key validation occurs.
        self.headers = {"User-Agent": "UCV-Escher-Research/1.0", **(headers or {})}
        self.timeout, self.min_interval = timeout, min_interval
        self.get_attempts, self.sleep, self.clock = get_attempts, sleep, clock
        self.last_request = None
        self.busy_retries = 0
        context = ssl.create_default_context()
        # Some python.org macOS installations have no default CA file. Augment
        # system trust with the optional standard bundle; never disable TLS checks.
        try:
            import certifi
        except ImportError:
            pass
        else:
            context.load_verify_locations(cafile=certifi.where())
        self.opener = build_opener(_NoRedirect, HTTPSHandler(context=context))

    def _once(self, method, path, body):
        if self.last_request is not None:
            delay = self.min_interval - (self.clock() - self.last_request)
            if delay > 0:
                self.sleep(delay)
        self.last_request = self.clock()
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(self.base_url + path, data=encoded, method=method,
                          headers={"Accept": "application/json", "Content-Type": "application/json", **self.headers})
        with self.opener.open(request, timeout=self.timeout) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("Response too large")
        return json.loads(data)

    def request(self, method, path, body=None):
        if method not in ("GET", "POST") or not path.startswith("/") or path.startswith("//"):
            raise ValueError("Unsupported HTTP request")
        attempts = self.get_attempts if method == "GET" else self.post_attempts
        for attempt in range(attempts):
            if self.deadline is not None and self.clock() >= self.deadline:
                raise TimeoutError("Benchmark wall budget exhausted before request")
            try:
                return self._once(method, path, body)
            except HTTPError as exc:
                transient = exc.code == 429 or exc.code >= 500
                if method == "POST" and exc.code in self.retryable_post_statuses:
                    if attempt + 1 == attempts:
                        raise APIError(f"GTO engine busy: HTTP {exc.code} after {attempts} attempts") from None
                    self.busy_retries += 1
                    print(f"GTO engine busy (HTTP {exc.code}); retry {attempt + 2}/{attempts} "
                          f"in {min(2 ** (attempt + 1), 32)}s", flush=True)
                elif method == "POST" and transient:
                    raise AmbiguousOutcome(f"POST returned HTTP {exc.code}; mutation was not retried") from None
                elif not transient or attempt + 1 == attempts:
                    raise APIError(f"HTTP {exc.code}") from None
            except (URLError, TimeoutError, OSError, ValueError, HTTPException) as exc:
                # No URLs, headers or server bodies in exception text: they can contain secrets.
                if method == "POST":
                    reason_type = type(getattr(exc, "reason", exc)).__name__
                    raise AmbiguousOutcome(f"POST response unavailable or invalid ({reason_type}); mutation was not retried") from None
                if attempt + 1 == attempts:
                    raise APIError("GET response unavailable or invalid") from None
            delay = min(2 ** (attempt + 1), 32) if method == "POST" else min(2 ** attempt, 4)
            if self.deadline is not None:
                delay = min(delay, max(0, self.deadline - self.clock()))
            self.sleep(delay)


class GTOWizardTransport(JSONTransport):
    # Same status codes, attempt limit and backoff as gtowizard-ai's client.
    # These explicit HTTP responses are distinct from a lost/invalid response.
    post_attempts = 32
    retryable_post_statuses = (502, 503, 504)

    def _once(self, method, path, body):
        try:
            return super()._once(method, path, body)
        except HTTPError as exc:
            if exc.code != 409:
                raise
            # A conflict can mean a usage limit or a hand-state problem. Keep
            # the provider's explanation instead of guessing or retrying it.
            message = f"HTTP 409 on {method} {path.split('?')[0]}"
            try:
                with exc:
                    data = json.loads(exc.read(4096))
                detail = data.get("detail") if isinstance(data, dict) else None
                if isinstance(detail, str):
                    for value in self.headers.values():
                        if value:
                            detail = detail.replace(str(value), "[REDACTED]")
                    message += ": " + " ".join(detail.split())[:512]
            except (ValueError, OSError, HTTPException):
                pass
            raise APIError(message) from None


def _mutation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if getattr(self, "_blocked", False):
            raise AmbiguousOutcome("Session stopped after an earlier error; no further mutations allowed")
        try:
            return method(self, *args, **kwargs)
        except (APIError, ProtocolError):
            self._blocked = True
            raise
    return guarded


class SlumbotClient:
    provider = "slumbot"

    def __init__(self, *, action_menu=DEFAULT_ACTION_MENU, transport=None, username=None, password=None):
        if bool(username) != bool(password):
            raise ValueError("Provide both Slumbot credentials or neither")
        self.action_menu = tuple(action_menu)
        self.transport = transport or JSONTransport("https://slumbot.com")
        self.token = None
        self.current = None
        self._counter = 0
        if username:
            response = self.transport.request("POST", "/slumbot/api/login",
                                              {"username": username, "password": password})
            self._token(response, required=True)

    def _token(self, response, *, required=False):
        if not isinstance(response, dict) or response.get("error_msg"):
            raise ProtocolError("Slumbot returned an invalid/error response")
        token = response.get("token")
        if token is not None:
            if not isinstance(token, str) or not token:
                raise ProtocolError("Invalid Slumbot token")
            self.token = token
        if required and not self.token:
            raise ProtocolError("Missing Slumbot token")

    @_mutation
    def new_hand(self):
        if self.current is not None and not self.current.terminal:
            raise ProtocolError("Finish the current hand before starting another")
        data = {} if self.token is None else {"token": self.token}
        response = self.transport.request("POST", "/slumbot/api/new_hand", data)
        self._token(response, required=True)
        self._counter += 1
        self.current = parse_slumbot(response, self._counter, self.action_menu)
        return self.current

    @_mutation
    def act(self, index):
        if self.current is None:
            raise ProtocolError("No current Slumbot hand")
        sent = self.current.action(index)
        kind, amount = sent
        action = kind if amount is None else f"b{amount}"
        response = self.transport.request("POST", "/slumbot/api/act", {"token": self.token, "incr": action})
        self._token(response, required=True)
        after = parse_slumbot(response, self.current.hand_id, self.action_menu)
        check_successor(self.current, after, sent)
        self.current = after
        return after


class GTOWizardClient:
    provider = "gtow"

    def __init__(self, api_key=None, *, action_menu=DEFAULT_ACTION_MENU, transport=None):
        if transport is None and not api_key:
            raise ValueError("GTO Wizard requires an approved API key")
        self.action_menu = tuple(action_menu)
        # Match the official GTO client: opponent computation may exceed 30s.
        self.transport = transport or GTOWizardTransport("https://researcher.gtowizard.com",
            headers={"X-API-Key": api_key}, timeout=180)
        self.current = None
        self.recoveries = 0

    @_mutation
    def new_hand(self):
        if self.current is not None and not self.current.terminal:
            raise ProtocolError("Finish the current hand before starting another")
        payload = self.transport.request("POST", "/hands", {"game_name": "HUNL 200BB"})
        self.current = parse_gtow(payload, self.action_menu)
        return self.current

    def _in_progress(self):
        result = self.transport.request("GET", "/hands/in-progress?" + urlencode({"game_name": "HUNL 200BB"}))
        if not isinstance(result, list) or any(not isinstance(p, dict) for p in result):
            raise ProtocolError("Invalid GTO Wizard in-progress response")
        return result

    def resume_hand(self, hand_id):
        matches = [p for p in self._in_progress() if p.get("hand_id") == hand_id]
        if len(matches) != 1:
            raise AmbiguousOutcome("Requested hand is not uniquely present among unfinished hands")
        self.current = parse_gtow(matches[0], self.action_menu)
        return self.current

    @_mutation
    def act(self, index):
        if self.current is None:
            raise ProtocolError("No current GTO Wizard hand")
        sent = self.current.action(index)
        kind, amount = sent
        body = {"action": kind}
        if amount is not None:
            body["amount"] = amount
        before = self.current
        try:
            payload = self.transport.request("POST", f"/hands/{before.hand_id}/act", body)
            after = parse_gtow(payload, self.action_menu, hero=before.hero)
            check_successor(before, after, sent)
        except AmbiguousOutcome:
            # Only continue if the authoritative read proves our action was applied.
            # An unchanged state may still have an in-flight request; never resend.
            matches = [p for p in self._in_progress() if p.get("hand_id") == before.hand_id]
            if len(matches) != 1:
                raise AmbiguousOutcome("Hand no longer in progress; terminal outcome unresolved, stopping") from None
            after = parse_gtow(matches[0], self.action_menu, hero=before.hero)
            try:
                check_successor(before, after, sent)
            except ProtocolError:
                raise AmbiguousOutcome("In-progress state does not prove the action was applied; stopping") from None
            self.recoveries += 1
        self.current = after
        return after
