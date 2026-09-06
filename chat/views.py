import hashlib
import json
import logging
import threading
from pathlib import Path
from time import perf_counter

from django.conf import settings
from django.core.cache import cache
from django.db import IntegrityError, transaction
from django.contrib import messages as django_messages
from django.contrib.admin.views.decorators import staff_member_required
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from rest_framework import status
from rest_framework.exceptions import ParseError, ValidationError
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
    throttle_classes,
)
from rest_framework.response import Response

from .api_permissions import WidgetAccessPermission, _resolve_request_origin
from .api_throttles import (
    WidgetEventsThrottle,
    WidgetFeedbackThrottle,
    WidgetHandoffThrottle,
    WidgetKeyRateThrottle,
    WidgetLeadsThrottle,
    WidgetRateThrottle,
)
from .conversation_tokens import make_conversation_token, verify_conversation_token
from .intents import detect_intent, normalize_question_key
from .models import (
    AdminNotification,
    AnalyticsEvent,
    Conversation,
    Feedback,
    HandoffRequest,
    Lead,
    Message,
    ProviderSettings,
    UnansweredQuestion,
    WidgetConfig,
)
from .services import CapacityLimitedError, CorpusConfigError
from rag.retriever import EmbeddingDimensionMismatchError
from .serializers import (
    ChatRequestSerializer,
    EventSerializer,
    FeedbackSerializer,
    HandoffSerializer,
    HistoryRequestSerializer,
    LeadSerializer,
)

logger = logging.getLogger(__name__)
_rag_service = None
_rag_artifact_signature = None
_rag_lock = threading.RLock()
_config_cache_key = "ai-support:widget-config"
ERROR_LOG = Path(settings.BASE_DIR) / "error_tracebacks.log"
EVENT_TYPES = {choice[0] for choice in AnalyticsEvent.EVENT_CHOICES}
CLIENT_EVENT_TYPES = {"widget_loaded", "fallback_triggered"}
MAX_HISTORY_MESSAGES = 50
FALLBACK_MESSAGE = (
    "در حال حاضر دستیار موقتاً در دسترس نیست. لطفاً چند لحظه دیگر تلاش کنید."
)


def _parse_json(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, Response(
            {"error": "invalid_json", "message": "Request body must be valid JSON."},
            status=400,
        )
    if not isinstance(data, dict):
        return None, Response(
            {
                "error": "invalid_payload",
                "message": "Request body must be a JSON object.",
            },
            status=400,
        )
    return data, None


def _validate(serializer):
    try:
        serializer.is_valid(raise_exception=True)
    except ParseError:
        return None, Response(
            {"error": "invalid_json", "message": "Request body must be valid JSON."},
            status=400,
        )
    except ValidationError as exc:
        detail = exc.detail
        if isinstance(detail, dict) and "message" in detail:
            errors = detail["message"]
            first = errors[0] if isinstance(errors, list) else errors
            code = getattr(first, "code", "")
            if code == "max_length":
                return None, Response(
                    {
                        "error": "message_too_long",
                        "message": "The message is too long.",
                    },
                    status=413,
                )
            if code in {"blank", "required"}:
                return None, Response(
                    {"error": "message_required", "message": "The message is required."},
                    status=400,
                )
            return None, Response(
                {"error": "invalid_message", "message": "The message is invalid."},
                status=400,
            )
        if isinstance(detail, dict) and "conversation_id" in detail:
            return None, Response(
                {
                    "error": "invalid_conversation",
                    "message": "The conversation id is invalid.",
                },
                status=400,
            )
        return None, Response(
            {"error": "invalid_payload", "message": "Invalid request payload."},
            status=400,
        )
    return serializer.validated_data, None


def get_widget_config():
    config = cache.get(_config_cache_key)
    if config is None:
        config = WidgetConfig.objects.first()
        if config is None:
            try:
                config = WidgetConfig.objects.create()
            except IntegrityError:
                config = WidgetConfig.objects.first()
        cache.set(_config_cache_key, config, timeout=10)
    return config


def widget_config_payload(config):
    # Resolve icon URL
    icon_url = ""
    if config.icon_type == "custom" and config.custom_icon_file:
        icon_url = config.custom_icon_file.url
    payload = {
        "business_name": config.business_name,
        "website_url": config.website_url,
        "title": config.title,
        "subtitle": config.subtitle,
        "greeting": config.greeting,
        "primary_color": config.primary_color,
        "secondary_color": config.secondary_color,
        "accent_color": config.accent_color,
        "header_badge": config.header_badge,
        "bot_avatar_text": config.bot_avatar_text,
        "input_placeholder": config.input_placeholder,
        "theme_mode": config.theme_mode,
        "dark_mode": config.dark_mode,
        "bubble_style": config.bubble_style,
        "panel_width": config.panel_width,
        "panel_height": config.panel_height,
        "border_radius": config.border_radius,
        "mobile_fullscreen": config.mobile_fullscreen,
        "logo_url": config.logo_url,
        "font_family": config.font_family,
        "font_size": config.font_size,
        "position": config.position,
        "positionVerticalOffset": config.position_vertical_offset,
        "positionHorizontalOffset": config.position_horizontal_offset,
        "iconType": config.icon_type,
        "defaultIconChoice": config.default_icon_choice,
        "customIconUrl": icon_url,
        "showFeedback": config.show_feedback,
        "show_powered_by": config.show_powered_by,
        "show_timestamp": config.show_timestamp,
        "show_avatar": config.show_avatar,
        "enable_sounds": config.enable_sounds,
        "enable_animations": config.enable_animations,
        "enable_conversation_memory": bool(config.enable_conversation_memory),
        "context_window": config.context_window,
        "suggestions": config.suggestions or [],
        "faq_url": config.faq_url,
        "privacy_url": config.privacy_url,
        "support_email": config.support_email,
        # Streaming & citations
        "enable_streaming": bool(config.enable_streaming),
        "show_citations": bool(config.show_citations),
        # Lead capture
        "enable_lead_capture": bool(config.enable_lead_capture),
        "lead_form_title": config.lead_form_title,
        "lead_form_description": config.lead_form_description,
        # Human handoff
        "enable_handoff": bool(config.enable_handoff),
        "handoff_trigger": config.handoff_trigger,
        "handoff_message": config.handoff_message,
        "handoff_urls": {
            "telegram": config.handoff_telegram_url or "",
            "whatsapp": config.handoff_whatsapp_url or "",
            "contact_form": config.handoff_contact_url or "",
        },
    }
    return payload


def demo_page(request):
    """Serve the demo page with the widget bundle matching the configured theme.

    static/demo.html hardcodes widget.js and cannot change at runtime; the
    themed page therefore lives in chat/templates/demo.html and this public
    view renders it with the same theme→file mapping as the installation
    snippet and the live preview.
    """
    config = WidgetConfig.objects.first()
    provider = ProviderSettings.objects.first()
    scheme = "https" if request.is_secure() else "http"
    base = f"{scheme}://{request.get_host()}"
    theme_label = config.get_widget_theme_display() if config else "کلاسیک"
    return render(request, "demo.html", {
        "base": base,
        "widget_file": config.widget_bundle_file() if config else "widget.js",
        "theme_label": theme_label,
        "widget_key": provider.widget_public_key if provider else "",
    })


CORPUS_ARTIFACTS = (
    Path(settings.BASE_DIR) / "Data" / "chunks.json",
    Path(settings.BASE_DIR) / "Data" / "metadata.json",
    Path(settings.BASE_DIR) / "Data" / "embeddings.npy",
)


def get_rag_service():
    global _rag_service, _rag_artifact_signature
    signature = tuple(
        (str(path), path.stat().st_mtime_ns if path.exists() else 0)
        for path in CORPUS_ARTIFACTS
    )
    with _rag_lock:
        config = get_widget_config()
        provider = _get_provider_row()
        signature += (
            config.updated_at.timestamp(),
            provider.updated_at.timestamp() if provider is not None else 0,
        )
        if _rag_service is None or signature != _rag_artifact_signature:
            from .document_pipeline import corpus_lock
            from .services import RAGService, get_provider_values

            try:
                with corpus_lock(timeout=10):
                    _rag_service = RAGService(
                        config=config,
                        provider=get_provider_values(),
                    )
            except RuntimeError as exc:
                raise CorpusConfigError(str(exc)) from exc
            _rag_artifact_signature = signature
        else:
            _rag_service.apply_config(config)
        return _rag_service


def _get_provider_row():
    return ProviderSettings.objects.first()


def _track_metrics(request, latency_ms=None, error=False):
    """Record metrics for admin dashboard."""
    event_type = "error_occurred" if error else "assistant_answered"
    metadata = {}
    if latency_ms is not None:
        metadata["latency_ms"] = latency_ms
    metadata["origin"] = _resolve_request_origin(request)[:200]
    try:
        AnalyticsEvent.objects.create(
            event_type=event_type,
            path=request.headers.get("Referer", "")[:1000],
            metadata=metadata,
        )
    except Exception:
        logger.warning("Failed to track analytics event", exc_info=True)


def _notify_if_needed(error_type, detail):
    """Create a notification for critical errors (rate-limited to once per 5 min)."""
    now = int(perf_counter())
    notify_key = f"monitor:notify:{error_type}:{now // 300}"
    if cache.get(notify_key):
        return
    cache.set(notify_key, True, timeout=300)
    severity = "critical" if error_type in ("corpus_config", "capacity_limited") else "warning"
    AdminNotification.objects.get_or_create(
        title=f"Chat error: {error_type}",
        severity=severity,
        defaults={"message": detail[:500]},
    )


def _visitor_key(request):
    """Stable abuse-tracking fingerprint (hashed; no raw IP stored)."""
    ip = request.META.get("REMOTE_ADDR", "")
    if getattr(settings, "TRUST_X_FORWARDED_FOR", False):
        forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
    agent = request.META.get("HTTP_USER_AGENT", "")[:200]
    digest = hashlib.sha256(
        f"{ip}|{agent}|{settings.SECRET_KEY or 'dev'}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def _get_or_create_conversation(request, conversation_id, page_url=""):
    """Return (conversation, created). Reuses a valid id or creates one."""
    if conversation_id:
        existing = Conversation.objects.filter(conversation_id=conversation_id).first()
        if existing is not None:
            return existing, False
    create_kwargs = {
        "origin": _resolve_request_origin(request)[:300],
        "path": (page_url or "")[:1000],
        "visitor_key": _visitor_key(request),
    }
    if conversation_id:
        create_kwargs["conversation_id"] = conversation_id
    try:
        conversation = Conversation.objects.create(**create_kwargs)
        return conversation, True
    except IntegrityError:
        existing = Conversation.objects.filter(
            conversation_id=conversation_id
        ).first()
        if existing is None:
            raise
        return existing, False


def _build_history(conversation, config):
    """Server-side conversation context (#16 + conversation memory toggle)."""
    if getattr(config, "enable_conversation_memory", True) is False:
        return []
    window = getattr(config, "context_window", 6) or 0
    if window <= 0:
        return []
    window = min(window, 20)
    messages = conversation.messages.order_by("-created_at")[:window]
    history = [
        {"role": message.role, "content": message.content[:1000]}
        for message in reversed(list(messages))
    ]
    return history


def _record_unanswered(question, reason, intent, conversation=None):
    """Upsert the unanswered-question ledger (#9)."""
    question = str(question).strip()[:2000]
    if not question:
        return
    try:
        question_hash = hashlib.sha256(
            normalize_question_key(question).encode("utf-8")
        ).hexdigest()
        row, _created = UnansweredQuestion.objects.get_or_create(
            question_hash=question_hash,
            defaults={
                "question": question,
                "count": 1,
                "reason": reason[:20],
                "last_intent": intent or "",
                "conversation": conversation,
            },
        )
        if not _created:
            UnansweredQuestion.objects.filter(pk=row.pk).update(
                count=row.count + 1,
                reason=reason[:20],
                last_intent=intent or row.last_intent,
                conversation=conversation or row.conversation_id,
                updated_at=None,  # auto_now requires save; update via save below
            )
            row.refresh_from_db()
            row.save(update_fields=("updated_at",))
    except Exception:
        logger.warning("Failed to record unanswered question", exc_info=True)


def _persist_user_message(conversation, content, intent):
    message = Message.objects.create(
        conversation=conversation,
        role="user",
        content=content,
        intent=intent,
    )
    conversation.message_count += 1
    conversation.last_intent = intent
    if not conversation.first_intent:
        conversation.first_intent = intent
    conversation.save(update_fields=("message_count", "last_intent", "first_intent", "updated_at"))
    return message


def _attach_rule_actions(question, intent, result):
    """Evaluate business rules and attach non-blocking actions to result."""
    try:
        from .business_rules import evaluate_rules

        actions = evaluate_rules(
            message=question,
            intent=intent,
            used_fallback=bool(result.get("used_fallback")),
            confidence=result.get("confidence"),
        )
        if actions:
            result["rule_actions"] = actions
    except Exception:
        logger.warning("Business rule evaluation failed", exc_info=True)
    return result


def _persist_assistant_message(conversation, result, latency_ms):
    message = Message.objects.create(
        conversation=conversation,
        role="assistant",
        content=result.get("answer", ""),
        intent=conversation.last_intent or "",
        sources=result.get("sources") or [],
        used_fallback=bool(result.get("used_fallback")),
        confidence=result.get("confidence"),
        latency_ms=latency_ms,
    )
    conversation.message_count += 1
    conversation.save(update_fields=("message_count", "updated_at"))
    return message


def _sse_event(event_name, data):
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@api_view(["GET"])
@authentication_classes([])
@permission_classes([])
@throttle_classes([])
def health(request):
    from django.db import connection

    checks = {"database": "ok", "corpus": "ok"}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        checks["database"] = "error"

    existing = [path.exists() for path in CORPUS_ARTIFACTS]
    if all(existing):
        checks["corpus"] = "ok"
    elif not any(existing):
        checks["corpus"] = "ok"
    else:
        checks["corpus"] = "error"
    healthy = all(value == "ok" for value in checks.values())
    return Response(
        {
            "status": "ok" if healthy else "degraded",
            "service": "ai-support-platform",
            "version": settings.APP_VERSION,
            "mode": "single-site",
            "checks": checks,
        },
        status=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@api_view(["GET"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetRateThrottle, WidgetKeyRateThrottle])
def widget_config(request):
    response = Response(widget_config_payload(get_widget_config()))
    response["Cache-Control"] = "public, max-age=10, stale-while-revalidate=60"
    return response


def _handle_chat_failure(request, exc):
    """Shared error handling for chat + chat_stream (feature #19)."""
    if isinstance(exc, (CorpusConfigError, EmbeddingDimensionMismatchError)):
        logger.error("Corpus/provider configuration error: %s", exc)
        _track_metrics(request, error=True)
        _notify_if_needed("corpus_config", str(exc))
        return Response(
            {
                "error": "corpus_config",
                "message": "The knowledge base needs attention. Please check the documents and provider settings.",
            },
            status=503,
        )
    if isinstance(exc, CapacityLimitedError):
        return Response(
            {
                "error": "capacity_limited",
                "message": "The assistant is busy. Please try again shortly.",
            },
            status=429,
            headers={"Retry-After": str(exc.retry_after)},
        )
    logger.exception("Chat request failed: %s", type(exc).__name__)
    try:
        if settings.DEBUG:
            with ERROR_LOG.open("a", encoding="utf-8") as error_file:
                error_file.write(f"---\n{type(exc).__name__}: {exc}\n")
    except OSError:
        pass
    _track_metrics(request, error=True)
    _notify_if_needed(type(exc).__name__, str(exc))
    return Response(
        {
            "error": "backend_error",
            "message": FALLBACK_MESSAGE,
        },
        status=503,
    )


def _normalize_rag_result(result):
    """Accept both the structured dict and a bare string answer.

    Keeps view logic resilient to custom adapters and older mocks that
    still return a plain string from the RAG layer.
    """
    if isinstance(result, str):
        return {
            "answer": result,
            "sources": [],
            "used_fallback": False,
            "reason": "",
            "confidence": None,
        }
    return result if isinstance(result, dict) else {"answer": ""}


@api_view(["POST"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetRateThrottle, WidgetKeyRateThrottle])
def chat(request):
    raw_data, parse_error = _parse_json(request)
    if parse_error:
        return parse_error
    if "message" in raw_data and not isinstance(raw_data["message"], str):
        return Response(
            {"error": "invalid_message", "message": "The message must be a string."},
            status=400,
        )
    data, validation_error = _validate(ChatRequestSerializer(data=raw_data))
    if validation_error:
        return validation_error

    config = get_widget_config()
    conversation, _created = _get_or_create_conversation(
        request, data.get("conversation_id", ""), data.get("page_url", "")
    )
    intent = detect_intent(data["message"])
    _persist_user_message(conversation, data["message"], intent)
    history = _build_history(conversation, config)

    started_at = perf_counter()
    try:
        result = _normalize_rag_result(
            get_rag_service().ask(data["message"], history=history)
        )
    except Exception as exc:
        return _handle_chat_failure(request, exc)

    answer = str(result.get("answer", "")).strip()
    if not answer:
        answer = FALLBACK_MESSAGE
        result["answer"] = answer
    result = _attach_rule_actions(data["message"], intent, result)
    latency_ms = round((perf_counter() - started_at) * 1000)
    _track_metrics(request, latency_ms=latency_ms)

    message = _persist_assistant_message(conversation, result, latency_ms)
    if result.get("used_fallback"):
        _record_unanswered(
            data["message"],
            result.get("reason") or "low_confidence",
            intent,
            conversation,
        )
        AnalyticsEvent.objects.create(
            event_type="fallback_triggered",
            path=request.headers.get("Referer", "")[:1000],
            metadata={"conversation_id": conversation.conversation_id},
        )

    response = Response(
        {
            "answer": answer,
            "conversation_id": conversation.conversation_id,
            "conversation_token": make_conversation_token(conversation.conversation_id),
            "message_id": message.pk,
            "citations": result.get("sources") or [],
            "intent": intent,
            "fallback": bool(result.get("used_fallback")),
            "latency_ms": latency_ms,
            "rule_actions": result.get("rule_actions") or [],
        }
    )
    response["Cache-Control"] = "no-store"
    return response


@api_view(["POST"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetRateThrottle, WidgetKeyRateThrottle])
def chat_stream(request):
    """Streaming chat over Server-Sent Events (#2).

    Event sequence:
      meta  → {conversation_id, conversation_token, intent}
      token → {t: "<text chunk>"}            (repeated)
      done  → {message_id, citations, fallback, latency_ms, ...}
      error → {code, message}                (only on failure)
    """
    raw_data, parse_error = _parse_json(request)
    if parse_error:
        return parse_error
    if "message" in raw_data and not isinstance(raw_data["message"], str):
        return Response(
            {"error": "invalid_message", "message": "The message must be a string."},
            status=400,
        )
    data, validation_error = _validate(ChatRequestSerializer(data=raw_data))
    if validation_error:
        return validation_error

    config = get_widget_config()
    conversation, _created = _get_or_create_conversation(
        request, data.get("conversation_id", ""), data.get("page_url", "")
    )
    intent = detect_intent(data["message"])
    _persist_user_message(conversation, data["message"], intent)
    history = _build_history(conversation, config)
    conversation_token = make_conversation_token(conversation.conversation_id)

    def event_stream():
        yield _sse_event(
            "meta",
            {
                "conversation_id": conversation.conversation_id,
                "conversation_token": conversation_token,
                "intent": intent,
            },
        )
        started_at = perf_counter()
        try:
            final_result = None
            for event in get_rag_service().stream_answer(
                data["message"], history=history
            ):
                if event["type"] == "token":
                    yield _sse_event("token", {"t": event["text"]})
                elif event["type"] == "done":
                    final_result = event.get("result") or {}
            if final_result is None:
                final_result = {"answer": "", "sources": [], "used_fallback": True, "confidence": 0}
            final_result = _normalize_rag_result(final_result)
            if not str(final_result.get("answer", "")).strip():
                final_result["answer"] = FALLBACK_MESSAGE
            final_result = _attach_rule_actions(data["message"], intent, final_result)
            latency_ms = round((perf_counter() - started_at) * 1000)
            message = _persist_assistant_message(conversation, final_result, latency_ms)
            _track_metrics(request, latency_ms=latency_ms)
            if final_result.get("used_fallback"):
                _record_unanswered(
                    data["message"],
                    final_result.get("reason") or "low_confidence",
                    intent,
                    conversation,
                )
            yield _sse_event(
                "done",
                {
                    "message_id": message.pk,
                    "conversation_id": conversation.conversation_id,
                    "conversation_token": conversation_token,
                    "citations": final_result.get("sources") or [],
                    "fallback": bool(final_result.get("used_fallback")),
                    "latency_ms": latency_ms,
                    "intent": intent,
                    "rule_actions": final_result.get("rule_actions") or [],
                },
            )
        except CapacityLimitedError:
            yield _sse_event(
                "error",
                {
                    "code": "capacity_limited",
                    "message": "The assistant is busy. Please try again shortly.",
                },
            )
        except (CorpusConfigError, EmbeddingDimensionMismatchError) as exc:
            logger.error("Corpus/provider configuration error: %s", exc)
            _track_metrics(request, error=True)
            _notify_if_needed("corpus_config", str(exc))
            yield _sse_event(
                "error",
                {
                    "code": "corpus_config",
                    "message": FALLBACK_MESSAGE,
                },
            )
        except Exception as exc:
            logger.exception("Chat stream failed: %s", type(exc).__name__)
            _track_metrics(request, error=True)
            _notify_if_needed(type(exc).__name__, str(exc))
            yield _sse_event(
                "error",
                {"code": "backend_error", "message": FALLBACK_MESSAGE},
            )

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-store"
    response["X-Accel-Buffering"] = "no"
    return response


@api_view(["GET"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetRateThrottle])
def history(request):
    """Conversation history — protected by the HMAC conversation token."""
    data, validation_error = _validate(
        HistoryRequestSerializer(data=request.query_params)
    )
    if validation_error:
        return validation_error
    token = data.get("token") or request.headers.get("X-Conversation-Token", "")
    if not verify_conversation_token(data["conversation_id"], token):
        return Response(
            {"error": "invalid_token", "message": "Conversation token is invalid."},
            status=403,
        )
    conversation = Conversation.objects.filter(
        conversation_id=data["conversation_id"]
    ).first()
    if conversation is None:
        return Response({"messages": []})
    messages = conversation.messages.order_by("created_at")[:MAX_HISTORY_MESSAGES]
    payload = [
        {
            "role": message.role,
            "content": message.content,
            "citations": message.sources or [],
            "created_at": message.created_at.isoformat(),
        }
        for message in messages
    ]
    response = Response(
        {
            "conversation_id": conversation.conversation_id,
            "messages": payload,
        }
    )
    response["Cache-Control"] = "no-store"
    return response


@api_view(["POST"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetEventsThrottle])
def events(request):
    raw_data, parse_error = _parse_json(request)
    if parse_error:
        return parse_error
    data, validation_error = _validate(EventSerializer(data=raw_data))
    if validation_error:
        return validation_error
    if data["event_type"] not in CLIENT_EVENT_TYPES:
        return Response({"error": "event_not_client_writable"}, status=403)
    try:
        AnalyticsEvent.objects.create(
            event_type=data["event_type"],
            path=request.headers.get("Referer", "")[:1000],
            metadata=data.get("metadata") or {},
        )
    except Exception:
        logger.warning("Failed to create analytics event", exc_info=True)
    return Response({"ok": True})


@api_view(["POST"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetFeedbackThrottle, WidgetKeyRateThrottle])
def feedback(request):
    """Per-message feedback (#10) with conversation-token ownership."""
    raw_data, parse_error = _parse_json(request)
    if parse_error:
        return parse_error
    data, validation_error = _validate(FeedbackSerializer(data=raw_data))
    if validation_error:
        return validation_error

    event_type = "answer_helpful" if data["helpful"] else "answer_not_helpful"
    metadata = {
        "question": data.get("question", "")[:500],
        "answer_preview": data.get("answer_preview", "")[:300],
        "session_id": data.get("session_id", "")[:100],
    }
    if data.get("comment"):
        metadata["comment"] = data["comment"][:500]

    feedback_row = None
    if data.get("message_id"):
        message = Message.objects.filter(pk=data["message_id"]).select_related(
            "conversation"
        ).first()
        if message is not None:
            conversation = message.conversation
            if conversation is not None and not verify_conversation_token(
                conversation.conversation_id,
                data.get("conversation_token") or request.headers.get("X-Conversation-Token", ""),
            ):
                return Response(
                    {"error": "invalid_token", "message": "Conversation token is invalid."},
                    status=403,
                )
            metadata["message_id"] = message.pk
            metadata["conversation_id"] = (
                conversation.conversation_id if conversation else ""
            )
            try:
                feedback_row, _created = Feedback.objects.update_or_create(
                    message=message,
                    defaults={
                        "helpful": data["helpful"],
                        "comment": data.get("comment", "")[:1000],
                    },
                )
            except Exception:
                logger.warning("Failed to persist per-message feedback", exc_info=True)

    try:
        AnalyticsEvent.objects.create(
            event_type=event_type,
            path=_resolve_request_origin(request)[:1000],
            metadata=metadata,
        )
    except Exception:
        # Log but don't fail the request — feedback is best-effort
        logger.warning(
            "Failed to persist feedback: %s error=%s",
            event_type,
            exc_info=True,
        )

    return Response({"ok": True, "feedback_saved": feedback_row is not None})


@api_view(["POST"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetLeadsThrottle, WidgetKeyRateThrottle])
def leads(request):
    """Lead capture (#11) — name + at least one contact channel."""
    raw_data, parse_error = _parse_json(request)
    if parse_error:
        return parse_error
    data, validation_error = _validate(LeadSerializer(data=raw_data))
    if validation_error:
        return validation_error

    # Honeypot hit — pretend success, store nothing.
    if data.get("honeypot_hit"):
        return Response({"ok": True})

    conversation = None
    if data.get("conversation_id"):
        if verify_conversation_token(
            data["conversation_id"],
            data.get("conversation_token") or request.headers.get("X-Conversation-Token", ""),
        ):
            conversation = Conversation.objects.filter(
                conversation_id=data["conversation_id"]
            ).first()

    lead = Lead.objects.create(
        name=data["name"],
        email=data.get("email", ""),
        phone=data.get("phone", ""),
        note=data.get("note", ""),
        source="widget_form",
        origin=_resolve_request_origin(request)[:300],
        conversation=conversation,
    )
    # Notify after commit: the background thread must not race the request
    # transaction (otherwise the row may be invisible to the worker).
    transaction.on_commit(
        lambda: threading.Thread(
            target=_notify_lead_async, args=(lead.pk,), daemon=True
        ).start()
    )
    return Response(
        {
            "ok": True,
            "message": "اطلاعات شما ثبت شد. به‌زودی با شما تماس می‌گیریم.",
        },
        status=201,
    )


def _notify_lead_async(lead_id):
    from .models import Lead as LeadModel
    from .notify import notify_lead_created

    try:
        lead = LeadModel.objects.get(pk=lead_id)
        notify_lead_created(lead)
        LeadModel.objects.filter(pk=lead_id).update(notified=True)
    except Exception:
        logger.warning("Lead notification failed", exc_info=True)


@api_view(["POST"])
@authentication_classes([])
@permission_classes([WidgetAccessPermission])
@throttle_classes([WidgetHandoffThrottle, WidgetKeyRateThrottle])
def handoff(request):
    """Human handoff (#12) — logs the request and returns available channels."""
    raw_data, parse_error = _parse_json(request)
    if parse_error:
        return parse_error
    data, validation_error = _validate(HandoffSerializer(data=raw_data))
    if validation_error:
        return validation_error

    conversation = None
    if data.get("conversation_id"):
        if verify_conversation_token(
            data["conversation_id"],
            data.get("conversation_token") or request.headers.get("X-Conversation-Token", ""),
        ):
            conversation = Conversation.objects.filter(
                conversation_id=data["conversation_id"]
            ).first()

    handoff_row = HandoffRequest.objects.create(
        conversation=conversation,
        channel=data["channel"],
        question=data["message"],
        origin=_resolve_request_origin(request)[:300],
    )
    if conversation is not None:
        Conversation.objects.filter(pk=conversation.pk).update(status="handed_off")
    transaction.on_commit(
        lambda: threading.Thread(
            target=_notify_handoff_async, args=(handoff_row.pk,), daemon=True
        ).start()
    )

    config = get_widget_config()
    channels = widget_config_payload(config).get("handoff_urls", {})
    support_email = config.handoff_email or config.support_email
    return Response(
        {
            "ok": True,
            "message": "درخواست شما ثبت شد؛ کارشناسان ما در تماس خواهند بود.",
            "channels": channels,
            "support_email": support_email,
        },
        status=201,
    )


def _notify_handoff_async(handoff_id):
    from .models import HandoffRequest as HandoffModel
    from .notify import notify_handoff_request

    try:
        row = HandoffModel.objects.get(pk=handoff_id)
        notify_handoff_request(row)
        HandoffModel.objects.filter(pk=handoff_id).update(notified=True)
    except Exception:
        logger.warning("Handoff notification failed", exc_info=True)


@staff_member_required
@require_POST
def _leads_update(request, pk):
    lead = get_object_or_404(Lead, pk=pk)
    action = request.POST.get("action", "")
    if action == "delete":
        lead.delete()
        django_messages.success(request, "سرنخ حذف شد.")
    else:
        django_messages.error(request, "عملیات نامعتبر.")
    return redirect("panel:leads")


@staff_member_required
@require_POST
def _handoff_update(request, pk):
    handoff = get_object_or_404(HandoffRequest, pk=pk)
    action = request.POST.get("action", "")
    if action == "progress":
        handoff.status = "in_progress"
        handoff.save(update_fields=("status",))
        django_messages.success(request, "وضعیت به «در جریان» تغییر کرد.")
    elif action == "done":
        handoff.status = "done"
        handoff.save(update_fields=("status",))
        django_messages.success(request, "درخواست به «انجام شد» تغییر کرد.")
    elif action == "delete":
        handoff.delete()
        django_messages.success(request, "درخواست حذف شد.")
    else:
        django_messages.error(request, "عملیات نامعتبر.")
    return redirect("panel:handoff")


# Backward compat aliases expected by panel_urls
leads_update = _leads_update
handoff_update = _handoff_update
