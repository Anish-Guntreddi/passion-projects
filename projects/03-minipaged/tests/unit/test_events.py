from __future__ import annotations

from minipaged.simulation.events import EventLog, EventType


def test_empty_log_has_length_zero_and_is_monotonic():
    log = EventLog()
    assert len(log) == 0
    assert log.is_monotonic()


def test_record_appends_and_returns_the_event():
    log = EventLog()
    event = log.record(1.0, "req-1", EventType.ARRIVED, foo=1)
    assert len(log) == 1
    assert event.timestamp == 1.0
    assert event.request_id == "req-1"
    assert event.event_type == EventType.ARRIVED
    assert event.detail == {"foo": 1}


def test_iteration_yields_events_in_append_order():
    log = EventLog()
    log.record(0.0, "req-1", EventType.ARRIVED)
    log.record(1.0, "req-1", EventType.ADMITTED)
    log.record(2.0, "req-2", EventType.ARRIVED)
    types = [e.event_type for e in log]
    assert types == [EventType.ARRIVED, EventType.ADMITTED, EventType.ARRIVED]


def test_for_request_filters_by_request_id():
    log = EventLog()
    log.record(0.0, "req-1", EventType.ARRIVED)
    log.record(1.0, "req-2", EventType.ARRIVED)
    log.record(2.0, "req-1", EventType.ADMITTED)
    events = log.for_request("req-1")
    assert [e.event_type for e in events] == [EventType.ARRIVED, EventType.ADMITTED]
    assert all(e.request_id == "req-1" for e in events)


def test_of_type_filters_by_event_type():
    log = EventLog()
    log.record(0.0, "req-1", EventType.ARRIVED)
    log.record(1.0, "req-2", EventType.ARRIVED)
    log.record(2.0, "req-1", EventType.COMPLETED)
    events = log.of_type(EventType.ARRIVED)
    assert len(events) == 2
    assert all(e.event_type == EventType.ARRIVED for e in events)


def test_is_monotonic_true_for_nondecreasing_timestamps():
    log = EventLog()
    log.record(0.0, "req-1", EventType.ARRIVED)
    log.record(0.0, "req-2", EventType.ARRIVED)
    log.record(1.0, "req-1", EventType.ADMITTED)
    assert log.is_monotonic()


def test_is_monotonic_false_for_out_of_order_timestamps():
    log = EventLog()
    log.record(2.0, "req-1", EventType.ARRIVED)
    log.record(1.0, "req-2", EventType.ARRIVED)
    assert not log.is_monotonic()


def test_as_dicts_is_json_serializable_shape():
    log = EventLog()
    log.record(3.0, "req-1", EventType.ADMISSION_REJECTED, reason="insufficient_free_capacity")
    dicts = log.as_dicts()
    assert dicts == [
        {
            "timestamp": 3.0,
            "request_id": "req-1",
            "event_type": "admission_rejected",
            "reason": "insufficient_free_capacity",
        }
    ]
    import json

    json.dumps(dicts)  # must not raise
