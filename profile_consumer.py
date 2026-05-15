# ============================================================
# profile_consumer.py — Phase 2 profile-update Kafka consumer
# Consumes the `profile_updates` topic produced by app.py and
# forwards each event to the Phase 2 profile-store worker.
#
# Run with:
#   set KAFKA_BOOTSTRAP_SERVERS=localhost:9092   # Windows
#   export KAFKA_BOOTSTRAP_SERVERS=localhost:9092  # Linux/macOS
#   python profile_consumer.py
# ============================================================

import json
import os
import signal
import sys
import time

KAFKA_BOOTSTRAP = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').strip()
KAFKA_TOPIC     = os.environ.get('KAFKA_PROFILE_TOPIC', 'profile_updates')
KAFKA_GROUP     = os.environ.get('KAFKA_CONSUMER_GROUP', 'profile-store-worker')


def handle_event(event):
    """Override / extend this with the Phase 2 profile-store update logic.
    For now we just log to stdout so the wiring can be verified end-to-end."""
    tid   = event.get('transaction_id')
    risk  = event.get('risk_score')
    dec   = event.get('decision')
    print(f"[profile_consumer] tid={tid} decision={dec} risk={risk:.4f}")


def main():
    try:
        from kafka import KafkaConsumer
    except ImportError:
        print("kafka-python is not installed. Run `pip install kafka-python`.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP} (topic={KAFKA_TOPIC})...")
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP.split(','),
            group_id=KAFKA_GROUP,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b.decode('utf-8')),
            consumer_timeout_ms=0,
        )
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}", file=sys.stderr)
        sys.exit(2)

    print("Consumer ready. Waiting for events (Ctrl+C to stop)...")

    def _shutdown(*_):
        print("\nShutting down consumer...")
        try:
            consumer.close()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _shutdown)

    for msg in consumer:
        try:
            handle_event(msg.value)
        except Exception as e:
            print(f"[profile_consumer] error handling event: {e}", file=sys.stderr)
            time.sleep(0.1)


if __name__ == '__main__':
    main()
