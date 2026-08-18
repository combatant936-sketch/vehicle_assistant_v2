import time
import psycopg2
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class PostgresSpanExporter(SpanExporter):

    def __init__(self, dsn, max_retries=10, retry_delay=2):
        # dsn example: "dbname=ragdb user=postgres password=... host=localhost port=5432"
        self.dsn = dsn
        self.conn = None
        for attempt in range(max_retries):
            try:
                self.conn = psycopg2.connect(dsn)
                break
            except psycopg2.OperationalError as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(retry_delay)

        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    start_time BIGINT,
                    end_time BIGINT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost REAL
                )
            """)
        self.conn.commit()

    def export(self, spans):
        with self.conn.cursor() as cur:
            for span in spans:
                attrs = dict(span.attributes or {})
                cur.execute(
                    """INSERT INTO spans
                       (name, start_time, end_time, input_tokens, output_tokens, cost)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        span.name,
                        span.start_time,
                        span.end_time,
                        attrs.get("input_tokens"),
                        attrs.get("output_tokens"),
                        attrs.get("cost"),
                    ),
                )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self, timeout_millis=None):
        return True