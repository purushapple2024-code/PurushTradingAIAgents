"""Market feeds: broker WebSockets, delayed polling fallback and the failover wrapper."""

from ptaa.fast.feeds.failover import FailoverFeed

__all__ = ["FailoverFeed"]
