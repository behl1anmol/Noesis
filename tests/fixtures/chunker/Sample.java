package dev.noesis.fixtures;

import java.util.ArrayDeque;
import java.util.Deque;

/** Sliding-window rate limiter used to exercise the cAST chunker on Java. */
public final class Sample {

    private final int maxRequests;
    private final long windowMillis;
    private final Deque<Long> timestamps = new ArrayDeque<>();

    public Sample(int maxRequests, long windowMillis) {
        if (maxRequests <= 0) {
            throw new IllegalArgumentException("maxRequests must be positive");
        }
        if (windowMillis <= 0) {
            throw new IllegalArgumentException("windowMillis must be positive");
        }
        this.maxRequests = maxRequests;
        this.windowMillis = windowMillis;
    }

    /** Returns true when the request at {@code nowMillis} is admitted. */
    public synchronized boolean tryAcquire(long nowMillis) {
        evictExpired(nowMillis);
        if (timestamps.size() >= maxRequests) {
            return false;
        }
        timestamps.addLast(nowMillis);
        return true;
    }

    /** Number of admits still available at {@code nowMillis}. */
    public synchronized int remaining(long nowMillis) {
        evictExpired(nowMillis);
        return maxRequests - timestamps.size();
    }

    private void evictExpired(long nowMillis) {
        long cutoff = nowMillis - windowMillis;
        while (!timestamps.isEmpty() && timestamps.peekFirst() <= cutoff) {
            timestamps.removeFirst();
        }
    }

    @Override
    public String toString() {
        return "Sample{max=" + maxRequests + ", windowMillis=" + windowMillis + "}";
    }
}
