// Sample module used to exercise the cAST chunker on JavaScript source.

const DEFAULT_CAPACITY = 32;

class EventBus {
  constructor(capacity = DEFAULT_CAPACITY) {
    this.capacity = capacity;
    this.handlers = new Map();
  }

  on(topic, handler) {
    if (!this.handlers.has(topic)) {
      this.handlers.set(topic, []);
    }
    const list = this.handlers.get(topic);
    if (list.length >= this.capacity) {
      throw new Error(`too many handlers for ${topic}`);
    }
    list.push(handler);
    return () => this.off(topic, handler);
  }

  off(topic, handler) {
    const list = this.handlers.get(topic) || [];
    const index = list.indexOf(handler);
    if (index >= 0) {
      list.splice(index, 1);
    }
  }

  emit(topic, payload) {
    const list = this.handlers.get(topic) || [];
    let delivered = 0;
    for (const handler of list) {
      handler(payload);
      delivered += 1;
    }
    return delivered;
  }
}

function debounce(fn, waitMs) {
  let timer = null;
  return function debounced(...args) {
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      fn.apply(this, args);
    }, waitMs);
  };
}

const slugify = (text) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

module.exports = { EventBus, debounce, slugify };
