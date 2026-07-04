// Package ring implements a small fixed-size ring buffer used to
// exercise the cAST chunker on Go source.
package ring

import (
	"errors"
	"fmt"
)

// ErrEmpty is returned when reading from an empty buffer.
var ErrEmpty = errors.New("ring: buffer is empty")

// Buffer is a fixed-capacity FIFO of strings.
type Buffer struct {
	items []string
	head  int
	size  int
}

// New allocates a Buffer with the given capacity.
func New(capacity int) (*Buffer, error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("ring: capacity must be positive, got %d", capacity)
	}
	return &Buffer{items: make([]string, capacity)}, nil
}

// Push appends an item, evicting the oldest entry when full.
func (b *Buffer) Push(item string) {
	tail := (b.head + b.size) % len(b.items)
	b.items[tail] = item
	if b.size < len(b.items) {
		b.size++
	} else {
		b.head = (b.head + 1) % len(b.items)
	}
}

// Pop removes and returns the oldest item.
func (b *Buffer) Pop() (string, error) {
	if b.size == 0 {
		return "", ErrEmpty
	}
	item := b.items[b.head]
	b.head = (b.head + 1) % len(b.items)
	b.size--
	return item, nil
}

// Len reports how many items are currently buffered.
func (b *Buffer) Len() int {
	return b.size
}

// Snapshot copies the buffered items in FIFO order.
func (b *Buffer) Snapshot() []string {
	out := make([]string, 0, b.size)
	for i := 0; i < b.size; i++ {
		out = append(out, b.items[(b.head+i)%len(b.items)])
	}
	return out
}
