# ---------—---------—---------—---------—---------—---------—---------—
# Contains common helper functions for other programs
# ---------—---------—---------—---------—---------—---------—---------—

from constants import *

# Gets the content within the angle brackets <...>
def _get_event_content(event):
    return event[event.index('<') + 1: event.index('>')]

# Takes a MIDI note event and returns it in its letter representation
def simplify_note_event(event):
    content = _get_event_content(event)
    new_content = NUM_TO_NOTE[int(content) % 12]
    new_event = event[:event.index('<') + 1] + new_content + '>'
    return new_event

# Takes a complex chord event and returns it as either a major or
# minor chord
def simplify_chord_event(event):
    content = _get_event_content(event)

    # Check if the chord has a flat root (second position)
    new_content = content[:2] if len(
        content) >= 2 and content[1] == 'b' else content[0]

    # Technically, lower/upper already differentiates minor and major,
    # but we make this very explicit
    is_minor = content[0].islower()
    if is_minor:
        new_content += 'm'
    new_event = event[:event.index('<') + 1] + new_content + '>'
    return new_event
