import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def print_scalars(path, name):
    print(f"--- Analyzing {name} ---")
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    # Find event file
    files = [f for f in os.listdir(path) if f.startswith('events.out.tfevents')]
    if not files:
        print("No event file found.")
        return
    
    event_file = os.path.join(path, files[0])
    print(f"Loading {event_file}...")
    
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        print(f"Found tags: {tags}")
        
        for tag in tags:
            # We are interested in losses
            if 'loss' in tag.lower() or 'score' in tag.lower() or 'iou' in tag.lower():
                events = ea.Scalars(tag)
                values = [e.value for e in events]
                steps = [e.step for e in events]
                if len(values) > 0:
                    print(f"\nTag: {tag}")
                    print(f"  Start: {values[0]:.4f} (Step {steps[0]})")
                    print(f"  End:   {values[-1]:.4f} (Step {steps[-1]})")
                    print(f"  Min:   {min(values):.4f}")
                    print(f"  Max:   {max(values):.4f}")
                    # Print a few points to show trend
                    if len(values) > 5:
                        indices = [int(i * (len(values)-1) / 5) for i in range(6)]
                        print("  Trend:", " -> ".join([f"{values[i]:.4f}" for i in indices]))
    except Exception as e:
        print(f"Error reading events: {e}")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print_scalars(os.path.join(path, 'train_iter'), 'Iteration Logs')
        print_scalars(os.path.join(path, 'train_epoch'), 'Epoch Logs')
    else:
        print("Usage: python analyze_logs.py <log_folder_path>")
