#!/usr/bin/env python3
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def load_workload(filename):
    """Load workload from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_token_distributions(requests):
    """Plot prompt and completion token distributions"""
    prompt_tokens = [r['prompt_tokens'] for r in requests]
    completion_tokens = [r['expected_completion_tokens'] for r in requests]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(prompt_tokens, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Prompt Tokens')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prompt Token Distribution')
    ax1.axvline(np.mean(prompt_tokens), color='red', linestyle='--', 
                label=f'Mean: {np.mean(prompt_tokens):.0f}')
    ax1.axvline(np.median(prompt_tokens), color='green', linestyle='--',
                label=f'Median: {np.median(prompt_tokens):.0f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.hist(completion_tokens, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Completion Tokens')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Completion Token Distribution')
    ax2.axvline(np.mean(completion_tokens), color='red', linestyle='--',
                label=f'Mean: {np.mean(completion_tokens):.0f}')
    ax2.axvline(np.median(completion_tokens), color='green', linestyle='--',
                label=f'Median: {np.median(completion_tokens):.0f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_arrival_pattern(requests):
    """Plot request arrival times"""
    arrival_times = [r['arrival_time'] for r in requests]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    sorted_times = sorted(arrival_times)
    ax1.plot(sorted_times, range(1, len(sorted_times) + 1), linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Cumulative Requests')
    ax1.set_title('Cumulative Request Arrivals')
    ax1.grid(alpha=0.3)
    
    window_size = max(1, len(requests) // 20)  # 5% window
    if len(sorted_times) > window_size:
        rates = []
        time_points = []
        for i in range(window_size, len(sorted_times)):
            time_diff = sorted_times[i] - sorted_times[i - window_size]
            if time_diff > 0:
                rate = window_size / time_diff
                rates.append(rate)
                time_points.append(sorted_times[i])
        
        ax2.plot(time_points, rates, linewidth=2, alpha=0.7)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Request Rate (req/s)')
        ax2.set_title(f'Request Rate Over Time (window={window_size} requests)')
        ax2.axhline(np.mean(rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rates):.2f} req/s')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_category_distribution(requests):
    """Plot category combination distribution"""
    categories = []
    for r in requests:
        if r.get('metadata'):
            prompt_cat = r['metadata'].get('prompt_category', 'unknown')
            completion_cat = r['metadata'].get('completion_category', 'unknown')
            categories.append(f"{prompt_cat}-{completion_cat}")
    
    if not categories:
        print("No category metadata found in workload")
        return None
    
    category_counts = Counter(categories)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = list(category_counts.keys())
    values = list(category_counts.values())
    percentages = [100 * v / len(requests) for v in values]
    
    bars = ax.bar(range(len(labels)), values, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    
    ax.set_xlabel('Category (Prompt-Completion)')
    ax.set_ylabel('Count')
    ax.set_title('Request Category Distribution')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_scatter_tokens(requests):
    prompt_tokens = [r['prompt_tokens'] for r in requests]
    completion_tokens = [r['expected_completion_tokens'] for r in requests]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = []
    categories = set()
    for r in requests:
        if r.get('metadata'):
            cat = f"{r['metadata'].get('prompt_category', 'unknown')}-{r['metadata'].get('completion_category', 'unknown')}"
            categories.add(cat)
            colors.append(cat)
        else:
            colors.append('unknown')
    
    # Create color map
    unique_cats = sorted(list(categories))
    color_map = {cat: f'C{i}' for i, cat in enumerate(unique_cats)}
    point_colors = [color_map.get(c, 'gray') for c in colors]
    
    scatter = ax.scatter(prompt_tokens, completion_tokens, 
                        c=point_colors, alpha=0.5, s=20)
    
    ax.set_xlabel('Prompt Tokens')
    ax.set_ylabel('Completion Tokens')
    ax.set_title('Prompt vs Completion Token Distribution')
    ax.grid(alpha=0.3)
    
    # Add legend if we have categories
    if unique_cats:
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=color_map[cat], markersize=8, label=cat)
                  for cat in unique_cats]
        ax.legend(handles=handles, title='Category', loc='best')
    
    plt.tight_layout()
    return fig


def print_text_summary(workload):
    """Print text-based summary of workload"""
    requests = workload['requests']
    
    print("\n" + "="*60)
    print("WORKLOAD SUMMARY")
    print("="*60)
    
    print(f"\nTotal Requests: {len(requests)}")
    
    # Token statistics
    prompt_tokens = [r['prompt_tokens'] for r in requests]
    completion_tokens = [r['expected_completion_tokens'] for r in requests]
    
    print(f"\nPrompt Tokens:")
    print(f"  Min: {min(prompt_tokens)}")
    print(f"  Max: {max(prompt_tokens)}")
    print(f"  Mean: {np.mean(prompt_tokens):.1f}")
    print(f"  Median: {np.median(prompt_tokens):.1f}")
    print(f"  Total: {sum(prompt_tokens):,}")
    
    print(f"\nCompletion Tokens:")
    print(f"  Min: {min(completion_tokens)}")
    print(f"  Max: {max(completion_tokens)}")
    print(f"  Mean: {np.mean(completion_tokens):.1f}")
    print(f"  Median: {np.median(completion_tokens):.1f}")
    print(f"  Total: {sum(completion_tokens):,}")
    
    # Arrival time statistics
    arrival_times = [r['arrival_time'] for r in requests]
    print(f"\nArrival Times:")
    print(f"  First: {min(arrival_times):.2f}s")
    print(f"  Last: {max(arrival_times):.2f}s")
    print(f"  Duration: {max(arrival_times) - min(arrival_times):.2f}s")
    
    if len(arrival_times) > 1:
        inter_arrival = np.diff(sorted(arrival_times))
        print(f"  Mean inter-arrival: {np.mean(inter_arrival):.3f}s")
        print(f"  Mean rate: {len(requests) / (max(arrival_times) - min(arrival_times)):.2f} req/s")
    
    # Category distribution
    categories = Counter()
    for r in requests:
        if r.get('metadata'):
            cat = f"{r['metadata'].get('prompt_category', 'unknown')}-{r['metadata'].get('completion_category', 'unknown')}"
            categories[cat] += 1
    
    if categories:
        print(f"\nCategory Distribution:")
        for cat, count in categories.most_common():
            print(f"  {cat}: {count} ({100*count/len(requests):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Visualize workload distribution')
    parser.add_argument('workload', default='workload.json', nargs='?',
                       help='Path to workload JSON file (default: workload.json)')
    parser.add_argument('--output', '-o', default='workload_analysis.png',
                       help='Output file for plots (default: workload_analysis.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print text summary, no plots')
    
    args = parser.parse_args()
    
    # Load workload
    print(f"Loading workload from {args.workload}...")
    workload = load_workload(args.workload)
    requests = workload['requests']
    
    # Print summary
    print_text_summary(workload)
    
    if args.summary_only:
        return

    print("\nGenerating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Token distributions
    print("  - Token distributions...")
    fig1 = plot_token_distributions(requests)
    
    # Plot 2: Arrival pattern
    print("  - Arrival patterns...")
    fig2 = plot_arrival_pattern(requests)
    
    # Plot 3: Category distribution
    print("  - Category distribution...")
    fig3 = plot_category_distribution(requests)
    
    # Plot 4: Scatter plot
    print("  - Token scatter plot...")
    fig4 = plot_scatter_tokens(requests)
    
    # Show plots
    if not args.no_show:
        plt.show()
    
    print(f"\nVisualizations complete!")
    print(f"Note: Save individual plots manually or use --no-show to skip display")


if __name__ == '__main__':
    main()


