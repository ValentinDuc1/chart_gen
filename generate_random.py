#!/usr/bin/env python3
"""
CLI Tool for Random Chart Generation
Quick command-line interface for generating random charts
"""

import argparse
import sys
from chart_generator import ChartGenerator
from random_data_generator import RandomDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate random charts for automated reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 random charts
  python generate_random.py -n 5

  # Generate specific chart types
  python generate_random.py -t line bar pie

  # Generate charts with custom output directory
  python generate_random.py -n 3 -o ./my_reports

  # Generate with specific seed for reproducibility
  python generate_random.py -n 10 --seed 42

  # Generate one of each chart type
  python generate_random.py --one-of-each
        """
    )
    
    parser.add_argument(
        '-n', '--num',
        type=int,
        default=1,
        help='Number of random charts to generate (default: 1)'
    )
    
    parser.add_argument(
        '-t', '--types',
        nargs='+',
        choices=['line', 'bar', 'horizontal_bar', 'pie', 'scatter', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution'],
        help='Specific chart types to generate (random if not specified)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./generated_charts',
        help='Output directory for charts (default: ./generated_charts)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--one-of-each',
        action='store_true',
        help='Generate one chart of each type'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='chart',
        help='Filename prefix (default: chart)'
    )
    
    args = parser.parse_args()
    
    # Initialize generators
    generator = ChartGenerator(output_dir=args.output)
    data_gen = RandomDataGenerator(seed=args.seed)
    
    print("🎨 Random Chart Generator")
    print("=" * 60)
    
    # Determine which charts to generate
    if args.one_of_each:
        chart_types = ['line', 'bar', 'horizontal_bar', 'pie', 'scatter', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution']
        print(f"Generating one chart of each type...")
    elif args.types:
        chart_types = args.types * ((args.num // len(args.types)) + 1)
        chart_types = chart_types[:args.num]
        print(f"Generating {args.num} chart(s) of types: {', '.join(set(args.types))}")
    else:
        chart_types = [None] * args.num
        print(f"Generating {args.num} random chart(s)...")
    
    print(f"Output directory: {args.output}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    print()
    
    # Generate charts
    results = []
    for i, chart_type in enumerate(chart_types, 1):
        filename_root = f"{args.prefix}_{i:03d}"
        
        spec = data_gen.generate_random_chart_spec(
            chart_type=chart_type,
            filename_root=filename_root
        )
        
        png_path, json_path = generator.generate_chart(**spec)
        results.append((spec['chart_type'], spec['title'], png_path, json_path))
        
        print(f"✓ Chart {i}/{len(chart_types)}: {spec['chart_type'].upper()}")
        print(f"  Title: {spec['title']}")
        print(f"  Files: {filename_root}.png / {filename_root}.json")
        print()
    
    # Summary
    print("=" * 60)
    print(f"✅ Successfully generated {len(results)} chart(s)!")
    print(f"📁 Location: {args.output}")
    print()
    
    # Count by type
    type_counts = {}
    for chart_type, _, _, _ in results:
        type_counts[chart_type] = type_counts.get(chart_type, 0) + 1
    
    print("Chart Type Summary:")
    for chart_type, count in sorted(type_counts.items()):
        print(f"  • {chart_type}: {count}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)