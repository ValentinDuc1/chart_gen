#!/usr/bin/env python3
"""
Diagnostic script to find the Random.randint() error
"""

import traceback
import sys

print("Testing random_data_generator.py...\n")

try:
    from random_data_generator import RandomDataGenerator
    print("✓ Import successful")
    
    gen = RandomDataGenerator()
    print("✓ Created RandomDataGenerator instance")
    
    print("\nGenerating time_series_histogram spec...")
    spec = gen.generate_random_chart_spec(
        chart_type='time_series_histogram',
        filename_root='test'
    )
    
    print("✓ Spec generated successfully!")
    print(f"  Keys: {list(spec.keys())}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    error_str = str(e)
    
    if "Random.randint()" in error_str:
        print("\n⚠️  You have a naming conflict!")
        print("\nPossible causes:")
        print("1. You have a file named 'random.py' in your directory")
        print("   → Delete it or rename it")
        print("\n2. You have 'Random' class imported somewhere")
        print("   → Check your imports")
        print("\n3. The random module is shadowed")
        print("\nQuick fix:")
        print("  Check if you have random.py in your directory:")
        print("    ls -la random.py")
        print("\n  If yes, rename it:")
        print("    mv random.py my_random.py")
    
    sys.exit(1)

print("\n" + "="*60)
print("✅ NO ERRORS FOUND - Your files are working correctly!")
print("="*60)
